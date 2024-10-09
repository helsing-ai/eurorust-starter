from collections import defaultdict
import itertools
import logging
import time
from itertools import count
from typing import Final
import numpy as np
import requests
import queue

import grpc
import torch
from tensordict import TensorDict
from tensordict.nn import NormalParamExtractor, TensorDictModule
from torch import nn
from torch.optim.adam import Adam
from torchrl.collectors import SyncDataCollector
from torchrl.envs import EnvBase, ParallelEnv
from torchrl.modules import ProbabilisticActor, TanhNormal
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from torchrl.data import (
    BinaryDiscreteTensorSpec,
    BoundedTensorSpec,
    CompositeSpec,
    UnboundedContinuousTensorSpec,
)


import drone_pb2
import drone_pb2_grpc

log: Final = logging.getLogger(__name__)

GRPC_SERVER_ADDRESS = "157.230.103.230:10301"
TOKEN_NAME = "fancy_rl"
NUM_TOKENS_PER_ENV = 5
NUM_ENVS = 128
HIDDEN_SIZE = 64


def normalize(value, range, new_range):
    new_value = ((value - range[0]) / (range[1] - range[0])) * (
        new_range[1] - new_range[0]
    ) + new_range[0]
    return new_value


class Drone:
    def __init__(self, auth: str, num_tokens: int) -> None:
        self._tokens = itertools.cycle(auth + f"_{i}" for i in range(num_tokens))
        self._cmd_queue = queue.SimpleQueue()

        # Get initial condition.
        while True:
            try:
                self._channel = grpc.insecure_channel(GRPC_SERVER_ADDRESS)
                self._controller_stub = drone_pb2_grpc.DroneControllerStub(
                    self._channel
                )
                self._current_token = next(self._tokens)

                # # Generate token.
                # headers = {"Authorization": f"Bearer {self._current_token}"}
                # requests.post(
                #     f"http://{GRPC_SERVER_ADDRESS}",
                #     headers=headers,
                # )

                metadata = (("authorization", f"Bearer {self._current_token}"),)
                self._controller_stream = self._controller_stub.DroneConnection(
                    iter(self._cmd_queue.get, None), metadata=metadata
                )

                response: drone_pb2.DroneServerMsg = next(self._controller_stream)

                break

            except grpc.RpcError as e:
                print(f"[{self._current_token}] gRPC error: {e}")
                details: str = e.details()
                if "You are in a cool-off period." in details:
                    continue

                time.sleep(5)
                continue

        if not response.HasField("start"):
            raise RuntimeError("Unexpected")

        self._location = response.start.drone_location
        self._boundary = response.start.boundary
        self._goal_region = response.start.goal
        self._goal = drone_pb2.Point(
            x=self._goal_region.minimal_point.x
            + (self._goal_region.maximal_point.x - self._goal_region.minimal_point.x)
            * 0.5,
            y=self._goal_region.minimal_point.y
            + (self._goal_region.maximal_point.y - self._goal_region.minimal_point.y)
            * 0.5,
            z=self._goal_region.minimal_point.z
            + (self._goal_region.maximal_point.z - self._goal_region.minimal_point.z)
            * 0.5,
        )
        self._truncated = False
        self._info = ""

    def __del__(self):
        self._channel.close()

    @property
    def token(self):
        return self._current_token

    @property
    def info(self):
        return self._info

    @property
    def location(self):
        return self._location.x, self._location.y, self._location.z

    @property
    def goal(self):
        return self._goal.x, self._goal.y, self._goal.z

    @property
    def truncated(self):
        return self._truncated

    def step(self, throttle: int, roll: int, pitch: int):
        self._cmd_queue.put(
            drone_pb2.DroneClientMsg(
                throttle=throttle,
                roll=roll,
                pitch=pitch,
            )
        )

        response: drone_pb2.DroneServerMsg = next(self._controller_stream)

        if response.HasField("update"):
            self._location = response.update.drone_location

        elif response.HasField("ended"):
            self._truncated = True
            self._info = (
                f"success: {response.ended.success}, details: {response.ended.details}"
            )

        else:
            raise RuntimeError("Unexpected.")


class DroneEnv(EnvBase):  # type: ignore[misc]
    def __init__(self, env_i: int) -> None:
        self._env_i = env_i
        self._batch_size = ()
        self._dtype = torch.float

        super().__init__(device=torch.device("cpu"), batch_size=self._batch_size)

        self._drone: Drone = None

        self.observation_spec = CompositeSpec(
            observation=UnboundedContinuousTensorSpec(
                shape=(*self._batch_size, 9), dtype=self._dtype
            ),
            info=CompositeSpec(
                location_cost=UnboundedContinuousTensorSpec(
                    shape=(*self._batch_size, 1), dtype=self._dtype
                ),
                location_reward=UnboundedContinuousTensorSpec(
                    shape=(*self._batch_size, 1), dtype=self._dtype
                ),
                energy_cost=UnboundedContinuousTensorSpec(
                    shape=(*self._batch_size, 1), dtype=self._dtype
                ),
                energy_reward=UnboundedContinuousTensorSpec(
                    shape=(*self._batch_size, 1), dtype=self._dtype
                ),
                shape=(*self._batch_size,),
            ),
            shape=(*self._batch_size,),
        )
        self.action_spec = CompositeSpec(
            action=BoundedTensorSpec(
                -1, 1, shape=(*self._batch_size, 3), dtype=self._dtype
            ),
            shape=(*self._batch_size,),
        )
        self.reward_spec = UnboundedContinuousTensorSpec(
            shape=(*self._batch_size, 1), dtype=self._dtype
        )
        self.done_spec = CompositeSpec(
            done=BinaryDiscreteTensorSpec(
                1, shape=(*self._batch_size, 1), dtype=torch.bool
            ),
            terminated=BinaryDiscreteTensorSpec(
                1, shape=(*self._batch_size, 1), dtype=torch.bool
            ),
            truncated=BinaryDiscreteTensorSpec(
                1, shape=(*self._batch_size, 1), dtype=torch.bool
            ),
            shape=(*self._batch_size,),
        )

    def _reset(self, _: TensorDict) -> TensorDict:
        self._prev_action = (0, 0, 0)
        self._prev_location = (0, 0, 0)

        # Generate a token name.
        auth = TOKEN_NAME + (f"_{self._env_i}" if self._env_i is not None else "")
        self._drone = Drone(
            auth, num_tokens=NUM_TOKENS_PER_ENV if self._env_i is not None else 1
        )
        print(f"[Env {self._env_i}] token: '{self._drone.token}'")

        observation, _, info = self._get_observation_reward_info(
            self._drone.location,
            self._prev_location,
            self._drone.goal,
            (0, 0, 0),
            (0, 0, 0),
        )
        terminated = False
        truncated = self._drone.truncated

        next_tensordict = TensorDict(
            {
                "observation": observation,
                "info": info,
                "done": [terminated or truncated],
                "terminated": [terminated],
                "truncated": [truncated],
            },
            device=torch.device("cpu"),
        )
        return next_tensordict

    def _step(self, tensordict: TensorDict) -> TensorDict:
        # Network output is [-1, 1]
        action_raw = (
            tensordict["action"][0].item(),
            tensordict["action"][1].item(),
            tensordict["action"][2].item(),
        )
        throttle = int(normalize(action_raw[0], (-1, 1), (0, 100)))
        roll = int(normalize(action_raw[1], (-1, 1), (-45, 45)))
        pitch = int(normalize(action_raw[2], (-1, 1), (-45, 45)))
        action = throttle, roll, pitch

        self._drone.step(*action)

        observation, reward, info = self._get_observation_reward_info(
            self._drone.location,
            self._prev_location,
            self._drone.goal,
            action,
            self._prev_action,
        )
        terminated = False
        truncated = self._drone.truncated

        self._prev_action = action
        self._prev_location = self._drone.location

        print(
            f"[Env {self._env_i}]",
            "Step:",
            action,
            tuple(f"{x:.2f}" for x in self._drone.location),
            tuple(f"{x:.2f}" for x in self._drone.goal),
            f"{reward.item():.3f}",
            self._drone.info,
        )

        next_tensordict = TensorDict(
            {
                "observation": observation,
                "info": info,
                "reward": [reward],
                "done": [terminated or truncated],
                "terminated": [terminated],
                "truncated": [truncated],
            },
            device=torch.device("cpu"),
        )

        return next_tensordict

    def _get_observation_reward_info(
        self,
        location: tuple[float, float, float],
        prev_location: tuple[float, float, float],
        goal: tuple[float, float, float],
        action: tuple[float, float, float],
        prev_action: tuple[float, float, float],
    ) -> TensorDict:
        # Observation
        location_error = np.array(
            [
                (goal[0] - location[0]),
                (goal[1] - location[1]),
                (goal[2] - location[2]),
            ]
        )
        location_error_encoding = location_error / 100

        direction = np.array(
            [
                (location[0] - prev_location[0]),
                (location[1] - prev_location[1]),
                (location[2] - prev_location[2]),
            ]
        )
        direction_encoding = direction / 100
        prev_action_encoding = (
            np.array(
                [
                    prev_action[0],
                    prev_action[1],
                    prev_action[2],
                ]
            )
            / 100
        )

        observation = torch.as_tensor(
            [
                *location_error_encoding,
                *direction_encoding,
                *prev_action_encoding,
            ],
            dtype=self._dtype,
        )

        # Reward
        location_error_cost = np.linalg.norm(location_error)
        location_error_reward = -1e-1 * location_error_cost
        energy_cost = action[0] ** 2 + action[1] ** 2 + action[2] ** 2
        energy_reward = 0 * energy_cost
        reward = torch.tensor(location_error_reward + energy_reward, dtype=self._dtype)

        # Info
        info = TensorDict(
            {
                "location_cost": location_error_cost,
                "location_reward": location_error_reward,
                "energy_cost": energy_cost,
                "energy_reward": energy_reward,
            }
        ).to(self._dtype)

        return observation, reward, info

    def _set_seed(self, seed):
        raise NotImplementedError


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    metrics = defaultdict(list)

    # Environment
    def _env_make(env_i: int) -> EnvBase:
        env = DroneEnv(env_i=env_i)
        return env

    # Training
    env = ParallelEnv(
        NUM_ENVS,
        _env_make,
        device=torch.device("cpu"),
        create_env_kwargs=[{"env_i": env_i} for env_i in range(NUM_ENVS)],
    )

    # Networks
    observation_size = env.observation_spec["observation"].shape[-1]
    action_size = env.action_spec.shape[-1]

    # Actor critic with separate networks.
    policy_net = TensorDictModule(
        module=nn.Sequential(
            *(
                nn.Linear(observation_size, HIDDEN_SIZE),
                nn.ReLU(),
                nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
                nn.ReLU(),
                nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
                nn.ReLU(),
                nn.Linear(HIDDEN_SIZE, 2 * action_size),
                NormalParamExtractor(),
            )
        ),
        in_keys=["observation"],
        out_keys=["loc", "scale"],
    ).to(torch.device("cpu"))

    actor_module = ProbabilisticActor(
        module=policy_net,
        in_keys=["loc", "scale"],
        out_keys=["action"],
        distribution_class=TanhNormal,
        return_log_prob=True,
    )

    value_net = TensorDictModule(
        module=nn.Sequential(
            nn.Linear(observation_size, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, 1),
        ),
        in_keys=["observation"],
        out_keys=["state_value"],
    ).to(torch.device("cpu"))
    critic_module = value_net

    checkpoint = torch.load("./drone_checkpoint_final.pt")
    if checkpoint:
        actor_module.load_state_dict(checkpoint["actor_module"])
        critic_module.load_state_dict(checkpoint["critic_module"])

    # Data Collection
    collector = SyncDataCollector(
        env, actor_module, frames_per_batch=512, device=torch.device("cpu")
    )
    collector_iter = collector.iterator()

    # Trainer
    advantage_module = GAE(gamma=0.99, lmbda=0.95, value_network=critic_module)

    loss_module = ClipPPOLoss(
        actor_module, critic_module, entropy_coef=0.01, normalize_advantage=True
    )

    optimizer = Adam(loss_module.parameters(), lr=1e-4)

    for step_num in count():
        start_time = time.perf_counter()

        # Trick to account for time needed to sample batch from the environment.
        batch = next(collector_iter)

        loss: TensorDict
        loss_sum: TensorDict

        for _ in range(8):
            with torch.no_grad():
                advantage_module(batch)
            loss = loss_module(batch)

            loss_sum = (
                loss["loss_objective"] + loss["loss_critic"] + loss["loss_entropy"]
            )
            loss_sum.backward()

            torch.nn.utils.clip_grad_norm_(loss_module.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        train_reward_mean = batch["next", "reward"].mean().item()
        metrics["train/time"] = time.perf_counter() - start_time
        metrics["train/loss_objective"] = loss["loss_objective"].item()
        metrics["train/loss_critic"] = loss["loss_critic"].item()
        metrics["train/loss_entropy"] = loss["loss_entropy"].item()
        metrics["train/loss"] = loss_sum.item()
        metrics["train/reward_mean"] = train_reward_mean

        log.info(f"STEP {step_num}")
        log.info(f"  time: {time.perf_counter() - start_time:.5f}")
        log.info(f"  loss objective: {loss['loss_objective'].item():.5f}")
        log.info(f"  loss critic: {loss['loss_critic'].item():.5f}")
        log.info(f"  loss entropy: {loss['loss_entropy'].item():.5f}")

        # TODO ideally one would log the metrics collected here (e.g., to neptune
        # or tensorboard). This was removed to make the code not dependent on
        # Helsing's internal tools.


if __name__ == "__main__":
    main()
