# RL Solution for the Helsing Drone Challenge

This repository contains a one-file solution for the Helsing Drone Challenge.
The solution uses only openly available libraries, such that it does not require
any internal Helsing packages to run.

We use TorchRL to do the training. It assumes that the simulation is accessible
through a gRPC interface, with a limitless number of tokens.

The challenge is solved using the PPO algorithm, which is not ideal for this exact
use-case, mainly due to its sample inefficiency. However, it is more than enough
to get a near-optimal policy.

## Dependencies

- `torchrl==0.5.0`
- `tensordict`
- `torch`
- `numpy`
- `requests`
- `grpcio`
- `grpcio-tools`

## Usage

To generate the gRPC/protobuf files:

```bash
python -m grpc_tools.protoc --python_out=. --pyi_out=. --grpc_python_out=. ./drone.proto -I.
```

To start a training:

```bash
python train_ppo.py
```
