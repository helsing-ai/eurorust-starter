from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class SimStatus(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    RUNNING: _ClassVar[SimStatus]
    FAILED: _ClassVar[SimStatus]
    SUCCEEDED: _ClassVar[SimStatus]
RUNNING: SimStatus
FAILED: SimStatus
SUCCEEDED: SimStatus

class ObserveRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class SimulationStates(_message.Message):
    __slots__ = ("sims",)
    SIMS_FIELD_NUMBER: _ClassVar[int]
    sims: _containers.RepeatedCompositeFieldContainer[OngoingSimulation]
    def __init__(self, sims: _Optional[_Iterable[_Union[OngoingSimulation, _Mapping]]] = ...) -> None: ...

class OngoingSimulation(_message.Message):
    __slots__ = ("sim_id", "description", "drone_location", "boundary", "goal", "status")
    SIM_ID_FIELD_NUMBER: _ClassVar[int]
    DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    DRONE_LOCATION_FIELD_NUMBER: _ClassVar[int]
    BOUNDARY_FIELD_NUMBER: _ClassVar[int]
    GOAL_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    sim_id: int
    description: str
    drone_location: Point
    boundary: Region
    goal: Region
    status: SimStatus
    def __init__(self, sim_id: _Optional[int] = ..., description: _Optional[str] = ..., drone_location: _Optional[_Union[Point, _Mapping]] = ..., boundary: _Optional[_Union[Region, _Mapping]] = ..., goal: _Optional[_Union[Region, _Mapping]] = ..., status: _Optional[_Union[SimStatus, str]] = ...) -> None: ...

class DroneClientMsg(_message.Message):
    __slots__ = ("throttle", "roll", "pitch")
    THROTTLE_FIELD_NUMBER: _ClassVar[int]
    ROLL_FIELD_NUMBER: _ClassVar[int]
    PITCH_FIELD_NUMBER: _ClassVar[int]
    throttle: int
    roll: int
    pitch: int
    def __init__(self, throttle: _Optional[int] = ..., roll: _Optional[int] = ..., pitch: _Optional[int] = ...) -> None: ...

class Point(_message.Message):
    __slots__ = ("x", "y", "z")
    X_FIELD_NUMBER: _ClassVar[int]
    Y_FIELD_NUMBER: _ClassVar[int]
    Z_FIELD_NUMBER: _ClassVar[int]
    x: float
    y: float
    z: float
    def __init__(self, x: _Optional[float] = ..., y: _Optional[float] = ..., z: _Optional[float] = ...) -> None: ...

class Region(_message.Message):
    __slots__ = ("minimal_point", "maximal_point")
    MINIMAL_POINT_FIELD_NUMBER: _ClassVar[int]
    MAXIMAL_POINT_FIELD_NUMBER: _ClassVar[int]
    minimal_point: Point
    maximal_point: Point
    def __init__(self, minimal_point: _Optional[_Union[Point, _Mapping]] = ..., maximal_point: _Optional[_Union[Point, _Mapping]] = ...) -> None: ...

class DroneServerMsg(_message.Message):
    __slots__ = ("start", "ended", "update")
    class SimOver(_message.Message):
        __slots__ = ("success", "details")
        SUCCESS_FIELD_NUMBER: _ClassVar[int]
        DETAILS_FIELD_NUMBER: _ClassVar[int]
        success: bool
        details: str
        def __init__(self, success: bool = ..., details: _Optional[str] = ...) -> None: ...
    class SimStart(_message.Message):
        __slots__ = ("drone_location", "boundary", "goal")
        DRONE_LOCATION_FIELD_NUMBER: _ClassVar[int]
        BOUNDARY_FIELD_NUMBER: _ClassVar[int]
        GOAL_FIELD_NUMBER: _ClassVar[int]
        drone_location: Point
        boundary: Region
        goal: Region
        def __init__(self, drone_location: _Optional[_Union[Point, _Mapping]] = ..., boundary: _Optional[_Union[Region, _Mapping]] = ..., goal: _Optional[_Union[Region, _Mapping]] = ...) -> None: ...
    class SimUpdate(_message.Message):
        __slots__ = ("drone_location",)
        DRONE_LOCATION_FIELD_NUMBER: _ClassVar[int]
        drone_location: Point
        def __init__(self, drone_location: _Optional[_Union[Point, _Mapping]] = ...) -> None: ...
    START_FIELD_NUMBER: _ClassVar[int]
    ENDED_FIELD_NUMBER: _ClassVar[int]
    UPDATE_FIELD_NUMBER: _ClassVar[int]
    start: DroneServerMsg.SimStart
    ended: DroneServerMsg.SimOver
    update: DroneServerMsg.SimUpdate
    def __init__(self, start: _Optional[_Union[DroneServerMsg.SimStart, _Mapping]] = ..., ended: _Optional[_Union[DroneServerMsg.SimOver, _Mapping]] = ..., update: _Optional[_Union[DroneServerMsg.SimUpdate, _Mapping]] = ...) -> None: ...
