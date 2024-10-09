# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# NO CHECKED-IN PROTOBUF GENCODE
# source: drone.proto
# Protobuf Python Version: 5.27.2
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import runtime_version as _runtime_version
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
_runtime_version.ValidateProtobufRuntimeVersion(
    _runtime_version.Domain.PUBLIC,
    5,
    27,
    2,
    '',
    'drone.proto'
)
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x0b\x64rone.proto\x12\x05\x64rone\"\x10\n\x0eObserveRequest\":\n\x10SimulationStates\x12&\n\x04sims\x18\x01 \x03(\x0b\x32\x18.drone.OngoingSimulation\"\xbe\x01\n\x11OngoingSimulation\x12\x0e\n\x06sim_id\x18\x01 \x01(\x05\x12\x13\n\x0b\x64\x65scription\x18\x02 \x01(\t\x12$\n\x0e\x64rone_location\x18\x03 \x01(\x0b\x32\x0c.drone.Point\x12\x1f\n\x08\x62oundary\x18\x04 \x01(\x0b\x32\r.drone.Region\x12\x1b\n\x04goal\x18\x05 \x01(\x0b\x32\r.drone.Region\x12 \n\x06status\x18\x06 \x01(\x0e\x32\x10.drone.SimStatus\"?\n\x0e\x44roneClientMsg\x12\x10\n\x08throttle\x18\x01 \x01(\x05\x12\x0c\n\x04roll\x18\x02 \x01(\x05\x12\r\n\x05pitch\x18\x03 \x01(\x05\"(\n\x05Point\x12\t\n\x01x\x18\x01 \x01(\x02\x12\t\n\x01y\x18\x02 \x01(\x02\x12\t\n\x01z\x18\x03 \x01(\x02\"R\n\x06Region\x12#\n\rminimal_point\x18\x01 \x01(\x0b\x32\x0c.drone.Point\x12#\n\rmaximal_point\x18\x02 \x01(\x0b\x32\x0c.drone.Point\"\x8d\x03\n\x0e\x44roneServerMsg\x12/\n\x05start\x18\x01 \x01(\x0b\x32\x1e.drone.DroneServerMsg.SimStartH\x00\x12.\n\x05\x65nded\x18\x02 \x01(\x0b\x32\x1d.drone.DroneServerMsg.SimOverH\x00\x12\x31\n\x06update\x18\x03 \x01(\x0b\x32\x1f.drone.DroneServerMsg.SimUpdateH\x00\x1a<\n\x07SimOver\x12\x0f\n\x07success\x18\x01 \x01(\x08\x12\x14\n\x07\x64\x65tails\x18\x02 \x01(\tH\x00\x88\x01\x01\x42\n\n\x08_details\x1an\n\x08SimStart\x12$\n\x0e\x64rone_location\x18\x01 \x01(\x0b\x32\x0c.drone.Point\x12\x1f\n\x08\x62oundary\x18\x02 \x01(\x0b\x32\r.drone.Region\x12\x1b\n\x04goal\x18\x03 \x01(\x0b\x32\r.drone.Region\x1a\x31\n\tSimUpdate\x12$\n\x0e\x64rone_location\x18\x01 \x01(\x0b\x32\x0c.drone.PointB\x06\n\x04\x64\x61ta*3\n\tSimStatus\x12\x0b\n\x07RUNNING\x10\x00\x12\n\n\x06\x46\x41ILED\x10\x01\x12\r\n\tSUCCEEDED\x10\x02\x32V\n\x0f\x44roneController\x12\x43\n\x0f\x44roneConnection\x12\x15.drone.DroneClientMsg\x1a\x15.drone.DroneServerMsg(\x01\x30\x01\x32L\n\rDroneObserver\x12;\n\x07Observe\x12\x15.drone.ObserveRequest\x1a\x17.drone.SimulationStates0\x01\x62\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'drone_pb2', _globals)
if not _descriptor._USE_C_DESCRIPTORS:
  DESCRIPTOR._loaded_options = None
  _globals['_SIMSTATUS']._serialized_start=884
  _globals['_SIMSTATUS']._serialized_end=935
  _globals['_OBSERVEREQUEST']._serialized_start=22
  _globals['_OBSERVEREQUEST']._serialized_end=38
  _globals['_SIMULATIONSTATES']._serialized_start=40
  _globals['_SIMULATIONSTATES']._serialized_end=98
  _globals['_ONGOINGSIMULATION']._serialized_start=101
  _globals['_ONGOINGSIMULATION']._serialized_end=291
  _globals['_DRONECLIENTMSG']._serialized_start=293
  _globals['_DRONECLIENTMSG']._serialized_end=356
  _globals['_POINT']._serialized_start=358
  _globals['_POINT']._serialized_end=398
  _globals['_REGION']._serialized_start=400
  _globals['_REGION']._serialized_end=482
  _globals['_DRONESERVERMSG']._serialized_start=485
  _globals['_DRONESERVERMSG']._serialized_end=882
  _globals['_DRONESERVERMSG_SIMOVER']._serialized_start=651
  _globals['_DRONESERVERMSG_SIMOVER']._serialized_end=711
  _globals['_DRONESERVERMSG_SIMSTART']._serialized_start=713
  _globals['_DRONESERVERMSG_SIMSTART']._serialized_end=823
  _globals['_DRONESERVERMSG_SIMUPDATE']._serialized_start=825
  _globals['_DRONESERVERMSG_SIMUPDATE']._serialized_end=874
  _globals['_DRONECONTROLLER']._serialized_start=937
  _globals['_DRONECONTROLLER']._serialized_end=1023
  _globals['_DRONEOBSERVER']._serialized_start=1025
  _globals['_DRONEOBSERVER']._serialized_end=1101
# @@protoc_insertion_point(module_scope)
