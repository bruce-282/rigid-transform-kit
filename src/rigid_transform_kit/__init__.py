"""rigid_transform_kit — frame-aware rigid transforms for vision-robot pipelines."""

from .core import Frame, RigidTransform
from .vision import CameraConfig, PickPoint
from .robot import BaseRobotAdapter, FanucAdapter, build_tcp_pose

__all__ = [
    "Frame",
    "RigidTransform",
    "CameraConfig",
    "PickPoint",
    "BaseRobotAdapter",
    "FanucAdapter",
    "build_tcp_pose",
]


def __getattr__(name: str):
    if name == "viz":
        from . import viz
        return viz
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
