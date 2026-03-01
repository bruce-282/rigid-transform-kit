"""rigid_transform_kit.robot — robot adapters and TCP pose builder."""

from .base import BaseRobotAdapter
from .fanuc import FanucAdapter
from .tcp import build_tcp_pose

__all__ = ["BaseRobotAdapter", "FanucAdapter", "build_tcp_pose"]
