"""
rigid_transform_kit.robot.fanuc
=================================
FANUC robot adapter — reference implementation.

FANUC convention:
    Position: (X, Y, Z) in mm
    Orientation: (W, P, R) — intrinsic ZYX Euler angles in degrees
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
from scipy.spatial.transform import Rotation

from ..core import Frame, RigidTransform
from .base import BaseRobotAdapter


class FanucAdapter(BaseRobotAdapter):
    """FANUC robot adapter with suction cup redundancy resolution.

    Parameters
    ----------
    tool_z_offset : float
        Flange -> TCP distance along Z (mm).
    tool_rotation : np.ndarray or None
        Additional rotation in tool definition (3x3).
    pos_unit : str
        "m" or "mm" — output position unit for to_robot_command.
    """

    def __init__(
        self,
        tool_z_offset: float = 0.0,
        tool_rotation: np.ndarray | None = None,
        pos_unit: str = "mm",
    ):
        super().__init__(tool_z_offset=tool_z_offset, tool_rotation=tool_rotation)
        self.pos_unit = pos_unit

    def resolve_redundancy(self, T_base2tcp: RigidTransform) -> RigidTransform:
        """Round suction cup: flip Z 180 if TCP X-axis aligns with base X.

        This avoids wrist joint limits and cable tangling on FANUC arms.
        """
        tcp_x = T_base2tcp.R[:, 0]
        base_x = np.array([1.0, 0.0, 0.0])

        if np.dot(base_x, tcp_x) > 0:
            Rz_180 = np.diag([-1.0, -1.0, 1.0])
            R_new = T_base2tcp.R @ Rz_180
            return RigidTransform.from_Rt(
                R_new, T_base2tcp.t,
                T_base2tcp.from_frame, T_base2tcp.to_frame,
            )
        return T_base2tcp

    def to_robot_command(self, T_base2flange: RigidTransform) -> Dict[str, Any]:
        assert T_base2flange.from_frame == Frame.BASE
        assert T_base2flange.to_frame == Frame.FLANGE or T_base2flange.to_frame == Frame.TCP

        cmd = T_base2flange.to_xyzwpr(degrees=True)
        if self.pos_unit == "m":
            cmd["X"] /= 1000.0
            cmd["Y"] /= 1000.0
            cmd["Z"] /= 1000.0
        return cmd
