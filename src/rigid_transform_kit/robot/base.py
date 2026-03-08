"""
rigid_transform_kit.robot.base
================================
Abstract robot adapter interface.

Robot team implements vendor-specific subclasses.
Vision team delivers T_base2tcp; adapter handles everything after.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

import numpy as np

from ..core import Frame, RigidTransform


class BaseRobotAdapter(ABC):
    """Abstract interface for robot-vendor-specific logic.

    Subclass contract — implement these two:
        1. resolve_redundancy()    -> handle gripper symmetry (e.g. suction Z-180)
        2. to_robot_command()      -> vendor-specific command format

    ``get_tool_transform()`` is provided as a concrete default using
    ``tool_z_offset`` and ``tool_rotation`` set in ``__init__``.
    Override it if your tool definition is more complex.

    The template method `plan_pick()` chains them in order.
    """

    def __init__(
        self,
        tool_z_offset: float = 0.0,
        tool_rotation: np.ndarray | None = None,
    ):
        self.tool_z_offset = tool_z_offset
        self.tool_rotation = tool_rotation

    # ---- tool (concrete, override if needed) ----

    def get_tool_transform(self) -> RigidTransform:
        """Return T_flange2tcp: flange -> tool center point.

        Default: Z-offset + optional rotation.
        """
        T = np.eye(4)
        T[2, 3] = self.tool_z_offset
        if self.tool_rotation is not None:
            T[:3, :3] = self.tool_rotation
        return RigidTransform(T, Frame.FLANGE, Frame.TCP)

    @abstractmethod
    def resolve_redundancy(self, T_base2tcp: RigidTransform) -> RigidTransform:
        """Choose optimal TCP orientation exploiting gripper symmetry.

        Examples:
            - Round suction cup: Z-axis 180 (2 candidates)
            - Rectangular suction: 0/90/180/270 (4 candidates)
            - Asymmetric gripper: identity (no redundancy)

        Consider: joint limits, cable routing, singularity avoidance.

        Parameters
        ----------
        T_base2tcp : RigidTransform (BASE -> TCP)

        Returns
        -------
        RigidTransform (BASE -> TCP), possibly modified orientation
        """
        ...

    @abstractmethod
    def to_robot_command(self, T_base2flange: RigidTransform) -> Dict[str, Any]:
        """Convert base->flange transform to vendor command format.

        Examples:
            FANUC  -> {"X": .., "Y": .., "Z": .., "W": .., "P": .., "R": ..}
            UR     -> {"x": .., "y": .., "z": .., "rx": .., "ry": .., "rz": ..}
            ABB    -> {"x": .., "y": .., "z": .., "qw": .., "qx": .., "qy": .., "qz": ..}

        Parameters
        ----------
        T_base2flange : RigidTransform (BASE -> FLANGE)
        """
        ...

    # ---- concrete (shared logic) ----

    def compute_flange_target(self, T_base2tcp: RigidTransform) -> RigidTransform:
        """TCP target -> Flange target.

        T_base2flange = T_base2tcp @ T_tcp2flange
                      = T_base2tcp @ inv(T_flange2tcp)
        """
        T_flange2tcp = self.get_tool_transform()
        assert T_flange2tcp.from_frame == Frame.FLANGE
        assert T_flange2tcp.to_frame == Frame.TCP
        return T_base2tcp @ T_flange2tcp.inv

    def plan_pick(self, T_base2tcp: RigidTransform) -> Dict[str, Any]:
        """Template method: full pipeline from TCP pose to robot command.

        1. resolve_redundancy  ->  optimal orientation
        2. compute_flange_target  ->  TCP -> flange
        3. to_robot_command  ->  vendor format
        """
        assert T_base2tcp.from_frame == Frame.BASE
        assert T_base2tcp.to_frame == Frame.TCP

        T_resolved = self.resolve_redundancy(T_base2tcp)
        T_flange = self.compute_flange_target(T_resolved)
        return self.to_robot_command(T_flange)
