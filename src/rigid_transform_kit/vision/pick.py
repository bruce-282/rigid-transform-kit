"""
rigid_transform_kit.vision.pick
=================================
AI detection result in camera frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import numpy as np

from ..core import Frame, RigidTransform, orthogonal_frame

if TYPE_CHECKING:
    from .camera import CameraConfig


@dataclass
class PickPoint:
    """AI detection result in camera frame.

    Attributes
    ----------
    p_cam : np.ndarray, shape (3,)
        3-D suction point in camera coordinates.
    n_cam : np.ndarray or None, shape (3,)
        Surface normal in camera coordinates (unit vector).
        None -> defaults to -Z in base when converted.
    long_axis_cam : np.ndarray or None, shape (3,)
        Long-axis direction of the object in camera coordinates.
        Used for full 3-axis TCP orientation (e.g. rectangular box picking).
        None -> TCP x-axis is determined automatically.
    confidence : float
        AI model confidence [0, 1].
    """

    p_cam: np.ndarray
    n_cam: Optional[np.ndarray] = None
    long_axis_cam: Optional[np.ndarray] = None
    confidence: float = 1.0

    def __post_init__(self):
        self.p_cam = np.asarray(self.p_cam, dtype=np.float64)
        if self.n_cam is not None:
            self.n_cam = np.asarray(self.n_cam, dtype=np.float64)
        if self.long_axis_cam is not None:
            self.long_axis_cam = np.asarray(self.long_axis_cam, dtype=np.float64)

    def get_orientation_frame_cam(self) -> np.ndarray:
        """Right-handed orientation in camera frame from n_cam and long_axis_cam.

        Vision convention: Z = n_cam (surface normal, no flip). X from
        long_axis_cam (projected). Uses :func:`~rigid_transform_kit.core.orthogonal_frame`.

        Returns
        -------
        R_cam : (3, 3) rotation matrix  [x | y | z]
        """
        z = (
            self.n_cam / np.linalg.norm(self.n_cam)
            if self.n_cam is not None
            else np.array([0.0, 0.0, -1.0])
        )
        return orthogonal_frame(z_axis=z, hint=self.long_axis_cam)

    # ── coordinate transform ──────────────────────────────────

    def to_base(
        self,
        cam_config: CameraConfig,
    ) -> RigidTransform:
        """Pick pose in base frame as 4x4 rigid transform.

        Returns
        -------
        T_base2pick : RigidTransform (BASE -> OBJECT)
            Translation = p_base, rotation = orthogonal frame from n_base and
            long_axis_base (Z = n_base, X from long_axis).
        """
        return self._to_base_impl(cam_config.T_cam2base)

    def to_base_transform(self, T_cam2base: RigidTransform) -> RigidTransform:
        """Pick pose in base frame when you only have T_cam2base (no CameraConfig)."""
        return self._to_base_impl(T_cam2base)

    def _to_base_impl(self, T_cam2base: RigidTransform) -> RigidTransform:
        p_base = T_cam2base.transform_point(self.p_cam)

        if self.n_cam is not None:
            n_base = T_cam2base.transform_direction(self.n_cam)
            n_base = n_base / np.linalg.norm(n_base)
        else:
            n_base = np.array([0.0, 0.0, -1.0])

        long_axis_base = None
        if self.long_axis_cam is not None:
            long_axis_base = T_cam2base.transform_direction(self.long_axis_cam)
            nrm = np.linalg.norm(long_axis_base)
            if nrm > 1e-12:
                long_axis_base = long_axis_base / nrm
            else:
                long_axis_base = None

        R_base = orthogonal_frame(z_axis=n_base, hint=long_axis_base)
        return RigidTransform.from_Rt(R_base, p_base, Frame.BASE, Frame.OBJECT)

