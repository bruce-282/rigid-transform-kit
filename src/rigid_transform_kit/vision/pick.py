"""
rigid_transform_kit.vision.pick
=================================
AI detection result in camera frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np

from ..core import Frame, RigidTransform

if TYPE_CHECKING:
    from .camera import CameraConfig


@dataclass
class PickPoint:
    """AI detection result in camera frame.

    Attributes
    ----------
    p_cam : np.ndarray, shape (3,)
        3-D suction point in camera coordinates (meters).
    n_cam : np.ndarray or None, shape (3,)
        Surface normal in camera coordinates (unit vector).
        None -> defaults to "look down" when converted.
    confidence : float
        AI model confidence [0, 1].
    """

    p_cam: np.ndarray
    n_cam: Optional[np.ndarray] = None
    confidence: float = 1.0

    def __post_init__(self):
        self.p_cam = np.asarray(self.p_cam, dtype=np.float64)
        if self.n_cam is not None:
            self.n_cam = np.asarray(self.n_cam, dtype=np.float64)

    def to_base(self, cam_config: CameraConfig) -> Tuple[np.ndarray, np.ndarray]:
        """Transform pick point and normal to base frame.

        Returns
        -------
        p_base : np.ndarray (3,)
        n_base : np.ndarray (3,), unit vector
        """
        T_cam2base = cam_config.T_cam2base

        p_base = T_cam2base.transform_point(self.p_cam)

        if self.n_cam is not None:
            n_base = T_cam2base.transform_direction(self.n_cam)
            n_base = n_base / np.linalg.norm(n_base)
        else:
            n_base = np.array([0.0, 0.0, -1.0])

        return p_base, n_base
