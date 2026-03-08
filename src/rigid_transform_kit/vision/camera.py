"""
rigid_transform_kit.vision.camera
===================================
Camera configuration and hand-eye calibration result.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core import Frame, RigidTransform


@dataclass
class CameraConfig:
    """Camera intrinsics + hand-eye calibration result.

    Parameters
    ----------
    T_base2cam : RigidTransform
        base -> camera transform (from hand-eye calibration).
    intrinsics : np.ndarray, shape (3, 3)
        Camera intrinsic matrix K.
    distortion : np.ndarray
        Distortion coefficients (OpenCV convention).
    depth_scale : float
        Raw depth value -> meters. e.g. 0.001 for mm sensors.
    """

    T_base2cam: RigidTransform
    intrinsics: np.ndarray
    distortion: np.ndarray
    depth_scale: float = 0.001

    def __post_init__(self):
        self.intrinsics = np.asarray(self.intrinsics, dtype=np.float64)
        self.distortion = np.asarray(self.distortion, dtype=np.float64)
        assert self.T_base2cam.from_frame == Frame.BASE
        assert self.T_base2cam.to_frame == Frame.CAMERA

    @property
    def T_cam2base(self) -> RigidTransform:
        """Convenience: cam -> base."""
        return self.T_base2cam.inv

    @classmethod
    def from_calibration_dict(
        cls,
        calib: dict,
        intrinsics,
        distortion,
        depth_scale: float = 0.001,
        calib_key: str = "base2cam",
        calib_convention: str = "base2cam",
    ) -> CameraConfig:
        """Create from calibration dict (default: calib["base2cam"] = T_base2cam).

        Parameters
        ----------
        calib : dict
            Must contain a 4x4 matrix under `calib_key`.
        calib_convention : str
            "base2cam" — matrix is T_base2cam (default).
            "cam2base" — matrix is T_cam2base (camera -> base).
        """
        mat = np.array(calib[calib_key], dtype=np.float64)

        if calib_convention == "cam2base":
            T_cam2base = RigidTransform(mat, Frame.CAMERA, Frame.BASE)
            T_base2cam = T_cam2base.inv
        elif calib_convention == "base2cam":
            T_base2cam = RigidTransform(mat, Frame.BASE, Frame.CAMERA)
        else:
            raise ValueError(f"Unknown calib_convention: {calib_convention}")

        return cls(
            T_base2cam=T_base2cam,
            intrinsics=np.asarray(intrinsics),
            distortion=np.asarray(distortion),
            depth_scale=depth_scale,
        )
