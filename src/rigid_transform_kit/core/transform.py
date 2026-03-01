"""
rigid_transform_kit.core.transform
====================================
Frame-aware rigid body transforms with compile-time-like frame chain validation.

Convention:
    RigidTransform(from_frame=A, to_frame=B)  ->  T_AB  ->  transforms points in B into A
    i.e.  p_A = T_AB @ p_B

    This follows the robotics "read right-to-left" convention:
        T_base2cam means "cam expressed in base" or equivalently "base <- cam"
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from .frame import Frame


@dataclass
class RigidTransform:
    """4x4 homogeneous rigid transform with explicit frame labels.

    Attributes
    ----------
    matrix : np.ndarray, shape (4, 4)
    from_frame : Frame
    to_frame : Frame
    """

    matrix: np.ndarray
    from_frame: Frame
    to_frame: Frame

    # ---- lifecycle ----

    def __post_init__(self):
        self.matrix = np.asarray(self.matrix, dtype=np.float64)
        if self.matrix.shape != (4, 4):
            raise ValueError(f"Expected (4,4) matrix, got {self.matrix.shape}")

    # ---- properties ----

    @property
    def R(self) -> np.ndarray:
        """Rotation part (3x3)."""
        return self.matrix[:3, :3]

    @property
    def t(self) -> np.ndarray:
        """Translation part (3,)."""
        return self.matrix[:3, 3]

    @property
    def inv(self) -> RigidTransform:
        """Inverse transform with swapped frames."""
        return RigidTransform(np.linalg.inv(self.matrix), self.to_frame, self.from_frame)

    # ---- operators ----

    def __matmul__(self, other: RigidTransform) -> RigidTransform:
        """Chain two transforms with frame compatibility check.

        T_AB @ T_BC  ->  T_AC
        """
        if not isinstance(other, RigidTransform):
            return NotImplemented
        if self.to_frame != other.from_frame:
            raise ValueError(
                f"Frame mismatch: {self} @ {other}  —  "
                f"need {self.to_frame} == {other.from_frame}"
            )
        return RigidTransform(
            self.matrix @ other.matrix,
            self.from_frame,
            other.to_frame,
        )

    # ---- point / direction transforms ----

    def transform_point(self, p: np.ndarray) -> np.ndarray:
        """Apply full transform (rotation + translation) to a 3-D point."""
        p = np.asarray(p, dtype=np.float64)
        return (self.matrix @ np.append(p, 1.0))[:3]

    def transform_points(self, pts: np.ndarray) -> np.ndarray:
        """Batch transform (N, 3) points."""
        pts = np.asarray(pts, dtype=np.float64)
        ones = np.ones((pts.shape[0], 1))
        homo = np.hstack([pts, ones])  # (N, 4)
        return (self.matrix @ homo.T).T[:, :3]

    def transform_direction(self, d: np.ndarray) -> np.ndarray:
        """Rotate a direction vector (translation ignored)."""
        return self.R @ np.asarray(d, dtype=np.float64)

    # ---- export helpers ----

    def to_vec6_euler(self, convention: str = "XYZ", degrees: bool = True) -> np.ndarray:
        """Return [x, y, z, r1, r2, r3] using scipy euler convention."""
        euler = Rotation.from_matrix(self.R).as_euler(convention, degrees=degrees)
        return np.concatenate([self.t, euler])

    def to_pos_quat_xyzw(self):
        """Return (pos(3,), quat(4,)) in [qx,qy,qz,qw] order (scipy default)."""
        return self.t.copy(), Rotation.from_matrix(self.R).as_quat()

    def to_pos_quat_wxyz(self):
        """Return (pos(3,), quat(4,)) in [qw,qx,qy,qz] order."""
        q_xyzw = Rotation.from_matrix(self.R).as_quat()
        q_wxyz = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])
        return self.t.copy(), q_wxyz

    # ---- factory methods ----

    @classmethod
    def from_matrix(cls, mat, from_frame: Frame, to_frame: Frame) -> RigidTransform:
        return cls(np.asarray(mat), from_frame, to_frame)

    @classmethod
    def from_Rt(cls, R, t, from_frame: Frame, to_frame: Frame) -> RigidTransform:
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        return cls(T, from_frame, to_frame)

    @classmethod
    def from_translation(cls, t, from_frame: Frame, to_frame: Frame) -> RigidTransform:
        T = np.eye(4)
        T[:3, 3] = np.asarray(t, dtype=np.float64)
        return cls(T, from_frame, to_frame)

    @classmethod
    def from_euler(
        cls, t, euler, from_frame: Frame, to_frame: Frame,
        convention: str = "XYZ", degrees: bool = True,
    ) -> RigidTransform:
        R = Rotation.from_euler(convention, euler, degrees=degrees).as_matrix()
        return cls.from_Rt(R, t, from_frame, to_frame)

    @classmethod
    def from_quat_xyzw(cls, t, quat, from_frame: Frame, to_frame: Frame) -> RigidTransform:
        R = Rotation.from_quat(quat).as_matrix()  # scipy expects xyzw
        return cls.from_Rt(R, t, from_frame, to_frame)

    @classmethod
    def identity(cls, frame: Frame) -> RigidTransform:
        return cls(np.eye(4), frame, frame)

    # ---- repr ----

    def __repr__(self) -> str:
        return f"T({self.from_frame.name}\u2192{self.to_frame.name})"
