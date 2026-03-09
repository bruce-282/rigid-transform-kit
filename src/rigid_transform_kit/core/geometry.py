"""
rigid_transform_kit.core.geometry
==================================
Orthogonal frame: build rotation matrix from Z-axis direction (and optional X-hint).
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def orthogonal_frame(
    z_axis: np.ndarray,
    hint: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Build a right-handed rotation matrix (3, 3) from Z-axis and optional X-hint.

    Z-axis = normalized *z_axis*. X-axis from *hint* (projected onto plane ⊥ z) or
    world-up heuristic. Y = Z × X. (TCP에서는 z_axis = -normal, 비전에서는 z_axis = normal.)

    Parameters
    ----------
    z_axis : (3,) array
        Direction that becomes the Z-axis (will be normalized).
    hint : (3,) array or None
        Preferred X-axis (e.g. object long axis). Projected onto plane ⊥ z_axis.
        If None, world-up heuristic.

    Returns
    -------
    R : (3, 3) array
        Rotation matrix [x | y | z], det(R) = 1. Use with :meth:`RigidTransform.from_Rt`.
    """
    z = np.asarray(z_axis, dtype=np.float64)
    z = z / np.linalg.norm(z)

    if hint is not None:
        hint = np.asarray(hint, dtype=np.float64)
        proj = hint - np.dot(hint, z) * z
        if np.linalg.norm(proj) > 1e-9:
            x = proj / np.linalg.norm(proj)
        else:
            hint = None

    if hint is None:
        up = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(z, up)) > 0.99:
            up = np.array([1.0, 0.0, 0.0])
        x = np.cross(up, z)
        x = x / np.linalg.norm(x)

    y = np.cross(z, x)
    R = np.column_stack([x, y, z])
    if np.linalg.det(R) < 0:
        R[:, 0] = -R[:, 0]
    return R
