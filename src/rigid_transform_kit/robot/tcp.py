"""
rigid_transform_kit.robot.tcp
===============================
TCP pose builder from position + surface normal.
"""

from __future__ import annotations

import numpy as np

from ..core import Frame, RigidTransform


def build_tcp_pose(
    p_base: np.ndarray,
    n_base: np.ndarray,
    contact_offset: float = 0.0,
) -> RigidTransform:
    """Build T_base2tcp from position and surface normal.

    Convention:
        TCP Z-axis = -normal (approach direction, pointing into surface)
        contact_offset > 0  ->  TCP is lifted away from surface along normal

    Parameters
    ----------
    p_base : target position in base frame (3,)
    n_base : surface normal in base frame (3,), unit vector
    contact_offset : standoff distance along normal (meters)

    Returns
    -------
    RigidTransform  T(BASE -> TCP)
    """
    z_axis = -n_base

    up = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(z_axis, up)) < 0.99:
        x_axis = np.cross(up, z_axis)
    else:
        x_axis = np.cross([1.0, 0.0, 0.0], z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)

    R = np.column_stack([x_axis, y_axis, z_axis])
    t = p_base + n_base * contact_offset

    return RigidTransform.from_Rt(R, t, Frame.BASE, Frame.TCP)
