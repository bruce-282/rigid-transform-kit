"""
rigid_transform_kit.robot.tcp
===============================
TCP pose builder from position + surface normal.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..core import Frame, RigidTransform


def _orthogonal_frame(
    approach: np.ndarray,
    hint: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Build a right-handed rotation matrix from an approach vector.

    Parameters
    ----------
    approach : (3,) unit vector — becomes the TCP Z-axis.
    hint : (3,) optional preferred X-axis direction (e.g. object long axis).
           Projected onto the plane perpendicular to *approach*
           to guarantee orthogonality.
           If None, a world-up heuristic is used.

    Returns
    -------
    R : (3, 3) rotation matrix  [x | y | z]
    """
    z = approach / np.linalg.norm(approach)

    if hint is not None:
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


def build_tcp_pose(
    p_base: np.ndarray,
    n_base: np.ndarray,
    contact_offset: float = 0.0,
    long_axis_hint: Optional[np.ndarray] = None,
) -> RigidTransform:
    """Build T_base2tcp from position and surface normal.

    Convention:
        TCP Z-axis = -normal (approach direction, pointing into surface)
        TCP X-axis = long_axis_hint (if given), else world-up heuristic
        contact_offset > 0  ->  TCP is lifted away from surface along normal

    Parameters
    ----------
    p_base : target position in base frame (3,)
    n_base : surface normal in base frame (3,), unit vector
    contact_offset : standoff distance along normal
    long_axis_hint : (3,) optional object long-axis direction in base frame.
                     Used to fully determine TCP orientation for rectangular picks.

    Returns
    -------
    RigidTransform  T(BASE -> TCP)
    """
    R = _orthogonal_frame(-n_base, hint=long_axis_hint)
    t = p_base + n_base * contact_offset

    return RigidTransform.from_Rt(R, t, Frame.BASE, Frame.TCP)
