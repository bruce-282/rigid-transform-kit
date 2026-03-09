"""
rigid_transform_kit.robot.tcp
===============================
TCP pose builder from position + surface normal.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..core import Frame, RigidTransform, orthogonal_frame


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
    R = orthogonal_frame(z_axis=-n_base, hint=long_axis_hint)
    t = p_base + n_base * contact_offset
    return RigidTransform.from_Rt(R, t, Frame.BASE, Frame.TCP)
