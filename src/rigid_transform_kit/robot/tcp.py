"""
rigid_transform_kit.robot.tcp
===============================
TCP pose builder from pick pose (4x4 RigidTransform).
"""

from __future__ import annotations

import numpy as np

from ..core import Frame, RigidTransform, is_orthogonal_frame


def build_tcp_pose(T_pick: RigidTransform) -> RigidTransform:
    """Build T_*2tcp from pick pose (4x4). Same frame as input (e.g. BASE or CAMERA).

    *T_pick* is the pick pose (Z = surface normal). TCP는 approach 방향이 Z이므로
    R만 Z축 반전(및 오른손 유지 위해 Y 반전): R_tcp = R_pick @ diag(1, -1, -1).
    Origin = pick 위치.

    Parameters
    ----------
    T_pick : RigidTransform (4x4)
        Pick pose (e.g. PickPoint.to_base() or from_Rt(R_cam, p_cam, CAMERA, OBJECT)).

    Returns
    -------
    RigidTransform  T(from_frame -> TCP)
    """
    R = T_pick.R
    if not is_orthogonal_frame(R):
        raise ValueError("T_pick rotation must be a valid 3x3 orthonormal (right-handed)")
    # approach 방향으로 Z축 반전 (Y도 반전해 오른손 좌표계 유지)
    R_tcp = R @ np.diag([1.0, -1.0, -1.0])
    return RigidTransform.from_Rt(R_tcp, T_pick.t, T_pick.from_frame, Frame.TCP)
