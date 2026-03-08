"""Pallet box pick-point extraction and TCP pose computation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import numpy as np

from rigid_transform_kit import PickPoint, RigidTransform, build_tcp_pose

log = logging.getLogger(__name__)


def _ensure_mm(p: np.ndarray) -> np.ndarray:
    """Convert to mm if values look like meters (median < 10)."""
    if np.median(np.abs(p)) < 10:
        return p * 1000.0
    return p


def extract_picks_from_boxes(box_paths: Sequence[Path]) -> list[PickPoint]:
    """Extract one :class:`PickPoint` per box PLY via RANSAC plane fitting + 2-D PCA.

    Requires Open3D (``pip install open3d``).
    """
    from utils import fit_plane, get_box_axes, load_box_pcd

    picks: list[PickPoint] = []
    for path in box_paths:
        box_pcd = load_box_pcd(path)
        if box_pcd is None:
            continue
        normal_cam, _, inlier_pcd = fit_plane(box_pcd)
        _, long_axis, center, info = get_box_axes(inlier_pcd, plane_normal=normal_cam)
        picks.append(
            PickPoint(p_cam=center, n_cam=normal_cam, long_axis_cam=long_axis)
        )
        log.info(
            "%s: center=(%.1f, %.1f, %.1f), long=%.1f, short=%.1f, aspect=%.2f",
            path.name, center[0], center[1], center[2],
            info["extent_long"], info["extent_short"], info["aspect_ratio"],
        )
    return picks


def _pick_to_base(
    pick: PickPoint, T_cam2base: RigidTransform,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Transform a single pick's position, normal, and long-axis hint to base frame."""
    p_cam_mm = _ensure_mm(pick.p_cam)
    p_base = T_cam2base.transform_point(p_cam_mm)

    n_cam = pick.n_cam if pick.n_cam is not None else np.array([0.0, 0.0, -1.0])
    n_base = T_cam2base.transform_direction(n_cam)
    n_base = n_base / (np.linalg.norm(n_base) + 1e-12)

    long_hint_base = None
    if pick.long_axis_cam is not None:
        long_hint_base = T_cam2base.transform_direction(pick.long_axis_cam)
        long_hint_base = long_hint_base / (np.linalg.norm(long_hint_base) + 1e-12)

    return p_base, n_base, long_hint_base


def picks_to_tcp_poses(
    picks: Sequence[PickPoint],
    T_cam2base: RigidTransform,
) -> list[RigidTransform]:
    """Convert camera-frame PickPoints to base-frame TCP poses."""
    tcp_poses: list[RigidTransform] = []
    for pick in picks:
        p_base, n_base, long_hint_base = _pick_to_base(pick, T_cam2base)
        tcp_poses.append(build_tcp_pose(p_base, n_base, long_axis_hint=long_hint_base))
    return tcp_poses


def picks_to_tcp_poses_base_and_cam(
    picks: Sequence[PickPoint],
    T_cam2base: RigidTransform,
) -> tuple[list[RigidTransform], list[RigidTransform], list[bool]]:
    """Compute TCP poses in both base and camera frames.

    Returns
    -------
    tcp_poses_base : base-frame TCP poses
    tcp_poses_cam  : camera-frame TCP poses
    has_axes       : per-pick flag — True when the pick has an explicit normal
    """
    tcp_poses_base: list[RigidTransform] = []
    tcp_poses_cam: list[RigidTransform] = []
    has_axes: list[bool] = []

    for pick in picks:
        p_base, n_base, long_hint_base = _pick_to_base(pick, T_cam2base)
        tcp_poses_base.append(build_tcp_pose(p_base, n_base, long_axis_hint=long_hint_base))

        p_cam_mm = _ensure_mm(pick.p_cam)
        n_cam = pick.n_cam if pick.n_cam is not None else np.array([0.0, 0.0, -1.0])
        long_hint_cam = pick.long_axis_cam
        tcp_poses_cam.append(build_tcp_pose(p_cam_mm, n_cam, long_axis_hint=long_hint_cam))

        has_axes.append(pick.n_cam is not None)

    return tcp_poses_base, tcp_poses_cam, has_axes
