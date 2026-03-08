"""
rigid-transform-kit / examples / visualize_pallet_sample.py
===========================================================
Visualize calibration, point cloud, and pick points in Rerun.

Requires: pip install rigid-transform-kit[viz]  (and open3d for PLY)

Usage:
  python examples/visualize_pallet_sample.py
  python examples/visualize_pallet_sample.py --data-dir /path/to/dataset
  python examples/visualize_pallet_sample.py --box-pcd box1.ply box2.ply
  python examples/visualize_pallet_sample.py --calibration cal.yml --pcd scene.ply
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from rigid_transform_kit import (
    Frame,
    PickPoint,
    RigidTransform,
    build_tcp_pose,
)
from rigid_transform_kit.viz import TransformVisualizer
from utils import (
    fit_plane,
    get_box_axes,
    load_box_pcd,
    load_extrinsics,
    load_ply_points,
    load_cam_targets,
    remove_statistical_outlier,
)

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "datasets" / "aw_pallet"


def parse_args():
    p = argparse.ArgumentParser(
        description="Visualize calibration, point cloud, and pick points in Rerun.",
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Base directory for default paths (default: {DEFAULT_DATA_DIR})",
    )
    p.add_argument(
        "--calibration",
        type=Path,
        default=DEFAULT_DATA_DIR / "calibration_result.yml",
        metavar="YAML",
        help="Camera calibration YAML (default: <data-dir>/calibration_result.yml), the format is like this: {config: {base2cam: [x, y, z, w, x, y, z, w]}}",
    )
    p.add_argument(
        "--cam-targets",
        type=Path,
        default=None,
        metavar="JSON",
        help="Target points JSON (optional), the format is like this: {cam_targets: [{p_cam: [x, y, z], n_cam: [nx, ny, nz], long_axis_cam: [lx, ly, lz]}]}",
    )
    p.add_argument(
        "--pcd",
        type=Path,
        default=DEFAULT_DATA_DIR / "pcd.ply",
        metavar="PLY",
        help="Point cloud PLY (default: <data-dir>/pcd.ply)",
    )
    p.add_argument(
        "--box-pcd",
        type=Path,
        nargs="*",
        default=[DEFAULT_DATA_DIR / "box_pcd1.ply", DEFAULT_DATA_DIR / "box_pcd2.ply",
        DEFAULT_DATA_DIR / "box_pcd3.ply", DEFAULT_DATA_DIR / "box_pcd4.ply",
        DEFAULT_DATA_DIR / "box_pcd5.ply", DEFAULT_DATA_DIR / "box_pcd6.ply"],
        metavar="PLY",
        help="Box PLY file(s) for OBB-based pick; one pick per file",
    )
    args = p.parse_args()
    return args


def main():
    args = parse_args()

    calib = load_extrinsics(args.calibration)
    T_base2cam = RigidTransform.from_matrix(calib["base2cam"], Frame.BASE, Frame.CAMERA)
    T_cam2base = T_base2cam.inv

    picks: list[PickPoint] = []

    if args.cam_targets is not None:
        picks = load_cam_targets(args.cam_targets)
        print(f"Loaded {len(picks)} cam_targets from {args.cam_targets}.")
    elif args.box_pcd:
        for path in args.box_pcd:
            box_pcd = load_box_pcd(path)
            if box_pcd is not None:
                #box_pcd, _ = remove_statistical_outlier(box_pcd)
                normal_cam, _, inlier_pcd = fit_plane(box_pcd)
                _, long_axis, center, info = get_box_axes(inlier_pcd, plane_normal=normal_cam)
                picks.append(
                    PickPoint(p_cam=center, n_cam=normal_cam, long_axis_cam=long_axis)
                )
                print(f"Box {path.name}: center={center}, long={info['extent_long']:.1f}, short={info['extent_short']:.1f}, aspect={info['aspect_ratio']:.2f}")
    else:
        print("No pick points found.")

    vis = TransformVisualizer("pallet_sample1", spawn=True)

    # ── world = base (robot) coordinate system, all in mm ──
    vis.log_transform(
        "world/base",
        RigidTransform.identity(Frame.BASE),
        axis_length=300.0,
        label="WORLD=BASE",
    )

    cam_pose = RigidTransform.from_Rt(
        T_cam2base.R, T_base2cam.t,
        Frame.BASE, Frame.CAMERA,
    )
    vis.log_transform(
        "world/camera",
        cam_pose,
        axis_length=200.0,
        label="CAMERA",
    )

    # ── Load PCD ──
    pts_cam_mm = None
    pts_base = None
    colors_cam = None
    ply_data = load_ply_points(args.pcd)
    if ply_data is not None:
        pts_cam_mm, colors_cam = ply_data
        if np.median(np.abs(pts_cam_mm)) < 10:
            pts_cam_mm = pts_cam_mm * 1000.0

        pts_base = T_cam2base.transform_points(pts_cam_mm)
        vis.log_points("world/pcd", pts_base, colors=colors_cam, radii=3.0)
        print(f"Logged {len(pts_base)} points from PLY (colors={'yes' if colors_cam is not None else 'no'}).")

    tcp_poses_base = []
    tcp_poses_cam = []
    has_axes = []
    for i, pick in enumerate(picks):
        p_cam_mm = pick.p_cam * 1000.0 if np.median(np.abs(pick.p_cam)) < 10 else pick.p_cam
        p_base = T_cam2base.transform_point(p_cam_mm)

        n_cam = pick.n_cam if pick.n_cam is not None else np.array([0.0, 0.0, -1.0])
        n_base = T_cam2base.transform_direction(n_cam)
        n_base = n_base / (np.linalg.norm(n_base) + 1e-12)

        long_hint_base = None
        long_hint_cam = None
        if pick.long_axis_cam is not None:
            long_hint_cam = pick.long_axis_cam
            long_hint_base = T_cam2base.transform_direction(long_hint_cam)
            long_hint_base = long_hint_base / (np.linalg.norm(long_hint_base) + 1e-12)

        tcp_base = build_tcp_pose(p_base, n_base, long_axis_hint=long_hint_base)
        tcp_cam = build_tcp_pose(p_cam_mm, n_cam, long_axis_hint=long_hint_cam)
        tcp_poses_base.append(tcp_base)
        tcp_poses_cam.append(tcp_cam)
        show = pick.n_cam is not None
        has_axes.append(show)
        tag = "TCP" if show else "center"
        print(f"Pick #{i}: {tag} ({tcp_base.t[0]:.1f}, {tcp_base.t[1]:.1f}, {tcp_base.t[2]:.1f}) mm")

    if tcp_poses_base:
        vis.log_tcp_poses(tcp_poses_base, parent_path="world/picks", axis_length=100.0, arrow_radius=2.0, show_axes=has_axes)

    # ── Scene in camera-frame view (second tab) ──
    vis.log_scene_in_camera(
        pts_cam=pts_cam_mm,
        colors=colors_cam,
        tcp_poses=tcp_poses_cam if tcp_poses_cam else None,
        show_axes=has_axes if tcp_poses_cam else None,
    )

    # ── Scene in base-frame view (third tab) ──
    vis.log_scene_base(
        pts_base=pts_base,
        colors=colors_cam,
        tcp_poses=tcp_poses_base if tcp_poses_base else None,
        show_axes=has_axes if tcp_poses_base else None,
    )

    print("\nRerun viewer - 'Overview (in Base)' / 'Scene (in Camera)' / 'Scene (in Base)' tab.")


if __name__ == "__main__":
    main()
