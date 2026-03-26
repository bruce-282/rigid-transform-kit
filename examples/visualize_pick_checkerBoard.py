"""
rigid-transform-kit / examples / visualize_pick_checkerBoard.py
================================================================
Checkerboard corner 추출 + pose 추정 (RGB only) 후 Rerun 시각화.

Util: utils.checkerboard — detect_corners, get_pose_from_corners, detect_checkerboard_pose
Input: RGB 이미지; optional PLY 포인트클라우드 (calibration 있으면 base frame으로 표시).

Requires: pip install rigid-transform-kit[viz] opencv-python

Usage:
  python examples/visualize_pick_checkerBoard.py --image rgb.png --intrinsics intrinsics.json --pattern-size 9 6 --square-size 25
  python examples/visualize_pick_checkerBoard.py --image rgb.png --intrinsics intrinsics.json --calibration calib.yml --pcd pointcloud.ply
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from rigid_transform_kit import FanucAdapter, Frame, RigidTransform
from rigid_transform_kit.app import (
    build_tcp_result,
    load_calibration,
    log_robot_commands,
    picks_to_tcp_poses_base_and_cam,
    pose_to_tcp_poses_base_and_cam,
    save_tcp_poses,
)
from rigid_transform_kit.viz import TransformVisualizer, save_recording

try:
    import cv2
except ImportError:
    cv2 = None

from utils import load_intrinsics_any, load_ply_points
from utils.checkerboard import checkerboard_to_pick_point, detect_checkerboard_pose, undistort_point_cloud

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(
        description="Checkerboard corner + pose from RGB image, visualize in Rerun.",
    )
    p.add_argument(
        "--image",
        type=Path,
        required=True,
        metavar="PATH",
        help="Input RGB image path",
    )
    p.add_argument(
        "--intrinsics",
        type=Path,
        required=True,
        metavar="JSON",
        help="Camera intrinsics JSON: {\"K\": [[3,3]], \"dist\": [5]}",
    )
    p.add_argument(
        "--pattern-size",
        type=int,
        nargs=2,
        default=[9, 6],
        metavar=("COLS", "ROWS"),
        help="Checkerboard inner corners (cols, rows). Default 9 6",
    )
    p.add_argument(
        "--square-size",
        type=float,
        default=25.0,
        metavar="MM",
        help="Square size in mm. Default 25",
    )
    p.add_argument(
        "--origin",
        type=str,
        default="center",
        choices=["center", "LT", "RB"],
        help="Board origin: center (default), LT (left-top), RB (right-bottom)",
    )
    p.add_argument(
        "--calibration",
        type=Path,
        default=None,
        metavar="YAML",
        help="Optional: hand-eye calibration (T_base2cam) to show board in base frame",
    )
    p.add_argument(
        "--save",
        type=Path,
        default=None,
        metavar="RRD",
        help="Save to .rrd and open with rerun",
    )
    p.add_argument(
        "--port",
        type=int,
        default=None,
        help="Rerun gRPC port if 9876 in use",
    )
    p.add_argument(
        "--axis-length",
        type=float,
        default=100.0,
        help="Axis length in mm for visualization. Default 100",
    )
    p.add_argument(
        "--pcd",
        type=Path,
        default=None,
        metavar="PLY",
        help="Optional: point cloud PLY (camera frame); with --calibration shown in base frame",
    )
    p.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        metavar="JSON",
        help="Save TCP poses to JSON (requires --calibration)",
    )
    p.add_argument(
        "--tool-z-offset",
        type=float,
        default=100.0,
        metavar="MM",
        help="Flange→TCP offset in mm (default: 200)",
    )
    p.add_argument(
        "--tool-rotation",
        type=float,
        nargs=3,
        default=None,
        metavar=("W", "P", "R"),
        help="Tool rotation as WPR (degrees, Fanuc xyz Euler). e.g. 180 0 105",
    )
    return p.parse_args()


def main():
    args = parse_args()
    if cv2 is None:
        raise ImportError("opencv-python required: pip install opencv-python")

    # Load image (RGB)
    img = cv2.imread(str(args.image))
    if img is None:
        raise FileNotFoundError(f"Could not load image: {args.image}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Load intrinsics
    K, dist = load_intrinsics_any(args.intrinsics)
    pattern_size = tuple(args.pattern_size)

    # Optional: load PLY once (for RGB-depth pose and visualization)
    ply_data = load_ply_points(args.pcd) if (args.pcd and args.pcd.exists()) else None
    points_cam_m, colors_cam = (ply_data if ply_data else (None, None))

    # Detect checkerboard pose (RGB-only or RGB+depth when points_cam_m given)
    T_cam2board, corners = detect_checkerboard_pose(
        img_rgb,
        pattern_size,
        args.square_size,
        K,
        dist,
        refine_corners=True,
        origin=args.origin,
        points_cam_m=points_cam_m,
    )
    if T_cam2board is None:
        log.error(
            "Checkerboard not detected in %s with --pattern-size %s. "
            "Inner corners = (cols, rows); e.g. 10x7 squares → 9 6.",
            args.image,
            pattern_size,
        )
        return

    log.info("Checkerboard detected. T_cam2board t = [%.1f, %.1f, %.1f] mm", *T_cam2board.t)

    # Calibration + TCP poses (pallet_box와 동일: PickPoint → picks_to_tcp_poses_base_and_cam)
    T_base2cam = None
    T_cam2base = None
    if args.calibration is not None and args.calibration.exists():
        T_base2cam, T_cam2base = load_calibration(args.calibration)
        log.info("Calibration loaded: board in base frame")


    pick = checkerboard_to_pick_point(T_cam2board)
    if pick is not None:
        tcp_poses_base, tcp_poses_cam, has_axes = picks_to_tcp_poses_base_and_cam([pick], T_cam2base)
    else:
        raise ValueError("Checkerboard not detected")
    # Fanuc robot commands + TCP save (pallet_box와 동일)
    tool_rotation_matrix = None
    if args.tool_rotation is not None:
        tool_rotation_matrix = Rotation.from_euler(
            "xyz", args.tool_rotation, degrees=True
        ).as_matrix()

    fanuc = FanucAdapter(
        pos_unit="mm",
        tool_z_offset=args.tool_z_offset,
        tool_rotation=tool_rotation_matrix,
    )
    robot_commands = [fanuc.plan_pick(pose) for pose in tcp_poses_base]
    flange_poses_base = [
        fanuc.compute_flange_target(fanuc.resolve_redundancy(p)) for p in tcp_poses_base
    ]
    labels = ["TCP" if ax else "center" for ax in has_axes]
    log_robot_commands(robot_commands, labels=labels, logger=log)
    if args.output is not None:
        result = build_tcp_result(tcp_poses_base, robot_commands, flange_poses=flange_poses_base)
        save_tcp_poses(result, args.output)
        log.info("Saved %d TCP+flange poses to %s", len(tcp_poses_base), args.output)

    pts_cam_mm = None
    pts_base = None

    # Rerun
    spawn = args.save is None
    if not spawn:
        log.info("Saving to file (spawn disabled).")
    port = args.port
    if port is None and os.environ.get("RERUN_PORT"):
        try:
            port = int(os.environ["RERUN_PORT"])
        except ValueError:
            pass
    vis = TransformVisualizer("checkerboard_pose", spawn=spawn, port=port)

    # World = base (pallet_box와 동일). base, camera, checkerboard
    vis.log_transform(
        "world/base",
        RigidTransform.identity(Frame.BASE),
        axis_length=args.axis_length,
        label="BASE",
    )
    if T_base2cam is not None:
        cam_pose = RigidTransform.from_Rt(
            T_cam2base.R, T_base2cam.t,
            Frame.BASE, Frame.CAMERA,
        )
        vis.log_transform(
            "world/camera",
            cam_pose,
            axis_length=args.axis_length,
            label="CAMERA",
        )
    else:
        vis.log_transform(
            "world/camera",
            RigidTransform.identity(Frame.CAMERA),
            axis_length=args.axis_length,
            label="CAMERA",
        )
    board_pose = tcp_poses_base[0] if tcp_poses_base else tcp_poses_cam[0]
    vis.log_transform(
        "world/checkerboard",
        board_pose,
        axis_length=args.axis_length,
        label="BOARD",
    )

    if tcp_poses_base:
        vis.log_tcp_poses(tcp_poses_base, parent_path="world/picks", axis_length=100.0, arrow_radius=2.0, show_axes=has_axes)
        vis.log_flange_poses(flange_poses_base, parent_path="world/flanges", axis_length=100.0)

    # ── Optional PLY (use undistorted for vis so it matches pose) ──
    if points_cam_m is not None:
        dist_arr = (
            np.asarray(dist, dtype=np.float64).ravel()[:5]
            if dist is not None
            else np.zeros(5, dtype=np.float64)
        )
        if np.any(np.abs(dist_arr) > 1e-10):
            pts_vis = undistort_point_cloud(points_cam_m, K, dist)
            valid = ~np.any(np.isnan(pts_vis), axis=1)
            pts_cam_m_vis = pts_vis[valid]
            colors_vis = colors_cam[valid] if colors_cam is not None else None
        else:
            pts_cam_m_vis = points_cam_m
            colors_vis = colors_cam
        pts_cam_mm = pts_cam_m_vis * 1000.0
        pts_base = T_cam2base.transform_points(pts_cam_mm) if T_cam2base is not None else None
        pts_world = pts_base if pts_base is not None else pts_cam_mm
        vis.log_points("world/pcd", pts_world, colors=colors_vis, radii=1.2)
        n_pts = len(pts_world)
        log.info("Logged %d points from PLY (colors=%s).", n_pts, "yes" if colors_cam is not None else "no")
        if n_pts > 500_000 and spawn:
            log.warning(
                "Large point cloud (%d). Use --save out.rrd then: rerun out.rrd to avoid gRPC errors.",
                n_pts,
            )
    elif args.pcd is not None:
        log.warning("PLY not loaded or not found: %s", args.pcd)

    # 2D projection: 좌표축 길이 = 체커보드 반쪽 (center 원점 기준)
    proj_axis_mm = max(pattern_size[0] - 1, pattern_size[1] - 1) * 0.5 * args.square_size
    vis.log_projection_2d(
        K,
        pts_cam=pts_cam_mm,
        colors=colors_vis if points_cam_m is not None else None,
        transforms=tcp_poses_cam,
        axis_length_mm=proj_axis_mm,
    )

    vis.log_scene_in_camera(
        pts_cam=pts_cam_mm,
        colors=colors_cam,
        tcp_poses=tcp_poses_cam or None,
        show_axes=has_axes or None,
    )

    vis.log_scene_base(
        pts_base=pts_base,
        colors=colors_cam,
        tcp_poses=tcp_poses_base or None,
        show_axes=has_axes or None,
    )

    if args.save is not None:
        save_recording(args.save)
        log.info("Saved to %s. Run: rerun %s", args.save, args.save)
        try:
            subprocess.run(["rerun", str(args.save)], check=False)
        except FileNotFoundError:
            pass

    if spawn:
        log.info("Rerun viewer - 'Overview (in Base)' / 'Scene (in Camera)' / 'Scene (in Base)' tab.")
        import rerun as rr
        rec = rr.get_global_data_recording()
        if rec is not None:
            try:
                rec.flush(timeout_sec=10.0)
            except Exception:  # noqa: BLE001
                pass
            time.sleep(1.0)


if __name__ == "__main__":
    main()
