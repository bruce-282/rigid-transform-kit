"""
rigid-transform-kit / examples / visualize_pallet_box.py
========================================================
Visualize calibration, point cloud, and pick points in Rerun.

Requires: pip install rigid-transform-kit[viz]  (and open3d for PLY)

Usage:
  python examples/visualize_pallet_box.py
  python examples/visualize_pallet_box.py --save output.rrd   # save then open (avoids spawn+save conflict)
  python examples/visualize_pallet_box.py --box-pcd box1.ply box2.ply

Large PCD (수십만~백만 포인트) 시 실시간 뷰어는 gRPC 종료 시 에러 로그가 날 수 있음.
에러 없이 보려면: --save out.rrd 로 저장 후 `rerun out.rrd` 로 열기.
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
    extract_picks_from_boxes,
    load_calibration,
    load_cam_targets,
    log_robot_commands,
    picks_to_tcp_poses_base_and_cam,
    save_tcp_poses,
)
from rigid_transform_kit.viz import TransformVisualizer, save_recording
from utils import load_intrinsics_any, load_ply_points
from utils.checkerboard import undistort_point_cloud


DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "datasets" / "aw_pallet"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


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
        "--intrinsics",
        type=Path,
        required=True,
        metavar="JSON",
        help="Camera intrinsics JSON: {\"K\": [[3,3]], \"dist\": [5]}",
    )
    p.add_argument(
        "--calibration",
        type=Path,
        default=DEFAULT_DATA_DIR / "calibration_result.yml",
        metavar="YAML",
        help="Camera calibration YAML (default: <data-dir>/calibration_result.yml)",
    )
    p.add_argument(
        "--cam-targets",
        type=Path,
        default=None,
        metavar="JSON",
        help="Target points JSON (optional)",
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
        default=[],
        metavar="PLY",
        help="Box PLY file(s) for OBB-based pick; one pick per file",
    )
    p.add_argument(
        "--save",
        type=Path,
        default=None,
        metavar="RRD",
        help="Save to .rrd file (use viewer Save when spawn=True; --save avoids spawn conflict)",
    )
    p.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        metavar="JSON",
        help="Save TCP (and flange if tool-z-offset set) poses to JSON",
    )
    p.add_argument(
        "--port",
        type=int,
        default=None,
        metavar="PORT",
        help="Rerun gRPC port (default 9876). Use if port in use (e.g. Windows 10048). Env: RERUN_PORT",
    )
    p.add_argument(
        "--tool-z-offset",
        type=float,
        default=0.0,
        metavar="MM",
        help="Flange→TCP Z offset in mm (default: 0). If 0, flange poses are not computed or shown.",
    )
    p.add_argument(
        "--tool-rotation",
        type=float,
        nargs=3,
        default=None,
        metavar=("W", "P", "R"),
        help="Tool rotation as WPR degrees (Fanuc xyz Euler). e.g. 0 0 105",
    )
    p.add_argument(
        "--show-xy-both",
        action="store_true",
        help="Show both +X/-X and +Y/-Y axes for TCP in scene and 2D projection.",
    )
    return p.parse_args()


log = logging.getLogger(__name__)


def main():
    args = parse_args()

    # If data-dir has cam_targets_simple.json and no --cam-targets given, use it (e.g. yonggin_pasto)
    K, dist = load_intrinsics_any(args.intrinsics)

    cam_targets_path = args.cam_targets
    if cam_targets_path is None:
        simple = args.data_dir / "cam_targets_simple.json"
        if simple.exists():
            cam_targets_path = simple
            log.info("Using cam targets: %s", cam_targets_path)

    T_base2cam, T_cam2base = load_calibration(args.calibration)

    if cam_targets_path is not None:
        picks = load_cam_targets(cam_targets_path)
        log.info("Loaded %d cam_targets from %s.", len(picks), cam_targets_path)
    elif args.box_pcd:
        picks = extract_picks_from_boxes(args.box_pcd)
    else:
        picks = []
        log.warning("No pick points found.")

    # rr.spawn() and rr.save() conflict — use spawn=False when saving to file
    # Large PCD (>500k) → auto-save to avoid gRPC transport errors on shutdown
    save_path = args.save
    ply_data = load_ply_points(args.pcd)
    if save_path is None and ply_data is not None:
        n_preview = len(ply_data[0])
        if n_preview > 500_000:
            save_path = args.data_dir / "box_palletizing.rrd"
            log.info("Large PCD (%d pts). Auto-saving to %s to avoid gRPC errors.", n_preview, save_path)
    spawn = save_path is None
    if not spawn:
        log.info("Saving to file (spawn disabled); viewer will open after save.")
    port = args.port
    if port is None and os.environ.get("RERUN_PORT"):
        try:
            port = int(os.environ["RERUN_PORT"])
        except ValueError:
            pass
    vis = TransformVisualizer("box_palletizing", spawn=spawn, port=port)

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
    colors_vis = None
    if ply_data is not None:
        pts_cam_m, colors_cam = ply_data
        dist_arr = (
            np.asarray(dist, dtype=np.float64).ravel()[:5]
            if dist is not None
            else np.zeros(5, dtype=np.float64)
        )
        if np.any(np.abs(dist_arr) > 1e-10):
            pts_vis = undistort_point_cloud(pts_cam_m, K, dist_arr)
            log.info("Undistorted points.")
            valid = ~np.any(np.isnan(pts_vis), axis=1)
            pts_cam_m_vis = pts_vis[valid]
            colors_vis = colors_cam[valid] if colors_cam is not None else None
        else:
            pts_cam_m_vis = pts_cam_m
            colors_vis = colors_cam
        pts_cam_mm = pts_cam_m_vis * 1000.0  # load_ply_points returns meters

        pts_base = T_cam2base.transform_points(pts_cam_mm)
        vis.log_points("world/pcd", pts_base, colors=colors_vis, radii=1.2)
        n_pts = len(pts_base)
        log.info("Logged %d points from PLY (colors=%s).", n_pts, "yes" if colors_vis is not None else "no")
        if n_pts > 500_000 and spawn:
            log.warning(
                "Large point cloud (%d points). For fewer gRPC errors, use --save out.rrd then: rerun out.rrd",
                n_pts,
            )

    # ── TCP poses (base + cam frames) ──
    tcp_poses_base, tcp_poses_cam, has_axes = picks_to_tcp_poses_base_and_cam(picks, T_cam2base)

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
    flange_commands = [fanuc.plan_pick(pose) for pose in tcp_poses_base]
    tcp_commands = [fanuc.to_robot_command(fanuc.resolve_redundancy(pose)) for pose in tcp_poses_base]

    log_robot_commands(tcp_commands,labels=["TCP"] * len(tcp_commands), logger=log)
    log_robot_commands(flange_commands, labels=["Flange"] * len(flange_commands), logger=log)

    flange_poses_base = None
    if tcp_poses_base:
        vis.log_tcp_poses(tcp_poses_base, parent_path="world/picks", axis_length=100.0, arrow_radius=2.0, show_axes=has_axes)
        if fanuc.tool_z_offset != 0:
            flange_poses_base = [
                fanuc.compute_flange_target(fanuc.resolve_redundancy(p)) for p in tcp_poses_base
            ]
            vis.log_flange_poses(flange_poses_base, parent_path="world/flanges", axis_length=100.0)

    if tcp_poses_base and args.output is not None:
        result = build_tcp_result(tcp_poses_base, tcp_commands, flange_poses=flange_poses_base)
        save_tcp_poses(result, args.output)
        log.info("Saved %d TCP+flange poses to %s", len(tcp_poses_base), args.output)

    vis.log_scene_in_camera(
        pts_cam=pts_cam_mm,
        colors=colors_vis,
        tcp_poses=tcp_poses_cam or None,
        show_axes=has_axes or None,
        show_xy_both=args.show_xy_both,
    )

    vis.log_scene_base(
        pts_base=pts_base,
        colors=colors_vis,
        tcp_poses=tcp_poses_base or None,
        show_axes=has_axes or None,
        show_xy_both=args.show_xy_both,
    )
    vis.log_projection_2d(
        K,
        pts_cam=pts_cam_mm,
        colors=colors_vis,
        transforms=tcp_poses_cam,
        axis_length_mm=100.0,
        show_xy_both=args.show_xy_both,
    )


    if save_path is not None:
        save_recording(save_path)
        log.info("Saved to %s", save_path)
        try:
            subprocess.run(["rerun", str(save_path)], check=False)
        except FileNotFoundError:
            log.info("Run: rerun %s", save_path)

    if spawn:
        log.info("\nRerun viewer - 'Overview (in Base)' / 'Scene (in Camera)' / 'Scene (in Base)' tab.")
        # Flush만 하고 disconnect()는 호출하지 않음 (disconnect 시 채널이 닫혀서 SDK teardown에서 에러 유발)
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
