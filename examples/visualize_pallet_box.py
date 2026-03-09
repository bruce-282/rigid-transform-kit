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

from rigid_transform_kit import FanucAdapter, Frame, RigidTransform
from rigid_transform_kit.app import (
    extract_picks_from_boxes,
    load_calibration,
    load_cam_targets,
    log_robot_commands,
    picks_to_tcp_poses_base_and_cam,
)
from rigid_transform_kit.viz import TransformVisualizer, save_recording
from utils import load_ply_points

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
        default=[DEFAULT_DATA_DIR / "box_pcd1.ply", DEFAULT_DATA_DIR / "box_pcd2.ply",
        DEFAULT_DATA_DIR / "box_pcd3.ply", DEFAULT_DATA_DIR / "box_pcd4.ply",
        DEFAULT_DATA_DIR / "box_pcd5.ply", DEFAULT_DATA_DIR / "box_pcd6.ply"],
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
        "--port",
        type=int,
        default=None,
        metavar="PORT",
        help="Rerun gRPC port (default 9876). Use if port in use (e.g. Windows 10048). Env: RERUN_PORT",
    )
    return p.parse_args()


log = logging.getLogger(__name__)


def main():
    args = parse_args()

    T_base2cam, T_cam2base = load_calibration(args.calibration)

    if args.cam_targets is not None:
        picks = load_cam_targets(args.cam_targets)
        log.info("Loaded %d cam_targets from %s.", len(picks), args.cam_targets)
    elif args.box_pcd:
        picks = extract_picks_from_boxes(args.box_pcd)
    else:
        picks = []
        log.warning("No pick points found.")

    # rr.spawn() and rr.save() conflict — use spawn=False when saving to file
    spawn = args.save is None
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
    ply_data = load_ply_points(args.pcd)
    if ply_data is not None:
        pts_cam_m, colors_cam = ply_data
        pts_cam_mm = pts_cam_m * 1000.0  # load_ply_points returns meters

        pts_base = T_cam2base.transform_points(pts_cam_mm)
        vis.log_points("world/pcd", pts_base, colors=colors_cam, radii=3.0)
        n_pts = len(pts_base)
        log.info("Logged %d points from PLY (colors=%s).", n_pts, "yes" if colors_cam is not None else "no")
        if n_pts > 500_000 and spawn:
            log.warning(
                "Large point cloud (%d points). For fewer gRPC errors, use --save out.rrd then: rerun out.rrd",
                n_pts,
            )

    # ── TCP poses (base + cam frames) ──
    tcp_poses_base, tcp_poses_cam, has_axes = picks_to_tcp_poses_base_and_cam(picks, T_cam2base)

    fanuc = FanucAdapter(pos_unit="mm", tool_z_offset=200.0)
    robot_commands = [fanuc.plan_pick(pose) for pose in tcp_poses_base]
    labels = ["TCP" if ax else "center" for ax in has_axes]
    log_robot_commands(robot_commands, labels=labels, logger=log)

    if tcp_poses_base:
        vis.log_tcp_poses(tcp_poses_base, parent_path="world/picks", axis_length=100.0, arrow_radius=2.0, show_axes=has_axes)
        # Flange 좌표계도 Overview (in Base)에 표시 (resolve_redundancy 후 compute_flange_target)
        flange_poses_base = [
            fanuc.compute_flange_target(fanuc.resolve_redundancy(p)) for p in tcp_poses_base
        ]
        vis.log_flange_poses(flange_poses_base, parent_path="world/flanges", axis_length=100.0)

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
        log.info("Saved to %s", args.save)
        try:
            subprocess.run(["rerun", str(args.save)], check=False)
        except FileNotFoundError:
            log.info("Run: rerun %s", args.save)

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
