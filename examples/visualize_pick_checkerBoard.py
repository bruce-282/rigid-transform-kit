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
import json
import logging
import os
import subprocess
import time
from pathlib import Path

import numpy as np

from rigid_transform_kit import Frame, RigidTransform
from rigid_transform_kit.viz import TransformVisualizer, save_recording

try:
    import cv2
except ImportError:
    cv2 = None

from utils import load_ply_points
from utils.checkerboard import detect_checkerboard_pose, undistort_point_cloud

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def load_intrinsics_for_checkerboard(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load K (3,3) and dist (5,) from JSON. Supports sensores.image or plain \"K\"+\"dist\"."""
    try:
        from utils import load_intrinsics
        return load_intrinsics(path)
    except (KeyError, ImportError):
        pass
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if "K" in data:
        K = np.array(data["K"], dtype=np.float64)
        dist = np.array(data.get("dist", [0.0] * 5), dtype=np.float64)
    else:
        raise KeyError("JSON must have sensores.image (intrinsic_matrix, distortion_coefficients) or K+dist")
    if len(dist) > 5:
        dist = dist[:5]
    return K, dist


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
        default=50.0,
        help="Axis length in mm for visualization. Default 50",
    )
    p.add_argument(
        "--pcd",
        type=Path,
        default=None,
        metavar="PLY",
        help="Optional: point cloud PLY (camera frame); with --calibration shown in base frame",
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
    K, dist = load_intrinsics_for_checkerboard(args.intrinsics)
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

    pts_cam_mm = None
    pts_base = None
    # colors_cam already set from ply_data above; do not overwrite

    # Optional base frame
    T_base2cam = None
    if args.calibration is not None and args.calibration.exists():
        from utils import load_extrinsics
        calib = load_extrinsics(args.calibration)
        T_base2cam = RigidTransform.from_matrix(calib["base2cam"], Frame.BASE, Frame.CAMERA)
        T_base2board = T_base2cam @ T_cam2board
        log.info("Calibration loaded: board in base frame")

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

    # World = camera frame (or base if we have calibration).
    # Robot base: right-handed, Z-up (identity = no transform; axes follow calibration).
    if T_base2cam is not None:
        vis.log_transform(
            "world/base",
            RigidTransform.identity(Frame.BASE),
            axis_length=args.axis_length,
            label="BASE",
        )
        vis.log_transform(
            "world/camera",
            RigidTransform.from_Rt(T_base2cam.R, T_base2cam.t, Frame.BASE, Frame.CAMERA),
            axis_length=args.axis_length,
            label="CAMERA",
        )
        vis.log_transform(
            "world/checkerboard",
            T_base2board,
            axis_length=args.axis_length,
            label="BOARD",
        )
    else:
        vis.log_transform(
            "world/camera",
            RigidTransform.identity(Frame.CAMERA),
            axis_length=args.axis_length,
            label="CAMERA",
        )
        vis.log_transform(
            "world/checkerboard",
            T_cam2board,
            axis_length=args.axis_length,
            label="BOARD",
        )

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
        if T_base2cam is not None:
            T_cam2base = T_base2cam.inv
            pts_base = T_cam2base.transform_points(pts_cam_mm)
        else:
            pts_base = None
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

    vis.log_scene_in_camera(
        pts_cam=pts_cam_mm,
        colors=colors_cam,
        tcp_poses=[T_cam2board],
        show_axes=[True],
    )

    vis.log_scene_base(
        pts_base=pts_base,
        colors=colors_cam,
        tcp_poses=[T_base2board] if T_base2cam is not None else None,
        show_axes=[True] if T_base2cam is not None else None,
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
