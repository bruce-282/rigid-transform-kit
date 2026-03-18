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
from utils.checkerboard import detect_checkerboard_pose

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

    # Detect checkerboard pose (RGB-only util)
    T_cam2board, corners = detect_checkerboard_pose(
        img_rgb,
        pattern_size,
        args.square_size,
        K,
        dist,
        refine_corners=True,
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

    # ── Optional PLY (camera frame; base frame if calibration given) ──
    if args.pcd is not None and args.pcd.exists():
        ply_data = load_ply_points(args.pcd)
        if ply_data is not None:
            pts_cam_m, colors_cam = ply_data
            pts_cam_mm = pts_cam_m * 1000.0
            if T_base2cam is not None:
                T_cam2base = T_base2cam.inv
                pts_world = T_cam2base.transform_points(pts_cam_mm)
            else:
                pts_world = pts_cam_mm
            vis.log_points("world/pcd", pts_world, colors=colors_cam, radii=3.0)
            n_pts = len(pts_world)
            log.info("Logged %d points from PLY.", n_pts)
            if n_pts > 500_000 and spawn:
                log.warning(
                    "Large point cloud (%d). Use --save out.rrd then: rerun out.rrd to avoid gRPC errors.",
                    n_pts,
                )
        else:
            log.warning("Could not load PLY: %s", args.pcd)
    elif args.pcd is not None:
        log.warning("PLY file not found: %s", args.pcd)

    if args.save is not None:
        save_recording(args.save)
        log.info("Saved to %s. Run: rerun %s", args.save, args.save)
        try:
            subprocess.run(["rerun", str(args.save)], check=False)
        except FileNotFoundError:
            pass

    if spawn:
        log.info("Rerun viewer: world/camera, world/checkerboard.")
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
