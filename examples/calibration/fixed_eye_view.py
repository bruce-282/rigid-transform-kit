"""
calibration / fixed_eye_view.py
================================
Fixed-eye (eye-to-hand) 시각화: 카메라가 고정, ``cam_to_base`` 로 직접 변환.

변환::

    p_base = cam_to_base @ p_cam

Usage::

  python examples/calibration/fixed_eye_view.py
  python examples/calibration/fixed_eye_view.py --save output/fixed_eye.rrd
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from rigid_transform_kit import Frame, RigidTransform
from rigid_transform_kit.viz import TransformVisualizer

from _viz_common import (
    add_common_args,
    finalize_viewer,
    load_4x4_matrices,
    load_and_preprocess_ply,
    log_camera_frustum,
    log_debug_link,
    log_origin_spheres,
    parse_tcp_vec6_from_filename,
)

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_DATA_DIR = REPO_ROOT / "datasets/fixed_eye_example"
_DEFAULT_PLY = next(_DEFAULT_DATA_DIR.glob("*.ply"), _DEFAULT_DATA_DIR / "scan.ply")
_DEFAULT_CAL = _DEFAULT_DATA_DIR / "fixed_eye_cal.yml"

_CAL_KEYS = ("cam_to_base", "base_to_cam", "flange_to_world", "world_to_flange")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fixed-eye viewer: PLY + cam_to_base → robot base frame (Rerun).",
    )
    p.add_argument("--ply", type=Path, default=_DEFAULT_PLY, help="PLY (camera frame).")
    p.add_argument("--calibration", type=Path, default=_DEFAULT_CAL, help="YAML with cam_to_base.")
    add_common_args(p)
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()

    # ── Calibration ──
    cal = load_4x4_matrices(args.calibration, _CAL_KEYS)

    # cam_to_base: p_base = M @ p_cam → from=CAMERA, to=BASE
    if "cam_to_base" in cal:
        T_cam_to_base = RigidTransform.from_matrix(cal["cam_to_base"], Frame.CAMERA, Frame.BASE)
        log.info("Loaded cam_to_base from %s", args.calibration)
    elif "base_to_cam" in cal:
        T_cam_to_base = RigidTransform.from_matrix(cal["base_to_cam"], Frame.BASE, Frame.CAMERA).inv
        log.info("Loaded base_to_cam (inverted) from %s", args.calibration)
    else:
        raise KeyError(f"Need 'cam_to_base' or 'base_to_cam': {args.calibration}")

    # flange_to_world: p_world = M @ p_flange → from=FLANGE, to=WORLD
    T_flange_to_world: RigidTransform | None = None
    if "flange_to_world" in cal:
        T_flange_to_world = RigidTransform.from_matrix(cal["flange_to_world"], Frame.FLANGE, Frame.WORLD)
        log.info("Loaded flange_to_world from %s", args.calibration)

    # ── TCP from filename ──
    vec6 = parse_tcp_vec6_from_filename(args.ply)
    T_flange_to_base: RigidTransform | None = None
    if vec6 is not None:
        T_flange_to_base = RigidTransform.from_vec6(vec6, Frame.FLANGE, Frame.BASE)
        log.info("TCP: xyz=(%.1f,%.1f,%.1f) WPR=(%.1f,%.1f,%.1f)", *vec6)
    else:
        log.warning("Could not parse TCP from filename: %s", args.ply.name)

    # ── PLY ──
    pts_cam_mm, colors = load_and_preprocess_ply(
        args.ply,
        no_depth_clip=args.no_depth_clip,
        depth_min_m=args.depth_min_m,
        depth_max_m=args.depth_max_m,
        depth_axis=args.depth_axis,
        max_points=args.max_points,
    )
    pts_base = T_cam_to_base.transform_points(pts_cam_mm)
    log.info("Transformed %d points to base frame.", len(pts_base))

    # ── Rerun ──
    save_path = args.save
    if save_path is None and len(pts_base) > 500_000:
        save_path = _DEFAULT_DATA_DIR / "fixed_eye.rrd"
    spawn = save_path is None
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)

    vis = TransformVisualizer("fixed_eye_view", spawn=spawn, port=args.port,
                              views=[("Fixed-Eye (base frame)", "world")])

    vis.log_transform("world/base", RigidTransform.identity(Frame.BASE),
                      axis_length=300.0, label="BASE")
    vis.log_transform("world/camera", T_cam_to_base,
                      axis_length=150.0, label="CAMERA")
    log_camera_frustum("world/camera/frustum", image_plane_distance=90.0)

    if T_flange_to_base is not None:
        vis.log_transform("world/flange", T_flange_to_base,
                          axis_length=200.0, label="FLANGE")
        if T_flange_to_world is not None:
            # world → flange → base (시각화 기준이 base이므로 역방향)
            # p_base = inv(T_flange_to_world @ inv(T_flange_to_base)) @ p_world
            T_world_to_base = RigidTransform.from_matrix(
                T_flange_to_base.matrix @ np.linalg.inv(T_flange_to_world.matrix),
                Frame.WORLD, Frame.BASE,
            )
            vis.log_transform("world/world_frame", T_world_to_base,
                              axis_length=240.0, label="WORLD")

    # origin spheres
    origins = [
        ("world/_origins/base", np.zeros(3), (255, 80, 80), "BASE"),
        ("world/_origins/camera", T_cam_to_base.t, (80, 120, 255), "CAMERA"),
    ]
    if T_flange_to_base is not None:
        origins.append(("world/_origins/flange", T_flange_to_base.t, (80, 255, 80), "FLANGE"))
    log_origin_spheres(origins)

    # point cloud
    if colors is None:
        colors = np.tile(np.array([[40, 180, 255]], dtype=np.uint8), (len(pts_base), 1))
    vis.log_points("world/pcd", pts_base, colors=colors, radii=1.2)
    log.info("Logged %d points.", len(pts_base))

    # debug links
    log_debug_link("world/_debug/base_to_cam", np.zeros(3), T_cam_to_base.t)
    if T_flange_to_base is not None:
        log_debug_link("world/_debug/base_to_flange", np.zeros(3), T_flange_to_base.t,
                       color=(80, 255, 80, 120))

    finalize_viewer(save_path, spawn=spawn, app_name="Fixed-Eye (base frame)")


if __name__ == "__main__":
    main()
