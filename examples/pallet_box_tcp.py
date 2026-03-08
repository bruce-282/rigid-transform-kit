"""
rigid-transform-kit / examples / pallet_box_tcp.py
===================================================
Compute TCP poses from pallet box point clouds (no visualization).

Loads calibration + box PLY files, extracts pick points via
plane fitting + 2D PCA, transforms to base frame, and builds
TCP poses.  Results are printed to stdout and optionally saved
to a JSON file.

Usage:
  python examples/pallet_box_tcp.py
  python examples/pallet_box_tcp.py --data-dir /path/to/dataset
  python examples/pallet_box_tcp.py --box-pcd box1.ply box2.ply
  python examples/pallet_box_tcp.py --output tcp_poses.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from rigid_transform_kit import (
    Frame,
    PickPoint,
    RigidTransform,
    build_tcp_pose,
)
from utils import (
    fit_plane,
    get_box_axes,
    load_box_pcd,
    load_extrinsics,
    load_cam_targets,
)

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "datasets" / "aw_pallet"


def parse_args():
    p = argparse.ArgumentParser(
        description="Compute TCP poses from pallet box point clouds.",
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
        help="Target points JSON (optional, alternative to --box-pcd)",
    )
    p.add_argument(
        "--box-pcd",
        type=Path,
        nargs="*",
        default=[
            DEFAULT_DATA_DIR / "box_pcd1.ply",
            DEFAULT_DATA_DIR / "box_pcd2.ply",
            DEFAULT_DATA_DIR / "box_pcd3.ply",
        ],
        metavar="PLY",
        help="Box PLY file(s) for pick point extraction; one pick per file",
    )
    p.add_argument(
        "--output", "-o",
        type=Path,
        default="datasets/output/tcp_poses.json",
        metavar="JSON",
        help="Save TCP poses to JSON file",
    )
    return p.parse_args()


def extract_picks_from_boxes(box_paths: list[Path]) -> list[PickPoint]:
    """Extract one PickPoint per box PLY via plane fitting + 2D PCA."""
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
        print(
            f"  {path.name}: center=({center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f}), "
            f"long={info['extent_long']:.1f}, short={info['extent_short']:.1f}, "
            f"aspect={info['aspect_ratio']:.2f}"
        )
    return picks


def picks_to_tcp_poses(
    picks: list[PickPoint],
    T_cam2base: RigidTransform,
) -> list[RigidTransform]:
    """Convert camera-frame PickPoints to base-frame TCP poses."""
    tcp_poses: list[RigidTransform] = []
    for pick in picks:
        p_cam_mm = pick.p_cam
        if np.median(np.abs(p_cam_mm)) < 10:
            p_cam_mm = p_cam_mm * 1000.0
        p_base = T_cam2base.transform_point(p_cam_mm)

        n_cam = pick.n_cam if pick.n_cam is not None else np.array([0.0, 0.0, -1.0])
        n_base = T_cam2base.transform_direction(n_cam)
        n_base = n_base / (np.linalg.norm(n_base) + 1e-12)

        long_hint = None
        if pick.long_axis_cam is not None:
            long_hint = T_cam2base.transform_direction(pick.long_axis_cam)
            long_hint = long_hint / (np.linalg.norm(long_hint) + 1e-12)

        tcp_poses.append(build_tcp_pose(p_base, n_base, long_axis_hint=long_hint))
    return tcp_poses


def save_tcp_poses(poses: list[RigidTransform], path: Path) -> None:
    """Save TCP poses as JSON (4x4 matrices, mm)."""
    data = []
    for i, pose in enumerate(poses):
        data.append({
            "index": i,
            "position_mm": pose.t.tolist(),
            "rotation_3x3": pose.R.tolist(),
            "matrix_4x4": pose.matrix.tolist(),
        })
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"\nSaved {len(data)} TCP poses to {path}")


def main():
    args = parse_args()

    # ── Load calibration ──
    calib = load_extrinsics(args.calibration)
    T_base2cam = RigidTransform.from_matrix(calib["base2cam"], Frame.BASE, Frame.CAMERA)
    T_cam2base = T_base2cam.inv
    print(f"Calibration loaded: {args.calibration}")

    # ── Extract pick points ──
    picks: list[PickPoint] = []

    if args.cam_targets is not None:
        picks = load_cam_targets(args.cam_targets)
        print(f"Loaded {len(picks)} cam_targets from {args.cam_targets}")

    elif args.box_pcd:
        print(f"Extracting picks from {len(args.box_pcd)} box PLY files:")
        picks.extend(extract_picks_from_boxes(args.box_pcd))

    if not picks:
        print("No pick points found.")
        return

    # ── Compute TCP poses ──
    tcp_poses = picks_to_tcp_poses(picks, T_cam2base)

    print(f"\n{'#':>3}  {'X':>10}  {'Y':>10}  {'Z':>10}  (mm, base frame)")
    print("-" * 50)
    for i, pose in enumerate(tcp_poses):
        print(f"{i:3d}  {pose.t[0]:10.1f}  {pose.t[1]:10.1f}  {pose.t[2]:10.1f}")

    # ── Save (optional) ──
    if args.output is not None:
        save_tcp_poses(tcp_poses, args.output)


if __name__ == "__main__":
    main()
