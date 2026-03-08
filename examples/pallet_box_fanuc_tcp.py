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
import logging
from pathlib import Path

from rigid_transform_kit import FanucAdapter
from rigid_transform_kit.app import (
    build_tcp_result,
    extract_picks_from_boxes,
    load_calibration,
    load_cam_targets,
    log_robot_commands,
    log_tcp_flange_detail,
    picks_to_tcp_poses,
    save_tcp_poses,
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
    p.add_argument(
        "--tool-z-offset",
        type=float,
        default=100.0,
        metavar="MM",
        help="Flange→TCP offset along Z in mm (default: 0 = no tool offset)",
    )
    p.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable DEBUG logging for the full pipeline trace",
    )
    return p.parse_args()


log = logging.getLogger(__name__)


def main():
    args = parse_args()
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    _, T_cam2base = load_calibration(args.calibration)
    log.info("Calibration loaded: %s", args.calibration)

    if args.cam_targets is not None:
        picks = load_cam_targets(args.cam_targets)
        log.info("Loaded %d cam_targets from %s", len(picks), args.cam_targets)
    elif args.box_pcd:
        log.info("Extracting picks from %d box PLY files:", len(args.box_pcd))
        picks = extract_picks_from_boxes(args.box_pcd)
    else:
        picks = []

    if not picks:
        log.warning("No pick points found.")
        return

    tcp_poses = picks_to_tcp_poses(picks, T_cam2base)

    fanuc = FanucAdapter(tool_z_offset=args.tool_z_offset, pos_unit="mm")

    robot_commands = [fanuc.plan_pick(pose) for pose in tcp_poses]
    result = build_tcp_result(tcp_poses, robot_commands)

    log_robot_commands(robot_commands, logger=log)

    if args.verbose:
        flange_poses = [
            fanuc.compute_flange_target(fanuc.resolve_redundancy(p))
            for p in tcp_poses
        ]
        log_tcp_flange_detail(tcp_poses, flange_poses, logger=log)

    if args.output is not None:
        save_tcp_poses(result, args.output)
        log.info("Saved %d TCP poses to %s", len(tcp_poses), args.output)


if __name__ == "__main__":
    main()
