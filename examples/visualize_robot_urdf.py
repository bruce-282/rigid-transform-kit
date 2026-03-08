"""
rigid-transform-kit / examples / visualize_robot_urdf.py
=========================================================
Visualize a robot URDF model in Rerun.

Requires: ``pip install rigid-transform-kit[viz]``

URDF sources:
    FANUC  — https://github.com/ros-industrial/fanuc
    UR     — https://github.com/ros-industrial/universal_robot
    ABB    — https://github.com/ros-industrial/abb
    KUKA   — https://github.com/ros-industrial/kuka_experimental

Usage:
    python examples/visualize_robot_urdf.py
    python examples/visualize_robot_urdf.py --urdf path/to/robot.urdf
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from rigid_transform_kit.viz import TransformVisualizer

DEFAULT_URDF = Path(__file__).resolve().parent.parent / "data" / "robot" / "fanuc_m710ic_description" / "urdf" / "m710ic70.urdf"


def main():
    parser = argparse.ArgumentParser(
        description="Visualize a robot URDF model in Rerun.",
    )
    parser.add_argument(
        "--urdf",
        type=Path,
        default=DEFAULT_URDF,
        help=f"Path to a .urdf file (default: {DEFAULT_URDF})",
    )
    args = parser.parse_args()

    if not args.urdf.exists():
        print(f"URDF not found: {args.urdf}")
        return

    vis = TransformVisualizer("robot_urdf_viewer", spawn=True)

    print(f"Loading URDF: {args.urdf}")
    urdf_tree = vis.load_urdf(args.urdf)

    joint_names = [
        j.name for j in urdf_tree.joints()
        if j.joint_type in ("revolute", "prismatic", "continuous")
    ]
    print(f"  Joints ({len(joint_names)}): {joint_names}")

    # ── Gentle joint animation ──
    n_steps = 200
    amplitude = math.radians(30)
    trajectory = []
    for step in range(n_steps):
        t = step / n_steps * 2 * math.pi
        snapshot = {}
        for idx, name in enumerate(joint_names):
            phase = idx * math.pi / len(joint_names)
            snapshot[name] = amplitude * math.sin(t + phase)
        trajectory.append(snapshot)

    print(f"Animating {len(joint_names)} joints ({n_steps} steps)...")
    vis.animate_joints(trajectory, dt=0.05, urdf_tree=urdf_tree)

    print("Done. Rerun viewer에서 확인하세요.")


if __name__ == "__main__":
    main()
