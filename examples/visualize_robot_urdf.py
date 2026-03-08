"""
rigid-transform-kit / examples / visualize_robot_urdf.py
=========================================================
URDF robot visualization + picking pipeline in Rerun.

Works with any robot vendor (FANUC, UR, ABB, KUKA, ...)
as long as a valid URDF file is provided.

Requires: ``pip install rigid-transform-kit[viz]``

URDF sources:
    FANUC  — https://github.com/ros-industrial/fanuc
    UR     — https://github.com/ros-industrial/universal_robot
    ABB    — https://github.com/ros-industrial/abb
    KUKA   — https://github.com/ros-industrial/kuka_experimental

Usage:
    python visualize_robot_urdf.py --urdf path/to/robot.urdf
    python visualize_robot_urdf.py --urdf fanuc_m10ia.urdf
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from rigid_transform_kit import (
    CameraConfig,
    PickPoint,
    build_tcp_pose,
)
from rigid_transform_kit.robot import FanucAdapter
from rigid_transform_kit.viz import TransformVisualizer


def main():
    parser = argparse.ArgumentParser(
        description="Visualize a robot URDF with picking pipeline in Rerun.",
    )
    parser.add_argument(
        "--urdf",
        type=str,
        default=None,
        help="Path to a .urdf file. If omitted, only the picking pipeline is shown.",
    )
    parser.add_argument(
        "--animate",
        action="store_true",
        help="Animate the robot joints with a sine wave demo.",
    )
    args = parser.parse_args()

    vis = TransformVisualizer("robot_picking_viz", spawn=True)

    # ── Load URDF (optional) ──────────────────────────────────

    urdf_tree = None
    if args.urdf:
        urdf_path = Path(args.urdf)
        print(f"Loading URDF: {urdf_path}")
        urdf_tree = vis.load_urdf(urdf_path)

        joint_names = [j.name for j in urdf_tree.joints() if j.joint_type in ("revolute", "prismatic", "continuous")]
        print(f"  Joints ({len(joint_names)}): {joint_names}")

    # ── Picking pipeline visualization ────────────────────────

    calibration_yml = {
        "camera_calibration": [
            [0.9998,  0.0175, -0.0087,  0.150],
            [-0.0174, 0.9998,  0.0052, -0.320],
            [0.0088, -0.0050,  0.9999,  1.200],
            [0.0,     0.0,     0.0,     1.0],
        ]
    }

    K = np.array([
        [610.0,   0.0, 320.0],
        [  0.0, 610.0, 240.0],
        [  0.0,   0.0,   1.0],
    ])
    D = np.zeros(5)

    cam_config = CameraConfig.from_calibration_dict(
        calib=calibration_yml,
        intrinsics=K,
        distortion=D,
        depth_scale=1.0,
        calib_convention="cam2base",
    )

    robot = FanucAdapter(tool_z_offset=0.100, pos_unit="mm")

    picks = [
        PickPoint(p_cam=[0.050, 0.120, 0.800], n_cam=[0.01, -0.02, -0.998]),
        PickPoint(p_cam=[0.060, 0.110, 0.790], n_cam=[0.0, 0.0, -1.0]),
        PickPoint(p_cam=[-0.030, 0.096, 0.810]),
    ]

    for i, pick in enumerate(picks):
        p_base, n_base = pick.to_base(cam_config)
        T_base2tcp = build_tcp_pose(p_base, n_base, contact_offset=0.005)
        T_base2flange = robot.compute_flange_target(
            robot.resolve_redundancy(T_base2tcp),
        )

        vis.log_picking_pipeline(
            cam_config, pick, T_base2tcp, T_base2flange,
            index=i,
        )

        cmd = robot.plan_pick(T_base2tcp)
        print(
            f"Pick #{i}: "
            f"X={cmd['X']:.1f} Y={cmd['Y']:.1f} Z={cmd['Z']:.1f} "
            f"W={cmd['W']:.2f} P={cmd['P']:.2f} R={cmd['R']:.2f}"
        )

    # ── Animate joints (optional) ─────────────────────────────

    if args.animate and urdf_tree is not None:
        joints = [j for j in urdf_tree.joints() if j.joint_type in ("revolute", "continuous")]
        print(f"\nAnimating {len(joints)} revolute joints...")

        trajectory = []
        for step in range(300):
            snapshot = {}
            for idx, joint in enumerate(joints):
                sin_val = math.sin(step * (0.02 + idx / 100.0))
                angle = joint.limit_lower + (sin_val + 1.0) / 3.0 * (joint.limit_upper - joint.limit_lower)
                snapshot[joint.name] = angle
            trajectory.append(snapshot)

        vis.animate_joints(trajectory, dt=0.03)
        print("Animation logged.")

    print("\nRerun viewer에서 확인하세요.")


if __name__ == "__main__":
    main()
