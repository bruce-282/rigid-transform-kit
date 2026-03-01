"""
rigid-transform-kit / examples / visualize_pipeline.py
======================================================
Picking pipeline + Rerun 3D visualization.

Requires: ``pip install rigid-transform-kit[viz]``
"""

import numpy as np

from rigid_transform_kit import (
    Frame,
    RigidTransform,
    CameraConfig,
    PickPoint,
    build_tcp_pose,
)
from rigid_transform_kit.robot import FanucAdapter
from rigid_transform_kit.viz import TransformVisualizer


def main():
    # ── Setup ─────────────────────────────────────────────────

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

    # ── Simulated picks ───────────────────────────────────────

    picks = [
        PickPoint(p_cam=[0.050, 0.120, 0.800], n_cam=[0.01, -0.02, -0.998]),
        PickPoint(p_cam=[0.060, 0.110, 0.790], n_cam=[0.0, 0.0, -1.0]),
        PickPoint(p_cam=[-0.030, 0.096, 0.810]),
    ]

    # ── Visualize ─────────────────────────────────────────────

    vis = TransformVisualizer("picking_pipeline", spawn=True)

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

    print("\nRerun viewer에서 3D 시각화를 확인하세요.")


if __name__ == "__main__":
    main()
