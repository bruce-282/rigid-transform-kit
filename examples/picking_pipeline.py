"""
rigid-transform-kit / examples / picking_pipeline.py
=====================================================
Full picking pipeline example:
    AI raw output → parse → vision transform → robot command

Shows how to replace the tangled legacy code with clean, traceable steps.
"""

import numpy as np
from typing import List, Optional
from dataclasses import dataclass

from rigid_transform_kit import (
    Frame,
    RigidTransform,
    CameraConfig,
    PickPoint,
    build_tcp_pose,
)
from rigid_transform_kit.robot import FanucAdapter


# ============================================================
# 1. AI Result Parser (프로젝트별 — kit 외부 코드)
# ============================================================

@dataclass
class PickingAIResult:
    """Picking AI 원시 출력 → 정제된 PickPoint + box_type 리스트.

    Responsibilities:
        - unknown 타입 필터링
        - legacy string format 파싱 ("x,y,z,rx,ry,rz")
        - box type 매핑 (e.g. AI label → 현장 코드)
    """

    pick_points: List[PickPoint]
    box_types: List[str]

    @classmethod
    def from_raw(
        cls,
        results: dict,
        exclude_types: set = {"unknown"},
        box_type_mapping: Optional[dict] = None,
    ) -> "PickingAIResult":
        """Parse raw AI backend output.

        Expected `results` format::

            {
                "box_types": ["typeA", "unknown", "typeB"],
                "suction_pts": [
                    "100.0,200.0,500.0,0.1,0.2,-0.9",   # str (legacy)
                    [100.0, 200.0, 500.0, 0.1, 0.2, -0.9],  # list (modern)
                    ...
                ]
            }

        Each suction_pts entry: [x, y, z, nx, ny, nz]
            - xyz: suction point in camera frame (mm from AI → converted to meters)
            - nxnynz: surface normal in camera frame (optional, can be 3-element)
        """
        raw_types = results["box_types"]
        raw_suction = results["suction_pts"]

        pick_points = []
        box_types = []

        for box_type, suction_raw in zip(raw_types, raw_suction):
            if box_type in exclude_types:
                continue

            coords = _parse_suction_pts(suction_raw)

            p_cam = np.array(coords[:3])
            n_cam = np.array(coords[3:6]) if len(coords) >= 6 else None

            pick_points.append(PickPoint(p_cam=p_cam, n_cam=n_cam))

            mapped = box_type
            if box_type_mapping:
                mapped = box_type_mapping.get(box_type, box_type)
            box_types.append(mapped)

        return cls(pick_points=pick_points, box_types=box_types)


def _parse_suction_pts(raw) -> List[float]:
    """Legacy format compatibility: str / list / tuple → float list."""
    if isinstance(raw, str):
        return [float(x.strip()) for x in raw.split(",")]
    if isinstance(raw, (list, tuple)):
        return [float(x) for x in raw]
    raise ValueError(f"Unexpected suction_pts format: {type(raw)}")


# ============================================================
# 2. Pipeline
# ============================================================

def run_picking_pipeline():
    """
    Complete pipeline: AI output → robot pick commands.

    Transform chain:
        p_cam  ──T_cam2base──▶  p_base  ──build_tcp_pose──▶  T_base2tcp
                                                                  │
                     robot_adapter.plan_pick()                    │
                          ┌───────────────────────────────────────┘
                          ▼
                  resolve_redundancy (Z-180° flip)
                          │
                  compute_flange_target (T_base2tcp @ T_tcp2flange)
                          │
                  to_robot_command (X,Y,Z,W,P,R)
    """

    # ── Setup (앱 초기화 시 한번) ────────────────────────────

    # Hand-eye calibration result (from calibration.yml)
    # 기존 코드의 camera_calibration = T_cam2base (4x4)
    calibration_yml = {
        "camera_calibration": [
            [0.9998,  0.0175, -0.0087,  150.0],
            [-0.0174, 0.9998,  0.0052, -320.0],
            [0.0088, -0.0050,  0.9999, 1200.0],
            [0.0,     0.0,     0.0,      1.0],
        ]
    }

    # Camera intrinsics (from camera SDK / calibration)
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
        depth_scale=0.001,
        calib_key="camera_calibration",
        calib_convention="cam2base",
    )

    robot = FanucAdapter(
        tool_z_offset=0.100,  # flange → suction tip: 100mm
        pos_unit="mm",
    )

    # Box type mapping (AI label → 현장 코드)
    box_type_mapping = {
        "box_A": "DW_A_001",
        "box_B": "DW_B_002",
        "box_C": "DW_C_003",
    }

    # ── Per-cycle (매 picking cycle마다) ─────────────────────

    # Simulated AI output (normally from picking AI inference)
    ai_raw_results = {
        "box_types": ["box_A", "unknown", "box_B", "box_C"],
        "suction_pts": [
            "50.5, 120.3, 800.0, 0.01, -0.02, -0.998",  # legacy string format
            "999, 999, 999, 0, 0, -1",                     # unknown → filtered out
            [60.0, 110.0, 790.0, 0.0, 0.0, -1.0],         # modern list format
            [-30.2, 95.7, 810.5],                           # no normal → default look-down
        ],
    }

    # Step 1: Parse AI output
    ai_result = PickingAIResult.from_raw(
        results=ai_raw_results,
        exclude_types={"unknown"},
        box_type_mapping=box_type_mapping,
    )

    print(f"Parsed {len(ai_result.pick_points)} valid picks "
          f"(filtered from {len(ai_raw_results['box_types'])} detections)")
    print(f"Box types: {ai_result.box_types}")
    print()

    # Step 2-4: Vision transform → Robot command
    robot_commands = []

    for i, (pick, box_type) in enumerate(
        zip(ai_result.pick_points, ai_result.box_types)
    ):
        # Step 2: Camera → Base frame
        p_base, n_base = pick.to_base(cam_config)

        # Step 3: Build TCP target pose
        T_base2tcp = build_tcp_pose(
            p_base, n_base,
            contact_offset=0.005,  # 5mm standoff
        )

        # Step 4: Robot command (redundancy + tool offset + vendor format)
        cmd = robot.plan_pick(T_base2tcp)
        robot_commands.append(cmd)

        # Debug: trace the full chain
        print(f"── Pick #{i} ({box_type}) ──")
        print(f"  p_cam:       {pick.p_cam}")
        print(f"  n_cam:       {pick.n_cam}")
        print(f"  p_base:      {p_base}")
        print(f"  n_base:      {n_base}")
        print(f"  T_base2tcp:  {T_base2tcp}")
        print(f"  FANUC cmd:   X={cmd['X']:.1f} Y={cmd['Y']:.1f} Z={cmd['Z']:.1f} "
              f"W={cmd['W']:.2f} P={cmd['P']:.2f} R={cmd['R']:.2f}")
        print()

    return robot_commands, ai_result.box_types


if __name__ == "__main__":
    commands, types = run_picking_pipeline()
    print(f"\n✓ Generated {len(commands)} robot commands")
