# rigid-transform-kit

Frame-aware rigid transforms for industrial vision-robot pipelines.

## Why

산업용 비전-로봇 시스템에서 좌표 변환은 항상 버그의 온상이다:
- `camera_calibration`이 `T_cam2base`인지 `T_base2cam`인지 헷갈림
- `np.linalg.inv`를 잘못 적용
- 변환 체인에서 프레임 순서 실수

`RigidTransform`은 **프레임 라벨을 강제**해서 이런 실수를 런타임에 잡아준다.

```python
T_base2cam @ T_flange2tcp  # → ValueError: Frame mismatch!
T_base2cam @ T_cam2tcp     # → OK: T(BASE→TCP)
```

## Install

```bash
# base (numpy + scipy만)
pip install -e .

# with Rerun 3D visualization
pip install -e ".[viz]"

# with dev/test tools
pip install -e ".[dev]"
```

### Build Scripts

```bash
# Linux (pip)
./scripts/build/build_linux_pip.sh          # base
./scripts/build/build_linux_pip.sh viz      # with viz

# Linux (uv)
./scripts/build/build_linux_uv.sh 3.11 viz,dev

# Windows (uv, PowerShell)
.\scripts\build\build_win_uv.ps1 3.11 viz
```



**의존 방향:** `vision → core ← robot`, `viz → core + vision + robot`

- **core** — 프레임 enum + 변환 수학. 양쪽 다 의존
- **vision** — 카메라 설정, 피킹 포인트. `core`만 의존
- **robot** — TCP 포즈 빌더, 어댑터. `core`만 의존. vision 몰라도 됨
- **viz** — Rerun 3D 시각화. 전체를 시각화하므로 모두 참조 (optional dependency)

## Quick Start

```python
from rigid_transform_kit import CameraConfig, PickPoint, build_tcp_pose
from rigid_transform_kit.robot import FanucAdapter

# 1. Camera config (vision 도메인)
cam_config = CameraConfig.from_calibration_dict(
    calib={"camera_calibration": T_cam2base_4x4.tolist()},
    intrinsics=K, distortion=D,
)

# 2. AI output → base frame (vision 도메인)
pick = PickPoint(p_cam=ai_result["suction_xyz"], n_cam=ai_result["normal"])
p_base, n_base = pick.to_base(cam_config)

# 3. TCP pose (robot 도메인)
T_base2tcp = build_tcp_pose(p_base, n_base, contact_offset=0.005)

# 4. Robot command (robot 도메인)
robot = FanucAdapter(tool_z_offset=0.10)
cmd = robot.plan_pick(T_base2tcp)
# {"X": ..., "Y": ..., "Z": ..., "W": ..., "P": ..., "R": ...}
```

### 3D Visualization (optional)

```python
from rigid_transform_kit.viz import TransformVisualizer

vis = TransformVisualizer("my_pipeline")
vis.log_picking_pipeline(cam_config, pick, T_base2tcp)
```

## File Structure

```
src/rigid_transform_kit/
├── __init__.py              # top-level re-export
├── core/
│   ├── __init__.py          # Frame, RigidTransform
│   ├── frame.py             # Frame enum
│   └── transform.py         # RigidTransform
├── vision/
│   ├── __init__.py          # CameraConfig, PickPoint
│   ├── camera.py            # CameraConfig
│   └── pick.py              # PickPoint
├── robot/
│   ├── __init__.py          # BaseRobotAdapter, FanucAdapter, build_tcp_pose
│   ├── base.py              # BaseRobotAdapter (ABC)
│   ├── fanuc.py             # FanucAdapter (reference impl)
│   └── tcp.py               # build_tcp_pose
└── viz/
    ├── __init__.py          # TransformVisualizer
    └── visualizer.py        # Rerun visualization (optional)

scripts/build/
├── build_linux_pip.sh       # Linux pip build
├── build_linux_uv.sh        # Linux uv build
└── build_win_uv.ps1         # Windows uv build

examples/
├── picking_pipeline.py      # Full pipeline example
└── visualize_pipeline.py    # Pipeline + Rerun 3D visualization
```

## Extending

### New robot vendor

```python
from rigid_transform_kit.robot import BaseRobotAdapter

class URAdapter(BaseRobotAdapter):
    def get_tool_transform(self) -> RigidTransform: ...
    def resolve_redundancy(self, T_base2tcp): ...
    def to_robot_command(self, T_base2flange): ...
```

### New frame

```python
from rigid_transform_kit.core import Frame
# Frame enum is open for extension if needed via custom subclass
```
