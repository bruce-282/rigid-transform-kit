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
pip install -e .
```

## Architecture

```
Vision Engineer (concrete)          Robot Engineer (abstract → implement)
─────────────────────────           ─────────────────────────────────────
CameraConfig                        BaseRobotAdapter
  └─ hand-eye calibration              ├─ get_tool_transform()
PickPoint                              ├─ resolve_redundancy()
  └─ AI output → base frame            └─ to_robot_command()
build_tcp_pose()                    
  └─ position + normal → T_base2tcp FanucAdapter (example)
                                    URAdapter (TODO)
        ──── responsibility boundary ────
              T_base2tcp handoff
```

## Quick Start

```python
from rigid_transform_kit import CameraConfig, PickPoint, build_tcp_pose
from rigid_transform_kit.adapters import FanucAdapter

# 1. Camera config
cam_config = CameraConfig.from_calibration_dict(
    calib={"camera_calibration": T_cam2base_4x4.tolist()},
    intrinsics=K, distortion=D,
)

# 2. AI output → base frame
pick = PickPoint(p_cam=ai_result["suction_xyz"], n_cam=ai_result["normal"])
p_base, n_base = pick.to_base(cam_config)

# 3. TCP pose (vision team's final output)
T_base2tcp = build_tcp_pose(p_base, n_base, contact_offset=0.005)

# 4. Robot command (robot team's domain)
robot = FanucAdapter(tool_z_offset=0.10)
cmd = robot.plan_pick(T_base2tcp)
# {"X": ..., "Y": ..., "Z": ..., "W": ..., "P": ..., "R": ...}
```

## File Structure

```
rigid_transform_kit/
├── __init__.py          # public API
├── core.py              # Frame, RigidTransform
├── vision.py            # CameraConfig, PickPoint, build_tcp_pose
└── adapters/
    ├── __init__.py
    ├── base.py          # BaseRobotAdapter (ABC)
    └── fanuc.py         # FanucAdapter (reference impl)
tests/
├── test_core.py
├── test_vision.py
└── test_adapters.py
```

## Extending

### New robot vendor

```python
from rigid_transform_kit.adapters.base import BaseRobotAdapter

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
