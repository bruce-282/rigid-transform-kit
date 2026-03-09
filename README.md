# rigid-transform-kit

Frame-aware rigid transforms for industrial vision-robot pipelines.

```python
T_base2cam @ T_flange2tcp  # → ValueError: Frame mismatch!
T_base2cam @ T_cam2tcp     # → OK: T(BASE→TCP)
```

## Install

```bash
# base (numpy, scipy, pyyaml)
pip install -e .

# with Rerun 3D visualization (rerun-sdk, plyfile)
pip install -e ".[viz]"

# with dev/test tools
pip install -e ".[dev]"
```

- **open3d** — `examples/visualize_pallet_box.py`에서 `--box-pcd`(PLY/박스 PCD)를 쓰려면 별도 설치.  
  open3d는 Python 3.8~3.12만 지원하므로, 해당 예제용으로는 3.12 이하 권장.

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

# 1. Camera config (vision 도메인, base2cam 통일)
cam_config = CameraConfig.from_calibration_dict(
    calib={"base2cam": T_base2cam_4x4.tolist()},
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

### Example scripts

```bash
# 캘리브레이션 + PCD + 피킹 포인트 시각화 (pip install -e ".[viz]" 필요)
python examples/visualize_pallet_box.py --calibration path/to/calib.yml --pcd scene.ply

# 결과를 파일로 저장 후 뷰어로 열기 (실시간 뷰어 대신, gRPC 종료 오류 회피)
python examples/visualize_pallet_box.py --calibration path/to/calib.yml --pcd scene.ply --save out.rrd
rerun out.rrd

# 포트 9876 사용 중일 때 (Windows 10048 등): --port 9877 또는 set RERUN_PORT=9877
python examples/visualize_pallet_box.py --calibration path/to/calib.yml --pcd scene.ply --port 9877
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
├── app/
│   ├── __init__.py          # load_calibration, extract_picks_from_boxes, ...
│   ├── io.py                # calibration I/O
│   └── pallet.py            # pallet/box picking helpers
└── viz/
    ├── __init__.py          # TransformVisualizer
    ├── visualizer.py        # Rerun visualization (optional)
    └── urdf_viewer.py       # URDF loading for Rerun

src/utils/                    # shared utilities (calibration, PCD, datasets)
├── dataset_loader.py        # load_extrinsics (YAML), load_ply_points, ...
└── pcd_processing.py        # open3d-based outlier removal, downsampling (optional)

data/robot/                   # URDF robot descriptions
├── fanuc_m710ic_description/
│   ├── urdf/m710ic70.urdf
│   └── meshes/m710ic50/...  # STL meshes (visual + collision)
└── fanuc_r2000ic_description/
    ├── urdf/r2000ic165f.urdf
    └── meshes/r2000ic165f/... # STL meshes (visual + collision)

scripts/build/
├── build_linux_pip.sh       # Linux pip build
├── build_linux_uv.sh        # Linux uv build
└── build_win_uv.ps1         # Windows uv build

examples/
├── visualize_pallet_box.py      # Calibration + PCD + pick points in Rerun ([viz], optional open3d)
├── visualize_robot_urdf.py    # URDF robot + pipeline visualization
└── pallet_box_fanuc_tcp.py    # Pallet/box → Fanuc TCP example
```

## URDF Robot Models (optional)

`data/robot/` 디렉토리에 로봇 URDF description 패키지를 관리한다.
현재 포함된 모델:

| 패키지 | 로봇 | Payload | Reach |
|--------|------|---------|-------|
| `fanuc_m710ic_description` | FANUC M-710iC/70 | 70 kg | 2050 mm |
| `fanuc_r2000ic_description` | FANUC R-2000iC/165F | 165 kg | 2655 mm |

### 새 로봇 모델 추가하기

#### 1. 디렉토리 구조 만들기

`data/robot/` 아래에 `<vendor>_<model>_description/` 폴더를 만든다.
반드시 아래 구조를 따라야 `load_urdf()`가 `package://` URI를 올바르게 해석한다.

```
data/robot/<vendor>_<model>_description/
├── urdf/
│   └── <model>.urdf          # 순수 URDF (xacro 아님)
├── meshes/
│   └── <variant>/
│       ├── visual/           # 시각화용 STL
│       │   ├── base_link.stl
│       │   ├── link_1.stl
│       │   └── ...
│       └── collision/        # 충돌 검사용 STL
│           ├── base_link.stl
│           ├── link_1.stl
│           └── ...
├── package.xml               # ROS 패키지 메타 (선택)
├── LICENSE
└── README.md
```

#### 2. URDF 작성 규칙

- **순수 URDF만 사용** — xacro 매크로(`${prefix}`, `${radians(...)}` 등)는 전부 풀어서 작성
- **mesh 경로** — `package://<패키지명>/meshes/...` 형태 사용
  ```xml
  <mesh filename="package://fanuc_r2000ic_description/meshes/r2000ic165f/visual/link_1.stl"/>
  ```
- **조인트 리밋** — 라디안 단위로 직접 기입 (도 → 라디안: `deg × π / 180`)
- **표준 프레임** — `base_link`, `link_1`~`link_6`, `flange`, `tool0` 이름 권장
- **fixed joints** — `base_link-base`, `joint_6-flange`, `link_6-tool0` 포함 권장

#### 3. URDF 소스 구하기

ros-industrial에서 대부분의 산업용 로봇 URDF를 구할 수 있다:

| 벤더 | 레포 |
|------|------|
| FANUC | https://github.com/ros-industrial/fanuc |
| UR | https://github.com/ros-industrial/universal_robot |
| ABB | https://github.com/ros-industrial/abb |
| KUKA | https://github.com/ros-industrial/kuka_experimental |
| Yaskawa | https://github.com/ros-industrial/motoman |

대부분 xacro 형태이므로, `radians()` 등의 매크로를 직접 계산해서 순수 URDF로 변환해야 한다.

#### 4. 시각화 실행

```bash
# URDF만 보기
python examples/visualize_robot_urdf.py --urdf data/robot/fanuc_r2000ic_description/urdf/r2000ic165f.urdf

# 조인트 애니메이션 포함
python examples/visualize_robot_urdf.py --urdf data/robot/fanuc_r2000ic_description/urdf/r2000ic165f.urdf --animate
```

#### 5. 코드에서 사용

```python
from rigid_transform_kit.viz import TransformVisualizer

vis = TransformVisualizer("my_robot", spawn=True)

# package:// URI 자동 해석 (urdf 파일의 grandparent = description 폴더의 parent)
tree = vis.load_urdf("data/robot/fanuc_r2000ic_description/urdf/r2000ic165f.urdf")

# 조인트 각도 설정
vis.set_joint_angles({"joint_1": 0.5, "joint_2": -0.3, "joint_3": 0.8})
```

> **`package://` 경로 해석 원리**: `load_urdf()`는 URDF 파일의 2단계 상위 디렉토리를
> `ROS_PACKAGE_PATH`에 자동 등록한다. 따라서 `urdf/<model>.urdf` 파일이
> `<pkg_name>/urdf/` 안에 있으면 별도 설정 없이 `package://<pkg_name>/meshes/...`가 해석된다.

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
