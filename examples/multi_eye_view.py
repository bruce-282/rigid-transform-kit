"""
examples / multi_eye_view.py
=============================
**설계 목표 (단일 좌표계: camera 1):** 3D 씬 안에서 모든 기하를 **cam1 좌표계**로 맞춥니다.

* **cam1 PLY** — cam1 좌표로 적혀 있다고 보고 ``world/scene/cam1`` 에 그대로 표시합니다.
* **cam2 PLY** — extrinsic으로 **cam1 좌표**로 변환한 뒤 ``world/scene/cam2`` 에 표시합니다
  (``p_cam1 = T @ p_cam2``).
* **로봇 베이스 축** — ``base_to_cam`` (``p_cam1 = M @ p_base``)으로 **같은 cam1 좌표계** 안에
  ``world/scene/robot_base`` 로 그립니다.

**전제:** ``--cam1-ply`` 파일의 (x,y,z)가 실제로 **그 카메라 1 프레임**이어야 위가 한눈에 맞습니다.
캡처/export가 다른 프레임(예: 베이스)이면 스크립트가 자동으로 바꿔 주지는 않습니다.

**TCP (파일명):** stem이 ``[인덱스_]_x_y_z_W_P_R`` (mm, Fanuc xyz WPR °)이면
``examples/visualize_pallet_box.py`` 의 ``--tool-rotation`` 과 같은 vec6으로 TCP를 읽어
``world/scene/tcp_from_filename`` 에 표시합니다 (기본 ON). 해석은 ``--tcp-pose-frame``:
로봇 **base**면 ``base_to_cam`` 으로 cam1에 올리고, 이미 **cam1**이면 그대로 그립니다.

* extrinsic YAML: ``cam2_pose_matrix`` (OpenCV ``!!opencv-matrix`` 또는 4x4 중첩 리스트)
  — **cam1 기준 cam2**: ``p_cam1 = T @ p_cam2`` (동차 좌표).

Requires: ``pip install -e ".[viz]"`` (open3d: PLY 로드 시 권장)

기본으로 각 PLY 점을 **카메라 깊이 축(Z, m)** 기준 ``0~1`` m 로 클립합니다.
끄려면 ``--no-depth-clip``, 범위 변경은 ``--depth-min-m`` / ``--depth-max-m`` / ``--depth-axis``.

기본으로 각 카메라 **RGB PNG + 인트린식 JSON** 을 읽어 ``Pinhole`` + ``Image`` 를 올립니다.

* **3D (cam1 좌표계):** ``world/scene/cam1_rgb``, ``world/scene/cam2_rgb`` — Pinhole 전용 entity.
  3D 콘텐츠가 있는 ``world/scene/cam1`` / ``cam2`` 와 **분리된 sibling**으로 두어야 Rerun 의
  2D subspace 규칙을 위반하지 않는다 ("pinhole's child frame ... does not form the root of a 2D subspace").
* **Spatial2DView 탭:** ``world`` 에 ``Transform3D`` 가 있으면 그 **아래** 어디에 두든 Pinhole 2D 루트가 깨질 수 있어,
  ``world`` 밖 최상위 ``rerun_2d/cam1``, ``rerun_2d/cam2`` 에만 동일 RGB를 한 번 더 로그합니다.

RGB 끄기: ``--no-rgb``.

3D Stereo 뷰에서 RGB 평면이 PLY를 너무 덮으면 ``--image-plane-mm-3d`` 를 더 줄이면 됩니다 (기본 200 mm).
이 값은 ``world/scene/cam{1,2}_rgb`` 의 Pinhole ``image_plane_distance`` 에 적용됩니다.

기본으로 ``datasets/_source_capture/base_to_cam_cam1_example.yml`` (``base_to_cam``)을 읽어
**cam1 좌표계** 안 ``world/scene/robot_base`` 에 베이스 축을 그립니다. 끄기: ``--no-robot-base``.

행렬은 **역을 쓰지 않고** 그대로 Rerun ``Transform3D``(ParentFromChild)에 넣습니다. 즉 YAML이
``p_cam1 = M @ p_base`` (**base→cam1**) 일 때 베이스 원점·축이 cam1에서 올바릅니다.
파일이 실제로 ``p_base = M @ p_cam1`` (**cam→base**) 이면 ``--invert-base-calibration`` 을 켜세요.

다른 YAML은 ``--base-calibration`` 으로 지정. 형식: ``base_to_cam`` 4x4 또는
``utils.load_extrinsics`` (``base2cam`` / ``camera_calibration``).

**Debug visualization (frame convention 검증용):**

* ``--debug-axes`` (기본 ON): 모든 frame의 축을 **큰 라벨 + 서로 다른 길이**로 그려 혼동 방지.
* ``--debug-origins`` (기본 ON): 각 frame 원점에 sphere 점을 찍어 원점 위치를 직관적으로 표시.
* ``--debug-links`` (기본 ON): cam1 원점에서 cam2, robot_base 원점까지 선을 그어 상대 배치 표시.
* ``--raw-base-transform`` (기본 OFF): ``rigid_transform_kit`` 을 우회하고 원시 Rerun API로
  base transform을 추가 로그 (``world/scene/robot_base_raw``). 두 entity가 정확히 겹치면
  기존 코드가 맞다는 확증이 됩니다.

Usage::

  uv run python examples/multi_eye_view.py

  uv run python examples/multi_eye_view.py \\
    --cam1-ply datasets/_source_capture/camera_primary/0_....ply \\
    --cam2-ply datasets/_source_capture/camera_secondary/0_....ply \\
    --extrinsic datasets/_source_capture/stereo_extrinsic_example.yml

  uv run python examples/multi_eye_view.py --save output/multi_view.rrd
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from rigid_transform_kit import Frame, RigidTransform
from rigid_transform_kit.viz import TransformVisualizer, save_recording

from utils import clip_depth_range, load_intrinsics_any, load_ply_points

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_PLY_CAM1 = (
    REPO_ROOT
    / "datasets/_source_capture/camera_primary/0_1006.205_486.709_-86.209_-179.513_-0.574_-128.436.ply"
)
_DEFAULT_PLY_CAM2 = (
    REPO_ROOT
    / "datasets/_source_capture/camera_secondary/0_1006.205_486.709_-86.209_-179.513_-0.574_-128.436.ply"
)
_DEFAULT_BASE_TO_CAM_YAML = REPO_ROOT / "datasets/_source_capture/base_to_cam_cam1_example.yml"


# ---------------------------------------------------------------------------
# Debug visualization helpers — frame convention 검증용
# ---------------------------------------------------------------------------
# 국룰: X=Red, Y=Green, Z=Blue (Rerun convention과 일치)
_AXIS_COLORS = [[255, 0, 0], [0, 255, 0], [0, 0, 255]]


def _log_frame_axes(
    entity: str,
    *,
    axis_length_mm: float,
    frame_label: str,
    origin_color: tuple[int, int, int] = (255, 255, 0),
    origin_radius: float = 8.0,
    with_origin: bool = True,
) -> None:
    """Log RGB-colored XYZ arrows + origin sphere at *entity* (local frame).

    *frame_label* 은 "CAM1", "CAM2", "BASE" 등 frame 이름. 축 라벨은
    ``"{frame_label}_X"`` 형식으로 찍혀서 여러 frame 겹쳐도 구분된다.
    """
    import rerun as rr

    rr.log(
        f"{entity}/axes",
        rr.Arrows3D(
            origins=[[0, 0, 0]] * 3,
            vectors=(np.eye(3) * axis_length_mm).tolist(),
            colors=_AXIS_COLORS,
            labels=[f"{frame_label}_X", f"{frame_label}_Y", f"{frame_label}_Z"],
        ),
        static=True,
    )
    if with_origin:
        rr.log(
            f"{entity}/origin",
            rr.Points3D(
                [[0, 0, 0]],
                colors=[list(origin_color)],
                radii=[origin_radius],
                labels=[frame_label],
            ),
            static=True,
        )


def _log_debug_link(
    entity: str,
    p_from_cam1: np.ndarray,
    p_to_cam1: np.ndarray,
    color: tuple[int, int, int] = (160, 160, 160),
) -> None:
    """cam1 좌표계 두 점을 잇는 회색 선분 (frame 간 상대 배치 가늠용)."""
    import rerun as rr

    rr.log(
        entity,
        rr.LineStrips3D(
            [[p_from_cam1.tolist(), p_to_cam1.tolist()]],
            colors=[list(color)],
            radii=[0.5],
        ),
        static=True,
    )


def _log_base_raw(
    entity: str,
    M_base2cam1: np.ndarray,
    *,
    axis_length_mm: float,
    frame_label: str = "BASE_RAW",
) -> None:
    """``rigid_transform_kit`` 을 우회해 원시 Rerun API로 base frame 로그.

    Rerun Transform3D 는 parent-from-child 이므로, entity parent 가 ``world/scene`` (=cam1)
    이면 ``p_cam1 = M @ p_base`` 인 M 을 **그대로** 넣으면 된다. 기존 ``vis.log_transform``
    결과와 이 결과가 겹치면 frame 선언이 올바르다는 sanity check 가 된다.
    """
    import rerun as rr

    R_bc = M_base2cam1[:3, :3]
    t_bc = M_base2cam1[:3, 3]
    quat_bc = Rotation.from_matrix(R_bc).as_quat().tolist()  # xyzw

    rr.log(
        entity,
        rr.Transform3D(
            translation=t_bc.tolist(),
            quaternion=rr.Quaternion(xyzw=quat_bc),
        ),
        static=True,
    )
    _log_frame_axes(
        entity,
        axis_length_mm=axis_length_mm,
        frame_label=frame_label,
        origin_color=(255, 80, 255),  # magenta: raw 용 (기존 base 와 구별)
    )


# ---------------------------------------------------------------------------
# 기존 IO 헬퍼
# ---------------------------------------------------------------------------
def _load_4x4_from_yaml(path: Path) -> np.ndarray:
    """Load 4x4 ``cam2_pose_matrix`` (or ``T_cam1_cam2``) from YAML."""
    if not path.exists():
        raise FileNotFoundError(path)

    try:
        import yaml
    except ImportError as e:
        raise ImportError("PyYAML required: pip install pyyaml") from e

    with open(path, encoding="utf-8") as f:
        doc = yaml.safe_load(f)

    if isinstance(doc, dict):
        for key in ("cam2_pose_matrix", "T_cam1_cam2", "T_cam1_to_cam2"):
            if key not in doc:
                continue
            obj = doc[key]
            if isinstance(obj, dict) and "data" in obj:
                rows = int(obj["rows"])
                cols = int(obj["cols"])
                flat = np.asarray(obj["data"], dtype=np.float64).ravel()
                if flat.size != rows * cols:
                    raise ValueError(f"{key}: data length {flat.size} != rows*cols {rows*cols}")
                return flat.reshape(rows, cols)
            arr = np.asarray(obj, dtype=np.float64)
            if arr.shape == (4, 4):
                return arr

    try:
        import cv2

        fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
        try:
            for key in ("cam2_pose_matrix", "T_cam1_cam2", "T_cam1_to_cam2"):
                node = fs.getNode(key)
                if not node.empty():
                    m = node.mat()
                    if m is not None and m.shape == (4, 4):
                        return np.asarray(m, dtype=np.float64)
        finally:
            fs.release()
    except ImportError:
        pass
    except Exception:
        pass

    raise KeyError(
        "YAML must contain cam2_pose_matrix or T_cam1_cam2 "
        "(4x4 nested list, opencv-matrix dict, or OpenCV FileStorage YAML)"
    )


def _load_rgb_hwc(rgb_path: Path) -> np.ndarray | None:
    """Load PNG as RGB uint8 (H,W,3), or None."""
    try:
        import cv2
    except ImportError:
        return None
    bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _read_image_resolution_json(path: Path) -> tuple[int, int]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    im = data.get("sensores", {}).get("image", data.get("image", {}))
    res = im.get("resolution") or {}
    w, h = int(res.get("width", 0)), int(res.get("height", 0))
    if w <= 0 or h <= 0:
        raise ValueError(f"Missing sensores.image.resolution in {path}")
    return w, h


def _log_rgb_pinhole(
    entity_path: str,
    rgb_path: Path,
    intrinsic_json: Path,
    *,
    image_plane_mm: float,
    rgb_hwc: np.ndarray | None = None,
) -> bool:
    """Log ``ViewCoordinates`` + ``Pinhole`` + ``Image`` at *entity_path*. Returns False if skipped.

    *rgb_hwc* — optional preloaded RGB ``uint8`` (H,W,3); avoids re-reading when logging the same
    image to multiple entities (e.g. 3D under ``Transform3D`` + flat path for ``Spatial2DView``).
    """
    if not rgb_path.exists():
        log.warning("RGB not found, skip: %s", rgb_path)
        return False
    if not intrinsic_json.exists():
        log.warning("Intrinsics JSON not found, skip RGB: %s", intrinsic_json)
        return False

    try:
        import cv2
    except ImportError as e:
        raise ImportError("opencv-python required for RGB: pip install opencv-python") from e

    import rerun as rr

    K, _ = load_intrinsics_any(intrinsic_json)
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    w, h = _read_image_resolution_json(intrinsic_json)

    if rgb_hwc is None:
        bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if bgr is None:
            log.warning("Failed to read image: %s", rgb_path)
            return False
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    else:
        rgb = rgb_hwc

    rr.log(entity_path, rr.ViewCoordinates.RDF, static=True)
    rr.log(
        entity_path,
        rr.Pinhole(
            focal_length=[fx, fy],
            principal_point=[cx, cy],
            width=w,
            height=h,
            image_plane_distance=float(image_plane_mm),
        ),
        static=True,
    )
    rr.log(entity_path, rr.Image(rgb), static=True)
    log.info("Logged RGB Pinhole at %s (%dx%d, plane=%.0f mm)", entity_path, w, h, image_plane_mm)
    return True


def _send_stereo_blueprint(*, rgb_tab_origins: list[tuple[str, str]]) -> None:
    """*rgb_tab_origins*: (tab_title, rerun entity origin) for each Spatial2DView."""
    import rerun as rr
    import rerun.blueprint as rrb

    tabs: list = [
        rrb.Spatial3DView(
            name="Stereo (camera 1)",
            origin="world/scene",
            contents=["+ world/scene/**"],
        ),
    ]
    for title, origin in rgb_tab_origins:
        tabs.append(
            rrb.Spatial2DView(
                name=title,
                origin=origin,
                contents=[f"+ {origin}", f"+ {origin}/**"],
            )
        )
    rr.send_blueprint(rrb.Blueprint(rrb.Tabs(*tabs)))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Two-camera PLY + robot base in one Rerun scene; unified camera 1 frame (cam1 PLY, cam2→cam1, base→cam1).",
    )
    p.add_argument(
        "--cam1-ply",
        type=Path,
        default=_DEFAULT_PLY_CAM1,
        help="Primary camera PLY (cam1 / scene reference)",
    )
    p.add_argument(
        "--cam2-ply",
        type=Path,
        default=_DEFAULT_PLY_CAM2,
        help="Secondary PLY (transformed into camera 1 frame via extrinsic)",
    )
    p.add_argument(
        "--extrinsic",
        type=Path,
        default=REPO_ROOT / "datasets/_source_capture/stereo_extrinsic_example.yml",
        help="YAML with cam2_pose_matrix (cam1 <- cam2)",
    )
    p.add_argument(
        "--cam1-rgb",
        type=Path,
        default=_DEFAULT_PLY_CAM1.with_suffix(".png"),
        help="Cam1 RGB image (PNG). Default: same stem as built-in primary PLY",
    )
    p.add_argument(
        "--cam2-rgb",
        type=Path,
        default=_DEFAULT_PLY_CAM2.with_suffix(".png"),
        help="Cam2 RGB image (PNG). Default: same stem as built-in secondary PLY",
    )
    p.add_argument(
        "--cam1-intrinsics",
        type=Path,
        default=_DEFAULT_PLY_CAM1.with_suffix(".json"),
        help="Cam1 intrinsics JSON (sensores.image). Default: same stem as built-in primary PLY",
    )
    p.add_argument(
        "--cam2-intrinsics",
        type=Path,
        default=_DEFAULT_PLY_CAM2.with_suffix(".json"),
        help="Cam2 intrinsics JSON. Default: same stem as built-in secondary PLY",
    )
    p.add_argument(
        "--image-plane-mm-3d",
        type=float,
        default=200.0,
        metavar="MM",
        help="Pinhole image plane distance (mm) for world/scene/*/rgb — smaller = less overlap on PLY. Default 200",
    )
    p.add_argument(
        "--image-plane-mm-2d",
        type=float,
        default=600.0,
        metavar="MM",
        help="Pinhole image_plane_distance (mm) for rerun_2d/* (2D tabs only; rarely affects 2D view). Default 600",
    )
    p.add_argument(
        "--no-rgb",
        action="store_true",
        help="Do not load PNG / Pinhole / 2D tabs",
    )
    p.add_argument(
        "--max-points",
        type=int,
        default=400_000,
        metavar="N",
        help="Random subsample per cloud if larger than N (0 = no limit)",
    )
    p.add_argument(
        "--depth-min-m",
        type=float,
        default=0.0,
        metavar="M",
        help="Depth clip lower bound in meters (camera axis, default Z). Default 0",
    )
    p.add_argument(
        "--depth-max-m",
        type=float,
        default=3.0,
        metavar="M",
        help="Depth clip upper bound in meters. Default 1",
    )
    p.add_argument(
        "--depth-axis",
        type=int,
        choices=(0, 1, 2),
        default=2,
        help="Which point column is depth for clipping (0=x,1=y,2=z). Default 2",
    )
    p.add_argument(
        "--no-depth-clip",
        action="store_true",
        help="Disable depth clipping (otherwise depth-min/max apply in meters)",
    )
    p.add_argument(
        "--port",
        type=int,
        default=None,
        help="Rerun gRPC port if needed",
    )
    p.add_argument(
        "--save",
        type=Path,
        default=None,
        metavar="RRD",
        help="Save recording to .rrd (disables live spawn; then opens: rerun FILE)",
    )
    p.add_argument(
        "--base-calibration",
        type=Path,
        default=_DEFAULT_BASE_TO_CAM_YAML,
        metavar="YAML",
        help="base→cam1 extrinsic YAML (default: datasets/_source_capture/base_to_cam_cam1_example.yml)",
    )
    p.add_argument(
        "--no-robot-base",
        action="store_true",
        help="Do not log robot base frame (ignore --base-calibration)",
    )
    p.add_argument(
        "--invert-base-calibration",
        action="store_true",
        help="YAML 4x4가 cam→base (p_base=M p_cam)일 때: inv(M)으로 base→cam으로 바꿔 로그",
    )
    p.add_argument(
        "--base-axis-mm",
        type=float,
        default=250.0,
        metavar="MM",
        help="Axis arrow length for robot base frame in 3D view (mm). Default 250",
    )
    # -- TCP pose from PLY filename (Fanuc WPR, same as visualize_pallet_box --tool-rotation) --
    p.add_argument(
        "--tcp-from-ply-name",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Parse cam1 PLY stem as [idx_]_x_y_z_W_P_R mm/deg and draw TCP (default: on).",
    )
    p.add_argument(
        "--tcp-pose-frame",
        choices=("base", "cam1"),
        default="base",
        help="vec6 frame: base (robot base → needs base_to_cam) or already in cam1 (default: base).",
    )
    p.add_argument(
        "--tcp-pose-axis-mm",
        type=float,
        default=120.0,
        metavar="MM",
        help="TCP axis arrow length in mm for tcp_from_filename (default 120).",
    )
    # -- Debug visualization -------------------------------------------------
    p.add_argument(
        "--no-debug-axes",
        action="store_true",
        help="Disable per-frame axes with labels (default: axes ON for all frames)",
    )
    p.add_argument(
        "--no-debug-origins",
        action="store_true",
        help="Disable origin sphere markers at each frame",
    )
    p.add_argument(
        "--no-debug-links",
        action="store_true",
        help="Disable gray link lines between cam1 origin and other frame origins",
    )
    p.add_argument(
        "--raw-base-transform",
        action="store_true",
        help="Also log base frame via raw Rerun API (world/scene/robot_base_raw) — sanity check for rigid_transform_kit",
    )
    return p.parse_args()


def load_base_to_cam_matrix(path: Path) -> np.ndarray:
    """Return 4x4 ``T_base2cam`` (mm). Supports ``base_to_cam`` in YAML or :func:`utils.load_extrinsics` files."""
    if not path.exists():
        raise FileNotFoundError(
            f"Base calibration YAML not found: {path}\n"
            "Pass a real file path (e.g. your ``base_to_cam`` YAML). "
            "Documentation placeholders like ``path/to/...`` are not valid paths."
        )
    try:
        import yaml
    except ImportError as e:
        raise ImportError("PyYAML required") from e

    with open(path, encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    node = doc.get("config", doc)
    if isinstance(node, dict) and "base_to_cam" in node:
        arr = np.asarray(node["base_to_cam"], dtype=np.float64)
        if arr.shape == (4, 4):
            return arr

    from utils import load_extrinsics

    return load_extrinsics(path)["base2cam"]


def parse_tcp_vec6_from_ply_filename(path: Path) -> np.ndarray | None:
    """Parse stem ``[idx_]_x_y_z_W_P_R`` → vec6 (mm, Fanuc xyz WPR degrees).

    Example stem: ``0_1006.205_486.709_-86.209_-179.513_-0.574_-128.436`` → skips leading
    index ``0``, then x,y,z,W,P,R. Same Euler convention as :func:`RigidTransform.from_vec6`
    (``convention='xyz'``) and ``visualize_pallet_box`` ``--tool-rotation``.
    """
    parts = path.stem.split("_")
    floats: list[float] = []
    for p in parts:
        try:
            floats.append(float(p))
        except ValueError:
            return None
    if len(floats) < 6:
        return None
    if len(floats) == 6:
        return np.array(floats, dtype=np.float64)
    if len(floats) == 7:
        return np.array(floats[1:7], dtype=np.float64)
    return np.array(floats[-6:], dtype=np.float64)


def _maybe_subsample(pts: np.ndarray, colors: np.ndarray | None, max_n: int) -> tuple[np.ndarray, np.ndarray | None]:
    if max_n <= 0 or pts.shape[0] <= max_n:
        return pts, colors
    rng = np.random.default_rng(0)
    idx = rng.choice(pts.shape[0], size=max_n, replace=False)
    c2 = colors[idx] if colors is not None else None
    return pts[idx], c2


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()

    # debug 옵션 (기본 ON, --no-* 로 off)
    debug_axes = not args.no_debug_axes
    debug_origins = not args.no_debug_origins
    debug_links = not args.no_debug_links

    base_cal_path = None if args.no_robot_base else args.base_calibration
    if base_cal_path is not None and not base_cal_path.exists():
        raise SystemExit(
            f"--base-calibration: file not found: {base_cal_path}\n"
            "Use an existing YAML (``base_to_cam`` 4x4 or hand-eye ``base2cam`` / ``camera_calibration``), "
            "or ``--no-robot-base`` to skip."
        )

    T = _load_4x4_from_yaml(args.extrinsic)
    R = T[:3, :3]
    t = T[:3, 3]
    quat_xyzw = Rotation.from_matrix(R).as_quat().tolist()

    ply1 = load_ply_points(args.cam1_ply)
    ply2 = load_ply_points(args.cam2_ply)
    if ply1 is None:
        raise SystemExit(f"No points in cam1 PLY: {args.cam1_ply}")
    if ply2 is None:
        raise SystemExit(f"No points in cam2 PLY: {args.cam2_ply}")

    pts1_m, col1 = ply1
    pts2_m, col2 = ply2

    if not args.no_depth_clip:
        n1, n2 = len(pts1_m), len(pts2_m)
        pts1_m, col1 = clip_depth_range(
            pts1_m,
            args.depth_min_m,
            args.depth_max_m,
            depth_axis=args.depth_axis,
            colors=col1,
        )
        pts2_m, col2 = clip_depth_range(
            pts2_m,
            args.depth_min_m,
            args.depth_max_m,
            depth_axis=args.depth_axis,
            colors=col2,
        )
        log.info(
            "Depth clip [%.3f, %.3f] m (axis=%d): cam1 %d -> %d, cam2 %d -> %d",
            args.depth_min_m,
            args.depth_max_m,
            args.depth_axis,
            n1,
            len(pts1_m),
            n2,
            len(pts2_m),
        )

    pts1_mm = pts1_m * 1000.0
    pts2_mm = pts2_m * 1000.0

    if col1 is None:
        col1 = np.tile(np.array([[255, 120, 40]], dtype=np.uint8), (len(pts1_mm), 1))
    if col2 is None:
        col2 = np.tile(np.array([[40, 180, 255]], dtype=np.uint8), (len(pts2_mm), 1))

    pts1_mm, col1 = _maybe_subsample(pts1_mm, col1, args.max_points)
    pts2_mm, col2 = _maybe_subsample(pts2_mm, col2, args.max_points)

    save_path = args.save
    spawn = save_path is None
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        log.info("Saving to %s (spawn disabled).", save_path)

    log.info(
        "Unified camera 1 frame: cam1 PLY as-is, cam2 via extrinsic, robot_base via base_to_cam. "
        "--cam1-ply=%s (PLY must be in cam1 coords).",
        args.cam1_ply,
    )

    vis = TransformVisualizer(
        "multi_eye_view",
        spawn=spawn,
        port=args.port,
        views=[("Stereo (camera 1)", "world/scene")],
    )
    import rerun as rr

    # ------------------------------------------------------------------
    # Robot base (cam1 좌표계)
    # ------------------------------------------------------------------
    M_base2cam: np.ndarray | None = None
    if base_cal_path is not None:
        M_base2cam = load_base_to_cam_matrix(base_cal_path)
        if args.invert_base_calibration:
            M_base2cam = np.linalg.inv(M_base2cam)
            log.info("Using inv(base matrix): YAML treated as cam→base, logging as base→cam1.")
        # RigidTransform: p_from = T @ p_to. Here p_cam1 = M @ p_base → from=CAMERA, to=BASE.
        T_cam1_from_base = RigidTransform.from_matrix(M_base2cam, Frame.CAMERA, Frame.BASE)
        vis.log_transform(
            "world/scene/robot_base",
            T_cam1_from_base,
            axis_length=args.base_axis_mm,
            label="ROBOT_BASE",
        )
        log.info("Logged robot base in camera 1 frame from %s", base_cal_path)

        # 디버그: base frame 에 라벨 달린 축 + 원점 sphere 추가
        if debug_axes:
            _log_frame_axes(
                "world/scene/robot_base",
                axis_length_mm=args.base_axis_mm,
                frame_label="BASE",
                origin_color=(255, 255, 0),  # yellow
                origin_radius=12.0,
                with_origin=debug_origins,
            )

        # 디버그: rigid_transform_kit 우회한 raw 버전도 같이 찍어 sanity check
        if args.raw_base_transform:
            _log_base_raw(
                "world/scene/robot_base_raw",
                M_base2cam,
                axis_length_mm=args.base_axis_mm * 0.7,  # 살짝 작게 → 겹쳐도 구분
                frame_label="BASE_RAW",
            )
            log.info(
                "Raw base transform also logged at world/scene/robot_base_raw "
                "(should visually coincide with world/scene/robot_base)"
            )

    # ------------------------------------------------------------------
    # Cam1 (world/scene 와 동일 좌표계)
    # ------------------------------------------------------------------
    axis_mm = 80.0
    if debug_axes:
        # 라벨 포함 버전 (디버그용) — 축 색은 기존과 동일 RGB
        _log_frame_axes(
            "world/scene/cam1",
            axis_length_mm=axis_mm,
            frame_label="CAM1",
            origin_color=(255, 80, 80),  # reddish
            origin_radius=8.0,
            with_origin=debug_origins,
        )
    else:
        # 기존 코드 유지 (라벨 없이)
        rr.log(
            "world/scene/cam1/axes",
            rr.Arrows3D(
                origins=[[0, 0, 0]] * 3,
                vectors=(np.eye(3) * axis_mm).tolist(),
                colors=[[220, 40, 40], [40, 220, 40], [40, 80, 220]],
                labels=["cam1 X", "cam1 Y", "cam1 Z"],
            ),
            static=True,
        )
    rr.log("world/scene/cam1/pcd", rr.Points3D(pts1_mm, colors=col1, radii=[1.2]), static=True)

    # ------------------------------------------------------------------
    # Cam2 (cam1 좌표계에 extrinsic 으로 배치)
    # ------------------------------------------------------------------
    rr.log(
        "world/scene/cam2",
        rr.Transform3D(translation=t.tolist(), quaternion=rr.Quaternion(xyzw=quat_xyzw)),
        static=True,
    )
    if debug_axes:
        _log_frame_axes(
            "world/scene/cam2",
            axis_length_mm=axis_mm,
            frame_label="CAM2",
            origin_color=(80, 180, 255),  # bluish
            origin_radius=8.0,
            with_origin=debug_origins,
        )
    else:
        rr.log(
            "world/scene/cam2/axes",
            rr.Arrows3D(
                origins=[[0, 0, 0]] * 3,
                vectors=(np.eye(3) * axis_mm).tolist(),
                colors=[[220, 40, 40], [40, 220, 40], [40, 80, 220]],
                labels=["cam2 X", "cam2 Y", "cam2 Z"],
            ),
            static=True,
        )
    rr.log("world/scene/cam2/pcd", rr.Points3D(pts2_mm, colors=col2, radii=[1.2]), static=True)

    # ------------------------------------------------------------------
    # 디버그: cam1 ↔ cam2 / cam1 ↔ base 연결선 (상대 배치 한눈에)
    # ------------------------------------------------------------------
    if debug_links:
        cam1_origin_in_cam1 = np.zeros(3)
        cam2_origin_in_cam1 = t  # T[:3, 3] 이 cam1 좌표계 상의 cam2 원점
        _log_debug_link(
            "world/scene/_debug/cam1_to_cam2",
            cam1_origin_in_cam1,
            cam2_origin_in_cam1,
            color=(140, 140, 200),
        )
        log.info(
            "cam1→cam2 link: translation %s mm (|t|=%.1f mm)",
            np.round(t, 2).tolist(),
            float(np.linalg.norm(t)),
        )
        if M_base2cam is not None:
            base_origin_in_cam1 = M_base2cam[:3, 3]
            _log_debug_link(
                "world/scene/_debug/cam1_to_base",
                cam1_origin_in_cam1,
                base_origin_in_cam1,
                color=(200, 200, 120),
            )
            log.info(
                "cam1→base link: translation %s mm (|t|=%.1f mm)",
                np.round(base_origin_in_cam1, 2).tolist(),
                float(np.linalg.norm(base_origin_in_cam1)),
            )

    # ------------------------------------------------------------------
    # TCP pose from PLY filename: stem [idx_]_x_y_z_W_P_R (mm, Fanuc xyz WPR deg)
    # Same convention as examples/visualize_pallet_box.py --tool-rotation.
    # ------------------------------------------------------------------
    if args.tcp_from_ply_name:
        vec6 = parse_tcp_vec6_from_ply_filename(args.cam1_ply)
        if vec6 is None:
            log.info(
                "TCP from filename: stem %r is not [idx_]_x_y_z_W_P_R — skip.",
                args.cam1_ply.stem,
            )
        else:
            log.info(
                "TCP from filename: [x,y,z,W,P,R] = %s (mm, deg)",
                np.round(vec6, 3).tolist(),
            )
            if args.tcp_pose_frame == "cam1":
                T_cam_tcp = RigidTransform.from_vec6(
                    vec6, Frame.CAMERA, Frame.TCP, convention="xyz", degrees=True
                )
            else:
                T_base_tcp = RigidTransform.from_vec6(
                    vec6, Frame.BASE, Frame.TCP, convention="xyz", degrees=True
                )
                if M_base2cam is None:
                    log.warning(
                        "TCP (--tcp-pose-frame=base): need base_to_cam; skip. "
                        "Use --tcp-pose-frame=cam1 or provide --base-calibration."
                    )
                    T_cam_tcp = None
                else:
                    T_cam_tcp = RigidTransform.from_matrix(
                        M_base2cam @ T_base_tcp.matrix,
                        Frame.CAMERA,
                        Frame.TCP,
                    )
            if T_cam_tcp is not None:
                vis.log_tcp_pose(
                    T_cam_tcp,
                    parent_path="world/scene",
                    label="tcp",
                    axis_length=args.tcp_pose_axis_mm,
                    arrow_radius=max(2.0, args.tcp_pose_axis_mm * 0.04),
                    show_axes=True,
                )
                log.info(
                    "Logged world/scene/tcp (cam1 frame, --tcp-pose-frame=%s).",
                    args.tcp_pose_frame,
                )

    # ------------------------------------------------------------------
    # RGB pinhole
    # ------------------------------------------------------------------
    # Rerun rule: Pinhole entity is the root of a 2D subspace. Its entity MUST NOT have
    # 3D siblings under it (Arrows3D, Points3D), otherwise the subspace is ambiguous and
    # you get "The pinhole's child frame ... does not form the root of a 2D subspace" warnings.
    # Solution: give each Pinhole its OWN dedicated entity that is a SIBLING of the 3D cam
    # entity (not a child), and put ONLY 2D content under it (Image).
    #
    #   world/scene/cam1        ← 3D only (axes, pcd, origin)
    #   world/scene/cam1_rgb    ← Pinhole + Image only  (same frame as cam1)
    #   world/scene/cam2        ← Transform3D + 3D (axes, pcd, origin)
    #   world/scene/cam2_rgb    ← same Transform3D + Pinhole + Image only
    rgb_tabs: list[tuple[str, str]] = []
    if not args.no_rgb:
        # --- CAM1 ---
        # cam1 shares world/scene frame (no Transform3D), so cam1_rgb needs no Transform3D either.
        c1_rgb = _load_rgb_hwc(args.cam1_rgb)
        if c1_rgb is not None:
            _log_rgb_pinhole(
                "world/scene/cam1_rgb",
                args.cam1_rgb,
                args.cam1_intrinsics,
                image_plane_mm=args.image_plane_mm_3d,
                rgb_hwc=c1_rgb,
            )
            if _log_rgb_pinhole(
                "rerun_2d/cam1",
                args.cam1_rgb,
                args.cam1_intrinsics,
                image_plane_mm=args.image_plane_mm_2d,
                rgb_hwc=c1_rgb,
            ):
                rgb_tabs.append(("Cam1 RGB", "rerun_2d/cam1"))
        else:
            _log_rgb_pinhole(
                "world/scene/cam1_rgb",
                args.cam1_rgb,
                args.cam1_intrinsics,
                image_plane_mm=args.image_plane_mm_3d,
            )

        # --- CAM2 ---
        # cam2_rgb needs its own Transform3D (same as cam2) so that the frustum sits at the
        # cam2 pose inside world/scene.
        c2_rgb = _load_rgb_hwc(args.cam2_rgb)
        rgb2_exists = args.cam2_rgb.exists() and args.cam2_intrinsics.exists()
        if rgb2_exists:
            rr.log(
                "world/scene/cam2_rgb",
                rr.Transform3D(
                    translation=t.tolist(),
                    quaternion=rr.Quaternion(xyzw=quat_xyzw),
                ),
                static=True,
            )
            if c2_rgb is not None:
                _log_rgb_pinhole(
                    "world/scene/cam2_rgb",
                    args.cam2_rgb,
                    args.cam2_intrinsics,
                    image_plane_mm=args.image_plane_mm_3d,
                    rgb_hwc=c2_rgb,
                )
                if _log_rgb_pinhole(
                    "rerun_2d/cam2",
                    args.cam2_rgb,
                    args.cam2_intrinsics,
                    image_plane_mm=args.image_plane_mm_2d,
                    rgb_hwc=c2_rgb,
                ):
                    rgb_tabs.append(("Cam2 RGB", "rerun_2d/cam2"))
            else:
                _log_rgb_pinhole(
                    "world/scene/cam2_rgb",
                    args.cam2_rgb,
                    args.cam2_intrinsics,
                    image_plane_mm=args.image_plane_mm_3d,
                )

    _send_stereo_blueprint(rgb_tab_origins=rgb_tabs)

    log.info(
        "Logged cam1=%d pts, cam2=%d pts (cam2 frame under Transform; units mm).",
        len(pts1_mm),
        len(pts2_mm),
    )

    if save_path is not None:
        save_recording(save_path)
        log.info("Saved to %s", save_path)
        try:
            subprocess.run(["rerun", str(save_path)], check=False)
        except FileNotFoundError:
            log.info("Run: rerun %s", save_path)

    if spawn:
        import rerun as rr

        rec = rr.get_global_data_recording()
        if rec is not None:
            try:
                rec.flush(timeout_sec=10.0)
            except Exception:  # noqa: BLE001
                pass
            time.sleep(1.0)


if __name__ == "__main__":
    main()