"""
examples / multi_eye_view.py
=============================
두 카메라 PLY를 cam1 좌표계로 맞춰 Rerun 한 scene에 표시.

* extrinsic YAML: ``cam2_pose_matrix`` (OpenCV ``!!opencv-matrix`` 또는 4x4 중첩 리스트)
  — **cam1 기준 cam2**: ``p_cam1 = T @ p_cam2`` (동차 좌표).

Requires: ``pip install -e ".[viz]"`` (open3d: PLY 로드 시 권장)

기본으로 각 PLY 점을 **카메라 깊이 축(Z, m)** 기준 ``0~1`` m 로 클립합니다.
끄려면 ``--no-depth-clip``, 범위 변경은 ``--depth-min-m`` / ``--depth-max-m`` / ``--depth-axis``.

기본으로 각 카메라 **RGB PNG + 인트린식 JSON** 을 읽어 ``Pinhole`` + ``Image`` 를 올립니다.

* **3D (cam1 좌표계):** ``world/scene/cam1/rgb``, ``world/scene/cam2/rgb`` (cam2는 extrinsic 아래)
* **Spatial2DView 탭:** ``world`` 에 ``Transform3D`` 가 있으면 그 **아래** 어디에 두든 Pinhole 2D 루트가 깨질 수 있어,
  ``world`` 밖 최상위 ``rerun_2d/cam1``, ``rerun_2d/cam2`` 에만 동일 RGB를 한 번 더 로그합니다.

RGB 끄기: ``--no-rgb``.

3D Stereo 뷰에서 RGB 평면이 PLY를 너무 덮으면 ``--image-plane-mm-3d`` 를 더 줄이면 됩니다 (기본 200 mm).

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
            name="Stereo (cam1 frame)",
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
    p = argparse.ArgumentParser(description="Two-camera PLY in one Rerun scene (cam1 frame).")
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
        help="Secondary camera PLY (transformed into cam1 frame via extrinsic)",
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
    return p.parse_args()


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

    vis = TransformVisualizer(
        "multi_eye_view",
        spawn=spawn,
        port=args.port,
        views=[("Stereo (cam1 frame)", "world/scene")],
    )
    import rerun as rr

    axis_mm = 80.0
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

    rr.log(
        "world/scene/cam2",
        rr.Transform3D(translation=t.tolist(), quaternion=rr.Quaternion(xyzw=quat_xyzw)),
        static=True,
    )
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

    rgb_tabs: list[tuple[str, str]] = []
    if not args.no_rgb:
        # 3D: scene tree. 2D tabs: top-level rerun_2d/* (NOT under world — avoids Pinhole / tf# error).
        c1_rgb = _load_rgb_hwc(args.cam1_rgb)
        if c1_rgb is not None:
            _log_rgb_pinhole(
                "world/scene/cam1/rgb",
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
                "world/scene/cam1/rgb",
                args.cam1_rgb,
                args.cam1_intrinsics,
                image_plane_mm=args.image_plane_mm_3d,
            )

        c2_rgb = _load_rgb_hwc(args.cam2_rgb)
        if c2_rgb is not None:
            _log_rgb_pinhole(
                "world/scene/cam2/rgb",
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
                "world/scene/cam2/rgb",
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
