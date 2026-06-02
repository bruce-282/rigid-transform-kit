"""
examples / calibration / reconstruction.py
============================================
다중 프레임 PLY를 ``global_poses.yaml`` 의 ``global_pose`` 로 **글로벌 좌표계**에
맞춰 Rerun 으로 시각화합니다 (``multi_eye_view`` 와 동일한 씬 패턴).

* 각 프레임: PLY 로드 → ``p_global = global_pose @ p_local`` (행렬로 점 변환 후 로그).
* 프레임 축: ``Transform3D(global_pose)`` 로 표시 (RGB Pinhole 이 카메라 뷰).
* 기준 프레임: ``global_pose`` 가 단위행렬인 프레임 (예: ``frame_3``) = 글로벌 원점.
* RGB: ``N.png`` 를 프레임별 3D Pinhole + 2D 카메라 탭에 표시 (인트린식 JSON 없으면 이미지 크기로 추정).

Usage::

  uv run python examples/calibration/reconstruction.py
  uv run python examples/calibration/reconstruction.py \\
    --poses datasets/0518_JG_Swing_LH_raw_zig/global_poses.yaml \\
    --dataset-dir datasets/0518_JG_Swing_LH_raw_zig
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from rigid_transform_kit.viz import TransformVisualizer
from utils import clip_depth_range, load_intrinsics_any, load_ply_points

from _viz_common import (
    add_common_args,
    finalize_viewer,
    log_debug_link,
    log_origin_spheres,
    subsample,
)

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_DATASET = REPO_ROOT / "datasets/0518_JG_Swing_LH_raw_zig"
_DEFAULT_POSES = _DEFAULT_DATASET / "global_poses.yaml"

_AXIS_COLORS = [[255, 0, 0], [0, 255, 0], [0, 0, 255]]
_FRAME_PALETTE: list[tuple[int, int, int]] = [
    (255, 80, 80),
    (80, 180, 255),
    (80, 255, 80),
    (255, 200, 80),
    (200, 80, 255),
    (80, 255, 255),
    (255, 120, 180),
    (180, 180, 80),
]


@dataclass(frozen=True)
class FrameEntry:
    """One reconstruction frame."""

    key: str
    ply_path: Path
    global_pose: np.ndarray  # 4x4, p_global = M @ p_local


def _frame_sort_key(name: str) -> int:
    m = re.search(r"(\d+)", name)
    return int(m.group(1)) if m else 0


def _load_poses_yaml(poses_path: Path) -> dict[str, dict]:
    try:
        import yaml
    except ImportError as e:
        raise ImportError("PyYAML required: pip install pyyaml") from e

    with open(poses_path, encoding="utf-8") as f:
        doc = yaml.safe_load(f)

    frames_raw = doc.get("frames") if isinstance(doc, dict) else None
    if not isinstance(frames_raw, dict) or not frames_raw:
        raise ValueError(f"No 'frames' dict in {poses_path}")
    return frames_raw


def _discover_ply_files(dataset_dir: Path) -> list[Path]:
    """All ``N.ply`` (numeric stem) in *dataset_dir*, sorted by N."""
    plies: list[tuple[int, Path]] = []
    for path in dataset_dir.glob("*.ply"):
        stem = path.stem
        if stem.isdigit():
            plies.append((int(stem), path))
    return [p for _, p in sorted(plies, key=lambda x: x[0])]


def load_global_pose_frames(
    poses_path: Path,
    dataset_dir: Path,
) -> list[FrameEntry]:
    """Every existing ``N.ply`` under *dataset_dir* with ``global_pose`` from YAML."""
    frames_raw = _load_poses_yaml(poses_path)
    ply_files = _discover_ply_files(dataset_dir)

    if not ply_files:
        raise FileNotFoundError(f"No numeric *.ply in {dataset_dir}")

    entries: list[FrameEntry] = []
    for ply_path in ply_files:
        num = int(ply_path.stem)
        key = f"frame_{num}"
        block = frames_raw.get(key)
        if not isinstance(block, dict) or "global_pose" not in block:
            raise KeyError(
                f"{ply_path.name}: missing frames.{key}.global_pose in {poses_path.name}"
            )

        M = np.asarray(block["global_pose"], dtype=np.float64)
        if M.shape != (4, 4):
            raise ValueError(f"{key}: global_pose must be 4x4, got {M.shape}")

        entries.append(FrameEntry(key=key, ply_path=ply_path, global_pose=M))

    log.info(
        "Frames: %d PLY files under %s (poses from %s)",
        len(entries), dataset_dir, poses_path.name,
    )
    for e in entries:
        log.info("  %s -> %s", e.key, e.ply_path.name)
    return entries


def _ply_array_to_scene_mm(pts: np.ndarray) -> np.ndarray:
    """``load_ply_points`` 출력 → Rerun 씬 mm.

    reconstruction PLY는 이미 mm인데, 무효 (0,0,0) 점 때문에 median≈0 이면
    loader가 m로 착각하지 않음 → ``* 1000`` 하면 1000배 틀어짐.
    """
    extent = float(np.max(np.ptp(pts, axis=0)))
    if extent <= 30.0:
        return (pts * 1000.0).astype(np.float64)
    return pts.astype(np.float64)


def _drop_invalid_points(
    pts_mm: np.ndarray,
    colors: np.ndarray | None,
    *,
    min_norm_mm: float = 0.5,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Remove padding / invalid points at origin."""
    mask = np.linalg.norm(pts_mm, axis=1) >= min_norm_mm
    pts2 = pts_mm[mask]
    if colors is None:
        return pts2, None
    return pts2, colors[mask]


def _transform_points_mm(pts_mm: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Apply 4x4 ``p_global = M @ p_local`` (homogeneous, mm)."""
    n = pts_mm.shape[0]
    ones = np.ones((n, 1), dtype=np.float64)
    homog = np.hstack([pts_mm.astype(np.float64), ones])
    return (homog @ M.T)[:, :3]


def _matrix_to_transform_args(M: np.ndarray) -> tuple[list[float], list[float]]:
    """4x4 ``p_global = M @ p_local`` → Rerun Transform3D (translation, quat xyzw)."""
    R = M[:3, :3]
    t = M[:3, 3]
    quat_xyzw = Rotation.from_matrix(R).as_quat().tolist()
    return t.tolist(), quat_xyzw


def _load_rgb_hwc(rgb_path: Path) -> np.ndarray | None:
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


def _log_rgb_pinhole_hwc(
    entity_path: str,
    rgb_hwc: np.ndarray,
    *,
    image_plane_mm: float,
    focal_length: float | None = None,
) -> None:
    """Pinhole + Image from RGB array (no intrinsics JSON)."""
    import rerun as rr

    h, w = rgb_hwc.shape[:2]
    if focal_length is None:
        focal_length = float(max(w, h)) * 0.9
    fx = fy = focal_length
    cx, cy = w / 2.0, h / 2.0

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
    rr.log(entity_path, rr.Image(rgb_hwc), static=True)


def _log_rgb_pinhole_json(
    entity_path: str,
    rgb_hwc: np.ndarray,
    intrinsic_json: Path,
    *,
    image_plane_mm: float,
) -> bool:
    """Pinhole + Image using intrinsics JSON (multi_eye_view style)."""
    import rerun as rr

    K, _ = load_intrinsics_any(intrinsic_json)
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    w, h = _read_image_resolution_json(intrinsic_json)

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
    rr.log(entity_path, rr.Image(rgb_hwc), static=True)
    return True


def _resolve_intrinsics_json(
    dataset_dir: Path,
    frame_num: int,
    intrinsics_dir: Path | None,
) -> Path | None:
    if intrinsics_dir is not None:
        for name in (f"{frame_num}.json", f"frame_{frame_num}.json"):
            p = intrinsics_dir / name
            if p.exists():
                return p
    for name in (f"{frame_num}.json",):
        p = dataset_dir / name
        if p.exists():
            return p
    return None


def _log_frame_rgb(
    frame_key: str,
    rgb_path: Path,
    M: np.ndarray,
    *,
    plane_3d: float,
    plane_2d: float,
    intrinsic_json: Path | None,
) -> list[tuple[str, str]]:
    """Log RGB at frame pose (3D) + flat 2D tab. Returns blueprint tab entries."""
    import rerun as rr

    if not rgb_path.exists():
        log.warning("RGB not found, skip: %s", rgb_path)
        return []

    rgb_hwc = _load_rgb_hwc(rgb_path)
    if rgb_hwc is None:
        log.warning("Failed to read RGB: %s", rgb_path)
        return []

    tabs: list[tuple[str, str]] = []
    entity_3d = f"world/scene/{frame_key}_rgb"
    entity_2d = f"rerun_2d/{frame_key}"
    tf_args = None if np.allclose(M, np.eye(4)) else _matrix_to_transform_args(M)

    if tf_args is not None:
        t_list, quat_xyzw = tf_args
        rr.log(
            entity_3d,
            rr.Transform3D(
                translation=t_list,
                quaternion=rr.Quaternion(xyzw=quat_xyzw),
            ),
            static=True,
        )

    if intrinsic_json is not None and intrinsic_json.exists():
        _log_rgb_pinhole_json(entity_3d, rgb_hwc, intrinsic_json, image_plane_mm=plane_3d)
        if _log_rgb_pinhole_json(entity_2d, rgb_hwc, intrinsic_json, image_plane_mm=plane_2d):
            tabs.append((f"{frame_key} RGB", entity_2d))
    else:
        _log_rgb_pinhole_hwc(entity_3d, rgb_hwc, image_plane_mm=plane_3d)
        _log_rgb_pinhole_hwc(entity_2d, rgb_hwc, image_plane_mm=plane_2d)
        tabs.append((f"{frame_key} RGB", entity_2d))

    log.info("Logged RGB %s -> %s, %s", rgb_path.name, entity_3d, entity_2d)
    return tabs


def _send_reconstruction_blueprint(*, rgb_tab_origins: list[tuple[str, str]]) -> None:
    import rerun as rr
    import rerun.blueprint as rrb

    tabs: list = [
        rrb.Spatial3DView(
            name="Reconstruction (global)",
            origin="world/scene",
            contents=["+ world/scene/**"],
            line_grid=False,
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


def _log_frame_scene(
    frame_key: str,
    pts_local_mm: np.ndarray,
    colors: np.ndarray,
    M: np.ndarray,
    label: str,
    origin_color: tuple[int, int, int],
) -> np.ndarray:
    """Log frame axes (Transform3D) + PCD already in global frame.

    ``multi_eye_view`` cam2 와 동일: 축은 ``Transform3D`` 로 두고,
    점은 ``p_global = global_pose @ p_local`` 를 **직접** 계산해 ``world/scene/pcd/`` 에 올림.
    """
    import rerun as rr

    pts_global_mm = _transform_points_mm(pts_local_mm, M)
    entity = f"world/scene/{frame_key}"
    tf_args = None if np.allclose(M, np.eye(4)) else _matrix_to_transform_args(M)

    if tf_args is not None:
        t_list, quat_xyzw = tf_args
        rr.log(
            entity,
            rr.Transform3D(
                translation=t_list,
                quaternion=rr.Quaternion(xyzw=quat_xyzw),
            ),
            static=True,
        )

    rr.log(
        f"{entity}/axes",
        rr.Arrows3D(
            origins=[[0, 0, 0]] * 3,
            vectors=(np.eye(3) * 80.0).tolist(),
            colors=_AXIS_COLORS,
            labels=[f"{label}_X", f"{label}_Y", f"{label}_Z"],
        ),
        static=True,
    )
    log_origin_spheres([(f"{entity}/origin", np.zeros(3), origin_color, label)])

    # 글로벌 좌표 점 — world/scene 직하위 (Transform3D 이중 적용 방지)
    rr.log(
        f"world/scene/pcd/{frame_key}",
        rr.Points3D(pts_global_mm, colors=colors, radii=[1.2]),
        static=True,
    )
    return pts_global_mm


def _load_frame_points(
    ply_path: Path,
    frame_key: str,
    *,
    depth_clip: bool,
    depth_min_m: float,
    depth_max_m: float,
    depth_axis: int,
    max_points: int,
    fallback_color: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Load PLY → (points mm, colors uint8)."""
    log.info("Loading PLY %s (%s)...", frame_key, ply_path.name)
    loaded = load_ply_points(ply_path)
    if loaded is None:
        raise SystemExit(f"No points in PLY: {ply_path}")

    pts_raw, colors = loaded
    n_raw = len(pts_raw)
    pts_mm = _ply_array_to_scene_mm(pts_raw)
    pts_mm, colors = _drop_invalid_points(pts_mm, colors)
    log.info(
        "  %s valid pts %d / %d, extent mm %s",
        frame_key,
        len(pts_mm),
        n_raw,
        np.round(np.ptp(pts_mm, axis=0), 1).tolist(),
    )

    if depth_clip:
        pts_m = pts_mm / 1000.0
        pts_m, colors = clip_depth_range(
            pts_m, depth_min_m, depth_max_m,
            depth_axis=depth_axis, colors=colors,
        )
        pts_mm = pts_m * 1000.0
        log.info(
            "  %s depth clip [%.2f, %.2f] m: %d pts",
            frame_key, depth_min_m, depth_max_m, len(pts_mm),
        )

    if colors is None:
        colors = np.tile(
            np.array([list(fallback_color)], dtype=np.uint8),
            (len(pts_mm), 1),
        )
    pts_mm, colors = subsample(pts_mm, colors, max_points)
    log.info("  %s: using %d pts for display", frame_key, len(pts_mm))
    return pts_mm, colors


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Multi-frame PLY reconstruction in global frame (global_poses.yaml).",
    )
    p.add_argument(
        "--poses",
        type=Path,
        default=_DEFAULT_POSES,
        help="YAML with frames.<name>.global_pose and source_path",
    )
    p.add_argument(
        "--dataset-dir",
        type=Path,
        default=None,
        help="Directory containing N.ply files (default: parent of --poses)",
    )
    p.add_argument(
        "--frames",
        type=str,
        default=None,
        help="Comma-separated frame keys to show (default: all *.ply in dataset-dir)",
    )
    p.add_argument(
        "--no-merged-pcd",
        action="store_true",
        help="Do not log merged global point cloud at world/scene/merged",
    )
    p.add_argument(
        "--no-rgb",
        action="store_true",
        help="Do not load N.png / Pinhole / 2D camera tabs",
    )
    p.add_argument(
        "--image-plane-mm-3d",
        type=float,
        default=80.0,
        metavar="MM",
        help="Pinhole image_plane_distance for 3D RGB (default 80)",
    )
    p.add_argument(
        "--image-plane-mm-2d",
        type=float,
        default=400.0,
        metavar="MM",
        help="Pinhole image_plane_distance for 2D tabs (default 400)",
    )
    p.add_argument(
        "--intrinsics-dir",
        type=Path,
        default=None,
        help="Optional dir with N.json intrinsics; else estimate from image size",
    )
    add_common_args(p)
    # 8 frames × large PLY; depth in meters (PLY Z ≈ 0–1100 mm → ~1.2 m max)
    p.set_defaults(max_points=300_000, depth_max_m=1.2)
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()

    poses_path = args.poses.resolve()
    dataset_dir = (
        args.dataset_dir.resolve()
        if args.dataset_dir is not None
        else poses_path.parent
    )

    entries = load_global_pose_frames(poses_path, dataset_dir)
    if args.frames:
        allowed = {s.strip() for s in args.frames.split(",") if s.strip()}
        entries = [e for e in entries if e.key in allowed]
        if not entries:
            raise SystemExit(f"No frames matched --frames={args.frames!r}")

    depth_clip = not args.no_depth_clip
    save_path = args.save
    spawn = save_path is None
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)

    vis = TransformVisualizer(
        "reconstruction",
        spawn=spawn,
        port=args.port,
        views=[("Reconstruction (global)", "world/scene")],
    )

    import rerun as rr

    global_origin = np.zeros(3)
    ref_key: str | None = None
    merged_pts: list[np.ndarray] = []
    merged_cols: list[np.ndarray] = []

    for i, entry in enumerate(entries):
        M = entry.global_pose
        if ref_key is None and np.allclose(M, np.eye(4)):
            ref_key = entry.key

        color = _FRAME_PALETTE[i % len(_FRAME_PALETTE)]
        label = entry.key.upper().replace("_", "")

        pts_mm, colors = _load_frame_points(
            entry.ply_path,
            entry.key,
            depth_clip=depth_clip,
            depth_min_m=args.depth_min_m,
            depth_max_m=args.depth_max_m,
            depth_axis=args.depth_axis,
            max_points=args.max_points,
            fallback_color=color,
        )

        pts_global_mm = _log_frame_scene(
            entry.key, pts_mm, colors, M, label, color,
        )
        log.info(
            "Logged %s: %d pts (global_pose applied), |t|=%.1f mm",
            entry.key, len(pts_global_mm), float(np.linalg.norm(M[:3, 3])),
        )

        if not args.no_merged_pcd:
            merged_pts.append(pts_global_mm)
            merged_cols.append(colors)

        t_global = M[:3, 3]
        log_debug_link(
            f"world/scene/_debug/origin_to_{entry.key}",
            global_origin,
            t_global,
            color=(*color, 120),
        )

    if not args.no_merged_pcd and merged_pts:
        all_pts = np.vstack(merged_pts)
        all_cols = np.vstack(merged_cols)
        rr.log(
            "world/scene/merged/pcd",
            rr.Points3D(all_pts, colors=all_cols, radii=[1.0]),
            static=True,
        )
        log.info(
            "Merged global PCD: %d pts from %d frames",
            len(all_pts), len(entries),
        )

    if ref_key:
        log.info("Reference frame (identity global_pose): %s", ref_key)

    rgb_tabs: list[tuple[str, str]] = []
    if not args.no_rgb:
        for entry in entries:
            num = int(entry.ply_path.stem)
            rgb_path = dataset_dir / f"{num}.png"
            intrinsic_json = _resolve_intrinsics_json(
                dataset_dir, num, args.intrinsics_dir,
            )
            rgb_tabs += _log_frame_rgb(
                entry.key,
                rgb_path,
                entry.global_pose,
                plane_3d=args.image_plane_mm_3d,
                plane_2d=args.image_plane_mm_2d,
                intrinsic_json=intrinsic_json,
            )
        _send_reconstruction_blueprint(rgb_tab_origins=rgb_tabs)

    log.info("Done: %d / %d frames visualized.", len(entries), len(entries))
    finalize_viewer(save_path, spawn=spawn, app_name="Reconstruction (global)")


if __name__ == "__main__":
    main()
