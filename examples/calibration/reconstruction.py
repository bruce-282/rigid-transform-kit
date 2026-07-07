"""
examples / calibration / reconstruction.py
============================================
다중 프레임 PLY를 ``global_poses.yaml`` 의 ``global_pose`` 로 **글로벌 좌표계**에
맞춰 Rerun 으로 시각화합니다 (``multi_eye_view`` 와 동일한 씬 패턴).

* 각 프레임: PLY 샘플링 → ``p_global = global_pose @ p_local`` (행렬로 점 변환 후 로그).
* 프레임 축: ``Transform3D(global_pose)`` 로 표시 (RGB Pinhole 이 카메라 뷰).
* 기준 프레임: ``global_pose`` 가 단위행렬인 프레임 (예: ``frame_3``) = 글로벌 원점.
* ``world/scene/frame_*`` — 축·프레임 pose만.
* ``world/pcd/frame_*`` — 글로벌 좌표 점군 (``merged`` 선택).
* ``world/rgb/frame_*`` — 3D Pinhole + 2D ``rerun_2d/01`` …
* ``world/_debug/`` — 프레임 간 연결선.
* ``world/mesh/scene_mesh`` — ``scene_mesh.ply`` 정점 샘플링 (Points3D).
* 프레임 PLY — 무효점 제거 후 **랜덤 샘플링** (기본 8만 점/프레임).

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
    static: bool = True,
) -> None:
    """Pinhole + Image from RGB array (no intrinsics JSON).

    ``static=False`` logs the Image on the active timeline (sequential mode);
    ViewCoordinates / Pinhole stay static (constant intrinsics).
    """
    import rerun as rr

    h, w = rgb_hwc.shape[:2]
    if focal_length is None:
        focal_length = float(max(w, h)) * 0.9
    fx = fy = focal_length
    cx, cy = w / 2.0, h / 2.0

    rr.log(entity_path, rr.ViewCoordinates.RDF, static=static)
    rr.log(
        entity_path,
        rr.Pinhole(
            focal_length=[fx, fy],
            principal_point=[cx, cy],
            width=w,
            height=h,
            image_plane_distance=float(image_plane_mm),
        ),
        static=static,
    )
    rr.log(entity_path, rr.Image(rgb_hwc), static=static)


def _log_rgb_pinhole_json(
    entity_path: str,
    rgb_hwc: np.ndarray,
    intrinsic_json: Path,
    *,
    image_plane_mm: float,
    static: bool = True,
) -> bool:
    """Pinhole + Image using intrinsics JSON (multi_eye_view style)."""
    import rerun as rr

    K, _ = load_intrinsics_any(intrinsic_json)
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    w, h = _read_image_resolution_json(intrinsic_json)

    rr.log(entity_path, rr.ViewCoordinates.RDF, static=static)
    rr.log(
        entity_path,
        rr.Pinhole(
            focal_length=[fx, fy],
            principal_point=[cx, cy],
            width=w,
            height=h,
            image_plane_distance=float(image_plane_mm),
        ),
        static=static,
    )
    rr.log(entity_path, rr.Image(rgb_hwc), static=static)
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
    frame_num: int,
    rgb_path: Path,
    M: np.ndarray,
    *,
    plane_3d: float,
    plane_2d: float,
    intrinsic_json: Path | None,
) -> bool:
    """Log RGB: 3D at frame pose + 2D under ``rerun_2d/{num:02d}`` (tree pick)."""
    import rerun as rr

    if not rgb_path.exists():
        log.warning("RGB not found, skip: %s", rgb_path)
        return False

    rgb_hwc = _load_rgb_hwc(rgb_path)
    if rgb_hwc is None:
        log.warning("Failed to read RGB: %s", rgb_path)
        return False

    entity_3d = f"world/rgb/{frame_key}"
    entity_2d = f"rerun_2d/{frame_num:02d}"
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
        _log_rgb_pinhole_json(entity_2d, rgb_hwc, intrinsic_json, image_plane_mm=plane_2d)
    else:
        _log_rgb_pinhole_hwc(entity_3d, rgb_hwc, image_plane_mm=plane_3d)
        _log_rgb_pinhole_hwc(entity_2d, rgb_hwc, image_plane_mm=plane_2d)

    log.info("Logged RGB %s -> %s, %s", rgb_path.name, entity_3d, entity_2d)
    return True


def _log_scene_mesh(
    mesh_path: Path,
    *,
    max_points: int = 150_000,
) -> bool:
    """Log ``scene_mesh.ply`` as sampled Points3D at ``world/mesh/scene_mesh``."""
    import rerun as rr

    try:
        import open3d as o3d
    except ImportError:
        log.warning("open3d required for scene mesh")
        return False

    if not mesh_path.exists():
        return False

    log.info("Loading scene mesh %s (sampled points)...", mesh_path.name)
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if mesh.is_empty():
        log.warning("Scene mesh empty: %s", mesh_path)
        return False

    verts = _ply_array_to_scene_mm(np.asarray(mesh.vertices, dtype=np.float64))
    colors: np.ndarray | None = None
    if mesh.has_vertex_colors():
        c = np.asarray(mesh.vertex_colors, dtype=np.float64)
        colors = (np.clip(c, 0, 1) * 255).astype(np.uint8)

    n_verts = len(verts)
    verts, colors = subsample(verts, colors, max_points)
    log.info("  scene_mesh: %d -> %d verts", n_verts, len(verts))

    rr.log(
        "world/mesh/scene_mesh",
        rr.Points3D(verts, colors=colors, radii=[0.8]),
        static=True,
    )
    return True


def _send_reconstruction_blueprint(*, include_rgb_tab: bool) -> None:
    """3D + optional single 2D tab (프레임마다 탭을 만들지 않음 — 24탭이면 UI에서 누락됨)."""
    import rerun as rr
    import rerun.blueprint as rrb

    tabs: list = [
        rrb.Spatial3DView(
            name="Reconstruction (global)",
            origin="world",
            contents=[
                "+ world/scene/**",
                "+ world/pcd/**",
                "+ world/rgb/**",
                "+ world/mesh/**",
                "+ world/_debug/**",
            ],
            line_grid=False,
        ),
    ]
    if include_rgb_tab:
        tabs.append(
            rrb.Spatial2DView(
                name="Camera images",
                origin="rerun_2d",
                contents=["+ rerun_2d/**"],
            )
        )
    rr.send_blueprint(rrb.Blueprint(rrb.Tabs(*tabs)))


def _order_entries_nearest(entries: list) -> list:
    """Greedy nearest-neighbor chain by camera translation (mm).

    파일 번호 순서 대신 **공간상 가장 가까운 프레임**을 차례로 이어 붙여
    인접 프레임끼리 부드럽게 진행되도록 재정렬합니다. 시작점은 기준 프레임
    (global_pose == identity) 이 있으면 그것, 없으면 첫 항목.
    """
    if len(entries) <= 2:
        return list(entries)

    remaining = list(entries)
    start_idx = 0
    for j, e in enumerate(remaining):
        if np.allclose(e.global_pose, np.eye(4)):
            start_idx = j
            break
    ordered = [remaining.pop(start_idx)]
    while remaining:
        last_t = ordered[-1].global_pose[:3, 3]
        k = int(np.argmin([
            float(np.linalg.norm(e.global_pose[:3, 3] - last_t)) for e in remaining
        ]))
        ordered.append(remaining.pop(k))
    return ordered


def _order_entries(entries: list, mode: str) -> list:
    """Order frames for sequential playback: ``file`` (numeric) or ``nearest``."""
    if mode == "nearest":
        return _order_entries_nearest(entries)
    return list(entries)


def _send_sequential_blueprint(*, include_rgb_tab: bool, include_snr_tab: bool = False) -> None:
    """Sequential playback: single moving camera + current-frame PCD over a timeline."""
    import rerun as rr
    import rerun.blueprint as rrb

    tabs: list = [
        rrb.Spatial3DView(
            name="Sequential (global)",
            origin="world",
            contents=["+ world/**"],
            line_grid=False,
        ),
    ]
    if include_rgb_tab:
        tabs.append(
            rrb.Spatial2DView(
                name="Camera image",
                origin="rerun_2d",
                contents=["+ rerun_2d/**"],
            )
        )
    if include_snr_tab:
        tabs.append(
            rrb.Spatial2DView(
                name="SNR map",
                origin="rerun_snr",
                contents=["+ rerun_snr/**"],
            )
        )
    rr.send_blueprint(rrb.Blueprint(rrb.Tabs(*tabs)))


def _log_sequential(
    entries,
    dataset_dir: Path,
    args,
    *,
    timeline: str = "step",
    accumulate: bool = False,
) -> None:
    """Log frames on a timeline (camera + PCD per step).

    * ``accumulate=False`` (기본): 공유 엔티티(``world/pcd/current``, ``world/cam``)에
      기록 → 재생하면 프레임이 **한 개씩 교체**되며 지나갑니다.
    * ``accumulate=True``: 프레임마다 **고유 엔티티**(``world/pcd/frame_N`` 등)에
      기록 → 재생할수록 이전 프레임 위에 **순차적으로 쌓입니다** (reconstruction 누적).

    재생 순서는 ``--order`` 로 결정 (``file`` 번호순 / ``nearest`` 공간 인접순).
    타임라인 값은 파일 번호가 아니라 **재생 순서(step, 1-based)** 입니다.
    """
    import rerun as rr

    depth_clip = not args.no_depth_clip
    entries = _order_entries(entries, args.order)
    log.info(
        "Sequential order (%s): %s",
        args.order, " -> ".join(str(int(e.ply_path.stem)) for e in entries),
    )

    # CloudCompare 느낌: 화면 픽셀 단위(UI points)로 일정한 작은 점.
    # --point-size-mm 지정 시 물리 반지름(mm, 줌에 따라 커짐).
    if args.point_size_mm is not None:
        pcd_radii = [float(args.point_size_mm)]
    else:
        pcd_radii = rr.Radius.ui_points([float(args.point_size)])

    if not accumulate:
        # 교체 모드: 카메라 로컬 축은 고정(Transform3D 가 프레임마다 이동) → 한 번만 static.
        rr.log(
            "world/cam/axes",
            rr.Arrows3D(
                origins=[[0, 0, 0]] * 3,
                vectors=(np.eye(3) * 80.0).tolist(),
                colors=_AXIS_COLORS,
                labels=["cam_X", "cam_Y", "cam_Z"],
            ),
            static=True,
        )

    rgb_ok = 0
    snr_ok = 0
    for i, entry in enumerate(entries):
        num = int(entry.ply_path.stem)
        step = i + 1  # 재생 순서 (파일 번호가 아님)
        M = entry.global_pose
        color = _FRAME_PALETTE[i % len(_FRAME_PALETTE)]

        # 누적: 프레임별 고유 경로 / 교체: 공유 경로.
        pcd_path = f"world/pcd/{entry.key}" if accumulate else "world/pcd/current"
        cam_path = f"world/cam/{entry.key}" if accumulate else "world/cam"
        # 2D "Camera image" 탭은 항상 공유 경로 → 재생 순서(step)대로 이미지가 갱신됨.
        rgb_2d_path = "rerun_2d/current"

        rr.set_time(timeline, sequence=step)

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
        pts_global_mm = _transform_points_mm(pts_mm, M)
        rr.log(pcd_path, rr.Points3D(pts_global_mm, colors=colors, radii=pcd_radii))

        t_list, quat_xyzw = _matrix_to_transform_args(M)
        rr.log(
            cam_path,
            rr.Transform3D(
                translation=t_list,
                quaternion=rr.Quaternion(xyzw=quat_xyzw),
            ),
        )
        if accumulate:
            # 누적 모드는 카메라마다 고유 경로 → 축도 프레임별로 로깅.
            rr.log(
                f"{cam_path}/axes",
                rr.Arrows3D(
                    origins=[[0, 0, 0]] * 3,
                    vectors=(np.eye(3) * 80.0).tolist(),
                    colors=_AXIS_COLORS,
                ),
            )

        log.info(
            "Seq step %d = %s: %d pts, |t|=%.1f mm",
            step, entry.key, len(pts_global_mm), float(np.linalg.norm(M[:3, 3])),
        )

        intrinsic_json = _resolve_intrinsics_json(dataset_dir, num, args.intrinsics_dir)

        def _log_frustum_img(path: str, img: np.ndarray) -> None:
            """3D view: Pinhole + Image (projected onto the camera image plane)."""
            if intrinsic_json is not None and intrinsic_json.exists():
                _log_rgb_pinhole_json(
                    path, img, intrinsic_json,
                    image_plane_mm=args.image_plane_mm_3d, static=False,
                )
            else:
                _log_rgb_pinhole_hwc(
                    path, img, image_plane_mm=args.image_plane_mm_3d, static=False,
                )

        # ---- RGB: 3D frustum image (with Pinhole) + plain 2D "Camera image" tab ----
        if not args.no_rgb:
            rgb_path = dataset_dir / f"{num}.png"
            rgb_hwc = _load_rgb_hwc(rgb_path) if rgb_path.exists() else None
            if rgb_hwc is None:
                log.warning("RGB not found/readable, skip: %s", rgb_path)
            else:
                _log_frustum_img(f"{cam_path}/rgb", rgb_hwc)
                rr.log(rgb_2d_path, rr.Image(rgb_hwc))  # 2D 탭: 이미지만 (Pinhole 금지)
                rgb_ok += 1

        # ---- SNR map: plain 2D "SNR map" tab (timeline-synced, shared path) ----
        if not args.no_snr:
            snr_path = dataset_dir / f"{num}{args.snr_suffix}.png"
            snr_hwc = _load_rgb_hwc(snr_path) if snr_path.exists() else None
            if snr_hwc is None:
                log.warning("SNR not found/readable, skip: %s", snr_path)
            else:
                rr.log("rerun_snr/current", rr.Image(snr_hwc))  # 2D 탭: 이미지만
                snr_ok += 1

    _send_sequential_blueprint(include_rgb_tab=rgb_ok > 0, include_snr_tab=snr_ok > 0)
    log.info(
        "Sequential (%s): %d frames on timeline '%s' (RGB %d, SNR %d).",
        "accumulate" if accumulate else "replace",
        len(entries), timeline, rgb_ok, snr_ok,
    )


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
    점은 ``p_global = global_pose @ p_local`` → ``world/pcd/{frame_key}`` (scene 과 분리).
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

    rr.log(
        f"world/pcd/{frame_key}",
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
    """Load PLY → subsample → optional depth clip (points mm, colors uint8)."""
    log.info("Loading PLY %s (%s)...", frame_key, ply_path.name)
    loaded = load_ply_points(ply_path)
    if loaded is None:
        raise SystemExit(f"No points in PLY: {ply_path}")

    pts_raw, colors = loaded
    n_raw = len(pts_raw)
    pts_mm = _ply_array_to_scene_mm(pts_raw)
    pts_mm, colors = _drop_invalid_points(pts_mm, colors)
    n_valid = len(pts_mm)

    if colors is None:
        colors = np.tile(
            np.array([list(fallback_color)], dtype=np.uint8),
            (n_valid, 1),
        )
    pts_mm, colors = subsample(pts_mm, colors, max_points)

    if depth_clip and len(pts_mm) > 0:
        pts_m = pts_mm / 1000.0
        pts_m, colors = clip_depth_range(
            pts_m, depth_min_m, depth_max_m,
            depth_axis=depth_axis, colors=colors,
        )
        pts_mm = pts_m * 1000.0

    log.info(
        "  %s: %d / %d raw → %d sampled pts",
        frame_key, n_valid, n_raw, len(pts_mm),
    )
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
        "--sequential",
        action="store_true",
        help="Play frames one-at-a-time on a timeline (camera + PCD replaced each step) "
             "instead of overlaying all frames statically.",
    )
    p.add_argument(
        "--accumulate",
        action="store_true",
        help="With --sequential: keep each frame on its own entity so frames stack up "
             "over the timeline (reconstruction builds up) instead of being replaced.",
    )
    p.add_argument(
        "--order",
        choices=["file", "nearest"],
        default="file",
        help="Sequential playback order: 'file' = numeric filename order (1,2,3,…), "
             "'nearest' = greedy nearest-neighbor path by camera position so adjacent "
             "steps are spatially close (default file).",
    )
    p.add_argument(
        "--point-size",
        type=float,
        default=1.5,
        metavar="PX",
        help="Point size in screen pixels (UI points) for --sequential PCD — small = "
             "CloudCompare-like crisp dots (default 1.5). See also --point-size-mm.",
    )
    p.add_argument(
        "--point-size-mm",
        type=float,
        default=None,
        metavar="MM",
        help="Use a fixed physical point radius in mm instead of screen-pixel sizing "
             "(overrides --point-size; points grow/shrink with zoom).",
    )
    p.add_argument(
        "--no-merged-pcd",
        action="store_true",
        help="Do not log merged global point cloud at world/pcd/merged",
    )
    p.add_argument(
        "--scene-mesh",
        type=Path,
        default=None,
        help="Scene mesh PLY (default: <dataset-dir>/scene_mesh.ply if exists)",
    )
    p.add_argument(
        "--no-scene-mesh",
        action="store_true",
        help="Do not load scene_mesh.ply",
    )
    p.add_argument(
        "--max-mesh-points",
        type=int,
        default=150_000,
        metavar="N",
        help="Max points for scene_mesh.ply vertex sampling (default 150000)",
    )
    p.add_argument(
        "--no-rgb",
        action="store_true",
        help="Do not load N.png / Pinhole / 2D camera tabs",
    )
    p.add_argument(
        "--no-snr",
        action="store_true",
        help="Do not load N<snr-suffix>.png SNR maps (sequential mode only)",
    )
    p.add_argument(
        "--snr-suffix",
        type=str,
        default="_snr_map",
        help="Filename suffix for per-frame SNR map PNG (default '_snr_map' -> N_snr_map.png)",
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
    # 프레임 PLY: 샘플링 우선 (기본 8만/프레임)
    p.set_defaults(max_points=80_000, depth_max_m=1.2)
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
        views=[("Reconstruction (global)", "world")],
    )

    import rerun as rr

    if args.sequential or args.accumulate:
        _log_sequential(entries, dataset_dir, args, accumulate=args.accumulate)
        if not args.no_scene_mesh:
            mesh_path = args.scene_mesh or (dataset_dir / "scene_mesh.ply")
            _log_scene_mesh(mesh_path.resolve(), max_points=args.max_mesh_points)
        log.info("Done (sequential): %d frames.", len(entries))
        finalize_viewer(save_path, spawn=spawn, app_name="Sequential (global)")
        return

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
            f"world/_debug/to_{entry.key}",
            global_origin,
            t_global,
            color=(*color, 120),
        )

    if not args.no_merged_pcd and merged_pts:
        all_pts = np.vstack(merged_pts)
        all_cols = np.vstack(merged_cols)
        rr.log(
            "world/pcd/merged",
            rr.Points3D(all_pts, colors=all_cols, radii=[1.0]),
            static=True,
        )
        log.info(
            "Merged global PCD: %d pts from %d frames",
            len(all_pts), len(entries),
        )

    if ref_key:
        log.info("Reference frame (identity global_pose): %s", ref_key)

    if not args.no_scene_mesh:
        mesh_path = args.scene_mesh or (dataset_dir / "scene_mesh.ply")
        _log_scene_mesh(mesh_path.resolve(), max_points=args.max_mesh_points)

    if not args.no_rgb:
        rgb_ok = 0
        for entry in entries:
            num = int(entry.ply_path.stem)
            rgb_path = dataset_dir / f"{num}.png"
            intrinsic_json = _resolve_intrinsics_json(
                dataset_dir, num, args.intrinsics_dir,
            )
            if _log_frame_rgb(
                entry.key,
                num,
                rgb_path,
                entry.global_pose,
                plane_3d=args.image_plane_mm_3d,
                plane_2d=args.image_plane_mm_2d,
                intrinsic_json=intrinsic_json,
            ):
                rgb_ok += 1
        _send_reconstruction_blueprint(include_rgb_tab=rgb_ok > 0)
        log.info(
            "RGB: %d / %d frames (2D tab: Camera images → pick rerun_2d/01 … in tree)",
            rgb_ok, len(entries),
        )

    log.info("Done: %d / %d frames visualized.", len(entries), len(entries))
    finalize_viewer(save_path, spawn=spawn, app_name="Reconstruction (global)")


if __name__ == "__main__":
    main()
