"""
calibration._viz_common
========================
Shared utilities for calibration visualization scripts
(fixed_eye_view, hand_eye_view, multi_eye_view).

Naming convention (from/to):
    변수명 T_X_to_Y = "X 점을 Y로 보내는 행렬" = p_Y = M @ p_X
    RigidTransform 프레임 라벨도 동일하게: from=X, to=Y
    예: T_cam_to_base = from=CAMERA, to=BASE

    주의: RigidTransform 라이브러리의 내부 컨벤션(from=output, to=input)과
    반대이므로, RigidTransform의 @ 체인 연산자는 사용하지 않고
    .matrix 로 직접 합성합니다.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import time
from pathlib import Path
from typing import Sequence

import numpy as np

from rigid_transform_kit import Frame, RigidTransform
from rigid_transform_kit.viz import TransformVisualizer, save_recording
from utils import clip_depth_range, load_ply_points

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# YAML helpers
# ---------------------------------------------------------------------------

def load_4x4_matrices(path: Path, keys: Sequence[str]) -> dict[str, np.ndarray]:
    """Load 4x4 matrices from YAML for the given *keys*.

    Returns a dict of ``{key: np.ndarray(4,4)}`` for each key found.
    """
    try:
        import yaml
    except ImportError as e:
        raise ImportError("PyYAML required: pip install pyyaml") from e

    with open(path, encoding="utf-8") as f:
        doc = yaml.safe_load(f)

    if not isinstance(doc, dict):
        return {}

    result: dict[str, np.ndarray] = {}
    for key in keys:
        if key not in doc:
            continue
        arr = np.asarray(doc[key], dtype=np.float64)
        if arr.shape == (4, 4):
            result[key] = arr
        elif arr.size == 16:
            result[key] = arr.reshape(4, 4)
    return result


# ---------------------------------------------------------------------------
# PLY filename → TCP vec6
# ---------------------------------------------------------------------------

def parse_tcp_vec6_from_filename(path: Path) -> np.ndarray | None:
    """Parse stem ``[idx_]x_y_z_W_P_R`` → vec6 (mm, Fanuc xyz WPR degrees).

    Returns None if the filename cannot be parsed as 6+ floats.
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


# ---------------------------------------------------------------------------
# Point cloud pipeline
# ---------------------------------------------------------------------------

def subsample(
    pts: np.ndarray, colors: np.ndarray | None, max_n: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Random subsample to *max_n* points (0 = no limit)."""
    if max_n <= 0 or pts.shape[0] <= max_n:
        return pts, colors
    rng = np.random.default_rng(0)
    idx = rng.choice(pts.shape[0], size=max_n, replace=False)
    c2 = colors[idx] if colors is not None else None
    return pts[idx], c2


def load_and_preprocess_ply(
    ply_path: Path,
    *,
    no_depth_clip: bool = False,
    depth_min_m: float = 0.0,
    depth_max_m: float = 3.0,
    depth_axis: int = 2,
    max_points: int = 500_000,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Load PLY, filter NaN, auto-detect mm→m, depth-clip, subsample.

    Returns ``(pts_mm, colors)`` in **millimeters** (cam frame).
    """
    ply_data = load_ply_points(ply_path)
    if ply_data is None:
        raise SystemExit(f"No points in PLY: {ply_path}")

    pts_m, colors = ply_data

    valid = ~np.any(np.isnan(pts_m), axis=1)
    if not np.all(valid):
        n_before = len(pts_m)
        pts_m = pts_m[valid]
        colors = colors[valid] if colors is not None else None
        log.info("Filtered NaN: %d -> %d pts", n_before, len(pts_m))

    if len(pts_m) > 0 and np.median(np.abs(pts_m)) > 100:
        pts_m = pts_m / 1000.0
        log.info("PLY appears to be in mm; converted to meters.")

    if not no_depth_clip:
        n_before = len(pts_m)
        pts_m, colors = clip_depth_range(
            pts_m, depth_min_m, depth_max_m,
            depth_axis=depth_axis, colors=colors,
        )
        log.info(
            "Depth clip [%.2f, %.2f] m (axis=%d): %d -> %d pts",
            depth_min_m, depth_max_m, depth_axis, n_before, len(pts_m),
        )

    pts_mm = pts_m * 1000.0
    pts_mm, colors = subsample(pts_mm, colors, max_points)
    return pts_mm, colors


# ---------------------------------------------------------------------------
# Rerun visualization helpers
# ---------------------------------------------------------------------------

def log_origin_spheres(
    frames: list[tuple[str, np.ndarray, tuple[int, int, int], str]],
) -> None:
    """Log colored origin spheres. Each entry: (entity_path, position, rgb, label)."""
    import rerun as rr

    for path, pos, color, label in frames:
        rr.log(
            path,
            rr.Points3D([pos.tolist()], colors=[color], radii=[6.0], labels=[label]),
            static=True,
        )


def log_debug_link(
    entity_path: str,
    start: np.ndarray,
    end: np.ndarray,
    color: tuple[int, ...] = (180, 180, 180, 120),
    radius: float = 1.5,
) -> None:
    """Log a single debug line strip between two 3D points."""
    import rerun as rr

    rr.log(
        entity_path,
        rr.LineStrips3D(
            [[start.tolist(), end.tolist()]],
            colors=[list(color)],
            radii=[radius],
        ),
        static=True,
    )


def log_camera_frustum(
    entity_path: str,
    *,
    focal_length: float = 300.0,
    width: int = 640,
    height: int = 480,
    image_plane_distance: float = 60.0,
) -> None:
    """Log a camera frustum wireframe."""
    import rerun as rr

    rr.log(
        entity_path,
        rr.Pinhole(
            focal_length=focal_length,
            width=width,
            height=height,
            image_plane_distance=image_plane_distance,
        ),
        static=True,
    )


# ---------------------------------------------------------------------------
# Rerun save / spawn
# ---------------------------------------------------------------------------

def finalize_viewer(
    save_path: Path | None,
    *,
    spawn: bool,
    app_name: str = "",
) -> None:
    """Save .rrd and/or keep Rerun viewer alive."""
    if save_path is not None:
        save_recording(save_path)
        log.info("Saved to %s", save_path)
        try:
            subprocess.run(["rerun", str(save_path)], check=False)
        except FileNotFoundError:
            log.info("Run: rerun %s", save_path)

    if spawn:
        if app_name:
            log.info("Rerun viewer open — '%s' tab.", app_name)
        import rerun as rr

        rec = rr.get_global_data_recording()
        if rec is not None:
            try:
                rec.flush(timeout_sec=10.0)
            except Exception:
                pass
            time.sleep(1.0)


# ---------------------------------------------------------------------------
# Common argparse arguments
# ---------------------------------------------------------------------------

def add_common_args(p: argparse.ArgumentParser) -> None:
    """Add shared CLI arguments for calibration viewers."""
    p.add_argument(
        "--save", type=Path, default=None, metavar="RRD",
        help="Save to .rrd file instead of live spawn.",
    )
    p.add_argument(
        "--port", type=int, default=None,
        help="Rerun gRPC port.",
    )
    p.add_argument(
        "--no-depth-clip", action="store_true",
        help="Disable depth clipping.",
    )
    p.add_argument(
        "--depth-min-m", type=float, default=0.0,
        help="Depth clip min (meters, default 0).",
    )
    p.add_argument(
        "--depth-max-m", type=float, default=3.0,
        help="Depth clip max (meters, default 3.0).",
    )
    p.add_argument(
        "--depth-axis", type=int, default=2,
        help="Depth axis index (default 2 = Z).",
    )
    p.add_argument(
        "--max-points", type=int, default=500_000,
        help="Max points to display (subsample if larger, 0 = no limit).",
    )
