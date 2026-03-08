"""
Load calibration, intrinsics, suction points, and PLY from a vision/pallet dataset.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from rigid_transform_kit import CameraConfig, PickPoint


def load_extrinsics(calibration_yml: Path) -> dict:
    """
    Load hand-eye (camera) calibration from YAML.

    Prefers config.base2cam (T_base2cam) if present and returns T_cam2base (inverse).
    Otherwise uses config.camera_calibration (T_cam2base).
    Matrix is returned as stored (typically translation in mm).

    Returns
    -------
    dict with "camera_calibration": np.ndarray (4, 4) T_cam2base;
    if translation is in mm (max |t| > 10), also "camera_calibration_m": matrix in meters.
    """
    if yaml is None:
        raise ImportError("PyYAML required: pip install pyyaml")

    if not calibration_yml.exists():
        raise FileNotFoundError(f"Calibration not found: {calibration_yml}")

    with open(calibration_yml, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    config = data.get("config", data)
    if "base2cam" in config:
        base2cam = np.array(config["base2cam"], dtype=np.float64)
        mat = np.linalg.inv(base2cam)
    else:
        mat = np.array(config["camera_calibration"], dtype=np.float64)

    out = {"camera_calibration": mat}
    if np.max(np.abs(mat[:3, 3])) > 10:
        mat_m = mat.copy()
        mat_m[:3, 3] *= 0.001
        out["camera_calibration_m"] = mat_m
    return out


def load_intrinsics(intrinsic_json: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Load camera intrinsics from JSON.

    Expects sensores.image or image with intrinsic_matrix (9 elements) and
    optional distortion_coefficients.

    Returns
    -------
    K : np.ndarray (3, 3)
    dist : np.ndarray (5,)
    """
    if not intrinsic_json.exists():
        raise FileNotFoundError(f"Intrinsic not found: {intrinsic_json}")

    with open(intrinsic_json, encoding="utf-8") as f:
        intrinsic_data = json.load(f)

    sensor = intrinsic_data.get("sensores", {}).get("image", intrinsic_data.get("image", {}))
    if not sensor:
        raise KeyError("intrinsic JSON must contain sensores.image or image")

    K_flat = sensor["intrinsic_matrix"]
    K = np.array(
        [
            [K_flat[0], K_flat[1], K_flat[2]],
            [K_flat[3], K_flat[4], K_flat[5]],
            [K_flat[6], K_flat[7], K_flat[8]],
        ],
        dtype=np.float64,
    )
    dist = np.array(sensor.get("distortion_coefficients", [0.0] * 5), dtype=np.float64)
    if len(dist) > 5:
        dist = dist[:5]
    return K, dist


def load_suction_pts(suction_pts_path: Path, cam_config: "CameraConfig") -> list["PickPoint"]:
    """
    Load suction_pts.json and convert to list of PickPoint.

    Format in file: original_suction_pts is list of items like
      [[[x, y, z], [nx, ny] or [nx, ny, nz] or [nx, ny, angle]]]
    We assume (x,y,z) is in camera frame in mm; normal is normalized to unit vector.
    """
    from rigid_transform_kit import PickPoint

    if not suction_pts_path.exists():
        return []

    with open(suction_pts_path, encoding="utf-8") as f:
        raw = f.read().strip()

    try:
        data = json.loads(raw)
        arr = data.get("original_suction_pts", data.get("suction_pts", []))
    except json.JSONDecodeError:
        start = raw.find("[[")
        end = raw.rfind("]]") + 2
        if start == -1 or end <= start:
            return []
        arr = json.loads(raw[start:end])

    picks = []
    for item in arr:
        if not item:
            continue
        if isinstance(item[0], list) and len(item[0]) >= 2:
            group = item[0]
        else:
            group = item
        if not isinstance(group, (list, tuple)) or len(group) < 1:
            continue
        pos = np.array(group[0], dtype=np.float64)
        p_cam = pos[:3] / 1000.0

        n_cam = None
        if len(group) >= 2:
            norm = np.array(group[1], dtype=np.float64)
            if len(norm) >= 2:
                if len(norm) >= 3:
                    n_cam = norm[:3]
                else:
                    n_cam = np.array([float(norm[0]), float(norm[1]), 0.0])
                n_norm = np.linalg.norm(n_cam)
                if n_norm > 1e-9:
                    n_cam = n_cam / n_norm
                else:
                    n_cam = np.array([0.0, 0.0, -1.0])

        picks.append(PickPoint(p_cam=p_cam, n_cam=n_cam))
    return picks


def save_suction_pts(
    picks: list["PickPoint"],
    suction_pts_path: Path | None = None,
    out_path: Path | None = None,
) -> None:
    """
    Save suction points as JSON: list of {"p_cam": [x,y,z], "n_cam": [nx,ny,nz]}
    (meters, unit normal).
    """
    out_path = out_path or (
        (suction_pts_path.with_name(suction_pts_path.stem + "_clean.json"))
        if suction_pts_path else None
    )
    if out_path is None:
        return
    data = []
    for p in picks:
        entry = {"p_cam": p.p_cam.tolist()}
        if p.n_cam is not None:
            entry["n_cam"] = p.n_cam.tolist()
        data.append(entry)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"Wrote suction points to {out_path}")


def load_ply_points(path: Path) -> tuple[np.ndarray, np.ndarray | None] | None:
    """Load point cloud from PLY; return (points Nx3 in meters, colors Nx3 uint8 or None)."""
    try:
        import open3d as o3d
        pcd = o3d.io.read_point_cloud(str(path))
        pts = np.asarray(pcd.points, dtype=np.float64)
        if pts.size == 0:
            return None
        if np.median(pts) > 100:
            pts = pts / 1000.0
        colors = None
        if pcd.has_colors():
            c = np.asarray(pcd.colors, dtype=np.float64)
            colors = (np.clip(c, 0, 1) * 255).astype(np.uint8)
        return (pts, colors)
    except ImportError:
        return _load_ply_ascii(path)


def _load_ply_ascii(path: Path) -> tuple[np.ndarray, np.ndarray | None] | None:
    """Minimal ASCII PLY reader: x,y,z and optional red,green,blue."""
    try:
        with open(path, encoding="utf-8") as f:
            lines = f.readlines()
        props = []
        n_vertex = 0
        for line in lines:
            line = line.strip()
            if line == "end_header":
                break
            if line.startswith("element vertex"):
                n_vertex = int(line.split()[-1])
            elif line.startswith("property"):
                parts = line.split()
                if len(parts) >= 2:
                    props.append(parts[1].lower())
        if n_vertex <= 0:
            return None
        idx_x = props.index("x") if "x" in props else 0
        idx_y = props.index("y") if "y" in props else 1
        idx_z = props.index("z") if "z" in props else 2
        color_props = []
        for c in ("red", "green", "blue"):
            if c in props:
                color_props.append(props.index(c))
        if len(color_props) != 3:
            color_props = []
        header_end = next(i for i, L in enumerate(lines) if L.strip() == "end_header")
        pts = []
        colors = [] if color_props else None
        for line in lines[header_end + 1 : header_end + 1 + n_vertex]:
            parts = line.split()
            if len(parts) < 3:
                continue
            pts.append([float(parts[idx_x]), float(parts[idx_y]), float(parts[idx_z])])
            if color_props:
                r = int(float(parts[color_props[0]]))
                g = int(float(parts[color_props[1]]))
                b = int(float(parts[color_props[2]]))
                if max(r, g, b) <= 1:
                    r, g, b = int(r * 255), int(g * 255), int(b * 255)
                colors.append([r, g, b])
        if not pts:
            return None
        pts = np.array(pts, dtype=np.float64)
        if np.median(pts) > 100:
            pts = pts / 1000.0
        out_colors = np.array(colors, dtype=np.uint8) if colors else None
        return (pts, out_colors)
    except Exception:
        return None


def load_box_pcd(path: Path):
    """Load box point cloud from PLY as an Open3D PointCloud."""
    try:
        import open3d as o3d
        pcd = o3d.io.read_point_cloud(str(path))
        if len(pcd.points) == 0:
            return None
        return pcd
    except Exception:
        return None
