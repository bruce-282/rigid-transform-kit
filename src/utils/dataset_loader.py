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

    Always returns T_base2cam (base2cam). Reads config.base2cam or config.camera_calibration;
    when the file has camera_calibration (T_cam2base), inverts to get base2cam.

    Returns
    -------
    dict with "base2cam": np.ndarray (4, 4) T_base2cam (as stored, typically mm);
    if translation is in mm (max |t| > 10), also "base2cam_m": T_base2cam in meters.
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
    else:
        cam2base = np.array(config["camera_calibration"], dtype=np.float64)
        base2cam = np.linalg.inv(cam2base)

    out = {"base2cam": base2cam}
    if np.max(np.abs(base2cam[:3, 3])) > 10:
        base2cam_m = base2cam.copy()
        base2cam_m[:3, 3] *= 0.001
        out["base2cam_m"] = base2cam_m
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


def load_cam_targets(path: Path) -> list["PickPoint"]:
    """
    Load pick points from JSON.

    Expected format (unit: mm, camera frame)::

        {
          "cam_targets": [
            {
              "p_cam": [x, y, z],
              "n_cam": [nx, ny, nz],
              "long_axis_cam": [lx, ly, lz]
            },
            ...
          ]
        }

    * ``p_cam`` (required): position in camera frame, mm.
    * ``n_cam`` (optional): surface normal (unit vector). Defaults to [0, 0, -1].
    * ``long_axis_cam`` (optional): long-axis direction (unit vector).
    """
    from rigid_transform_kit import Frame, PickPoint, RigidTransform

    if not path.exists():
        return []

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    arr = data.get("cam_targets", [])

    picks = []
    for item in arr:
        if "vec6_cam" in item:
            # vec6: [x, y, z, rx, ry, rz] (mm, degrees, WPR/xyz)
            vec6 = np.array(item["vec6_cam"], dtype=np.float64)
            T_cam2pick = RigidTransform.from_vec6(
                vec6, Frame.CAMERA, Frame.OBJECT, convention="xyz", degrees=True
            )
            p_cam = T_cam2pick.t
            R = T_cam2pick.R
            n_cam = R[:, 2].copy()
            long_axis_cam = R[:, 0].copy()
            picks.append(PickPoint(p_cam=p_cam, n_cam=n_cam, long_axis_cam=long_axis_cam))
            continue

        p_cam = np.array(item["p_cam"], dtype=np.float64)

        n_cam = None
        if "n_cam" in item:
            nv = np.array(item["n_cam"], dtype=np.float64)
            n_norm = np.linalg.norm(nv)
            n_cam = nv / n_norm if n_norm > 1e-9 else np.array([0.0, 0.0, -1.0])

        long_axis_cam = None
        if "long_axis_cam" in item:
            lv = np.array(item["long_axis_cam"], dtype=np.float64)
            l_norm = np.linalg.norm(lv)
            if l_norm > 1e-9:
                long_axis_cam = lv / l_norm

        picks.append(PickPoint(p_cam=p_cam, n_cam=n_cam, long_axis_cam=long_axis_cam))
    return picks

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
