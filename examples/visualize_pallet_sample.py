"""
rigid-transform-kit / examples / visualize_pallet_sample1.py
============================================================
Visualize pallet_sample1: calibration, point cloud, suction points.

Uses:
  - data/vision/pallet_sample1/palletizing_robot.picking_zone_camera.calibration.yml
  - data/vision/pallet_sample1/picking_zone_camera_*_intrinsic.json
  - data/vision/pallet_sample1/picking_zone_camera_*_pcd.ply
  - data/vision/pallet_sample1/picking_zone_camera_*_suction_pts.json

Requires: pip install rigid-transform-kit[viz]  (and open3d for PLY)
Run: python examples/visualize_pallet_sample1.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

try:
    import yaml
except ImportError:
    raise ImportError("This example requires PyYAML. Install with: pip install pyyaml")

from rigid_transform_kit import (
    CameraConfig,
    Frame,
    PickPoint,
    RigidTransform,
    build_tcp_pose,
)
from rigid_transform_kit.robot import FanucAdapter
from rigid_transform_kit.viz import TransformVisualizer
from utils import fit_plane, get_box_axes


# -----------------------------------------------------------------------------
# Data paths (pallet_sample1)
# -----------------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parent.parent / "datasets" / "aw_pallet"
CALIBRATION_YML = DATA_DIR / "calibration_result.yml"
INTRINSIC_JSON = DATA_DIR / "intrinsics.json"
#SUCTION_PTS_JSON = DATA_DIR / "suction_pts.json"
SUCTION_PTS_JSON = None
PCD_PLY = DATA_DIR / "pcd.ply"
box_pcd1 = DATA_DIR / "box_pcd1.ply"




def load_calibration_and_intrinsics():
    """Load camera_calibration from YAML (config.camera_calibration) and K, D from intrinsic JSON."""
    if not CALIBRATION_YML.exists():
        raise FileNotFoundError(f"Calibration not found: {CALIBRATION_YML}")

    with open(CALIBRATION_YML, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    # calibration.yml has root key "config"
    config = data.get("config", data)
    mat = np.array(config["camera_calibration"], dtype=np.float64)
    # Hand-eye calibration is often in mm; convert translation to meters for visualization
    if np.max(np.abs(mat[:3, 3])) > 10:
        mat = mat.copy()
        mat[:3, 3] *= 0.001
    calib = {"camera_calibration": mat}

    if not INTRINSIC_JSON.exists():
        raise FileNotFoundError(f"Intrinsic not found: {INTRINSIC_JSON}")

    with open(INTRINSIC_JSON, encoding="utf-8") as f:
        intrinsic_data = json.load(f)

    # Use "image" sensor (RGB); scanner has different K for depth)
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

    cam_config = CameraConfig.from_calibration_dict(
        calib=calib,
        intrinsics=K,
        distortion=dist,
        depth_scale=0.001,  # mm -> m
        calib_convention="cam2base",
    )
    return cam_config


def load_suction_pts(cam_config: CameraConfig):
    """
    Load suction_pts.json and convert to list of PickPoint.

    Format in file: original_suction_pts is list of items like
      [[[x, y, z], [nx, ny] or [nx, ny, nz] or [nx, ny, angle]]]
    We assume (x,y,z) is in camera frame in mm; normal is normalized to unit vector.
    """
    if not SUCTION_PTS_JSON.exists():
        return []

    with open(SUCTION_PTS_JSON, encoding="utf-8") as f:
        raw = f.read().strip()

    # Handle "original_suction_pts : \n[[..." style (key with space before colon = invalid JSON)
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
        # item can be [[[x,y,z], [nx,ny,...]]] or [[x,y,z], [nx,ny,...]]
        if isinstance(item[0], list) and len(item[0]) >= 2:
            group = item[0]  # [[x,y,z], [nx,ny]]
        else:
            group = item
        if not isinstance(group, (list, tuple)) or len(group) < 1:
            continue
        pos = np.array(group[0], dtype=np.float64)
        # pos in mm -> meters
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


def save_suction_pts(picks: list[PickPoint], out_path: Path | None = None):
    """
    Save suction points as JSON: list of {"p_cam": [x,y,z], "n_cam": [nx,ny,nz]}
    (meters, unit normal).
    """
    out_path = out_path or SUCTION_PTS_JSON.with_name(
        SUCTION_PTS_JSON.stem + "_clean.json"
    )
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
        # Assume points are in mm (common for robot vision)
        if np.median(pts) > 100:
            pts = pts / 1000.0
        colors = None
        if pcd.has_colors():
            # Open3D returns [0,1] float; Rerun expects 0-255
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

def main():
    cam_config = load_calibration_and_intrinsics()

    picks: list[PickPoint] = []

    if SUCTION_PTS_JSON is not None:
        picks = load_suction_pts(cam_config)
        print(f"Loaded {len(picks)} suction points.")
        if picks:
            save_suction_pts(picks)
    elif box_pcd is not None:
        box_pts = load_box_pcd(box_pcd)
        if box_pts is not None:
            normal_cam, _, inlier_pcd = fit_plane(box_pts)
            _, long_axis, center, info = get_box_axes(inlier_pcd)
            picks = [PickPoint(p_cam=center, n_cam=normal_cam, long_axis_cam=long_axis)]
            print(f"Box: center={center}, extent={info['extent_sorted']}, aspect={info['aspect_ratio']:.2f}")



    vis = TransformVisualizer("pallet_sample1", spawn=True)

    # ── world = base (robot) coordinate system, all in mm ──
    raw_mat = np.array(
        yaml.safe_load(open(CALIBRATION_YML, encoding="utf-8"))
        .get("config", {})["camera_calibration"],
        dtype=np.float64,
    )
    T_cam2base_mm = RigidTransform.from_matrix(raw_mat, Frame.CAMERA, Frame.BASE)

    vis.log_transform(
        "world/base",
        RigidTransform.identity(Frame.BASE),
        axis_length=300.0,
        label="BASE",
    )

    T_base2cam_mm = T_cam2base_mm.inv
    cam_pose = RigidTransform.from_Rt(
        T_cam2base_mm.R, T_base2cam_mm.t,
        Frame.BASE, Frame.CAMERA,
    )
    vis.log_transform(
        "world/camera",
        cam_pose,
        axis_length=200.0,
        label="CAMERA",
    )

    ply_data = load_ply_points(PCD_PLY)
    if ply_data is not None:
        pts_cam, colors_cam = ply_data
        if np.median(np.abs(pts_cam)) < 10:
            pts_cam = pts_cam * 1000.0
        pts_base = T_cam2base_mm.transform_points(pts_cam)
        vis.log_points(
            "world/pcd",
            pts_base,
            colors=colors_cam,
            radii=3.0,
        )
        print(f"Logged {len(pts_base)} points from PLY (colors={'yes' if colors_cam is not None else 'no'}).")

    tcp_poses = []
    for i, pick in enumerate(picks):
        p_cam_mm = pick.p_cam * 1000.0 if np.median(np.abs(pick.p_cam)) < 10 else pick.p_cam
        p_base = T_cam2base_mm.transform_point(p_cam_mm)

        n_cam = pick.n_cam if pick.n_cam is not None else np.array([0.0, 0.0, -1.0])
        n_base = T_cam2base_mm.transform_direction(n_cam)
        n_base = n_base / (np.linalg.norm(n_base) + 1e-12)

        long_hint = None
        if pick.long_axis_cam is not None:
            long_hint = T_cam2base_mm.transform_direction(pick.long_axis_cam)
            long_hint = long_hint / (np.linalg.norm(long_hint) + 1e-12)

        tcp_pose = build_tcp_pose(p_base, n_base, long_axis_hint=long_hint)
        tcp_poses.append(tcp_pose)
        print(f"Pick #{i}: TCP ({tcp_pose.t[0]:.1f}, {tcp_pose.t[1]:.1f}, {tcp_pose.t[2]:.1f}) mm")

    if tcp_poses:
        vis.log_tcp_poses(tcp_poses, parent_path="world/picks", axis_length=80.0)

    print("\nRerun viewer에서 확인하세요.")


if __name__ == "__main__":
    main()
