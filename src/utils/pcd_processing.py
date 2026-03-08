import open3d as o3d
import numpy as np
from typing import Tuple, Dict


def fit_plane(
    pcd: o3d.geometry.PointCloud,
    distance_threshold: float = 5.0,
    ransac_n: int = 3,
    num_iterations: int = 1000,
) -> Tuple[np.ndarray, np.ndarray, o3d.geometry.PointCloud]:
    """RANSAC 평면 피팅.

    Returns
    -------
    normal_cam : (3,) 법선벡터 (카메라 좌표계, 카메라를 향하도록 보정됨)
    target_cam : (3,) inlier centroid (카메라 좌표계)
    inlier_pcd : PointCloud  inlier 포인트 클라우드 (OBB 등 후속 처리용)
    """
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations,
    )
    a, b, c, d = plane_model

    normal_cam = np.array([a, b, c], dtype=np.float64)
    normal_cam /= np.linalg.norm(normal_cam)

    inlier_pcd = pcd.select_by_index(inliers)
    target_cam = np.asarray(inlier_pcd.points).mean(axis=0)

    if np.dot(normal_cam, -target_cam) < 0:
        normal_cam = -normal_cam

    return normal_cam, target_cam, inlier_pcd


def get_box_axes(
    inlier_pcd: o3d.geometry.PointCloud,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """OBB로 장축/단축/법선 방향 및 박스 중심 추출.

    Parameters
    ----------
    inlier_pcd : 평면 inlier PointCloud (fit_plane 반환값)

    Returns
    -------
    normal   : (3,) 두께 방향 (extent 최소축, ≈ 평면 법선)
    long_axis : (3,) 장축 방향
    center   : (3,) OBB 중심 (카메라 좌표계)
    info     : dict  extent, short_axis, aspect_ratio, is_square 등
    """
    obb = inlier_pcd.get_oriented_bounding_box()
    R_obb = np.asarray(obb.R)
    extent = np.asarray(obb.extent)

    axes_order = np.argsort(extent)  # [두께, 단축, 장축]

    normal = R_obb[:, axes_order[0]]
    short_axis = R_obb[:, axes_order[1]]
    long_axis = R_obb[:, axes_order[2]]
    center = np.asarray(obb.center)

    info = {
        "extent": extent,
        "extent_sorted": extent[axes_order],
        "normal": normal,
        "short_axis": short_axis,
        "long_axis": long_axis,
        "aspect_ratio": extent[axes_order[2]] / max(extent[axes_order[1]], 1e-9),
        "is_square": extent[axes_order[2]] / max(extent[axes_order[1]], 1e-9) < 1.2,
    }

    return normal, long_axis, center, info
