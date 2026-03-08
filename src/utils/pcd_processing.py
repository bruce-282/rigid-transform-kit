import open3d as o3d
import numpy as np
from typing import Tuple, Dict


def remove_statistical_outlier(
    pcd: o3d.geometry.PointCloud,
    nb_neighbors: int = 20,
    std_ratio: float = 2.0,
) -> Tuple[o3d.geometry.PointCloud, np.ndarray]:
    """통계적 이상치 제거: 이웃과의 평균 거리가 표준편차 기준으로 먼 점 제거.

    Parameters
    ----------
    pcd : PointCloud
    nb_neighbors : 각 점에서 참조할 이웃 개수
    std_ratio : 평균 거리 + std_ratio * std 보다 먼 점을 outlier로 제거

    Returns
    -------
    filtered_pcd : PointCloud  inlier만 남긴 포인트 클라우드
    inlier_indices : np.ndarray  유지된 점의 인덱스
    """
    filtered, ind = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors,
        std_ratio=std_ratio,
    )
    return filtered, np.asarray(ind)


def remove_radius_outlier(
    pcd: o3d.geometry.PointCloud,
    nb_points: int = 16,
    radius: float = 1.0,
) -> Tuple[o3d.geometry.PointCloud, np.ndarray]:
    """반경 이상치 제거: 반경 내 이웃이 nb_points 미만인 점 제거.

    Parameters
    ----------
    pcd : PointCloud
    nb_points : 반경 내 최소 이웃 개수
    radius : 이웃 탐색 반경 (점 좌표 단위)

    Returns
    -------
    filtered_pcd : PointCloud
    inlier_indices : np.ndarray
    """
    filtered, ind = pcd.remove_radius_outlier(
        nb_points=nb_points,
        radius=radius,
    )
    return filtered, np.asarray(ind)


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
