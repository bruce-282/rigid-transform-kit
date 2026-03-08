import open3d as o3d
import numpy as np
from typing import Tuple, Dict


def remove_statistical_outlier(
    pcd: o3d.geometry.PointCloud,
    nb_neighbors: int = 20,
    std_ratio: float = 3.0,
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
    distance_threshold: float = 4.0,
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
    plane_normal: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """평면 투영 + 2D PCA로 장축/단축 방향 및 박스 중심 추출.

    3D OBB는 평면 inlier처럼 거의 2D인 점군에서 장축 방향이
    불안정할 수 있으므로, 평면 법선 방향을 제거한 뒤 2D PCA로
    장/단축을 구한다.

    Parameters
    ----------
    inlier_pcd : 평면 inlier PointCloud (fit_plane 반환값)
    plane_normal : (3,) 평면 법선. None이면 내부에서 추정한다.

    Returns
    -------
    normal    : (3,) 평면 법선 (카메라를 향하는 방향)
    long_axis : (3,) 장축 방향 (3D, 단위벡터)
    center    : (3,) inlier 중심 (카메라 좌표계)
    info      : dict  extent_long, extent_short, short_axis, aspect_ratio, is_square
    """
    pts = np.asarray(inlier_pcd.points)
    center = pts.mean(axis=0)

    if plane_normal is None:
        cov3 = np.cov(pts, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov3)
        plane_normal = eigvecs[:, 0]
    normal = plane_normal / (np.linalg.norm(plane_normal) + 1e-12)
    if np.dot(normal, -center) < 0:
        normal = -normal

    centered = pts - center
    proj_along_n = np.outer(centered @ normal, normal)
    pts_2d = centered - proj_along_n

    cov = np.cov(pts_2d, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)

    order = np.argsort(eigvals)[::-1]  # [장축, 단축, (≈0 법선)]
    long_axis = eigvecs[:, order[0]]
    short_axis = eigvecs[:, order[1]]

    long_axis = long_axis / (np.linalg.norm(long_axis) + 1e-12)
    short_axis = short_axis / (np.linalg.norm(short_axis) + 1e-12)

    proj_long = centered @ long_axis
    proj_short = centered @ short_axis
    extent_long = proj_long.max() - proj_long.min()
    extent_short = proj_short.max() - proj_short.min()

    info = {
        "extent_long": extent_long,
        "extent_short": extent_short,
        "normal": normal,
        "short_axis": short_axis,
        "long_axis": long_axis,
        "aspect_ratio": extent_long / max(extent_short, 1e-9),
        "is_square": extent_long / max(extent_short, 1e-9) < 1.2,
    }

    return normal, long_axis, center, info
