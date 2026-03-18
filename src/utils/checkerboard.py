"""
Checkerboard corner detection and pose estimation from RGB image.
Optional: RGB + point cloud (RGB-depth) for 3D corner lookup and 3D-3D pose.

Requires: opencv-python, scipy (pip install opencv-python scipy).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.spatial import cKDTree

try:
    import cv2
except ImportError:
    cv2 = None  # type: ignore[assignment]

from rigid_transform_kit import Frame, PickPoint, RigidTransform


def _require_cv2() -> None:
    if cv2 is None:
        raise ImportError("opencv-python required for checkerboard utils: pip install opencv-python")


def _flatten_pixels(pixel_2d: np.ndarray) -> Tuple[np.ndarray, Tuple[int, ...]]:
    """Flatten pixel array to (M, 2), return (flat_uv, leading_shape) for reshape."""
    p = np.asarray(pixel_2d, dtype=np.float64)
    leading_shape = p.shape[:-1]
    flat_uv = p.reshape(-1, 2)
    return flat_uv, leading_shape


def undistort_point_cloud(
    points_cam: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
) -> np.ndarray:
    """Convert point cloud from distorted to undistorted camera space.

    Points from depth + distorted image back-projection are reprojected so that
    pinhole projection (no dist) yields undistorted (u,v). Same unit as input (e.g. meters).

    Parameters
    ----------
    points_cam : (N, 3)
        Points in camera frame (e.g. meters); Z > 0 only are processed.
    K : (3, 3)
        Camera matrix.
    dist : (5,) or None
        Distortion coefficients; if None or all zero, returns copy.

    Returns
    -------
    (N, 3)
        Points in undistorted camera space; invalid Z filled with NaN.
    """
    _require_cv2()
    P = np.asarray(points_cam, dtype=np.float64)
    K_np = np.asarray(K, dtype=np.float64)
    d = (
        np.asarray(dist, dtype=np.float64).ravel()[:5]
        if dist is not None
        else np.zeros(5, dtype=np.float64)
    )
    out = np.full_like(P, np.nan)
    valid = P[:, 2] > 0
    if not np.any(valid):
        return out
    Pv = P[valid]
    rvec = np.zeros(3, dtype=np.float64)
    tvec = np.zeros(3, dtype=np.float64)
    if np.all(np.abs(d) < 1e-10):
        out[valid] = Pv
        return out
    proj_d, _ = cv2.projectPoints(
        Pv.reshape(-1, 1, 3), rvec, tvec, K_np, d
    )
    pts_norm = cv2.undistortPoints(proj_d, K_np, d, P=None)
    pts_norm = pts_norm.reshape(-1, 2)
    z = Pv[:, 2]
    x_n, y_n = pts_norm[:, 0], pts_norm[:, 1]
    out[valid] = np.column_stack([x_n * z, y_n * z, z])
    return out


def find_3d_points_from_2d(
    intrinsic_matrix: np.ndarray,
    points_cam: np.ndarray,
    pixel_2d: np.ndarray,
    *,
    dist_coeffs: np.ndarray | None = None,
    k: int = 12,
    method: str = "plane",
    depth_gate: float | None = 10.0,
    idw_power: float = 2.0,
    gaussian_sigma_px: float | None = 2.0,
    eps: float = 1e-9,
) -> np.ndarray:
    """Look up 3D positions (camera frame) for 2D pixels using a point cloud.

    *points_cam* is (N, 3) in camera frame (meters). Returns (M, 3) in same unit (meters).
    Use undistorted image + undistorted PCD so pixel_2d and projection share pinhole (no dist).

    Parameters
    ----------
    intrinsic_matrix : (3, 3)
        Camera matrix (fx, fy, cx, cy).
    points_cam : (N, 3)
        Point cloud in camera frame (meters); only points with Z > 0 are used.
    pixel_2d : (M, 2) or (M, 1, 2)
        Query pixel coordinates (u, v); same space as projection (undistorted if dist_coeffs not set).
    k : int
        Number of nearest neighbors per pixel.
    method : "plane" or "idw"
        "plane": fit plane to neighbors, ray-plane intersection; fallback to IDW.
        "idw": inverse-distance weighting.
    depth_gate : float or None
        If set, drop neighbors whose Z differs from median Z by more than this (mm).
        None to disable.
    idw_power, gaussian_sigma_px, eps
        Weights and numerical stability.

    Returns
    -------
    out : (*leading_shape, 3)
        3D positions in camera frame (meters). NaN where lookup failed.
    """
    flat_uv, leading_shape = _flatten_pixels(pixel_2d)
    K = np.asarray(intrinsic_matrix, dtype=np.float64)
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    P = np.asarray(points_cam, dtype=np.float64)
    valid = P[:, 2] > 0
    P = P[valid]

    if P.size == 0:
        out = np.full((*leading_shape, 3), np.nan, dtype=np.float64)
        return out

    if dist_coeffs is not None and np.any(np.abs(np.asarray(dist_coeffs).ravel()[:5]) > 1e-10):
        _require_cv2()
        rvec = np.zeros(3, dtype=np.float64)
        tvec = np.zeros(3, dtype=np.float64)
        d = np.asarray(dist_coeffs, dtype=np.float64).ravel()[:5]
        proj_2d, _ = cv2.projectPoints(
            P.reshape(-1, 1, 3), rvec, tvec, K, d
        )
        proj = proj_2d.reshape(-1, 2)
    else:
        z = P[:, 2]
        u = fx * P[:, 0] / z + cx
        v = fy * P[:, 1] / z + cy
        proj = np.column_stack((u, v))

    tree = cKDTree(proj)
    k_eff = min(k, proj.shape[0])
    dists, idxs = tree.query(flat_uv, k=k_eff)
    if k_eff == 1:
        dists = np.reshape(dists, (-1, 1))
        idxs = np.reshape(idxs, (-1, 1))

    out = np.full((flat_uv.shape[0], 3), np.nan, dtype=np.float64)

    for i in range(flat_uv.shape[0]):
        uv = flat_uv[i]
        di = dists[i]
        ii = idxs[i].astype(int)

        neigh_P = P[ii].astype(np.float64)
        neigh_d = np.asarray(di, dtype=np.float64)

        if depth_gate is not None and neigh_P.shape[0] >= 3:
            z_med = np.median(neigh_P[:, 2])
            keep = np.abs(neigh_P[:, 2] - z_med) <= (depth_gate * 1e-3)
            neigh_P = neigh_P[keep]
            neigh_d = neigh_d[keep]

        if neigh_P.shape[0] < 3:
            w = 1.0 / (neigh_d + eps) ** idw_power
            w = w / (np.sum(w) + eps)
            out[i] = np.sum(w[:, None] * neigh_P, axis=0)
            continue

        if method == "plane":
            C = np.mean(neigh_P, axis=0)
            X = neigh_P - C
            _, _, Vt = np.linalg.svd(X, full_matrices=False)
            n = Vt[-1]
            nn = np.linalg.norm(n)
            if nn < 1e-12:
                method_local = "idw"
            else:
                n = n / nn
                d0 = -np.dot(n, C)
                ray = np.array(
                    [(uv[0] - cx) / fx, (uv[1] - cy) / fy, 1.0],
                    dtype=np.float64,
                )
                denom = np.dot(n, ray)
                if abs(denom) < 1e-9:
                    method_local = "idw"
                else:
                    t = -d0 / denom
                    if t > 0:
                        out[i] = t * ray
                        continue
                    method_local = "idw"

            if method_local == "idw":
                if gaussian_sigma_px is not None:
                    w = np.exp(
                        -(neigh_d**2) / (2.0 * (gaussian_sigma_px**2) + eps)
                    )
                else:
                    w = 1.0 / (neigh_d + eps) ** idw_power
                w = w / (np.sum(w) + eps)
                out[i] = np.sum(w[:, None] * neigh_P, axis=0)
        else:
            if gaussian_sigma_px is not None:
                w = np.exp(
                    -(neigh_d**2) / (2.0 * (gaussian_sigma_px**2) + eps)
                )
            else:
                w = 1.0 / (neigh_d + eps) ** idw_power
            w = w / (np.sum(w) + eps)
            out[i] = np.sum(w[:, None] * neigh_P, axis=0)

    return out.reshape(*leading_shape, 3)


def marker_3d_pose(
    marker_3d_cam: np.ndarray,
    object_points_board: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Rigid pose from 3D-3D correspondence (Kabsch). T_cam2board: p_cam = R @ p_board + t.

    Parameters
    ----------
    marker_3d_cam : (N, 3)
        Corner 3D positions in camera frame (mm).
    object_points_board : (N, 3)
        Same corners in board frame (mm).

    Returns
    -------
    R : (3, 3), t : (3,)
        Rotation and translation (mm); RigidTransform.from_Rt(R, t, CAMERA, MARKER).
    """
    P = np.asarray(marker_3d_cam, dtype=np.float64).reshape(-1, 3)
    Q = np.asarray(object_points_board, dtype=np.float64).reshape(-1, 3)
    if P.shape[0] != Q.shape[0] or P.shape[0] < 3:
        raise ValueError("marker_3d_pose requires at least 3 corresponding point pairs")

    p_mean = np.mean(P, axis=0)
    q_mean = np.mean(Q, axis=0)
    P_c = P - p_mean
    Q_c = Q - q_mean
    H = Q_c.T @ P_c
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    t = p_mean - R @ q_mean
    return R, t


def build_object_points(
    pattern_size: Tuple[int, int],
    square_size_mm: float,
    origin: str = "center",
) -> np.ndarray:
    """Board-frame 3D positions of inner corners (mm), same order as findChessboardCorners).

    Parameters
    ----------
    origin : {"center", "LT", "RB"}
        Where the board frame origin is placed.
        - "center": 가장 중심 코너 (기본값)
        - "LT": 좌상단(0,0) 코너
        - "RB": 우하단(cols-1, rows-1) 코너
    """
    n = pattern_size[0] * pattern_size[1]
    objp = np.zeros((n, 3), dtype=np.float64)
    grid = np.mgrid[0 : pattern_size[0], 0 : pattern_size[1]].T.reshape(-1, 2)
    objp[:, :2] = grid * square_size_mm

    if origin.lower() == "center":
        # Shift so that the most central corner becomes the origin.
        center_ix = pattern_size[0] // 2
        center_iy = pattern_size[1] // 2
        center = np.array([center_ix * square_size_mm, center_iy * square_size_mm, 0.0])
        objp -= center
    elif origin.upper() == "RB":
        rb = np.array(
            [(pattern_size[0] - 1) * square_size_mm, (pattern_size[1] - 1) * square_size_mm, 0.0]
        )
        objp -= rb
    # origin == "LT" → no shift (0,0) corner already at origin
    return objp


def _score_corners(gray: np.ndarray, corners: np.ndarray, refined: np.ndarray) -> float:
    """Higher = better. Prefer corners that needed less subpixel refinement."""
    diff = np.linalg.norm(refined.reshape(-1, 2) - corners.reshape(-1, 2), axis=1)
    mean_refinement = float(np.mean(diff))
    return 1.0 / (1.0 + mean_refinement)


def detect_corners(
    image: np.ndarray,
    pattern_size: Tuple[int, int],
    *,
    refine: bool = True,
) -> Tuple[np.ndarray | None, bool]:
    """Detect checkerboard inner corners in an image.

    Parameters
    ----------
    image : np.ndarray (H, W) or (H, W, 3)
        RGB or BGR image; will be converted to grayscale.
    pattern_size : (cols, rows)
        Number of inner corners (e.g. (9, 6) for 9x6 inner corners).
    refine : bool
        If True, refine corner positions with subpixel accuracy.

    Returns
    -------
    corners : np.ndarray (N, 1, 2) or None
        Detected corners in image coordinates. None if not found.
    ret : bool
        True if checkerboard was found.
    """
    _require_cv2()
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = np.asarray(image, dtype=np.uint8)

    flag_combinations = [
        cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK,
        cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE,
        cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FILTER_QUADS,
        cv2.CALIB_CB_ADAPTIVE_THRESH
        | cv2.CALIB_CB_NORMALIZE_IMAGE
        | cv2.CALIB_CB_FILTER_QUADS
        | cv2.CALIB_CB_FAST_CHECK,
        cv2.CALIB_CB_ADAPTIVE_THRESH,
        0,
    ]

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    best_corners = None
    best_score = -1.0

    for flags in flag_combinations:
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, flags)
        if not ret or corners is None:
            continue
        if refine:
            corners_refined = cv2.cornerSubPix(
                gray, corners.copy(), (11, 11), (-1, -1), criteria=criteria
            )
        else:
            corners_refined = corners
        score = _score_corners(gray, corners, corners_refined)
        if score > best_score:
            best_score = score
            best_corners = corners_refined

    if best_corners is None:
        return None, False
    return best_corners, True


def get_pose_from_corners(
    corners: np.ndarray,
    pattern_size: Tuple[int, int],
    square_size_mm: float,
    K: np.ndarray,
    dist: np.ndarray,
    *,
    origin: str = "center",
) -> RigidTransform:
    """Compute T_cam2board (camera to checkerboard frame) from detected corners.

    Board frame: origin at first corner, X along first row, Y along first column,
    Z pointing out of the board (right-handed). Uses SOLVEPNP_ITERATIVE then
    solvePnPRefineLM for a tighter pose.

    Parameters
    ----------
    corners : np.ndarray (N, 1, 2) or (N, 2)
        Image coordinates of inner corners (same order as object_points).
    pattern_size : (cols, rows)
        Number of inner corners.
    square_size_mm : float
        Side length of one square in mm.
    K : np.ndarray (3, 3)
        Camera intrinsic matrix.
    dist : np.ndarray (5,) or (1, 5) or None
        Distortion coefficients (k1, k2, p1, p2, k3). None → zeros(5).

    Returns
    -------
    T_cam2board : RigidTransform (CAMERA -> MARKER)
        Pose of the checkerboard in camera frame. Translation in mm.
    """
    _require_cv2()
    corners_2d = np.asarray(corners, dtype=np.float64).reshape(-1, 1, 2)
    object_points = build_object_points(pattern_size, square_size_mm, origin=origin).reshape(
        -1, 1, 3
    )

    camera_matrix = np.asarray(K, dtype=np.float64)
    if dist is None:
        dist_coeffs = np.zeros(5, dtype=np.float64)
    else:
        dist_coeffs = np.asarray(dist, dtype=np.float64).ravel()[:5]

    success, rvec, tvec = cv2.solvePnP(
        object_points,
        corners_2d,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        raise RuntimeError("solvePnP failed")

    rvec, tvec = cv2.solvePnPRefineLM(
        object_points,
        corners_2d,
        camera_matrix,
        dist_coeffs,
        rvec,
        tvec,
    )

    R, _ = cv2.Rodrigues(rvec)
    t = tvec.flatten()
    return RigidTransform.from_Rt(R, t, Frame.CAMERA, Frame.MARKER)


def detect_checkerboard_pose(
    image: np.ndarray,
    pattern_size: Tuple[int, int],
    square_size_mm: float,
    K: np.ndarray,
    dist: np.ndarray,
    *,
    refine_corners: bool = True,
    origin: str = "center",
    points_cam_m: np.ndarray | None = None,
) -> Tuple[RigidTransform | None, np.ndarray | None]:
    """Detect checkerboard and return T_cam2board. RGB-only or RGB+depth (point cloud).

    If *points_cam_m* is given (N, 3) in camera frame (meters), uses 3D lookup from
    the point cloud and 3D-3D pose (Kabsch) for higher accuracy.

    Parameters
    ----------
    image : np.ndarray (H, W, 3)
        RGB image.
    pattern_size : (cols, rows)
        Inner corners, e.g. (9, 6).
    square_size_mm : float
        Square size in mm.
    K : np.ndarray (3, 3)
        Camera intrinsics.
    dist : np.ndarray (5,) or None
        Distortion (ignored when points_cam_m is used).
    refine_corners : bool
        Subpixel refinement.
    origin : {"center", "LT", "RB"}
        보드 원점 위치. "center"(기본), "LT"(좌상단), "RB"(우하단).
    points_cam_m : np.ndarray (N, 3) or None
        Optional point cloud in camera frame (meters). If provided, pose is from
        find_3d_points_from_2d + marker_3d_pose (RGB-depth path).

    Returns
    -------
    T_cam2board : RigidTransform or None
        Pose if detected, else None.
    corners : np.ndarray or None
        Detected corners (N, 1, 2) for drawing; None if not found.
    """
    img_for_detection = np.asarray(image)
    points_for_lookup = points_cam_m

    if points_cam_m is not None and points_cam_m.size >= 3:
        dist_arr = (
            np.asarray(dist, dtype=np.float64).ravel()[:5]
            if dist is not None
            else np.zeros(5, dtype=np.float64)
        )
        if np.any(np.abs(dist_arr) > 1e-10):
            img_for_detection = cv2.undistort(img_for_detection, np.asarray(K, dtype=np.float64), dist_arr)
            points_for_lookup = undistort_point_cloud(points_cam_m, K, dist_arr)

    corners, ret = detect_corners(img_for_detection, pattern_size, refine=refine_corners)
    if not ret or corners is None:
        return None, None

    if points_cam_m is not None and points_cam_m.size >= 3:
        marker_3d_m = find_3d_points_from_2d(
            K,
            points_for_lookup,
            corners,
            dist_coeffs=None,
            k=12,
            method="plane",
            depth_gate=10.0,
        )
        marker_3d_m = np.asarray(marker_3d_m, dtype=np.float64).reshape(-1, 3)
        marker_3d_mm = marker_3d_m * 1000.0
        valid = np.asarray(~np.any(np.isnan(marker_3d_mm), axis=1)).reshape(-1)
        if np.sum(valid) < 3:
            T = get_pose_from_corners(
                corners, pattern_size, square_size_mm, K, None, origin=origin
            )
            return T, corners
        objp = build_object_points(pattern_size, square_size_mm, origin=origin)
        R, t = marker_3d_pose(marker_3d_mm[valid], objp[valid])
        T = RigidTransform.from_Rt(R, t, Frame.CAMERA, Frame.MARKER)
        return T, corners

    T = get_pose_from_corners(corners, pattern_size, square_size_mm, K, dist, origin=origin)
    return T, corners


def checkerboard_to_pick_point(T_cam2board: RigidTransform) -> PickPoint:
    """Convert T_cam2board to PickPoint for use with picks_to_tcp_poses_base_and_cam.

    Extracts p_cam (origin), n_cam (Z = surface normal), long_axis_cam (X).
    """
    p_cam = np.asarray(T_cam2board.t, dtype=np.float64)
    n_cam = np.asarray(-T_cam2board.R[:, 2], dtype=np.float64)  # 180도: 보드 Z → 접근 방향
    nrm = np.linalg.norm(n_cam)
    if nrm > 1e-12:
        n_cam = n_cam / nrm
    long_axis_cam = np.asarray(T_cam2board.R[:, 0], dtype=np.float64)
    nrm = np.linalg.norm(long_axis_cam)
    if nrm > 1e-12:
        long_axis_cam = long_axis_cam / nrm
    return PickPoint(p_cam=p_cam, n_cam=n_cam, long_axis_cam=long_axis_cam)
