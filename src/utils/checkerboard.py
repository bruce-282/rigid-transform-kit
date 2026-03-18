"""
Checkerboard corner detection and pose estimation from RGB image.

Requires: opencv-python (pip install opencv-python).
Input: RGB image only (later: RGB + PLY optional).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None  # type: ignore[assignment]

from rigid_transform_kit import Frame, RigidTransform


def _require_cv2() -> None:
    if cv2 is None:
        raise ImportError("opencv-python required for checkerboard utils: pip install opencv-python")


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
        gray = image

    # Try default first, then with flags that help under uneven lighting
    flags = 0
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, flags)
    if not ret or corners is None:
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, flags)
    if not ret or corners is None:
        return None, False

    if refine:
        win = (5, 5)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
        corners = cv2.cornerSubPix(gray, corners, win, (-1, -1), criteria)

    return corners, True


def get_pose_from_corners(
    corners: np.ndarray,
    pattern_size: Tuple[int, int],
    square_size_mm: float,
    K: np.ndarray,
    dist: np.ndarray,
) -> RigidTransform:
    """Compute T_cam2board (camera to checkerboard frame) from detected corners.

    Board frame: origin at first corner, X along first row, Y along first column,
    Z pointing out of the board (right-handed).

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
    dist : np.ndarray (5,) or (1, 5)
        Distortion coefficients (k1, k2, p1, p2, k3).

    Returns
    -------
    T_cam2board : RigidTransform (CAMERA -> MARKER)
        Pose of the checkerboard in camera frame. Translation in mm.
    """
    _require_cv2()
    # Object points: 3D coordinates of inner corners in board frame (mm)
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), dtype=np.float64)
    objp[:, :2] = np.mgrid[0 : pattern_size[0], 0 : pattern_size[1]].T.reshape(-1, 2)
    objp *= square_size_mm

    pts = np.asarray(corners, dtype=np.float64)
    if pts.ndim == 3:
        pts = pts.reshape(-1, 2)
    dist = np.asarray(dist, dtype=np.float64).ravel()[:5]

    ret, rvec, tvec = cv2.solvePnP(objp, pts, K, dist)
    if not ret:
        raise RuntimeError("solvePnP failed")

    R, _ = cv2.Rodrigues(rvec)
    t = tvec.ravel()
    return RigidTransform.from_Rt(R, t, Frame.CAMERA, Frame.MARKER)


def detect_checkerboard_pose(
    image: np.ndarray,
    pattern_size: Tuple[int, int],
    square_size_mm: float,
    K: np.ndarray,
    dist: np.ndarray,
    *,
    refine_corners: bool = True,
) -> Tuple[RigidTransform | None, np.ndarray | None]:
    """Detect checkerboard in image and return T_cam2board (RGB-only pipeline).

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
    dist : np.ndarray (5,)
        Distortion.
    refine_corners : bool
        Subpixel refinement.

    Returns
    -------
    T_cam2board : RigidTransform or None
        Pose if detected, else None.
    corners : np.ndarray or None
        Detected corners (N, 1, 2) for drawing; None if not found.
    """
    corners, ret = detect_corners(image, pattern_size, refine=refine_corners)
    if not ret or corners is None:
        return None, None
    T = get_pose_from_corners(corners, pattern_size, square_size_mm, K, dist)
    return T, corners
