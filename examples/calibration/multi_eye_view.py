"""
examples / calibration / multi_eye_view.py
==========================================
모든 기하를 **cam1 좌표계**로 통일하여 Rerun 으로 시각화합니다.

* cam1 PLY — 그대로 표시.
* cam2 PLY — extrinsic (cam1_to_cam2) 역행렬로 cam1 프레임으로 변환.
* 로봇 베이스 — base_to_cam (p_cam1 = M @ p_base) 으로 배치.
* TCP — PLY 파일명 vec6 파싱 (선택).
* RGB Pinhole — 각 카메라 PNG + 인트린식 JSON (선택).

Usage::

  uv run python examples/calibration/multi_eye_view.py
  uv run python examples/calibration/multi_eye_view.py \\
    --cam1-ply path/to/cam1.ply --cam2-ply path/to/cam2.ply \\
    --extrinsic path/to/multi_eye_cal.yml
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from rigid_transform_kit import Frame, RigidTransform
from rigid_transform_kit.viz import TransformVisualizer

from utils import clip_depth_range, load_intrinsics_any, load_ply_points

from _viz_common import (
    finalize_viewer,
    load_4x4_matrices,
    log_camera_frustum,
    log_debug_link,
    log_origin_spheres,
    parse_tcp_vec6_from_filename,
    subsample,
)

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_PLY_CAM1 = REPO_ROOT / "datasets/multi_eye_example/cam1/cam1.ply"
_DEFAULT_PLY_CAM2 = REPO_ROOT / "datasets/multi_eye_example/cam2/cam2.ply"


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------
_AXIS_COLORS = [[255, 0, 0], [0, 255, 0], [0, 0, 255]]
def _load_4x4_from_yaml(path: Path) -> np.ndarray:
    """Load 4x4 ``cam1_to_cam2`` (or ``cam2_pose_matrix``, ``T_cam1_cam2``) from YAML."""
    if not path.exists():
        raise FileNotFoundError(path)

    try:
        import yaml
    except ImportError as e:
        raise ImportError("PyYAML required: pip install pyyaml") from e

    with open(path, encoding="utf-8") as f:
        doc = yaml.safe_load(f)

    if isinstance(doc, dict):
        for key in ("cam1_to_cam2", "cam2_pose_matrix", "T_cam1_cam2", "T_cam1_to_cam2"):
            if key not in doc:
                continue
            obj = doc[key]
            if isinstance(obj, dict) and "data" in obj:
                rows = int(obj["rows"])
                cols = int(obj["cols"])
                flat = np.asarray(obj["data"], dtype=np.float64).ravel()
                if flat.size != rows * cols:
                    raise ValueError(f"{key}: data length {flat.size} != rows*cols {rows*cols}")
                return flat.reshape(rows, cols)
            arr = np.asarray(obj, dtype=np.float64)
            if arr.shape == (4, 4):
                return arr

    try:
        import cv2

        fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
        try:
            for key in ("cam1_to_cam2", "cam2_pose_matrix", "T_cam1_cam2", "T_cam1_to_cam2"):
                node = fs.getNode(key)
                if not node.empty():
                    m = node.mat()
                    if m is not None and m.shape == (4, 4):
                        return np.asarray(m, dtype=np.float64)
        finally:
            fs.release()
    except ImportError:
        pass
    except Exception:
        pass

    raise KeyError(
        "YAML must contain cam1_to_cam2, cam2_pose_matrix, or T_cam1_cam2 "
        "(4x4 nested list, opencv-matrix dict, or OpenCV FileStorage YAML)"
    )


def _try_load_base_to_cam1(path: Path) -> np.ndarray | None:
    """Return 4x4 ``base_to_cam1`` (p_cam1 = M @ p_base) from *path* if present, else None."""
    direct = load_4x4_matrices(path, ("base_to_cam1", "base_to_cam"))
    for key in ("base_to_cam1", "base_to_cam"):
        if key in direct:
            return direct[key]
    inverse = load_4x4_matrices(path, ("cam_to_base", "cam1_to_base"))
    for key in ("cam_to_base", "cam1_to_base"):
        if key in inverse:
            return np.linalg.inv(inverse[key])
    return None


def _load_rgb_hwc(rgb_path: Path) -> np.ndarray | None:
    """Load PNG as RGB uint8 (H,W,3), or None."""
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


def _log_rgb_pinhole(
    entity_path: str,
    rgb_path: Path,
    intrinsic_json: Path,
    *,
    image_plane_mm: float,
    rgb_hwc: np.ndarray | None = None,
) -> bool:
    """Log ``ViewCoordinates`` + ``Pinhole`` + ``Image`` at *entity_path*. Returns False if skipped.

    *rgb_hwc* — optional preloaded RGB ``uint8`` (H,W,3); avoids re-reading when logging the same
    image to multiple entities (e.g. 3D under ``Transform3D`` + flat path for ``Spatial2DView``).
    """
    if not rgb_path.exists():
        log.warning("RGB not found, skip: %s", rgb_path)
        return False
    if not intrinsic_json.exists():
        log.warning("Intrinsics JSON not found, skip RGB: %s", intrinsic_json)
        return False

    try:
        import cv2
    except ImportError as e:
        raise ImportError("opencv-python required for RGB: pip install opencv-python") from e

    import rerun as rr

    K, _ = load_intrinsics_any(intrinsic_json)
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    w, h = _read_image_resolution_json(intrinsic_json)

    if rgb_hwc is None:
        bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if bgr is None:
            log.warning("Failed to read image: %s", rgb_path)
            return False
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    else:
        rgb = rgb_hwc

    rr.log(entity_path, rr.ViewCoordinates.RDF, static=True)
    rr.log(
        entity_path,
        rr.Pinhole(
            focal_length=[fx, fy],
            principal_point=[cx, cy],
            width=w,
            height=h,
            image_plane_distance=float(image_plane_mm),
        ),
        static=True,
    )
    rr.log(entity_path, rr.Image(rgb), static=True)
    log.info("Logged RGB Pinhole at %s (%dx%d, plane=%.0f mm)", entity_path, w, h, image_plane_mm)
    return True


def _send_stereo_blueprint(*, rgb_tab_origins: list[tuple[str, str]]) -> None:
    """*rgb_tab_origins*: (tab_title, rerun entity origin) for each Spatial2DView."""
    import rerun as rr
    import rerun.blueprint as rrb

    tabs: list = [
        rrb.Spatial3DView(
            name="Stereo (camera 1)",
            origin="world/scene",
            contents=["+ world/scene/**"],
            line_grid=False,
        ),
    ]
    for title, origin in rgb_tab_origins:
        tabs.append(
            rrb.Spatial2DView(
                name=title,
                origin=origin,
                contents=[f"+ {origin}", f"+ {origin}/**"],
            )
        )
    rr.send_blueprint(rrb.Blueprint(rrb.Tabs(*tabs)))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Two-camera PLY + robot base in one Rerun scene; unified camera 1 frame (cam1 PLY, cam2→cam1, base→cam1).",
    )
    p.add_argument(
        "--cam1-ply",
        type=Path,
        default=_DEFAULT_PLY_CAM1,
        help="Primary camera PLY (cam1 / scene reference)",
    )
    p.add_argument(
        "--cam2-ply",
        type=Path,
        default=_DEFAULT_PLY_CAM2,
        help="Secondary PLY (transformed into camera 1 frame via extrinsic)",
    )
    p.add_argument(
        "--extrinsic",
        type=Path,
        default=REPO_ROOT / "datasets/multi_eye_example/multi_eye_cal.yml",
        help="YAML with cam1_to_cam2 (stereoCalibrate output, cam1→cam2). Internally inverted to cam2→cam1 for p_cam1 = T @ p_cam2.",
    )
    p.add_argument(
        "--cam1-rgb",
        type=Path,
        default=_DEFAULT_PLY_CAM1.with_suffix(".png"),
        help="Cam1 RGB image (PNG). Default: same stem as built-in primary PLY",
    )
    p.add_argument(
        "--cam2-rgb",
        type=Path,
        default=_DEFAULT_PLY_CAM2.with_suffix(".png"),
        help="Cam2 RGB image (PNG). Default: same stem as built-in secondary PLY",
    )
    p.add_argument(
        "--cam1-intrinsics",
        type=Path,
        default=_DEFAULT_PLY_CAM1.with_suffix(".json"),
        help="Cam1 intrinsics JSON (sensores.image). Default: same stem as built-in primary PLY",
    )
    p.add_argument(
        "--cam2-intrinsics",
        type=Path,
        default=_DEFAULT_PLY_CAM2.with_suffix(".json"),
        help="Cam2 intrinsics JSON. Default: same stem as built-in secondary PLY",
    )
    p.add_argument(
        "--image-plane-mm-3d",
        type=float,
        default=200.0,
        metavar="MM",
        help="Pinhole image plane distance (mm) for world/scene/*/rgb — smaller = less overlap on PLY. Default 200",
    )
    p.add_argument(
        "--image-plane-mm-2d",
        type=float,
        default=600.0,
        metavar="MM",
        help="Pinhole image_plane_distance (mm) for rerun_2d/* (2D tabs only; rarely affects 2D view). Default 600",
    )
    p.add_argument(
        "--no-rgb",
        action="store_true",
        help="Do not load PNG / Pinhole / 2D tabs",
    )
    p.add_argument(
        "--max-points",
        type=int,
        default=400_000,
        metavar="N",
        help="Random subsample per cloud if larger than N (0 = no limit)",
    )
    p.add_argument(
        "--depth-min-m",
        type=float,
        default=0.0,
        metavar="M",
        help="Depth clip lower bound in meters (camera axis, default Z). Default 0",
    )
    p.add_argument(
        "--depth-max-m",
        type=float,
        default=3.0,
        metavar="M",
        help="Depth clip upper bound in meters. Default 1",
    )
    p.add_argument(
        "--depth-axis",
        type=int,
        choices=(0, 1, 2),
        default=2,
        help="Which point column is depth for clipping (0=x,1=y,2=z). Default 2",
    )
    p.add_argument(
        "--no-depth-clip",
        action="store_true",
        help="Disable depth clipping (otherwise depth-min/max apply in meters)",
    )
    p.add_argument(
        "--port",
        type=int,
        default=None,
        help="Rerun gRPC port if needed",
    )
    p.add_argument(
        "--save",
        type=Path,
        default=None,
        metavar="RRD",
        help="Save recording to .rrd (disables live spawn; then opens: rerun FILE)",
    )
    p.add_argument(
        "--no-robot-base",
        action="store_true",
        help="Do not log robot base frame (even if base_to_cam1 is in extrinsic YAML)",
    )
    p.add_argument(
        "--invert-base-calibration",
        action="store_true",
        help="YAML 4x4가 cam→base (p_base=M p_cam)일 때: inv(M)으로 base→cam으로 바꿔 로그",
    )
    # -- TCP pose from PLY filename --
    p.add_argument(
        "--tcp-from-ply-name",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Parse cam1 PLY stem as [idx_]_x_y_z_W_P_R mm/deg and draw TCP (default: on).",
    )
    p.add_argument(
        "--tcp-pose-frame",
        choices=("base", "cam1"),
        default="base",
        help="vec6 frame: base (robot base → needs base_to_cam) or already in cam1 (default: base).",
    )
    p.add_argument(
        "--tcp-pose-axis-mm",
        type=float,
        default=120.0,
        metavar="MM",
        help="TCP axis arrow length in mm for tcp_from_filename (default 120).",
    )
    return p.parse_args()



def _maybe_log_tcp(
    vis: TransformVisualizer,
    args: argparse.Namespace,
    M_base_to_cam1: np.ndarray | None,
) -> None:
    """Parse TCP vec6 from PLY filename and log it in cam1 frame."""
    vec6 = parse_tcp_vec6_from_filename(args.cam1_ply)
    if vec6 is None:
        log.info("TCP: stem %r not parseable — skip.", args.cam1_ply.stem)
        return

    log.info("TCP vec6: %s (mm, deg)", np.round(vec6, 3).tolist())

    if args.tcp_pose_frame == "cam1":
        T_cam_tcp = RigidTransform.from_vec6(
            vec6, Frame.CAMERA, Frame.TCP, convention="xyz", degrees=True,
        )
    else:
        T_base_tcp = RigidTransform.from_vec6(
            vec6, Frame.BASE, Frame.TCP, convention="xyz", degrees=True,
        )
        if M_base_to_cam1 is None:
            log.warning("TCP (base frame): base_to_cam not available — skip.")
            return
        T_cam_tcp = RigidTransform.from_matrix(
            M_base_to_cam1 @ T_base_tcp.matrix, Frame.CAMERA, Frame.TCP,
        )

    ax = args.tcp_pose_axis_mm
    vis.log_tcp_pose(
        T_cam_tcp, parent_path="world/scene", label="tcp",
        axis_length=ax, arrow_radius=max(2.0, ax * 0.04), show_axes=True,
    )
    log.info("Logged world/scene/tcp (frame=%s).", args.tcp_pose_frame)


def _log_cam_frame(
    entity: str,
    pts_mm: np.ndarray,
    colors: np.ndarray,
    label: str,
    origin_color: tuple[int, int, int],
    transform_args: tuple[list, list] | None = None,
) -> None:
    """Log axes + origin + pcd + frustum for one camera entity."""
    import rerun as rr

    if transform_args is not None:
        t_list, quat_xyzw = transform_args
        rr.log(entity, rr.Transform3D(
            translation=t_list, quaternion=rr.Quaternion(xyzw=quat_xyzw),
        ), static=True)

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
    rr.log(f"{entity}/pcd", rr.Points3D(pts_mm, colors=colors, radii=[1.2]), static=True)
    log_camera_frustum(f"{entity}/frustum", image_plane_distance=90.0)


def _log_cam_rgb(
    cam_name: str,
    rgb_path: Path,
    intrinsic_json: Path,
    plane_3d: float,
    plane_2d: float,
    transform_args: tuple[list, list] | None = None,
) -> list[tuple[str, str]]:
    """Log RGB Pinhole for one camera. Returns 2D tab entries for blueprint."""
    import rerun as rr

    tabs: list[tuple[str, str]] = []
    entity_3d = f"world/scene/{cam_name}_rgb"
    entity_2d = f"rerun_2d/{cam_name}"

    if transform_args is not None:
        t_list, quat_xyzw = transform_args
        rr.log(entity_3d, rr.Transform3D(
            translation=t_list, quaternion=rr.Quaternion(xyzw=quat_xyzw),
        ), static=True)

    rgb_hwc = _load_rgb_hwc(rgb_path)
    _log_rgb_pinhole(entity_3d, rgb_path, intrinsic_json,
                     image_plane_mm=plane_3d, rgb_hwc=rgb_hwc)
    if rgb_hwc is not None:
        if _log_rgb_pinhole(entity_2d, rgb_path, intrinsic_json,
                            image_plane_mm=plane_2d, rgb_hwc=rgb_hwc):
            tabs.append((f"{cam_name.upper()} RGB", entity_2d))
    return tabs


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()

    T_cam1_to_cam2 = _load_4x4_from_yaml(args.extrinsic)
    T_cam2_to_cam1 = np.linalg.inv(T_cam1_to_cam2)
    t = T_cam2_to_cam1[:3, 3]
    quat_xyzw = Rotation.from_matrix(T_cam2_to_cam1[:3, :3]).as_quat().tolist()

    M_base_to_cam1: np.ndarray | None = None
    if not args.no_robot_base:
        M_base_to_cam1 = _try_load_base_to_cam1(args.extrinsic)
        if M_base_to_cam1 is not None:
            log.info("Loaded base_to_cam1 from %s", args.extrinsic)

    # ── Load & preprocess PLY ──
    ply1 = load_ply_points(args.cam1_ply)
    ply2 = load_ply_points(args.cam2_ply)
    if ply1 is None:
        raise SystemExit(f"No points in cam1 PLY: {args.cam1_ply}")
    if ply2 is None:
        raise SystemExit(f"No points in cam2 PLY: {args.cam2_ply}")

    pts1_m, col1 = ply1
    pts2_m, col2 = ply2

    if not args.no_depth_clip:
        pts1_m, col1 = clip_depth_range(
            pts1_m, args.depth_min_m, args.depth_max_m,
            depth_axis=args.depth_axis, colors=col1,
        )
        pts2_m, col2 = clip_depth_range(
            pts2_m, args.depth_min_m, args.depth_max_m,
            depth_axis=args.depth_axis, colors=col2,
        )

    pts1_mm = pts1_m * 1000.0
    pts2_mm = pts2_m * 1000.0

    _DEFAULT_COLORS = {
        "cam1": np.array([[255, 120, 40]], dtype=np.uint8),
        "cam2": np.array([[40, 180, 255]], dtype=np.uint8),
    }
    if col1 is None:
        col1 = np.tile(_DEFAULT_COLORS["cam1"], (len(pts1_mm), 1))
    if col2 is None:
        col2 = np.tile(_DEFAULT_COLORS["cam2"], (len(pts2_mm), 1))

    pts1_mm, col1 = subsample(pts1_mm, col1, args.max_points)
    pts2_mm, col2 = subsample(pts2_mm, col2, args.max_points)

    # ── Rerun setup ──
    save_path = args.save
    spawn = save_path is None
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)

    vis = TransformVisualizer(
        "multi_eye_view", spawn=spawn, port=args.port,
        views=[("Stereo (camera 1)", "world/scene")],
    )

    # ── Robot base ──
    if M_base_to_cam1 is not None:
        if args.invert_base_calibration:
            M_base_to_cam1 = np.linalg.inv(M_base_to_cam1)
        T_base_to_cam1 = RigidTransform.from_matrix(M_base_to_cam1, Frame.CAMERA, Frame.BASE)
        vis.log_transform("world/scene/robot_base", T_base_to_cam1,
                          axis_length=300.0, label="ROBOT_BASE")
        log_origin_spheres([
            ("world/scene/robot_base/origin", np.zeros(3), (255, 255, 0), "BASE"),
        ])

    # ── Cam1 / Cam2 (axes + pcd + frustum) ──
    cam2_tf = (t.tolist(), quat_xyzw)
    _log_cam_frame("world/scene/cam1", pts1_mm, col1,
                   "CAM1", (255, 80, 80), transform_args=None)
    _log_cam_frame("world/scene/cam2", pts2_mm, col2,
                   "CAM2", (80, 180, 255), transform_args=cam2_tf)

    # ── Debug links ──
    cam1_origin = np.zeros(3)
    log_debug_link("world/scene/_debug/cam1_to_cam2", cam1_origin, t,
                   color=(140, 140, 200))
    if M_base_to_cam1 is not None:
        log_debug_link("world/scene/_debug/cam1_to_base", cam1_origin,
                       M_base_to_cam1[:3, 3], color=(200, 200, 120))

    # ── TCP ──
    if args.tcp_from_ply_name:
        _maybe_log_tcp(vis, args, M_base_to_cam1)

    # ── RGB Pinhole ──
    # Pinhole 은 3D entity 와 분리된 sibling entity 에 둬야 Rerun 2D subspace 규칙 충족.
    rgb_tabs: list[tuple[str, str]] = []
    if not args.no_rgb:
        rgb_tabs += _log_cam_rgb(
            "cam1", args.cam1_rgb, args.cam1_intrinsics,
            args.image_plane_mm_3d, args.image_plane_mm_2d,
        )
        rgb_tabs += _log_cam_rgb(
            "cam2", args.cam2_rgb, args.cam2_intrinsics,
            args.image_plane_mm_3d, args.image_plane_mm_2d,
            transform_args=cam2_tf,
        )

    _send_stereo_blueprint(rgb_tab_origins=rgb_tabs)
    log.info("Logged cam1=%d pts, cam2=%d pts.", len(pts1_mm), len(pts2_mm))
    finalize_viewer(save_path, spawn=spawn, app_name="Stereo (camera 1)")


if __name__ == "__main__":
    main()