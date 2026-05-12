"""
examples / fixed_eye_view.py
==============================
**설계 목표 (단일 좌표계: robot base):** Fixed-eye (eye-to-hand) calibration 결과를 이용해
카메라 포인트클라우드를 **로봇 base 좌표계**로 변환하여 Rerun에 시각화합니다.

카메라가 고정이므로 ``cam_to_base`` 행렬 하나로 직접 변환합니다.

* **PLY** — 카메라 좌표계 포인트. ``cam_to_base`` 로 base 좌표계로 변환.
* **Camera 축** — base 기준 고정 카메라 위치.
* **Flange 축** — 파일명에서 파싱한 TCP vec6 (캡처 당시 로봇 포즈).
* **World 축** — ``flange_to_world`` 가 있으면 표시 (optional).

변환::

    p_base = cam_to_base @ p_cam

PLY 파일명 규칙: ``[idx_]x_y_z_W_P_R`` (mm, Fanuc xyz WPR degrees).

Requires: ``pip install -e ".[viz]"`` (open3d: PLY 로드 시 권장)

Usage::

  python examples/fixed_eye_view.py

  python examples/fixed_eye_view.py \\
    --ply datasets/fixed_eye_example/1_....ply \\
    --calibration datasets/fixed_eye_example/fixed_eye_cal.yml

  python examples/fixed_eye_view.py --save output/fixed_eye.rrd
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import time
from pathlib import Path

import numpy as np

from rigid_transform_kit import Frame, RigidTransform
from rigid_transform_kit.viz import TransformVisualizer, save_recording

from utils import clip_depth_range, load_ply_points

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_DATA_DIR = REPO_ROOT / "datasets/fixed_eye_example"
_DEFAULT_PLY = next(_DEFAULT_DATA_DIR.glob("*.ply"), _DEFAULT_DATA_DIR / "scan.ply")
_DEFAULT_CAL = _DEFAULT_DATA_DIR / "fixed_eye_cal.yml"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load_fixed_eye_cal(path: Path) -> dict[str, np.ndarray]:
    """Load ``cam_to_base`` (or ``base_to_cam``) and optionally ``flange_to_world`` from YAML."""
    try:
        import yaml
    except ImportError as e:
        raise ImportError("PyYAML required: pip install pyyaml") from e

    with open(path, encoding="utf-8") as f:
        doc = yaml.safe_load(f)

    result: dict[str, np.ndarray] = {}
    for key in ("cam_to_base", "base_to_cam", "flange_to_world", "world_to_flange"):
        if key in doc:
            arr = np.asarray(doc[key], dtype=np.float64)
            if arr.shape == (4, 4):
                result[key] = arr
            elif arr.size == 16:
                result[key] = arr.reshape(4, 4)

    if "cam_to_base" not in result and "base_to_cam" not in result:
        raise KeyError(f"YAML must contain 'cam_to_base' or 'base_to_cam' 4x4 matrix: {path}")
    return result


def _parse_tcp_vec6_from_filename(path: Path) -> np.ndarray | None:
    """Parse stem ``[idx_]x_y_z_W_P_R`` -> vec6 (mm, Fanuc xyz WPR degrees)."""
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


def _maybe_subsample(
    pts: np.ndarray, colors: np.ndarray | None, max_n: int
) -> tuple[np.ndarray, np.ndarray | None]:
    if max_n <= 0 or pts.shape[0] <= max_n:
        return pts, colors
    rng = np.random.default_rng(0)
    idx = rng.choice(pts.shape[0], size=max_n, replace=False)
    c2 = colors[idx] if colors is not None else None
    return pts[idx], c2


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fixed-eye (eye-to-hand) viewer: PLY + cam_to_base → robot base frame (Rerun).",
    )
    p.add_argument(
        "--ply",
        type=Path,
        default=_DEFAULT_PLY,
        help="PLY point cloud (camera frame). Filename encodes TCP vec6.",
    )
    p.add_argument(
        "--calibration",
        type=Path,
        default=_DEFAULT_CAL,
        help="YAML with cam_to_base (or base_to_cam) 4x4.",
    )
    p.add_argument(
        "--save",
        type=Path,
        default=None,
        metavar="RRD",
        help="Save to .rrd file instead of live spawn.",
    )
    p.add_argument(
        "--port",
        type=int,
        default=None,
        help="Rerun gRPC port.",
    )
    p.add_argument(
        "--no-depth-clip",
        action="store_true",
        help="Disable depth clipping.",
    )
    p.add_argument(
        "--depth-min-m",
        type=float,
        default=0.0,
        help="Depth clip min (meters, default 0).",
    )
    p.add_argument(
        "--depth-max-m",
        type=float,
        default=3.0,
        help="Depth clip max (meters, default 3.0).",
    )
    p.add_argument(
        "--depth-axis",
        type=int,
        default=2,
        help="Depth axis index (default 2 = Z).",
    )
    p.add_argument(
        "--max-points",
        type=int,
        default=500_000,
        help="Max points to display (subsample if larger, 0 = no limit).",
    )
    p.add_argument(
        "--base-axis-mm",
        type=float,
        default=300.0,
        help="Base frame axis length in mm.",
    )
    p.add_argument(
        "--cam-axis-mm",
        type=float,
        default=150.0,
        help="Camera frame axis length in mm.",
    )
    p.add_argument(
        "--flange-axis-mm",
        type=float,
        default=200.0,
        help="Flange frame axis length in mm.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()

    # ── Load calibration ──
    cal = _load_fixed_eye_cal(args.calibration)

    # RigidTransform 라이브러리 컨벤션: from_frame=A, to_frame=B → p_A = M @ p_B
    #   from_frame = output(결과), to_frame = input(입력)
    #
    # YAML "cam_to_base" 행렬 = p_base = M @ p_cam (cam 점을 base로 보냄)
    #   → from=BASE, to=CAMERA  (output=BASE, input=CAMERA)
    # YAML "base_to_cam" 행렬 = p_cam = M @ p_base (base 점을 cam으로 보냄)
    #   → from=CAMERA, to=BASE  (output=CAMERA, input=BASE)
    #
    # 변수명은 행렬의 실제 동작(점 변환 방향) 기준으로 명명.

    if "cam_to_base" in cal:
        T_cam_to_base = RigidTransform.from_matrix(
            cal["cam_to_base"], Frame.BASE, Frame.CAMERA,
        )
        log.info("Loaded cam_to_base from %s", args.calibration)
    elif "base_to_cam" in cal:
        T_base_to_cam = RigidTransform.from_matrix(
            cal["base_to_cam"], Frame.CAMERA, Frame.BASE,
        )
        T_cam_to_base = T_base_to_cam.inv
        log.info("Loaded base_to_cam (inverted → cam_to_base) from %s", args.calibration)
    else:
        raise KeyError(f"YAML must contain 'cam_to_base' or 'base_to_cam': {args.calibration}")

    # flange_to_world (optional)
    # YAML "flange_to_world" = p_world = M @ p_flange → from=WORLD, to=FLANGE
    T_flange_to_world: RigidTransform | None = None
    if "flange_to_world" in cal:
        T_flange_to_world = RigidTransform.from_matrix(
            cal["flange_to_world"], Frame.WORLD, Frame.FLANGE,
        )
        log.info("Loaded flange_to_world from %s", args.calibration)


    # ── Parse TCP pose from PLY filename (capture-time robot pose) ──
    # vec6 = [x, y, z, W, P, R] → p_base = M @ p_flange
    # from=BASE, to=FLANGE (output=BASE, input=FLANGE)
    vec6 = _parse_tcp_vec6_from_filename(args.ply)
    T_base_to_flange: RigidTransform | None = None
    if vec6 is not None:
        T_base_to_flange = RigidTransform.from_vec6(vec6, Frame.BASE, Frame.FLANGE)
        log.info(
            "TCP from filename: xyz=(%.1f, %.1f, %.1f) mm, WPR=(%.1f, %.1f, %.1f) deg",
            *vec6,
        )
    else:
        log.warning("Could not parse TCP vec6 from filename: %s", args.ply.name)

    # ── Load PLY ──
    ply_data = load_ply_points(args.ply)
    if ply_data is None:
        raise SystemExit(f"No points in PLY: {args.ply}")

    pts_cam_m, colors = ply_data

    # Filter NaN
    valid = ~np.any(np.isnan(pts_cam_m), axis=1)
    if not np.all(valid):
        n_before = len(pts_cam_m)
        pts_cam_m = pts_cam_m[valid]
        colors = colors[valid] if colors is not None else None
        log.info("Filtered NaN: %d -> %d pts", n_before, len(pts_cam_m))

    # Detect mm and convert to m
    if len(pts_cam_m) > 0 and np.median(np.abs(pts_cam_m)) > 100:
        pts_cam_m = pts_cam_m / 1000.0
        log.info("PLY appears to be in mm; converted to meters.")

    if not args.no_depth_clip:
        n_before = len(pts_cam_m)
        pts_cam_m, colors = clip_depth_range(
            pts_cam_m,
            args.depth_min_m,
            args.depth_max_m,
            depth_axis=args.depth_axis,
            colors=colors,
        )
        log.info(
            "Depth clip [%.2f, %.2f] m (axis=%d): %d -> %d pts",
            args.depth_min_m, args.depth_max_m, args.depth_axis,
            n_before, len(pts_cam_m),
        )

    pts_cam_mm = pts_cam_m * 1000.0
    pts_cam_mm, colors = _maybe_subsample(pts_cam_mm, colors, args.max_points)

    # T_cam_to_base: from=BASE, to=CAMERA → p_base = M @ p_cam
    pts_base = T_cam_to_base.transform_points(pts_cam_mm)
    log.info("Transformed %d points to base frame.", len(pts_base))

    # ── Rerun setup ──
    save_path = args.save
    if save_path is None and len(pts_base) > 500_000:
        save_path = _DEFAULT_DATA_DIR / "fixed_eye.rrd"
        log.info("Large PCD (%d pts). Auto-saving to %s.", len(pts_base), save_path)
    spawn = save_path is None
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)

    vis = TransformVisualizer(
        "fixed_eye_view",
        spawn=spawn,
        port=args.port,
        views=[("Fixed-Eye (base frame)", "world")],
    )
    import rerun as rr

    # ── Log base frame ──
    vis.log_transform(
        "world/base",
        RigidTransform.identity(Frame.BASE),
        axis_length=args.base_axis_mm,
        label="BASE",
    )

    # ── Log camera frame (fixed, with frustum) ──
    # log_transform uses .t and .R directly from the matrix.
    # T_cam_to_base의 .t = base 좌표계에서 본 카메라 위치, .R = 카메라 방향
    vis.log_transform(
        "world/camera",
        T_cam_to_base,
        axis_length=args.cam_axis_mm,
        label="CAMERA",
    )
    rr.log(
        "world/camera/frustum",
        rr.Pinhole(
            focal_length=300.0,
            width=640,
            height=480,
            image_plane_distance=args.cam_axis_mm * 0.6,
        ),
        static=True,
    )

    # ── Log flange frame (capture-time robot pose) ──
    if T_base_to_flange is not None:
        vis.log_transform(
            "world/flange",
            T_base_to_flange,
            axis_length=args.flange_axis_mm,
            label="FLANGE",
        )

        # world frame via flange (optional)
        # T_flange_to_world: from=WORLD, to=FLANGE → p_world = M @ p_flange
        # T_flange_to_world.inv: from=FLANGE, to=WORLD → p_flange = M⁻¹ @ p_world
        # T_base_to_flange:     from=BASE, to=FLANGE  → p_base   = M  @ p_flange
        #
        # 체인: p_base = T_base_to_flange @ inv(T_flange_to_world) @ p_world
        if T_flange_to_world is not None:
            T_base_to_world = RigidTransform.from_matrix(
                T_base_to_flange.matrix @ np.linalg.inv(T_flange_to_world.matrix),
                Frame.BASE, Frame.WORLD,
            )
            vis.log_transform(
                "world/world_frame",
                T_base_to_world,
                axis_length=args.base_axis_mm * 0.8,
                label="WORLD",
            )

    # ── Log origin spheres ──
    origin_frames = [
        ("world/_origins/base", np.zeros(3), (255, 80, 80), "BASE"),
        ("world/_origins/camera", T_cam_to_base.t, (80, 120, 255), "CAMERA"),
    ]
    if T_base_to_flange is not None:
        origin_frames.append(
            ("world/_origins/flange", T_base_to_flange.t, (80, 255, 80), "FLANGE")
        )
    for path, pos, color, label in origin_frames:
        rr.log(
            path,
            rr.Points3D([pos.tolist()], colors=[color], radii=[6.0], labels=[label]),
            static=True,
        )

    # ── Log point cloud ──
    if colors is None:
        colors = np.tile(np.array([[40, 180, 255]], dtype=np.uint8), (len(pts_base), 1))
    vis.log_points("world/pcd", pts_base, colors=colors, radii=1.2)
    log.info("Logged %d points (base frame).", len(pts_base))

    # ── Debug: link lines ──
    cam_origin = T_cam_to_base.t
    rr.log(
        "world/_debug/base_to_cam",
        rr.LineStrips3D(
            [[[0, 0, 0], cam_origin.tolist()]],
            colors=[[180, 180, 180, 120]],
            radii=[1.5],
        ),
        static=True,
    )
    if T_base_to_flange is not None:
        rr.log(
            "world/_debug/base_to_flange",
            rr.LineStrips3D(
                [[[0, 0, 0], T_base_to_flange.t.tolist()]],
                colors=[[80, 255, 80, 120]],
                radii=[1.5],
            ),
            static=True,
        )

    # ── Save / spawn ──
    if save_path is not None:
        save_recording(save_path)
        log.info("Saved to %s", save_path)
        try:
            subprocess.run(["rerun", str(save_path)], check=False)
        except FileNotFoundError:
            log.info("Run: rerun %s", save_path)

    if spawn:
        log.info("Rerun viewer open — 'Fixed-Eye (base frame)' tab.")
        rec = rr.get_global_data_recording()
        if rec is not None:
            try:
                rec.flush(timeout_sec=10.0)
            except Exception:
                pass
            time.sleep(1.0)


if __name__ == "__main__":
    main()
