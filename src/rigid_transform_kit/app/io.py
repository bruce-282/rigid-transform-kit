"""I/O helpers: calibration loading, TCP pose saving, cam-target loading."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Sequence

from rigid_transform_kit import Frame, RigidTransform

if TYPE_CHECKING:
    from rigid_transform_kit import PickPoint

log = logging.getLogger(__name__)


def load_calibration(path: Path) -> tuple[RigidTransform, RigidTransform]:
    """Load hand-eye calibration and return ``(T_base2cam, T_cam2base)`` as RigidTransforms.

    Wraps :func:`utils.load_extrinsics` so callers never touch raw numpy matrices.
    """
    from utils import load_extrinsics

    calib = load_extrinsics(path)
    T_base2cam = RigidTransform.from_matrix(calib["base2cam"], Frame.BASE, Frame.CAMERA)
    T_cam2base = T_base2cam.inv
    return T_base2cam, T_cam2base


def build_tcp_result(
    poses: list[RigidTransform],
    robot_commands: list[dict] | None = None,
    flange_poses: list[RigidTransform] | None = None,
) -> dict:
    """Build serializable result dict for TCP poses.

    Returns ``{"_schema": {...}, "picks": {"0": {...}, "1": {...}, ...}}``.
    Use this for saving to file or sending to protocol.
    flange_poses: 펜던트에서 플랜지 위치 조정용 (robot_command와 동일 값).

    Parameters
    ----------
    poses : list of RigidTransform (TCP poses)
    robot_commands : optional list of vendor command dicts (e.g. from plan_pick)
    flange_poses : optional list of RigidTransform (flange poses for pendant)
    """
    schema = {
        "tcp_pose": "TCP (tool center point) in base frame: position_mm, rotation_3x3, matrix_4x4",
        "flange_pose": "Flange in base frame (when tool has z-offset/rotation): position_mm, rotation_3x3, matrix_4x4",
        "tcp_pose_xyzwpr": "TCP in FANUC XYZWPR (X,Y,Z mm; W,P,R deg), for reference",
        "flange_pose_xyzwpr": "Flange in FANUC XYZWPR (X,Y,Z mm; W,P,R deg), for pendant/controller",
    }
    picks: dict = {}
    for i, pose in enumerate(poses):
        # 인덱스별 맨 위: 보기 좋게 XYZWPR 요약 먼저
        pick: dict = {
            "tcp_pose_xyzwpr": pose.to_xyzwpr(degrees=True),
        }
        if flange_poses is not None and i < len(flange_poses):
            fp = flange_poses[i]
            pick["flange_pose_xyzwpr"] = fp.to_xyzwpr(degrees=True)
            pick["flange_pose"] = {
                "position_mm": fp.t.tolist(),
                "rotation_3x3": fp.R.tolist(),
                "matrix_4x4": fp.matrix.tolist(),
            }
        elif robot_commands is not None and i < len(robot_commands):
            pick["flange_pose_xyzwpr"] = robot_commands[i]
        pick["tcp_pose"] = {
            "position_mm": pose.t.tolist(),
            "rotation_3x3": pose.R.tolist(),
            "matrix_4x4": pose.matrix.tolist(),
        }
        picks[str(i)] = pick
    return {"_schema": schema, "picks": picks}


def save_tcp_poses(result: dict, path: Path) -> None:
    """Save TCP result to JSON file.

    Expects ``result`` from :func:`build_tcp_result`.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")


def load_cam_targets(path: Path) -> list["PickPoint"]:
    """Load camera-frame target points from JSON.

    Re-exports :func:`utils.load_cam_targets` for a single import entry-point.
    """
    from utils import load_cam_targets as _load

    return _load(path)


def log_items(
    items: Sequence[Any],
    formatter: Callable[[int, Any], str | tuple],
    *,
    header: str | None = None,
    logger: logging.Logger | None = None,
) -> None:
    """Log each item using a custom formatter.

    Parameters
    ----------
    items : sequence to log
    formatter : ``(index, item) -> str`` or ``(index, item) -> (fmt, *args)``
        Return a plain string, or ``(fmt_str, *args)`` for ``log.info(fmt_str, *args)``.
    header : optional header line before items
    logger : defaults to this module's logger
    """
    _log = logger or log
    if header:
        _log.info(header)
    for i, item in enumerate(items):
        out = formatter(i, item)
        if isinstance(out, tuple):
            _log.info(out[0], *out[1:])
        else:
            _log.info(out)


def format_xyzwpr(i: int, cmd: dict, label: str | None = None) -> tuple:
    """Formatter for FANUC-style XYZWPR command dict."""
    tag = f" [{label}]" if label else ""
    return (
        "Pick #%d%s: X=%.1f Y=%.1f Z=%.1f W=%.2f P=%.2f R=%.2f",
        i, tag,
        cmd["X"], cmd["Y"], cmd["Z"],
        cmd["W"], cmd["P"], cmd["R"],
    )


def format_tcp_flange(i: int, pair: tuple[RigidTransform, RigidTransform]) -> tuple:
    """Formatter for (tcp_pose, flange_pose) pair."""
    tcp, flange = pair
    return (
        "Pick #%d  TCP (%.1f, %.1f, %.1f) → Flange (%.1f, %.1f, %.1f)",
        i, *tcp.t, *flange.t,
    )


def log_robot_commands(
    commands: Sequence[dict],
    *,
    labels: Sequence[str] | None = None,
    logger: logging.Logger | None = None,
) -> None:
    """Log each robot command. Uses :func:`format_xyzwpr`; override via :func:`log_items`."""
    def fmt(i: int, cmd: dict):
        lbl = labels[i] if labels and i < len(labels) else None
        return format_xyzwpr(i, cmd, label=lbl)
    log_items(commands, fmt, logger=logger)


def log_tcp_flange_detail(
    tcp_poses: Sequence[RigidTransform],
    flange_poses: Sequence[RigidTransform],
    *,
    logger: logging.Logger | None = None,
) -> None:
    """Log TCP → Flange for each pick. Uses :func:`format_tcp_flange`; override via :func:`log_items`."""
    pairs = list(zip(tcp_poses, flange_poses))
    log_items(pairs, format_tcp_flange, header="Detail (TCP → Flange)", logger=logger)
