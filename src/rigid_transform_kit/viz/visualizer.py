"""
rigid_transform_kit.viz.visualizer
====================================
Optional 3D visualization via Rerun.

Install: ``pip install rigid-transform-kit[viz]``

Usage::

    from rigid_transform_kit.viz import TransformVisualizer

    vis = TransformVisualizer("my_pipeline")
    vis.log_frame(T_base2cam, axis_length=0.15)
    vis.log_pick_point(pick, cam_config)
    vis.log_tcp_pose(T_base2tcp)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np

from ..core import Frame, RigidTransform

if TYPE_CHECKING:
    from ..vision import CameraConfig, PickPoint

try:
    import rerun as rr

    _HAS_RERUN = True
except ImportError:
    _HAS_RERUN = False


def _require_rerun():
    if not _HAS_RERUN:
        raise ImportError(
            "rerun-sdk is required for visualization. "
            "Install with: pip install rigid-transform-kit[viz]"
        )


# ============================================================
# Color palette
# ============================================================

FRAME_COLORS = {
    Frame.BASE: [200, 200, 200],
    Frame.CAMERA: [50, 180, 50],
    Frame.FLANGE: [180, 130, 50],
    Frame.TCP: [220, 60, 60],
    Frame.OBJECT: [60, 60, 220],
    Frame.WORLD: [160, 160, 160],
    Frame.MARKER: [180, 50, 180],
}

AXIS_COLORS = [
    [220, 40, 40],   # X — red
    [40, 180, 40],   # Y — green
    [40, 80, 220],   # Z — blue
]


# ============================================================
# TransformVisualizer
# ============================================================

class TransformVisualizer:
    """Rerun-based 3D visualizer for rigid transform pipelines.

    Parameters
    ----------
    app_id : str
        Rerun application identifier.
    spawn : bool
        If True, spawn the Rerun viewer on init.
    """

    def __init__(self, app_id: str = "rigid_transform_kit", spawn: bool = True):
        _require_rerun()
        rr.init(app_id)
        if spawn:
            rr.spawn()

        rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    # ---- transform logging ----

    def log_transform(
        self,
        entity_path: str,
        transform: RigidTransform,
        *,
        axis_length: float = 0.1,
        label: Optional[str] = None,
    ) -> None:
        """Log a RigidTransform as a coordinate frame with XYZ axes.

        The entity is placed at the transform's translation,
        with orientation arrows showing the rotation.

        Parameters
        ----------
        entity_path : str
            Rerun entity path (e.g. "world/camera").
        transform : RigidTransform
            The transform to visualize. Translation = position, R = orientation.
        axis_length : float
            Length of the XYZ axis arrows (meters).
        label : str or None
            Optional label. Defaults to "T(FROM->TO)".
        """
        pos, quat_xyzw = transform.to_pos_quat_xyzw()

        rr.log(
            entity_path,
            rr.Transform3D(
                translation=pos.tolist(),
                quaternion=rr.Quaternion(xyzw=quat_xyzw.tolist()),
            ),
        )

        axes = np.eye(3) * axis_length
        rr.log(
            f"{entity_path}/axes",
            rr.Arrows3D(
                origins=[[0, 0, 0]] * 3,
                vectors=axes.tolist(),
                colors=AXIS_COLORS,
                labels=["X", "Y", "Z"] if label is None else [f"{label} X", f"{label} Y", f"{label} Z"],
            ),
        )

    def log_frame(
        self,
        transform: RigidTransform,
        *,
        axis_length: float = 0.1,
        parent_path: str = "world",
    ) -> None:
        """Log a transform under a standard entity hierarchy.

        Entity path is auto-generated: ``{parent_path}/{to_frame_name}``.
        """
        name = transform.to_frame.name.lower()
        self.log_transform(
            f"{parent_path}/{name}",
            transform,
            axis_length=axis_length,
            label=name.upper(),
        )

    # ---- chain visualization ----

    def log_transform_chain(
        self,
        transforms: Sequence[RigidTransform],
        *,
        axis_length: float = 0.08,
        parent_path: str = "world",
        show_connections: bool = True,
    ) -> None:
        """Log a chain of transforms and optionally connect them with lines.

        Parameters
        ----------
        transforms : sequence of RigidTransform
            Transforms to visualize. Each is logged as a frame.
        show_connections : bool
            If True, draw lines connecting consecutive frame origins.
        """
        positions = []

        for tf in transforms:
            self.log_frame(tf, axis_length=axis_length, parent_path=parent_path)
            positions.append(tf.t.tolist())

        if show_connections and len(positions) >= 2:
            rr.log(
                f"{parent_path}/_chain",
                rr.LineStrips3D(
                    [positions],
                    colors=[[180, 180, 180]],
                ),
            )

    # ---- pick point visualization ----

    def log_pick_point(
        self,
        pick: PickPoint,
        cam_config: CameraConfig,
        *,
        parent_path: str = "world",
        index: int = 0,
        normal_length: float = 0.05,
    ) -> None:
        """Log a PickPoint as a 3D point + surface normal arrow in base frame.

        Parameters
        ----------
        pick : PickPoint
        cam_config : CameraConfig
        index : int
            Index for labeling when visualizing multiple picks.
        normal_length : float
            Length of the normal arrow (meters).
        """
        p_base, n_base = pick.to_base(cam_config)

        rr.log(
            f"{parent_path}/picks/pt_{index}",
            rr.Points3D(
                [p_base.tolist()],
                colors=[[255, 200, 50]],
                radii=[0.005],
                labels=[f"pick_{index} (conf={pick.confidence:.2f})"],
            ),
        )

        rr.log(
            f"{parent_path}/picks/normal_{index}",
            rr.Arrows3D(
                origins=[p_base.tolist()],
                vectors=[(n_base * normal_length).tolist()],
                colors=[[100, 220, 255]],
            ),
        )

    def log_pick_points(
        self,
        picks: Sequence[PickPoint],
        cam_config: CameraConfig,
        *,
        parent_path: str = "world",
        normal_length: float = 0.05,
    ) -> None:
        """Log multiple pick points at once."""
        for i, pick in enumerate(picks):
            self.log_pick_point(
                pick, cam_config,
                parent_path=parent_path,
                index=i,
                normal_length=normal_length,
            )

    # ---- TCP pose ----

    def log_tcp_pose(
        self,
        T_base2tcp: RigidTransform,
        *,
        parent_path: str = "world",
        axis_length: float = 0.06,
        label: str = "tcp",
    ) -> None:
        """Log a TCP target pose."""
        self.log_transform(
            f"{parent_path}/{label}",
            T_base2tcp,
            axis_length=axis_length,
            label=label.upper(),
        )

    # ---- robot command result ----

    def log_flange_pose(
        self,
        T_base2flange: RigidTransform,
        *,
        parent_path: str = "world",
        axis_length: float = 0.06,
    ) -> None:
        """Log a flange target pose."""
        self.log_transform(
            f"{parent_path}/flange",
            T_base2flange,
            axis_length=axis_length,
            label="FLANGE",
        )

    # ---- point cloud helpers ----

    def log_points(
        self,
        entity_path: str,
        points: np.ndarray,
        *,
        colors: Optional[np.ndarray] = None,
        radii: Optional[float] = None,
    ) -> None:
        """Log a raw point cloud."""
        kwargs = {}
        if colors is not None:
            kwargs["colors"] = colors
        if radii is not None:
            kwargs["radii"] = [radii]
        rr.log(entity_path, rr.Points3D(points, **kwargs))

    # ---- full pipeline visualization ----

    def log_picking_pipeline(
        self,
        cam_config: CameraConfig,
        pick: PickPoint,
        T_base2tcp: RigidTransform,
        T_base2flange: Optional[RigidTransform] = None,
        *,
        parent_path: str = "world",
        index: int = 0,
    ) -> None:
        """Log the complete picking pipeline for one pick.

        Visualizes: base frame, camera frame, pick point, TCP pose,
        and optionally the flange pose.
        """
        self.log_transform(
            f"{parent_path}/base",
            RigidTransform.identity(Frame.BASE),
            axis_length=0.15,
            label="BASE",
        )

        self.log_frame(
            cam_config.T_cam2base,
            axis_length=0.10,
            parent_path=parent_path,
        )

        self.log_pick_point(
            pick, cam_config,
            parent_path=parent_path,
            index=index,
        )

        self.log_tcp_pose(
            T_base2tcp,
            parent_path=parent_path,
            label=f"tcp_{index}",
        )

        if T_base2flange is not None:
            self.log_transform(
                f"{parent_path}/flange_{index}",
                T_base2flange,
                axis_length=0.05,
                label=f"FLANGE_{index}",
            )
