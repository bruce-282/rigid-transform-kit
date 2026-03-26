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

from pathlib import Path
from typing import TYPE_CHECKING, Optional, Sequence, Union

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


def save_recording(path: Union[Path, str]) -> None:
    """Save the current Rerun recording to an .rrd file.

    Call after logging. Requires rerun-sdk.
    """
    _require_rerun()
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    rr.save(str(p))


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

DEFAULT_VIEWS: list[tuple[str, str]] = [
    ("Overview (in Base)", "world"),
    ("Scene (in Camera)", "cam_view"),
    ("Scene (in Base)", "scene_base"),
]

PROJ_2D_VIEW = "proj_2d"


class TransformVisualizer:
    """Rerun-based 3D visualizer for rigid transform pipelines.

    Parameters
    ----------
    app_id : str
        Rerun application identifier.
    spawn : bool
        If True, spawn the Rerun viewer on init.
    port : int or None
        gRPC port for the viewer (default 9876). If None, uses Rerun default.
        Use a different port if 9876 is already in use (e.g. os error 10048 on Windows).
    views : list of (name, origin) tuples or None
        Spatial3DView tabs to create in the Rerun blueprint.
        Each tuple is ``("Tab Name", "entity_origin")``.
        Defaults to :data:`DEFAULT_VIEWS`.
    """

    def __init__(
        self,
        app_id: str = "rigid_transform_kit",
        spawn: bool = True,
        port: Optional[int] = None,
        views: Optional[list[tuple[str, str]]] = None,
    ):
        _require_rerun()
        rr.init(app_id)
        if spawn:
            if port is not None:
                rr.spawn(port=port)
            else:
                rr.spawn()

        rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        rr.log("world", rr.Transform3D(translation=[0, 0, 0]), static=True)

        self._build_blueprint(views or DEFAULT_VIEWS)

    @staticmethod
    def _build_blueprint(views: list[tuple[str, str]]) -> None:
        """Create and send a tabbed blueprint (3D views + Projection 2D)."""
        import rerun.blueprint as rrb

        tabs = [
            rrb.Spatial3DView(name=name, origin=origin, contents=[f"+ {origin}/**"])
            for name, origin in views
        ]
        tabs.append(
            rrb.Spatial2DView(
                name="Projection 2D",
                origin=PROJ_2D_VIEW,
                contents=[f"+ {PROJ_2D_VIEW}/**"],
            )
        )
        rr.send_blueprint(rrb.Blueprint(rrb.Tabs(*tabs)))

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
        """Log a PickPoint as a 3-axis frame in base frame (from T_base2pick rotation).

        Parameters
        ----------
        pick : PickPoint
        cam_config : CameraConfig
        index : int
            Index for labeling when visualizing multiple picks.
        normal_length : float
            Length of the XYZ axis arrows (meters).
        """
        T_base2pick = pick.to_base(cam_config)
        self.log_transform(
            f"{parent_path}/picks/pick_{index}",
            T_base2pick,
            axis_length=normal_length,
            label=f"pick_{index} (conf={pick.confidence:.2f})",
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
        arrow_radius: Optional[float] = None,
        label: str = "tcp",
        show_axes: bool = True,
    ) -> None:
        """Log a TCP target pose as a point, optionally with 3-axis arrows.

        Parameters
        ----------
        T_base2tcp : RigidTransform
            TCP pose in base frame.
        parent_path : str
            Rerun entity parent path.
        axis_length : float
            Length of each axis arrow (same unit as T_base2tcp).
        arrow_radius : float or None
            Radius of each axis arrow shaft. Defaults to axis_length * 0.04.
        label : str
            Entity name and display label.
        show_axes : bool
            If False, only the origin point is shown (no XYZ arrows).
        """
        entity = f"{parent_path}/{label}"
        t = T_base2tcp.t
        R = T_base2tcp.R
        r = arrow_radius if arrow_radius is not None else axis_length * 0.04
        # Overview: use TCP origin as-is (no lift) for correct visualization
        rr.log(
            f"{entity}/origin",
            rr.Points3D(
                [t.tolist()],
                colors=[[255, 200, 50]],
                radii=[r * 2.5],
                labels=[label.upper()],
            ),
        )
        if show_axes:
            rr.log(
                f"{entity}/axes",
                rr.Arrows3D(
                    origins=[t.tolist()] * 3,
                    vectors=[
                        (R[:, 0] * axis_length).tolist(),
                        (R[:, 1] * axis_length).tolist(),
                        (R[:, 2] * axis_length).tolist(),
                    ],
                    colors=AXIS_COLORS,
                    radii=[r] * 3,
                ),
            )
        # vec6 (x,y,z, W,P,R deg) for inspection in blueprint
        vec6 = T_base2tcp.to_vec6_euler(convention="xyz", degrees=True)
        rr.log(
            f"{entity}/vec6",
            rr.TextDocument(
                f"[{vec6[0]:.2f}, {vec6[1]:.2f}, {vec6[2]:.2f}, {vec6[3]:.2f}, {vec6[4]:.2f}, {vec6[5]:.2f}]"
            ),
        )

    def log_tcp_poses(
        self,
        poses: Sequence[RigidTransform],
        *,
        parent_path: str = "world/picks",
        axis_length: float = 0.06,
        arrow_radius: Optional[float] = None,
        show_axes: Optional[Sequence[bool]] = None,
    ) -> None:
        """Log multiple TCP poses at once."""
        for i, pose in enumerate(poses):
            axes = show_axes[i] if show_axes is not None else True
            self.log_tcp_pose(
                pose,
                parent_path=parent_path,
                axis_length=axis_length,
                arrow_radius=arrow_radius,
                label=f"tcp_{i}",
                show_axes=axes,
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

    def log_flange_poses(
        self,
        poses: Sequence[RigidTransform],
        *,
        parent_path: str = "world/flanges",
        axis_length: float = 100.0,
    ) -> None:
        """Log multiple flange poses (e.g. from compute_flange_target). Shown in Overview (in Base)."""
        for i, T_base2flange in enumerate(poses):
            self.log_transform(
                f"{parent_path}/flange_{i}",
                T_base2flange,
                axis_length=axis_length,
                label=f"FLANGE_{i}",
            )

    # ---- scene view (generic) ----

    def _log_scene_view(
        self,
        prefix: str,
        pts: Optional[np.ndarray] = None,
        colors: Optional[np.ndarray] = None,
        tcp_poses: Optional[Sequence[RigidTransform]] = None,
        show_axes: Optional[Sequence[bool]] = None,
        *,
        radii: float = 1.2,
        axis_length: float = 100.0,
        show_origin: bool = False,
        origin_axis_length: float = 300.0,
        show_y_both: bool = False,
    ) -> None:
        """Log a scene view under *prefix* (PCD + pick orientations).

        This is the shared implementation behind all ``log_scene_*``
        public methods.  Each public wrapper simply forwards its
        arguments with the appropriate *prefix* and defaults.

        Parameters
        ----------
        prefix : str
            Rerun entity path prefix (e.g. ``"cam_view"``, ``"scene_base"``).
        pts : (N,3) array or None
            Point cloud (mm).
        colors : (N,3) uint8 array or None
            Per-point colors.
        tcp_poses : sequence of RigidTransform or None
            Pick/TCP poses. Z axis is flipped to show camera direction.
        show_axes : list of bool or None
            Per-pose flag; False = origin point only (no arrows).
        radii : float
            Point / arrow shaft radius.
        axis_length : float
            Length of pick orientation arrows (mm).
        show_origin : bool
            If True, draw XYZ axes at the coordinate-frame origin.
        origin_axis_length : float
            Length of origin axes (mm).
        """
        if show_origin:
            rr.log(f"{prefix}/origin", rr.Transform3D(translation=[0, 0, 0]), static=True)
            axes_vecs = np.eye(3) * origin_axis_length
            rr.log(
                f"{prefix}/origin/axes",
                rr.Arrows3D(
                    origins=[[0, 0, 0]] * 3,
                    vectors=axes_vecs.tolist(),
                    colors=AXIS_COLORS,
                    labels=["X", "Y", "Z"],
                ),
                static=True,
            )

        if pts is not None:
            kwargs = {}
            if colors is not None:
                kwargs["colors"] = colors
            rr.log(f"{prefix}/pcd", rr.Points3D(pts, radii=[radii], **kwargs), static=True)

        if tcp_poses:
            for i, pose in enumerate(tcp_poses):
                draw = show_axes[i] if show_axes is not None else True
                t = pose.t
                R = pose.R
                cam_dir = -R[:, 2]
                t_lifted = t + cam_dir * axis_length * 0.15
                rr.log(
                    f"{prefix}/pick_{i}/origin",
                    rr.Points3D(
                        [t_lifted.tolist()],
                        colors=[[255, 200, 50]],
                        radii=[radii * 2.5],
                        labels=[f"Pick_{i}"],
                    ),
                    static=True,
                )
                if draw:
                    z_vec = cam_dir * axis_length
                    x_vec = R[:, 0] * axis_length
                    y_vec = -R[:, 1] * axis_length
                    if show_y_both:
                        y_pos_vec = R[:, 1] * axis_length
                        rr.log(
                            f"{prefix}/pick_{i}/axes",
                            rr.Arrows3D(
                                origins=[t_lifted.tolist()] * 4,
                                vectors=[x_vec.tolist(), y_pos_vec.tolist(), y_vec.tolist(), z_vec.tolist()],
                                colors=[AXIS_COLORS[0], [80, 220, 80], AXIS_COLORS[1], AXIS_COLORS[2]],
                                radii=[radii] * 4,
                                labels=["X", "Y+", "Y-", "Z (->cam)"],
                            ),
                            static=True,
                        )
                        continue
                    rr.log(
                        f"{prefix}/pick_{i}/axes",
                        rr.Arrows3D(
                            origins=[t_lifted.tolist()] * 3,
                            vectors=[x_vec.tolist(), y_vec.tolist(), z_vec.tolist()],
                            colors=AXIS_COLORS,
                            radii=[radii] * 3,
                            labels=["X", "Y", "Z (->cam)"],
                        ),
                        static=True,
                    )

    # ---- scene view public wrappers ----

    def log_scene_in_camera(
        self,
        pts_cam: Optional[np.ndarray] = None,
        colors: Optional[np.ndarray] = None,
        tcp_poses: Optional[Sequence[RigidTransform]] = None,
        show_axes: Optional[Sequence[bool]] = None,
        *,
        radii: float = 1.2,
        axis_length: float = 100.0,
        show_y_both: bool = False,
    ) -> None:
        """Scene view in camera frame (no origin axes)."""
        self._log_scene_view(
            "cam_view", pts_cam, colors, tcp_poses, show_axes,
            radii=radii, axis_length=axis_length, show_origin=False, show_y_both=show_y_both,
        )

    def log_scene_base(
        self,
        pts_base: Optional[np.ndarray] = None,
        colors: Optional[np.ndarray] = None,
        tcp_poses: Optional[Sequence[RigidTransform]] = None,
        show_axes: Optional[Sequence[bool]] = None,
        *,
        radii: float = 1.2,
        axis_length: float = 100.0,
        show_y_both: bool = False,
    ) -> None:
        """Scene view in base frame (with origin axes)."""
        self._log_scene_view(
            "scene_base", pts_base, colors, tcp_poses, show_axes,
            radii=radii, axis_length=axis_length, show_origin=True, show_y_both=show_y_both,
        )

    # ---- point cloud helpers ----

    def log_points(
        self,
        entity_path: str,
        points: np.ndarray,
        *,
        colors: Optional[np.ndarray] = None,
        radii: Optional[float] = 1.2,
    ) -> None:
        """Log a raw point cloud. Default radii=1.2 (mm) for finer dots."""
        kwargs = {}
        if colors is not None:
            kwargs["colors"] = colors
        if radii is not None:
            kwargs["radii"] = [radii]
        rr.log(entity_path, rr.Points3D(points, **kwargs))

    def log_projection_2d(
        self,
        K: np.ndarray,
        *,
        pts_cam: Optional[np.ndarray] = None,
        colors: Optional[np.ndarray] = None,
        transforms: Optional[Sequence[RigidTransform]] = None,
        axis_length_mm: float = 50.0,
        point_radii: float = 1.0,
        arrow_radii: float = 0.8,
        base_path: Optional[str] = None,
    ) -> None:
        """Project point cloud and coordinate frames to 2D and log for Rerun 'Projection 2D' view.

        *pts_cam* in camera frame (mm), optional. *K* is (3,3). *transforms* in camera frame (e.g. T_cam2board).
        *arrow_radii*: 2D arrow shaft thickness in pixels (default 0.8).
        """
        base_path = base_path or PROJ_2D_VIEW
        K = np.asarray(K, dtype=np.float64)
        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])

        if pts_cam is not None:
            P = np.asarray(pts_cam, dtype=np.float64)
            valid = P[:, 2] > 0
        else:
            valid = np.array([], dtype=bool)
        if pts_cam is not None and np.any(valid):
            Pv = P[valid]
            u = fx * (Pv[:, 0] / Pv[:, 2]) + cx
            v = fy * (Pv[:, 1] / Pv[:, 2]) + cy
            uv = np.column_stack((u, v))
            kwargs = {"radii": [point_radii]}
            if colors is not None:
                kwargs["colors"] = colors[valid]
            rr.log(
                f"{base_path}/pcd",
                rr.Points2D(uv, draw_order=0.0, **kwargs),
                static=True,
            )

        if transforms:
            for i, T in enumerate(transforms):
                t = np.asarray(T.t, dtype=np.float64)
                R = np.asarray(T.R, dtype=np.float64)
                if t[2] <= 0:
                    continue
                u0 = fx * (t[0] / t[2]) + cx
                v0 = fy * (t[1] / t[2]) + cy
                axes_3d = R * axis_length_mm
                origins_2d = []
                vectors_2d = []
                for j in range(3):
                    end = t + axes_3d[:, j]
                    if end[2] <= 0:
                        continue
                    u1 = fx * (end[0] / end[2]) + cx
                    v1 = fy * (end[1] / end[2]) + cy
                    origins_2d.append([u0, v0])
                    vectors_2d.append([u1 - u0, v1 - v0])
                if origins_2d:
                    rr.log(
                        f"{base_path}/frame_{i}",
                        rr.Arrows2D(
                            origins=origins_2d,
                            vectors=vectors_2d,
                            colors=AXIS_COLORS[: len(origins_2d)],
                            labels=["X", "Y", "Z"][: len(origins_2d)],
                            radii=arrow_radii,
                            draw_order=50.0,
                        ),
                        static=True,
                    )

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

        # Camera pose in base frame:
        #   position = T_base2cam.t  (camera origin in base coords)
        #   rotation = T_cam2base.R  (camera axes expressed in base)
        cam_pose_in_base = RigidTransform.from_Rt(
            cam_config.T_cam2base.R, cam_config.T_base2cam.t,
            Frame.BASE, Frame.CAMERA,
        )
        self.log_transform(
            f"{parent_path}/camera",
            cam_pose_in_base,
            axis_length=0.10,
            label="CAMERA",
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

