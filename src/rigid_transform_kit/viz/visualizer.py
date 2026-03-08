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
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Union

import numpy as np

from ..core import Frame, RigidTransform

if TYPE_CHECKING:
    from ..vision import CameraConfig, PickPoint

try:
    import rerun as rr
    import rerun.urdf as rr_urdf

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
        rr.log("world", rr.Transform3D(translation=[0, 0, 0]), static=True)

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
        """Log a TCP target pose as a point with 3-axis arrows (XYZ = RGB).

        Parameters
        ----------
        T_base2tcp : RigidTransform
            TCP pose in base frame.
        parent_path : str
            Rerun entity parent path.
        axis_length : float
            Length of each axis arrow (same unit as T_base2tcp).
        label : str
            Entity name and display label.
        """
        entity = f"{parent_path}/{label}"
        t = T_base2tcp.t
        R = T_base2tcp.R

        rr.log(
            f"{entity}/origin",
            rr.Points3D(
                [t.tolist()],
                colors=[[255, 200, 50]],
                radii=[axis_length * 0.1],
                labels=[label.upper()],
            ),
        )
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
            ),
        )

    def log_tcp_poses(
        self,
        poses: Sequence[RigidTransform],
        *,
        parent_path: str = "world/picks",
        axis_length: float = 0.06,
    ) -> None:
        """Log multiple TCP poses at once."""
        for i, pose in enumerate(poses):
            self.log_tcp_pose(
                pose,
                parent_path=parent_path,
                axis_length=axis_length,
                label=f"tcp_{i}",
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

    # ---- URDF robot visualization ----

    @staticmethod
    def _resolve_urdf_packages(
        urdf_file: Path, pkg_search_dir: Path
    ) -> Path:
        """Replace ``package://`` URIs with absolute ``file:///`` paths.

        Returns a temporary URDF file with resolved paths so that
        Rerun's data-loader (which may run in a subprocess without
        inheriting ``ROS_PACKAGE_PATH``) can locate mesh assets.
        """
        import re
        import tempfile

        text = urdf_file.read_text(encoding="utf-8")

        def _replace(m: re.Match) -> str:
            pkg_name = m.group(1)
            rel_path = m.group(2)
            abs_path = (pkg_search_dir / pkg_name / rel_path).resolve()
            return str(abs_path)

        resolved = re.sub(
            r"package://([^/]+)/(.*?)(\")",
            lambda m: _replace(m) + '"',
            text,
        )

        if resolved == text:
            return urdf_file

        tmp = Path(tempfile.mktemp(suffix=".urdf", prefix="rr_"))
        tmp.write_text(resolved, encoding="utf-8")
        return tmp

    def load_urdf(
        self,
        urdf_path: Union[str, Path],
        *,
        package_path: Optional[Union[str, Path]] = None,
        static: bool = True,
    ) -> rr_urdf.UrdfTree:
        """Load a URDF file into the viewer and return the parsed tree.

        Works with any robot vendor (FANUC, UR, ABB, KUKA, ...) as long
        as a valid URDF + mesh files are provided.

        Parameters
        ----------
        urdf_path : str or Path
            Path to a ``.urdf`` file **or** a ROS package directory
            containing a ``urdf/`` subfolder.  When a directory is given
            the first ``.urdf`` file found under ``<dir>/urdf/`` is used.
        package_path : str, Path, or None
            Directory to resolve ``package://`` URIs in the URDF.
            If None, auto-detected as the parent of the package directory
            (works for standard ``<pkg>/urdf/robot.urdf`` layouts).
        static : bool
            If True, log the URDF geometry as a static (non-time-varying) resource.

        Returns
        -------
        rr.urdf.UrdfTree
            Parsed URDF tree, used for joint animation.
        """
        import os

        urdf_path = Path(urdf_path).resolve()

        if urdf_path.is_dir():
            urdf_dir = urdf_path / "urdf"
            candidates = sorted(urdf_dir.glob("*.urdf")) if urdf_dir.is_dir() else []
            if not candidates:
                candidates = sorted(urdf_path.rglob("*.urdf"))
            if not candidates:
                raise FileNotFoundError(
                    f"No .urdf file found in directory: {urdf_path}"
                )
            urdf_file = candidates[0]
            pkg_root = urdf_path
        else:
            urdf_file = urdf_path
            # Standard layout: <pkg_root>/urdf/robot.urdf
            pkg_root = urdf_file.parent.parent

        if not urdf_file.exists():
            raise FileNotFoundError(f"URDF not found: {urdf_file}")

        if package_path is not None:
            pkg_dir = Path(package_path).resolve()
        else:
            pkg_dir = pkg_root.parent

        # Also set ROS_PACKAGE_PATH for any code that reads it at runtime.
        pkg_dir_str = str(pkg_dir)
        existing = os.environ.get("ROS_PACKAGE_PATH", "")
        if pkg_dir_str not in existing.split(os.pathsep):
            os.environ["ROS_PACKAGE_PATH"] = (
                f"{pkg_dir_str}{os.pathsep}{existing}" if existing else pkg_dir_str
            )

        robot_entity_prefix = "world/robot"
        rr.log(robot_entity_prefix, rr.Transform3D(translation=[0, 0, 0]), static=True)

        resolved_urdf = self._resolve_urdf_packages(urdf_file, pkg_dir)
        try:
            rr.log_file_from_path(
                str(resolved_urdf),
                entity_path_prefix=robot_entity_prefix,
                static=static,
            )
            tree = rr_urdf.UrdfTree.from_file_path(
                str(resolved_urdf),
                entity_path_prefix=robot_entity_prefix,
            )
        finally:
            if resolved_urdf != urdf_file:
                resolved_urdf.unlink(missing_ok=True)

        self._urdf_tree = tree
        self._urdf_entity_prefix = robot_entity_prefix
        return tree

    def set_joint_angles(
        self,
        joint_angles: Dict[str, float],
        *,
        urdf_tree: Optional[rr_urdf.UrdfTree] = None,
    ) -> None:
        """Set robot joint angles and log the resulting transforms.

        Parameters
        ----------
        joint_angles : dict
            Mapping of joint name -> angle (radians).
            Joints not in the dict keep their URDF default.
        urdf_tree : UrdfTree or None
            Parsed URDF tree. If None, uses the tree from the last ``load_urdf`` call.
        """
        tree = urdf_tree or getattr(self, "_urdf_tree", None)
        if tree is None:
            raise RuntimeError("No URDF loaded. Call load_urdf() first.")

        for joint in tree.joints():
            if joint.joint_type in ("revolute", "prismatic", "continuous"):
                angle = joint_angles.get(joint.name, 0.0)
                transform = joint.compute_transform(angle, clamp=True)
                rr.log("transforms", transform)

    def animate_joints(
        self,
        joint_trajectory: List[Dict[str, float]],
        *,
        dt: float = 0.03,
        timeline: str = "robot_time",
        urdf_tree: Optional[rr_urdf.UrdfTree] = None,
    ) -> None:
        """Animate a sequence of joint angle snapshots.

        Parameters
        ----------
        joint_trajectory : list of dict
            Each element is a ``{joint_name: angle_rad}`` snapshot.
        dt : float
            Time step between snapshots (seconds).
        timeline : str
            Rerun timeline name.
        urdf_tree : UrdfTree or None
            Parsed URDF tree. If None, uses the tree from the last ``load_urdf`` call.
        """
        tree = urdf_tree or getattr(self, "_urdf_tree", None)
        if tree is None:
            raise RuntimeError("No URDF loaded. Call load_urdf() first.")

        t = 0.0
        for snapshot in joint_trajectory:
            rr.set_time(timeline, duration=t)
            self.set_joint_angles(snapshot, urdf_tree=tree)
            t += dt
