"""
rigid_transform_kit.viz.urdf_viewer
====================================
URDF robot visualization via Rerun.

Extends TransformVisualizer with URDF loading, joint control,
and animation capabilities.

Install: ``pip install rigid-transform-kit[viz]``
"""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Union

import numpy as np

from .visualizer import TransformVisualizer, _require_rerun

try:
    import rerun as rr
    import rerun.urdf as rr_urdf

    _HAS_RERUN = True
except ImportError:
    _HAS_RERUN = False


class UrdfVisualizer(TransformVisualizer):
    """TransformVisualizer with URDF robot support.

    Parameters
    ----------
    app_id : str
        Rerun application identifier.
    spawn : bool
        If True, spawn the Rerun viewer on init.
    """

    @staticmethod
    def _resolve_urdf_packages(
        urdf_file: Path, pkg_search_dir: Path
    ) -> Path:
        """Replace ``package://`` URIs with absolute ``file:///`` paths.

        Returns a temporary URDF file with resolved paths so that
        Rerun's data-loader (which may run in a subprocess without
        inheriting ``ROS_PACKAGE_PATH``) can locate mesh assets.
        """
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
        _require_rerun()
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
            pkg_root = urdf_file.parent.parent

        if not urdf_file.exists():
            raise FileNotFoundError(f"URDF not found: {urdf_file}")

        if package_path is not None:
            pkg_dir = Path(package_path).resolve()
        else:
            pkg_dir = pkg_root.parent

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
