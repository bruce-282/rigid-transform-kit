#!/usr/bin/env bash
# pasto_picking: checkerboard pose visualization (RGB + optional calibration)
# Run from repo root: ./datasets/pasto_picking/run_checkerboard.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

export RERUN_PORT=9877

python examples/visualize_pick_checkerBoard.py \
  --image "datasets/pasto_picking/rgb.png" \
  --intrinsics "datasets/pasto_picking/intrinsics.json" \
  --calibration "datasets/pasto_picking/calibration_result.yml" \
  --pcd "datasets/pasto_picking/pointcloud.ply" \
  --pattern-size 7 6 \
  --square-size 30 \
  --save "datasets/pasto_picking/checkerboard_pose.rrd" \
  --output "datasets/pasto_picking/tcp_poses.json" \
  --tool-z-offset 200 \
  "$@"
