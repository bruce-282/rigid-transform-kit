#!/usr/bin/env bash
# yonggin_pasto: pallet box visualization (calibration + PCD + cam_targets)
# Run from repo root: ./datasets/yonggin_pasto/run_pallet_box.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

python examples/visualize_pallet_box.py \
  --data-dir "datasets/yonggin2" \
  --calibration "datasets/yonggin2/picking_zone_camera.calibration.yml" \
  --pcd "datasets/yonggin2/picking_zone_camera_20260319_044652_pcd.ply" \
  --cam-targets "datasets/yonggin2/cam_targets_simple.json" \
  --tool-z-offset 585 \
  --tool-rotation 0 0 105 \
  --output "datasets/yonggin2/tcp_poses.json" \
  "$@"
