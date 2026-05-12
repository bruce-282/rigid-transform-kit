#!/usr/bin/env bash
# Example: run multi_eye_view.py with stereo point clouds
# Run from repo root: bash datasets/multi_eye_example/run_multi_eye.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$SCRIPT_DIR/../.."
cd "$REPO_ROOT"

python examples/calibration/multi_eye_view.py \
  --cam1-ply "datasets/multi_eye_example/cam1/0_314.612_915.754_-147.026_177.927_-23.441_127.045.ply" \
  --cam2-ply "datasets/multi_eye_example/cam2/0_314.612_915.754_-147.026_177.927_-23.441_127.045.ply" \
  --extrinsic "datasets/multi_eye_example/multi_eye_cal.yml" \
  --no-rgb \
  --save "datasets/multi_eye_example/multi_eye.rrd" \
  "$@"
