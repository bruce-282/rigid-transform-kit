#!/usr/bin/env bash
# Example: run multi_eye_view.py with stereo point clouds
# Run from repo root: bash datasets/multi_eye_example/run_multi_eye.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$SCRIPT_DIR/../.."
cd "$REPO_ROOT"

python examples/multi_eye_view.py \
  --cam1-ply "datasets/multi_eye_example/cam1/cam1.ply" \
  --cam2-ply "datasets/multi_eye_example/cam2/cam2.ply" \
  --extrinsic "datasets/multi_eye_example/multi_eye_cal.yml" \
  --no-rgb \
  --save "datasets/multi_eye_example/multi_eye.rrd" \
  "$@"
