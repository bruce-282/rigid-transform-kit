#!/usr/bin/env bash
# Example: run hand_eye_view.py with hand-eye calibration data
# Run from repo root: bash datasets/hand_eye_example/run_hand_eye.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$SCRIPT_DIR/../.."
cd "$REPO_ROOT"

python examples/calibration/hand_eye_view.py \
  --ply "datasets/hand_eye_example/0_-540.29_-57.19_723.23_96.60_24.34_-54.89.ply" \
  --calibration "datasets/hand_eye_example/hand_eye_cal.yml" \
  --save "datasets/hand_eye_example/hand_eye.rrd" \
  "$@"
