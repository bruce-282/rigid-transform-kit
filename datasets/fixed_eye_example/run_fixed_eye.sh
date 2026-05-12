#!/usr/bin/env bash
# Example: run fixed_eye_view.py with fixed-eye (eye-to-hand) calibration data
# Run from repo root: bash datasets/fixed_eye_example/run_fixed_eye.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$SCRIPT_DIR/../.."
cd "$REPO_ROOT"

python examples/calibration/fixed_eye_view.py \
  --ply "datasets/fixed_eye_example/1_-295.010_653.564_720.191_107.577_-4.997_-174.199.ply" \
  --calibration "datasets/fixed_eye_example/fixed_eye_cal.yml" \
  --save "datasets/fixed_eye_example/fixed_eye.rrd" \
  "$@"
