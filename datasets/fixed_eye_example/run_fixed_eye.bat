@echo off
REM Example: run fixed_eye_view.py with fixed-eye (eye-to-hand) calibration data
REM Run from repo root: datasets\fixed_eye_example\run_fixed_eye.bat

set SCRIPT_DIR=%~dp0
set REPO_ROOT=%SCRIPT_DIR%..\..
cd /d "%REPO_ROOT%"

python examples/calibration/fixed_eye_view.py ^
  --ply "datasets/fixed_eye_example/1_-295.010_653.564_720.191_107.577_-4.997_-174.199.ply" ^
  --calibration "datasets/fixed_eye_example/fixed_eye_cal.yml" ^
  --save "datasets/fixed_eye_example/fixed_eye.rrd" ^
  %*

pause
