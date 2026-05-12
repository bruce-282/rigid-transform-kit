@echo off
REM Example: run hand_eye_view.py with hand-eye calibration data
REM Run from repo root: datasets\hand_eye_example\run_hand_eye.bat

set SCRIPT_DIR=%~dp0
set REPO_ROOT=%SCRIPT_DIR%..\..
cd /d "%REPO_ROOT%"

python examples/calibration/hand_eye_view.py ^
  --ply "datasets/hand_eye_example/0_-540.29_-57.19_723.23_96.60_24.34_-54.89.ply" ^
  --calibration "datasets/hand_eye_example/hand_eye_cal.yml" ^
  --save "datasets/hand_eye_example/hand_eye.rrd" ^
  %*

pause
