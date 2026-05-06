@echo off
REM Example: run multi_eye_view.py with stereo point clouds
REM Run from repo root: datasets\multi_eye_example\run_multi_eye.bat

set SCRIPT_DIR=%~dp0
set REPO_ROOT=%SCRIPT_DIR%..\..
cd /d "%REPO_ROOT%"

python examples/multi_eye_view.py ^
  --cam1-ply "datasets/multi_eye_example/cam1/cam1.ply" ^
  --cam2-ply "datasets/multi_eye_example/cam2/cam2.ply" ^
  --extrinsic "datasets/multi_eye_example/multi_eye_cal.yml" ^
  --no-rgb ^
  --save "datasets/multi_eye_example/multi_eye.rrd" ^
  %*

pause
