@echo off
REM pasto_picking: checkerboard pose visualization (RGB + optional calibration)
REM Run from repo root: datasets\pasto_picking\run_checkerboard.bat

set SCRIPT_DIR=%~dp0
set REPO_ROOT=%SCRIPT_DIR%..\..
cd /d "%REPO_ROOT%"

set RERUN_PORT=9877

python examples/visualize_pick_checkerBoard.py ^
  --image "datasets/pasto_picking/rgb.png" ^
  --intrinsics "datasets/pasto_picking/intrinsics.json" ^
  --calibration "datasets/pasto_picking/calibration_result.yml" ^
  --pcd "datasets/pasto_picking/pointcloud.ply" ^
  --pattern-size 7 6 ^
  --square-size 30 ^
  --save "datasets/pasto_picking/checkerboard_pose.rrd" ^
  --output "datasets/pasto_picking/tcp_poses.json" ^
  --tool-z-offset 200 ^
  %*

echo.
pause
