@echo off
REM yonggin_pasto: pallet box visualization (calibration + PCD + cam_targets)
REM Run from repo root: datasets\yonggin_pasto\run_pallet_box.bat

set SCRIPT_DIR=%~dp0
set REPO_ROOT=%SCRIPT_DIR%..\..
cd /d "%REPO_ROOT%"

python examples/visualize_pallet_box.py ^
  --data-dir "datasets/yonggin_pasto" ^
  --calibration "datasets/yonggin_pasto/picking_zone_camera.calibration.yml" ^
  --pcd "datasets/yonggin_pasto/picking_zone_camera_20260319_044652_pcd.ply" ^
  --cam-targets "datasets/yonggin_pasto/cam_targets_simple.json" ^
  --tool-z-offset 585 ^
  --tool-rotation 0 0 105 ^
  --output "datasets/yonggin_pasto/tcp_poses.json" ^
  
  %*

echo.
pause
