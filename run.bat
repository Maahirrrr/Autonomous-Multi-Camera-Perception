@echo off
title Level 4 Autonomous Vehicle 360 Multi-Camera & 3D LiDAR Perception Stack
cd /d "%~dp0"

if exist ".venv\Scripts\activate.bat" (
    call ".venv\Scripts\activate.bat"
)

python run_bev_surround.py %*
pause
