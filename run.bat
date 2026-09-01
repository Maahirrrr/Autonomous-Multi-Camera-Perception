@echo off
title Level 4 Autonomous Vehicle 360 Multi-Camera & 3D LiDAR Perception Stack (RTX 4070)
cd /d "%~dp0"
call "C:\Users\MAHIR\Desktop\tmrl_env\Scripts\activate.bat"
python run_bev_surround.py
pause
