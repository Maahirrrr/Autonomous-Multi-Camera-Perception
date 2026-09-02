# Level 4 Autonomous Vehicle 360° Multi-Camera & 3D LiDAR Perception Stack

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![Tests](https://github.com/Maahirrrr/Autonomous-Multi-Camera-Perception/actions/workflows/tests.yml/badge.svg)](https://github.com/Maahirrrr/Autonomous-Multi-Camera-Perception/actions/workflows/tests.yml)
[![CPU-only friendly](https://img.shields.io/badge/GPU-optional-76B900.svg)](requirements-gpu.txt)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A high-performance, real-time Level 4 Autonomous Driving perception and visualization stack engineered with an **Apple-Tesla Glassmorphic UI/UX**, 360° surround perception, 64-beam LiDAR point cloud processing, 77GHz FMCW polar radar tracking, SAE J2735 V2X communication, and an authentic 3D Digital Twin highway physics simulator.

> Runs entirely on CPU out of the box. An NVIDIA GPU + PyTorch is optional
> and only used for an extra GPU-vs-CPU benchmark readout — see
> [Optional GPU acceleration](#optional-gpu-acceleration).

---

## 🏛️ System Architecture & File Structure

```
Autonomous-Multi-Camera-Perception/
├── run_bev_surround.py           # Master Level 4 Autonomous Cockpit GUI (60 FPS Locked)
├── config.py                     # Theme palette, layout & CLI-driven runtime settings
├── ui_widgets.py                 # Reusable HUD primitives (glass panels, glow, gradients, FPS overlay)
├── digital_twin_3d_renderer.py   # 3D Digital Twin Visualizer (Metallic paint, soft shadows, alloy wheels)
├── traffic_physics_simulator.py  # IDM car-following, MOBIL lane-change & quintic spline kinematics
├── lidar_3d_pointcloud_engine.py # 64-Beam LiDAR scanner physics & 77GHz FMCW polar radar Doppler
├── bev_transformer_engine.py     # 4-Camera IPM Homographies & log-odds occupancy grid (GPU optional)
├── multi_cam_simulator.py        # 4 Surround Cameras (Front/Left/Right/Rear) with 3-lane perspective
├── tests/                        # Automated unit test suite (12/12 tests passing)
│   ├── test_level4_perception.py # Point cloud, clustering & camera projection tests
│   ├── test_traffic_physics.py   # IDM car following & quintic polynomial spline tests
│   └── test_cinematic_systems.py # Weather attenuation, radar Doppler, V2X & particle systems
├── .github/workflows/tests.yml   # CI: runs the test suite headlessly on every push/PR
├── run.bat / run.sh              # 1-click launch scripts (Windows / macOS & Linux)
├── requirements.txt              # Core, CPU-only dependencies
├── requirements-gpu.txt          # Optional: PyTorch, for the GPU benchmark overlay
├── LICENSE                       # MIT License
└── README.md                     # Documentation & technical reference
```

---

## 🚀 Quickstart

### 1. Installation
```bash
git clone https://github.com/Maahirrrr/Autonomous-Multi-Camera-Perception.git
cd Autonomous-Multi-Camera-Perception

python -m venv .venv
# Windows:      .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Launch the Cockpit
```bash
python run_bev_surround.py
```
Or use the launch script for your platform (auto-activates `.venv` if present):
```powershell
run.bat        # Windows
```
```bash
./run.sh       # macOS / Linux
```

### 3. Command-Line Options & Flags

| Flag | Description |
| :--- | :--- |
| `--fullscreen` | Launch in fullscreen mode (HUD scales to fit automatically) |
| `--width N --height N` | Custom window resolution (default: `1280x800`) |
| `--fps N` | Target simulation frame rate (default: `60`) |
| `--seed N` | Fixed random seed for a reproducible traffic scenario |
| `--export out.mp4 --max-frames N` | Record the cockpit simulation session to an MP4 |
| `--no-fps-hud` | Hide the live performance FPS readout in the header bar |
| `--log-level DEBUG\|INFO\|WARNING\|ERROR` | Console log verbosity level (default: `INFO`) |

### 4. Run the Automated Test Suite
```bash
pytest tests/ -v
```

---

## ⚡ Optional: GPU Acceleration
The GPU-vs-CPU benchmark readout uses PyTorch and CUDA if available, but is entirely optional — the cockpit runs at 60 FPS on CPU without it. To enable it:
```bash
pip install -r requirements-gpu.txt
```

---

## 🎮 Interactive Controls

| Key | Action |
| :--- | :--- |
| **`TAB`** | Toggle between Autonomous Highway Pilot & Manual Override |
| **`W / S`** | Accelerate / Decelerate (in Manual Override mode) |
| **`A / D`** | Trigger Autonomous Left / Right Lane Change Maneuver |
| **`N`** | Toggle Night Mode (Thermal-IR Palette & Headlight Beams) |
| **`P`** | Cycle Weather Mode (`CLEAR` $\to$ `RAIN` $\to$ `FOG`) |
| **`L`** | Toggle LiDAR Point Cloud Projection on Cameras |
| **`R`** | Randomize Highway Traffic Scenario |
| **`SPACE`** | Pause / Resume Real-Time Simulation |
| **`ESC`** | Exit Cockpit |
| **`Mouse Drag`** | Orbit 3D Digital Twin Camera around Ego Tesla |

---

## 📄 License
This project is licensed under the terms of the [MIT License](LICENSE).
