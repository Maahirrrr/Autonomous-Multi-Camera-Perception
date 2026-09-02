# Level 4 Autonomous Vehicle 360° Multi-Camera & BEV Perception Stack

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![Tests](https://github.com/Maahirrrr/Autonomous-Multi-Camera-Perception/actions/workflows/tests.yml/badge.svg)](https://github.com/Maahirrrr/Autonomous-Multi-Camera-Perception/actions/workflows/tests.yml)
[![CPU-only friendly](https://img.shields.io/badge/GPU-optional-76B900.svg)](requirements-gpu.txt)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A high-performance, real-time Level 4 Autonomous Driving perception and visualization stack engineered with an **Apple-Tesla Glassmorphic UI/UX**, 360° surround multi-camera perception, 64-beam LiDAR point cloud processing, top-down Bird's-Eye View (BEV) occupancy mapping, and an authentic 3D Digital Twin highway physics simulator.

> Runs entirely on CPU out of the box at **60.0 FPS**. An NVIDIA GPU + PyTorch is optional and used only for GPU-accelerated IPM homography benchmarking.

---

## 🏛️ System Architecture & File Structure

```
Autonomous-Multi-Camera-Perception/
├── run_bev_surround.py           # Master Level 4 Autonomous Cockpit GUI (Locked 60 FPS)
├── config.py                     # Theme palette constants, layout & CLI argument parser
├── ui_widgets.py                 # Reusable HUD primitives (glass panels, glow, gradients, FPS)
├── digital_twin_3d_renderer.py   # 3D Digital Twin visualizer (Distinct 3D models, smooth orbit)
├── bev_transformer_engine.py     # Top-down BEV occupancy grid & IPM spatial fusion
├── traffic_physics_simulator.py  # IDM car-following, MOBIL lane changes & collision-free kinematics
├── lidar_3d_pointcloud_engine.py # 64-Beam LiDAR scanner physics & atmospheric attenuation
├── multi_cam_simulator.py        # 4 Surround HDR Cameras (Front, Left, Right, Rear Mirror)
├── tests/                        # Automated unit test suite (12/12 passing)
│   ├── test_level4_perception.py # Point cloud, ground segmentation & projection tests
│   ├── test_traffic_physics.py   # IDM car following & quintic polynomial spline tests
│   └── test_cinematic_systems.py # Atmospheric weather attenuation & particle physics tests
├── .github/workflows/tests.yml   # CI: Automated headless test execution on every push
├── run.bat / run.sh              # 1-click cross-platform launch scripts (Windows / macOS / Linux)
├── requirements.txt              # Core, lightweight CPU dependencies
├── requirements-gpu.txt          # Optional: PyTorch for GPU homography benchmarks
├── .gitignore                    # Git cleanliness rules for caches & media
├── LICENSE                       # MIT License
└── README.md                     # Technical reference & documentation
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

### 2. Launch Cockpit
```bash
python run_bev_surround.py
```
Or use the 1-click platform scripts:
```powershell
run.bat        # Windows
```
```bash
./run.sh       # macOS / Linux
```

### 3. Command-Line Options

| Flag | Description |
| :--- | :--- |
| `--fullscreen` | Launch in fullscreen mode |
| `--width N --height N` | Custom window resolution (default: `1280x800`) |
| `--fps N` | Target simulation frame rate (default: `60`) |
| `--seed N` | Fixed random seed for reproducible scenario generation |
| `--export out.mp4 --max-frames N` | Record and export simulation session to MP4 |
| `--no-fps-hud` | Hide live FPS readout in the header |
| `--log-level LEVEL` | Set log verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |

### 4. Run Automated Tests
```bash
pytest tests/ -v
```

---

## 🎮 Interactive Controls

| Key | Action |
| :--- | :--- |
| **`TAB`** | Toggle Autonomous Highway Pilot vs Manual Override |
| **`W / S`** | Accelerate / Decelerate (Manual Mode) |
| **`A / D`** | Execute Autonomous Left / Right Lane Change |
| **`N`** | Toggle Night FLIR Mode (Thermal-IR Palette) |
| **`P`** | Cycle Atmospheric Weather (`CLEAR` $\to$ `RAIN` $\to$ `FOG`) |
| **`L`** | Toggle LiDAR Point Cloud Projection on Cameras |
| **`R`** | Randomize Traffic Participants & Scenario |
| **`SPACE`** | Pause / Resume Simulation |
| **`ESC`** | Exit Cockpit |
| **`Mouse Drag`** | Orbit 3D Digital Twin Camera |

---

## 🔬 Core Mathematical & Engineering Models

1. **Intelligent Driver Model (IDM) Longitudinal Control:**
   $$a = a_{\text{max}} \left[ 1 - \left(\frac{v}{v_0}\right)^\delta - \left(\frac{s^*(v, \Delta v)}{s}\right)^2 \right]$$
2. **5th-Order Quintic Polynomial Lateral Trajectories:**
   $$x(\tau) = x_0 + \Delta x \left(10\tau^3 - 15\tau^4 + 6\tau^5\right), \quad \tau \in [0, 1]$$
3. **Beer-Lambert Atmospheric Weather Scattering:**
   $$I(d) = I_0 \cdot \exp(-\gamma_{\text{weather}} \cdot d)$$
4. **Log-Odds Probabilistic Occupancy Grid:**
   $$L_t(x, y) = \lambda \cdot L_{t-1}(x, y) + l_{\text{sensor}}(x, y) - l_0$$

---

## 📄 License
Licensed under the [MIT License](LICENSE).
