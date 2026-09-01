# Level 4 Autonomous Vehicle 360° Multi-Camera & 3D LiDAR Perception Stack

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Hardware](https://img.shields.io/badge/Hardware-NVIDIA%20RTX%204070-76B900.svg)](https://www.nvidia.com/)
[![Framerate](https://img.shields.io/badge/Performance-60%20FPS%20Locked-brightgreen.svg)]()
[![Tests](https://img.shields.io/badge/Tests-7%2F7%20Passed-success.svg)]()

A high-performance, real-time Level 4 Autonomous Driving perception and simulation stack integrating **4 surround-view cameras**, a **64-beam 3D LiDAR scanner**, **Inverse Perspective Mapping (IPM)**, **Fresnel Clothoid trajectory planning**, and **MOBIL + IDM autonomous overtaking intelligence**.

---

## 🏛️ System Architecture

```
multi_cam_360_bev_occupancy/
├── run_bev_surround.py           # Master Level 4 Autonomous Cockpit GUI (60 FPS on RTX 4070)
├── digital_twin_3d_renderer.py   # 3D Digital Twin world, 3D vehicles, suspension dynamics & lighting
├── traffic_physics_simulator.py  # IDM & MOBIL overtaking state machine, Ackermann steering & suspension
├── lidar_3d_pointcloud_engine.py # 64-Beam 3D LiDAR scanner, Lambertian reflectivity & camera projections
├── bev_transformer_engine.py     # 4-Camera IPM Homographies & Fresnel Clothoid path planner
├── multi_cam_simulator.py        # 4 Surround Cameras (Front, Rear, Left, Right) with live 3D perspective
├── tests/                        # Automated unit test suite (7/7 tests passing)
│   ├── test_level4_perception.py # Point cloud, RANSAC, camera projection & BEV map tests
│   └── test_traffic_physics.py   # IDM car-following, MOBIL overtaking & quintic S-curve tests
├── run.bat                       # 1-Click Launch Script
├── requirements.txt              # Production dependency specifications
└── README.md                     # Documentation & technical reference
```

---

## 🔬 Mathematical & Physical Models

### 1. 3D LiDAR Scanner & Lambertian Reflectivity
* **64-Beam Non-Linear Elevation:**
  $$\phi_i \in [-25.0^\circ, +15.0^\circ], \quad \theta_j \in [0^\circ, 360^\circ)$$
* **Lambertian Laser Surface Reflectivity:**
  $$I = I_0 \cdot \frac{\rho \cdot \cos(\alpha)}{\max(1.0, d^2)}$$

### 2. Point-to-Pixel Camera Projection Matrix
$$\mathbf{P}_{\text{cam}} = \mathbf{R}_{\text{cam}}^T (\mathbf{P}_{\text{world}} - \mathbf{T}_{\text{cam}})$$
$$u = f_x \frac{X_{\text{cam}}}{Z_{\text{cam}}} + c_x, \quad v = f_y \frac{Y_{\text{cam}}}{Z_{\text{cam}}} + c_y$$

### 3. Fresnel Integral Clothoid Trajectory Planner
$$x(s) \approx \frac{1}{6} \dot{\kappa} s^3 + \frac{1}{2} \kappa_0 s^2 + \theta_0 s, \quad y(s) \approx s - \frac{1}{2} \int_0^s x'(t)^2 dt$$

### 4. MOBIL & IDM Autonomous Overtaking Dynamics
* **Intelligent Driver Model (IDM):**
  $$a_{\text{IDM}} = a_{\text{max}} \left[ 1 - \left(\frac{v}{v_0}\right)^4 - \left(\frac{s^*(v, \Delta v)}{s}\right)^2 \right]$$
* **Smooth Quintic Polynomial Lateral S-Curve:**
  $$x(\tau) = x_{\text{start}} + \Delta x \cdot (10\tau^3 - 15\tau^4 + 6\tau^5), \quad \tau \in [0, 1]$$

---

## 🎮 Interactive Cockpit Controls

| Key / Input | Action |
| :--- | :--- |
| **`[TAB]`** | Toggle **Autonomous Highway Pilot** $\longleftrightarrow$ **Manual Driver Override** |
| **`[R]`** | **Randomize Traffic Scenario** (Re-rolls speeds, positions & vehicle models) |
| **`[W] / [S]`** | Manual Throttle (accelerate up to $130\text{ km/h}$) / Manual Brake |
| **`[A] / [D]`** | Manual Lane Change Left / Right (initiates smooth quintic $S$-curve) |
| **`[L]`** | Toggle 3D LiDAR Laser Points overlay on camera feeds |
| **`[Mouse Drag]`** | 360° Free 3D Camera Orbit around the Ego Vehicle |
| **`[SPACE]`** | Pause / Resume simulation |
| **`[ESC]`** | Exit application |

---

## 🚀 Quickstart

### 1. Launch Cockpit Simulator
Double-click `run.bat` or run via PowerShell:
```powershell
cd C:\Users\MAHIR\Desktop\multi_cam_360_bev_occupancy
python run_bev_surround.py
```

### 2. Run Automated Verification Test Suite
```powershell
pytest tests/ -v
```
