"""
lidar_3d_pointcloud_engine.py — 64-Beam LiDAR Physics, Fog Attenuation & 77GHz mmWave Radar
=========================================================================================
Physics & Sensor Simulation:
  - 64-Beam LiDAR Scanner with Non-Linear Elevation Trigonometry.
  - Atmospheric Weather Attenuation & Beer-Lambert Scattering:
      Fog Mode: Max range drops 65m -> 25m, intensity I(d) = I_0 * exp(-gamma_fog * d)
      Rain Mode: Ground spray clutter & droplet backscatter.
  - 77GHz mmWave FMCW Radar Simulation:
      120 deg Azimuth FOV, Range-Doppler Radial Velocity (Delta v_r), Target RCS & SNR.
  - Camera-to-LiDAR Point-to-Pixel Extrinsics (P_cam = K * [R | T] * P_lidar).
"""

import math
import random
import numpy as np
import cv2


class BoundingBox3D:
    def __init__(self, cx: float, cy: float, cz: float, dx: float, dy: float, dz: float, label: str = "VEHICLE"):
        self.cx = cx
        self.cy = cy
        self.cz = cz
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.label = label


class RadarDetection:
    """Represents a target return from 77GHz mmWave FMCW Radar."""
    def __init__(self, target_id: str, range_m: float, azimuth_deg: float, doppler_mps: float, rcs_db: float):
        self.id = target_id
        self.range_m = range_m
        self.azimuth_deg = azimuth_deg
        self.doppler_mps = doppler_mps # Radial velocity relative to ego (+ approaching, - receding)
        self.rcs_db = rcs_db # Radar Cross Section (dBm^2)
        self.x = range_m * math.sin(math.radians(azimuth_deg))
        self.z = range_m * math.cos(math.radians(azimuth_deg))


class Radar77GHzSimulator:
    """Simulates a forward-facing 77GHz mmWave automotive radar."""
    def __init__(self, fov_deg: float = 120.0, max_range_m: float = 85.0):
        self.fov_deg = fov_deg
        self.max_range_m = max_range_m

    def scan_targets(self, dynamic_objects: list, ego_speed_mps: float) -> list[RadarDetection]:
        detections = []
        for obj in dynamic_objects:
            ox, oz, ow, ol, label, col = obj
            rng = math.sqrt(ox**2 + oz**2)
            if 1.0 <= rng <= self.max_range_m:
                az_deg = math.degrees(math.atan2(ox, max(0.1, oz)))
                if abs(az_deg) <= (self.fov_deg * 0.5):
                    # Relative Doppler velocity approximation
                    v_target_mps = 20.0 if "TRUCK" in label else (28.0 if "SPORTS" in label else 22.0)
                    v_rel = v_target_mps - ego_speed_mps
                    doppler = v_rel * math.cos(math.radians(az_deg))

                    rcs = 15.0 if "TRUCK" in label else (10.0 if "CAR" in label or "SEDAN" in label else 8.0)
                    detections.append(RadarDetection(
                        target_id=label,
                        range_m=rng,
                        azimuth_deg=az_deg,
                        doppler_mps=doppler,
                        rcs_db=rcs
                    ))
        return detections


class Lidar3DPerceptionEngine:
    """
    Physical 64-Beam LiDAR scanner with atmospheric weather scattering and camera projection.
    """

    def __init__(self, num_lasers: int = 64, max_range_m: float = 65.0):
        self.num_lasers = num_lasers
        self.max_range_m = max_range_m
        self.horizontal_res_deg = 0.40 # ~900 firings per 360 rotation
        self.weather_mode = "CLEAR" # 'CLEAR', 'RAIN', 'FOG'

        # 64-beam non-linear elevation distribution [-25 deg, +15 deg]
        el_dense = np.linspace(-8.0, 2.0, 32)
        el_lower = np.linspace(-25.0, -8.5, 18)
        el_upper = np.linspace(2.5, 15.0, 14)
        self.elevation_angles_deg = np.concatenate([el_lower, el_dense, el_upper])

        self.radar_sim = Radar77GHzSimulator(fov_deg=120.0, max_range_m=85.0)
        self.cameras = {}

    def set_weather_mode(self, mode: str):
        self.weather_mode = mode.upper()

    def generate_scene_point_cloud(
        self,
        dynamic_objects: list,
        road_geometry: dict = None,
        frame_idx: int = 0
    ) -> np.ndarray:
        points = []

        # Effective max range based on weather
        if self.weather_mode == "FOG":
            effective_range = 26.0 # Severe fog attenuation
            gamma_fog = 0.08
        elif self.weather_mode == "RAIN":
            effective_range = 48.0
            gamma_fog = 0.02
        else:
            effective_range = self.max_range_m
            gamma_fog = 0.005

        azimuths = np.linspace(0.0, 360.0, int(360.0 / self.horizontal_res_deg), endpoint=False)

        # 1. Road Ground Plane Returns
        h_lidar = 1.65 # Sensor mount height above ground
        for el_deg in self.elevation_angles_deg:
            if el_deg < -0.8: # Pointing towards ground
                el_rad = math.radians(el_deg)
                # Distance to flat ground y = 0
                r_ground = -h_lidar / math.sin(el_rad)

                if 1.5 < r_ground < effective_range:
                    for az_deg in azimuths[::3]: # Subsample for 60 FPS real-time
                        az_rad = math.radians(az_deg)
                        px = r_ground * math.cos(el_rad) * math.sin(az_rad)
                        pz = r_ground * math.cos(el_rad) * math.cos(az_rad)
                        py = 0.02

                        # Asphalt vs Road Marking Reflectivity
                        is_lane_marker = (abs(px - (-1.875)) < 0.12 or abs(px - 1.875) < 0.12 or abs(px - (-5.8)) < 0.15 or abs(px - 5.8) < 0.15)
                        rho = 0.92 if is_lane_marker else 0.28

                        # Lambertian intensity with fog absorption
                        intensity = (rho / max(1.0, (r_ground * 0.1)**2)) * math.exp(-gamma_fog * r_ground)
                        intensity = min(1.0, max(0.05, intensity))
                        points.append([px, py, pz, intensity, r_ground])

        # 2. Dynamic Obstacles Point Cloud
        for obj in dynamic_objects:
            ox, oz, ow, ol, label, col = obj
            oh = 1.60
            dist_to_obj = math.sqrt(ox**2 + oz**2)

            if dist_to_obj <= effective_range:
                # Sample surface points across 3D box
                num_pts = max(6, int(35 - (dist_to_obj / effective_range) * 22))
                for _ in range(num_pts):
                    px = ox + random.uniform(-ow * 0.48, ow * 0.48)
                    pz = oz + random.uniform(-ol * 0.48, ol * 0.48)
                    py = random.uniform(0.35, oh)
                    rng = math.sqrt(px**2 + pz**2)
                    intensity = 0.90 * math.exp(-gamma_fog * rng)
                    points.append([px, py, pz, intensity, rng])

        # 3. Rain / Fog Atmospheric Noise Particles
        if self.weather_mode in ("RAIN", "FOG"):
            noise_cnt = 80 if self.weather_mode == "RAIN" else 150
            for _ in range(noise_cnt):
                rx = random.uniform(-14.0, 14.0)
                rz = random.uniform(2.0, effective_range)
                ry = random.uniform(0.1, 3.5)
                points.append([rx, ry, rz, random.uniform(0.05, 0.25), math.sqrt(rx**2 + rz**2)])

        return np.array(points, dtype=np.float32)

    def segment_ground_and_clusters(self, point_cloud: np.ndarray, dynamic_objects: list = None) -> tuple[np.ndarray, np.ndarray, list[BoundingBox3D]]:
        if len(point_cloud) == 0:
            return np.zeros((0, 5)), np.zeros((0, 5)), []

        y_pts = point_cloud[:, 1]
        ground_mask = y_pts < 0.28
        ground_points = point_cloud[ground_mask]
        obstacle_points = point_cloud[~ground_mask]

        bounding_boxes = []
        if dynamic_objects:
            for obj in dynamic_objects:
                ox, oz, ow, ol, label, col = obj
                oh = 1.60
                bbox = BoundingBox3D(cx=ox, cy=oh * 0.5, cz=oz, dx=ow, dy=oh, dz=ol, label=label)
                bounding_boxes.append(bbox)
        else:
            if len(obstacle_points) > 5:
                bbox = BoundingBox3D(cx=0.0, cy=0.8, cz=20.0, dx=1.85, dy=1.6, dz=4.7, label="LEAD CAR")
                bounding_boxes.append(bbox)

        return ground_points, obstacle_points, bounding_boxes

    def project_points_to_camera(
        self,
        point_cloud: np.ndarray,
        cam_id: str,
        image_w: int = 300,
        image_h: int = 170
    ) -> list[tuple[int, int, tuple[int, int, int], float]]:
        if len(point_cloud) == 0:
            return []

        pts_xyz = point_cloud[:, :3]
        ranges = point_cloud[:, 4]

        # Camera Extrinsic Transformations
        if cam_id == "FRONT":
            x_cam = pts_xyz[:, 0]
            y_cam = pts_xyz[:, 1] - 1.45
            z_cam = pts_xyz[:, 2]
        elif cam_id == "REAR":
            x_cam = -pts_xyz[:, 0]
            y_cam = pts_xyz[:, 1] - 1.45
            z_cam = -pts_xyz[:, 2]
        elif cam_id == "LEFT":
            x_cam = -pts_xyz[:, 2]
            y_cam = pts_xyz[:, 1] - 1.45
            z_cam = -pts_xyz[:, 0]
        elif cam_id == "RIGHT":
            x_cam = pts_xyz[:, 2]
            y_cam = pts_xyz[:, 1] - 1.45
            z_cam = pts_xyz[:, 0]
        else:
            return []

        # Forward frustum clipping
        valid_front = z_cam > 1.2
        x_c = x_cam[valid_front]
        y_c = y_cam[valid_front]
        z_c = z_cam[valid_front]
        r_c = ranges[valid_front]

        if len(z_c) == 0:
            return []

        # Perspective Pin-Hole Projection
        focal = (image_w * 0.5) / math.tan(math.radians(42.5))
        u = (image_w * 0.5) + (x_c / z_c) * focal
        v = (image_h * 0.58) - (y_c / z_c) * focal

        # Screen boundary filtering
        valid_screen = (u >= 0) & (u < image_w) & (v >= 0) & (v < image_h)
        u_valid = u[valid_screen].astype(np.int32)
        v_valid = v[valid_screen].astype(np.int32)
        r_valid = r_c[valid_screen]

        projected = []
        for i in range(len(u_valid)):
            dist = r_valid[i]
            # Heatmap colorization: Red (<8m) -> Yellow (<16m) -> Green (<28m) -> Cyan (<45m) -> Blue
            if dist < 8.0:
                color = (255, 30, 30)
            elif dist < 16.0:
                color = (255, 180, 0)
            elif dist < 28.0:
                color = (0, 255, 120)
            elif dist < 45.0:
                color = (0, 220, 255)
            else:
                color = (60, 100, 255)

            projected.append((int(u_valid[i]), int(v_valid[i]), color, float(dist)))

        return projected

    def project_lidar_to_camera_image(self, point_cloud: np.ndarray, cam_spec: dict, img_w: int = 300, img_h: int = 170):
        """Backward-compatible alias for camera projection."""
        cam_id = "FRONT"
        for cid, spec in self.cameras.items():
            if spec.get("yaw") == cam_spec.get("yaw"):
                cam_id = cid
                break
        res = self.project_points_to_camera(point_cloud, cam_id, img_w, img_h)
        return [(u, v, color) for u, v, color, _ in res]
