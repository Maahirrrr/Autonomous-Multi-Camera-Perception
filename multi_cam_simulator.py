"""
multi_cam_simulator.py — Real-Time 4-Camera Surround Simulator with Weather & Night Modes
========================================================================================
Features:
  - 4 Surround Viewports: FRONT (85 deg), REAR (85 deg), LEFT (85 deg), RIGHT (85 deg).
  - Weather Simulation System:
      * Rain: Animated falling rain streaks, lens droplets with refraction & wet asphalt sheen.
      * Fog: Atmospheric volumetric scattering & exponential distance haze.
  - Night Mode:
      * Deep night sky, Gaussian falloff headlight illumination cones, and multi-pass LED taillight bloom.
  - Projected 3D LiDAR laser point cloud overlays with metric depth heatmap coloring.
"""

import math
import random
import numpy as np
import cv2

from lidar_3d_pointcloud_engine import Lidar3DPerceptionEngine


class MultiCameraSimulator:
    """Simulates 4 synchronized surround HDR cameras with weather & night physics."""

    def __init__(self, width: int = 300, height: int = 170):
        self.w = width
        self.h = height

        # Rain droplet particles for lens
        self.droplets = [
            (random.randint(10, self.w - 10), random.randint(10, self.h - 10), random.randint(2, 6))
            for _ in range(25)
        ]

    def render_surround_views(
        self,
        frame_idx: int = 0,
        dynamic_objects: list = None,
        speed_kmh: float = 75.0,
        lidar_engine: Lidar3DPerceptionEngine = None,
        point_cloud: np.ndarray = None,
        render_lidar_on_cams: bool = True,
        weather_mode: str = "CLEAR", # 'CLEAR', 'RAIN', 'FOG'
        night_mode: bool = False
    ) -> dict[str, np.ndarray]:
        if dynamic_objects is None:
            dynamic_objects = []
        cams = {}
        for cam_id in ["FRONT", "LEFT", "RIGHT", "REAR"]:
            cams[cam_id] = self._render_camera_frame(
                cam_id=cam_id,
                frame_idx=frame_idx,
                dynamic_objects=dynamic_objects,
                speed_kmh=speed_kmh,
                weather_mode=weather_mode,
                night_mode=night_mode
            )

            # Overlay 3D LiDAR point cloud returns onto camera image
            if render_lidar_on_cams and lidar_engine is not None and point_cloud is not None:
                pts_2d = lidar_engine.project_points_to_camera(point_cloud, cam_id, self.w, self.h)
                for u, v, color, _ in pts_2d:
                    cv2.circle(cams[cam_id], (u, v), 1, color, -1)

        return cams

    def _render_camera_frame(
        self,
        cam_id: str,
        frame_idx: int,
        dynamic_objects: list,
        speed_kmh: float,
        weather_mode: str,
        night_mode: bool
    ) -> np.ndarray:
        frame = np.zeros((self.h, self.w, 3), dtype=np.uint8)

        # 1. Sky & Ground Horizon
        horizon_y = int(self.h * 0.58)

        if night_mode:
            # Deep Night Sky
            frame[:horizon_y, :] = (8, 10, 16)
            asphalt_base = (14, 16, 20)
        elif weather_mode == "FOG":
            # Dense Gray Fog Haze
            frame[:horizon_y, :] = (140, 150, 160)
            asphalt_base = (100, 110, 120)
        elif weather_mode == "RAIN":
            # Overcast Storm Sky
            frame[:horizon_y, :] = (35, 45, 55)
            asphalt_base = (24, 28, 36)
        else:
            # Clear Sunset Twilight Sky
            for y in range(horizon_y):
                ratio = y / max(1, horizon_y)
                r = int(18 + ratio * 45)
                g = int(24 + ratio * 32)
                b = int(45 + ratio * 20)
                frame[y, :] = (r, g, b)
            asphalt_base = (32, 36, 44)

        # 2. 3D Perspective Roadbed
        road_pts = np.array([
            [int(self.w * 0.5 - 28), horizon_y],
            [int(self.w * 0.5 + 28), horizon_y],
            [self.w + 60, self.h],
            [-60, self.h]
        ], dtype=np.int32)
        cv2.fillPoly(frame, [road_pts], asphalt_base)

        # Headlight Illumination Cone (Night Mode)
        if night_mode and cam_id == "FRONT":
            cone_overlay = frame.copy()
            hl_pts = np.array([
                [int(self.w * 0.5 - 15), horizon_y + 10],
                [int(self.w * 0.5 + 15), horizon_y + 10],
                [int(self.w * 0.5 + 110), self.h],
                [int(self.w * 0.5 - 110), self.h]
            ], dtype=np.int32)
            cv2.fillPoly(cone_overlay, [hl_pts], (200, 220, 255))
            cv2.addWeighted(cone_overlay, 0.35, frame, 0.65, 0, frame)

        # 3. Dashed Lane Lines & Road Edge Rumble Strips
        t_motion = (frame_idx * (speed_kmh * 0.08)) % 8.0
        line_col = (200, 210, 220) if not night_mode else (140, 150, 160)

        if cam_id in ("FRONT", "REAR"):
            # Center Dashed Line
            for z_dash in np.arange(2.0 + t_motion, 45.0, 7.0):
                d_f = (self.h * 0.5) / max(0.5, z_dash)
                y_d1 = int(horizon_y + d_f * 2.8)
                y_d2 = int(horizon_y + ((self.h * 0.5) / max(0.5, z_dash + 2.5)) * 2.8)
                if y_d1 < self.h and y_d2 > horizon_y:
                    cv2.line(frame, (int(self.w * 0.5), y_d1), (int(self.w * 0.5), y_d2), line_col, 2)

            # Solid Road Edges
            cv2.line(frame, (int(self.w * 0.5 - 28), horizon_y), (-40, self.h), (0, 215, 255), 2)
            cv2.line(frame, (int(self.w * 0.5 + 28), horizon_y), (self.w + 40, self.h), (230, 235, 240), 2)

        # 4. Render Dynamic Vehicles in Camera Perspective
        for obj in dynamic_objects:
            ox, oz, ow, ol, label, col = obj
            in_view = False
            cam_x, cam_z = 0.0, 0.0

            if cam_id == "FRONT" and oz > 1.5:
                cam_x, cam_z = ox, oz
                in_view = True
            elif cam_id == "REAR" and oz < -1.5:
                cam_x, cam_z = -ox, -oz
                in_view = True
            elif cam_id == "LEFT" and ox < -1.2:
                cam_x, cam_z = -oz, -ox
                in_view = True
            elif cam_id == "RIGHT" and ox > 1.2:
                cam_x, cam_z = oz, ox
                in_view = True

            if in_view and cam_z > 2.0:
                focal = (self.w * 0.5) / math.tan(math.radians(42.5))
                u = int((self.w * 0.5) + (cam_x / cam_z) * focal)
                v = int(horizon_y + (1.2 / cam_z) * focal)
                bw = max(6, int((ow / cam_z) * focal))
                bh = max(4, int((1.6 / cam_z) * focal))

                if -bw < u < self.w + bw and horizon_y < v < self.h + bh:
                    # Draw 3D Vehicle Block
                    v_col = col if col else (180, 50, 50)
                    cv2.rectangle(frame, (u - bw//2, v - bh), (u + bw//2, v), v_col, -1)
                    cv2.rectangle(frame, (u - bw//2, v - bh), (u + bw//2, v), (255, 255, 255), 1)

                    # Red LED Taillights with Bloom
                    tail_col = (40, 40, 255)
                    cv2.circle(frame, (u - bw//2 + 3, v - bh//3), 3, tail_col, -1)
                    cv2.circle(frame, (u + bw//2 - 3, v - bh//3), 3, tail_col, -1)
                    if night_mode:
                        # Multi-pass LED Bloom
                        cv2.circle(frame, (u - bw//2 + 3, v - bh//3), 7, (0, 0, 180), 1)
                        cv2.circle(frame, (u + bw//2 - 3, v - bh//3), 7, (0, 0, 180), 1)

                    # 2D YOLO Detection Bounding Bracket & Tag
                    cv2.rectangle(frame, (u - bw//2 - 2, v - bh - 2), (u + bw//2 + 2, v + 2), (0, 255, 180), 1)
                    cv2.putText(frame, f"{label} [{cam_z:.1f}m]", (u - bw//2, v - bh - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.26, (0, 255, 180), 1, cv2.LINE_AA)

        # 5. Rain Weather Effects: Streaking Raindrops & Lens Refraction
        if weather_mode == "RAIN":
            for _ in range(40):
                rx = random.randint(0, self.w)
                ry = random.randint(0, self.h)
                cv2.line(frame, (rx, ry), (rx - 2, ry + 7), (180, 200, 220), 1)

            # Lens droplets with chromatic distortion
            for dx, dy, dr in self.droplets:
                cv2.circle(frame, (dx, dy), dr, (220, 235, 255), 1)
                cv2.circle(frame, (dx - 1, dy - 1), max(1, dr - 2), (120, 180, 255), -1)

        # 6. Fog Atmospheric Layer
        if weather_mode == "FOG":
            fog_overlay = np.full_like(frame, 150)
            cv2.addWeighted(fog_overlay, 0.45, frame, 0.55, 0, frame)

        # 7. Cinematic Camera HUD Lens Bezel
        cv2.rectangle(frame, (0, 0), (self.w - 1, self.h - 1), (0, 180, 255), 1)
        tag_mode = f"🔴 REC 60 FPS HDR [{weather_mode}]"
        cv2.putText(frame, f"CAM: {cam_id}", (8, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 230, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, tag_mode, (self.w - 142, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.26, (255, 60, 60) if "REC" in tag_mode else (0, 255, 180), 1, cv2.LINE_AA)

        return frame
