"""
multi_cam_simulator.py — Apple-Tesla Level 4 Surround Camera Simulator
=======================================================================
Features:
  - 4 Synchronized Surround Feeds: FRONT (Top), LEFT (Left Flank), RIGHT (Right Flank), REAR (Bottom Mirror).
  - Accurate 3-Lane Perspective Highway Geometry with Dynamic Ego Lateral Position (ego_x).
  - Protected Optical Telemetry Zone (LiDAR points never obscure bottom text).
  - Boundary-Clamped Vehicle Bounding Boxes & Tags (Zero edge truncation).
  - Photorealistic Asphalt Micro-Texture, Specular Highlights & Mountain Silhouettes.
"""

import math
import random
import numpy as np
import cv2

from lidar_3d_pointcloud_engine import Lidar3DPerceptionEngine


class MultiCameraSimulator:
    """Simulates 4 synchronized surround HDR cameras with Apple-Tesla minimalist precision."""

    def __init__(self, width: int = 380, height: int = 135):
        self.w = width
        self.h = height

        # Asphalt grain noise
        np.random.seed(42)
        self.asphalt_noise = np.random.randint(-3, 2, (self.h, self.w), dtype=np.int8)

        # Lens droplets for rain mode
        random.seed(42)
        self.droplets = [
            (random.randint(10, self.w - 10), random.randint(10, self.h - 10), random.randint(2, 4))
            for _ in range(20)
        ]

    def render_surround_views(
        self,
        frame_idx: int = 0,
        dynamic_objects: list = None,
        speed_kmh: float = 75.0,
        ego_x: float = 0.0,
        lidar_engine: Lidar3DPerceptionEngine = None,
        point_cloud: np.ndarray = None,
        render_lidar_on_cams: bool = True,
        weather_mode: str = "CLEAR",
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
                ego_x=ego_x,
                weather_mode=weather_mode,
                night_mode=night_mode
            )

            # Project 3D LiDAR point cloud returns onto camera image (Protected Telemetry Zone)
            if render_lidar_on_cams and lidar_engine is not None and point_cloud is not None:
                pts_2d = lidar_engine.project_points_to_camera(point_cloud, cam_id, self.w, self.h)
                for u, v, color, _ in pts_2d:
                    # Clip points so they never obscure the bottom telemetry bar or top indicator
                    if 18 < v < self.h - 18 and 2 < u < self.w - 2:
                        cv2.circle(cams[cam_id], (u, v), 1, color, -1)

        return cams

    def _render_camera_frame(
        self,
        cam_id: str,
        frame_idx: int,
        dynamic_objects: list,
        speed_kmh: float,
        ego_x: float,
        weather_mode: str,
        night_mode: bool
    ) -> np.ndarray:
        frame = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        horizon_y = int(self.h * 0.52)
        focal = (self.w * 0.5) / math.tan(math.radians(42.5))
        cam_h = 1.45

        # -------------------------------------------------------------
        # 1. SKY GRADIENT (Apple-Tesla Dark Obsidian Sky)
        # -------------------------------------------------------------
        if night_mode:
            top_col = (6, 8, 14)
            bot_col = (14, 20, 32)
            for y in range(horizon_y):
                r = y / max(1, horizon_y)
                frame[y, :] = tuple(int(top_col[c] + (bot_col[c] - top_col[c]) * r) for c in range(3))
        elif weather_mode == "FOG":
            frame[:horizon_y, :] = (140, 148, 158)
        elif weather_mode == "RAIN":
            for y in range(horizon_y):
                r = y / max(1, horizon_y)
                frame[y, :] = (int(24 + r * 14), int(30 + r * 14), int(42 + r * 16))
        else:
            # Clean Twilight Horizon (RGB)
            for y in range(horizon_y):
                r = y / max(1, horizon_y)
                r_val = int(8 + r * 22)
                g_val = int(14 + r * 24)
                b_val = int(28 + r * 38)
                frame[y, :] = (r_val, g_val, b_val)

        # -------------------------------------------------------------
        # 2. MOUNTAIN SILHOUETTES AT HORIZON
        # -------------------------------------------------------------
        if weather_mode != "FOG":
            pts_m1 = [(0, horizon_y)]
            for x in range(0, self.w + 4, 4):
                my = int(horizon_y - 10 - 5.5 * math.sin(x * 0.022 + (1.0 if cam_id == "REAR" else 0.0)))
                pts_m1.append((x, my))
            pts_m1.append((self.w, horizon_y))
            cv2.fillPoly(frame, [np.array(pts_m1, dtype=np.int32)], (14, 20, 32))

            pts_m2 = [(0, horizon_y)]
            for x in range(0, self.w + 4, 4):
                my = int(horizon_y - 4 - 3.5 * math.sin(x * 0.045 + 1.2))
                pts_m2.append((x, my))
            pts_m2.append((self.w, horizon_y))
            cv2.fillPoly(frame, [np.array(pts_m2, dtype=np.int32)], (20, 28, 42))

        # -------------------------------------------------------------
        # 3. ROADBED SURFACE & ASPHALT GRAIN
        # -------------------------------------------------------------
        road_h = self.h - horizon_y
        for y in range(horizon_y, self.h):
            r = (y - horizon_y) / max(1, road_h)
            r_val = int(22 - r * 8)
            g_val = int(25 - r * 9)
            b_val = int(32 - r * 10)
            frame[y, :] = (r_val, g_val, b_val)

            if y % 4 == 0:
                noise_slice = self.asphalt_noise[y, :]
                for x in range(0, self.w, 6):
                    frame[y, x:x+3] = np.clip(frame[y, x:x+3].astype(np.int16) + noise_slice[x], 0, 255).astype(np.uint8)

        if weather_mode == "RAIN":
            sheen_y = int(horizon_y + 0.65 * road_h)
            if sheen_y + 3 < self.h:
                sheen_overlay = frame[sheen_y:sheen_y+3, :].copy()
                sheen_layer = np.full_like(sheen_overlay, (60, 85, 110))
                frame[sheen_y:sheen_y+3, :] = cv2.addWeighted(sheen_layer, 0.35, sheen_overlay, 0.65, 0)

        # -------------------------------------------------------------
        # 4. ACCURATE 3-LANE PERSPECTIVE PROJECTION
        # -------------------------------------------------------------
        t_motion = (frame_idx * (speed_kmh * 0.08)) % 8.0
        yellow_divider = (255, 200, 30)
        white_line = (230, 235, 245)

        if cam_id in ("FRONT", "REAR"):
            sign_dir = 1.0 if cam_id == "FRONT" else -1.0
            u_center = self.w * 0.5

            # Solid Left Yellow Divider (X = -5.8m)
            line_pts_left = []
            for z_m in np.arange(2.0, 55.0, 2.0):
                rel_x = (-5.8 - ego_x) * sign_dir
                u_p = int(u_center + (rel_x / z_m) * focal)
                v_p = int(horizon_y + (cam_h / z_m) * focal)
                if 0 <= v_p < self.h:
                    line_pts_left.append((u_p, v_p))
            if len(line_pts_left) > 1:
                cv2.polylines(frame, [np.array(line_pts_left, dtype=np.int32)], False, yellow_divider, 2, cv2.LINE_AA)

            # Solid Right White Line (X = +5.8m)
            line_pts_right = []
            for z_m in np.arange(2.0, 55.0, 2.0):
                rel_x = (5.8 - ego_x) * sign_dir
                u_p = int(u_center + (rel_x / z_m) * focal)
                v_p = int(horizon_y + (cam_h / z_m) * focal)
                if 0 <= v_p < self.h:
                    line_pts_right.append((u_p, v_p))
            if len(line_pts_right) > 1:
                cv2.polylines(frame, [np.array(line_pts_right, dtype=np.int32)], False, white_line, 2, cv2.LINE_AA)

            # Dashed Lane Dividers (X = -1.875m and X = +1.875m)
            for lane_x in (-1.875, 1.875):
                for z_dash in np.arange(2.0 + t_motion, 50.0, 7.5):
                    rel_x = (lane_x - ego_x) * sign_dir
                    u_d1 = int(u_center + (rel_x / z_dash) * focal)
                    v_d1 = int(horizon_y + (cam_h / z_dash) * focal)
                    u_d2 = int(u_center + (rel_x / (z_dash + 3.2)) * focal)
                    v_d2 = int(horizon_y + (cam_h / (z_dash + 3.2)) * focal)
                    if v_d1 < self.h and v_d2 > horizon_y:
                        cv2.line(frame, (u_d1, v_d1), (u_d2, v_d2), white_line, 2, cv2.LINE_AA)

        elif cam_id == "LEFT":
            u_edge = int(self.w * 0.70 + (ego_x / 5.8) * 40.0)
            cv2.line(frame, (u_edge, horizon_y), (u_edge + 80, self.h), yellow_divider, 2, cv2.LINE_AA)
            for g_x in range(max(0, u_edge - 90), u_edge, 28):
                cv2.line(frame, (g_x, horizon_y + 15), (g_x, horizon_y + 40), (70, 85, 105), 2)
            cv2.line(frame, (0, horizon_y + 20), (u_edge, horizon_y + 20), (90, 105, 125), 2)

        elif cam_id == "RIGHT":
            u_edge = int(self.w * 0.30 - (ego_x / 5.8) * 40.0)
            cv2.line(frame, (u_edge, horizon_y), (u_edge - 80, self.h), white_line, 2, cv2.LINE_AA)
            for g_x in range(u_edge, min(self.w, u_edge + 90), 28):
                cv2.line(frame, (g_x, horizon_y + 15), (g_x, horizon_y + 40), (70, 85, 105), 2)
            cv2.line(frame, (u_edge, horizon_y + 20), (self.w, horizon_y + 20), (90, 105, 125), 2)

        # -------------------------------------------------------------
        # 5. DYNAMIC VEHICLES (RGB Consistent & Boundary Clamped)
        # -------------------------------------------------------------
        visible_vehicles = []
        for obj in dynamic_objects:
            ox, oz, ow, ol, label, col_rgb = obj
            in_view = False
            cam_x, cam_z = 0.0, 0.0

            if cam_id == "FRONT" and oz > 1.2:
                cam_x, cam_z = ox, oz
                in_view = True
            elif cam_id == "REAR" and oz < -1.2:
                cam_x, cam_z = -ox, -oz
                in_view = True
            elif cam_id == "LEFT" and ox < -1.2:
                cam_x, cam_z = -oz, -ox
                in_view = True
            elif cam_id == "RIGHT" and ox > 1.2:
                cam_x, cam_z = oz, ox
                in_view = True

            if in_view and cam_z > 2.0:
                u = int((self.w * 0.5) + (cam_x / cam_z) * focal)
                v = int(horizon_y + (cam_h / cam_z) * focal)
                bw = max(8, int((ow / cam_z) * focal))
                bh = max(6, int((1.65 / cam_z) * focal))
                if -bw < u < self.w + bw and horizon_y < v < self.h + bh:
                    visible_vehicles.append((cam_z, u, v, bw, bh, label, col_rgb))

        visible_vehicles.sort(key=lambda item: item[0], reverse=True)

        drawn_labels = []
        for cam_z, u, v, bw, bh, label, col_rgb in visible_vehicles:
            veh_rgb = col_rgb if col_rgb else (218, 35, 38)
            dark_rgb = tuple(int(c * 0.68) for c in veh_rgb)
            top_h = max(2, int(bh * 0.32))

            # 3D Shaded Box
            cv2.rectangle(frame, (u - bw//2, v - bh), (u + bw//2, v - bh + top_h), veh_rgb, -1)
            cv2.rectangle(frame, (u - bw//2, v - bh + top_h), (u + bw//2, v), dark_rgb, -1)
            cv2.rectangle(frame, (u - bw//2, v - bh), (u + bw//2, v), (240, 245, 255), 1)

            # Windshield Glass with glint
            ws_w = int(bw * 0.75)
            ws_h = max(2, int(top_h * 0.75))
            ws_x = u - ws_w // 2
            ws_y = v - bh + 2
            cv2.rectangle(frame, (ws_x, ws_y), (ws_x + ws_w, ws_y + ws_h), (38, 48, 65), -1)
            cv2.line(frame, (ws_x + 1, ws_y + 1), (ws_x + 3, ws_y + 1), (255, 255, 255), 1)

            # Wheels
            cv2.circle(frame, (u - bw//2 + 3, v), 3, (12, 12, 12), -1)
            cv2.circle(frame, (u + bw//2 - 3, v), 3, (12, 12, 12), -1)

            # LED Taillights / Headlights
            for tx in (u - bw//2 + 3, u + bw//2 - 3):
                cv2.circle(frame, (tx, v - bh//3), 3, (245, 35, 35), -1)

            # Bounding Bracket
            cv2.rectangle(frame, (u - bw//2 - 2, v - bh - 2), (u + bw//2 + 2, v + 2), (0, 212, 255), 1)

            # Boundary-Clamped Smart Tag
            tag_str = f"{label} [{cam_z:.1f}m]"
            tag_w = len(tag_str) * 6 + 6
            tag_x = max(6, min(self.w - tag_w - 6, u - bw // 2))
            tag_y = v - bh - 4

            for prev_x, prev_y in drawn_labels:
                if abs(tag_x - prev_x) < 70 and abs(tag_y - prev_y) < 12:
                    tag_y = prev_y - 12

            drawn_labels.append((tag_x, tag_y))
            cv2.putText(frame, tag_str, (tag_x, tag_y),
                        cv2.FONT_HERSHEY_DUPLEX, 0.25, (0, 212, 255), 1, cv2.LINE_AA)

        # -------------------------------------------------------------
        # 6. RAIN / FOG ATMOSPHERIC LAYERS
        # -------------------------------------------------------------
        if weather_mode == "RAIN":
            for _ in range(25):
                rx = random.randint(0, self.w)
                ry = random.randint(0, self.h)
                cv2.line(frame, (rx, ry), (rx - 2, ry + 6), (170, 190, 215), 1)
            for dx, dy, dr in self.droplets:
                cv2.circle(frame, (dx, dy), dr, (215, 230, 250), 1)
        elif weather_mode == "FOG":
            fog_layer = np.full_like(frame, 135)
            cv2.addWeighted(fog_layer, 0.38, frame, 0.62, 0, frame)

        # -------------------------------------------------------------
        # 7. APPLE-TESLA MINIMALIST BEZEL & ASCII TELEMETRY
        # -------------------------------------------------------------
        cv2.rectangle(frame, (0, 0), (self.w - 1, self.h - 1), (30, 35, 47), 1)

        tag_text = {
            "FRONT": "FRONT | 1080p HDR",
            "LEFT":  "LEFT | 85 deg WIDE",
            "RIGHT": "RIGHT | 85 deg WIDE",
            "REAR":  "REAR MIRROR | 1080p"
        }.get(cam_id, cam_id)

        pill_w = len(tag_text) * 7 + 14
        cv2.rectangle(frame, (4, 4), (4 + pill_w, 18), (12, 15, 22), -1)
        cv2.rectangle(frame, (4, 4), (4 + pill_w, 18), (35, 44, 60), 1)
        cv2.putText(frame, tag_text, (8, 14), cv2.FONT_HERSHEY_DUPLEX, 0.26, (190, 215, 240), 1, cv2.LINE_AA)

        if frame_idx % 60 < 30:
            cv2.circle(frame, (self.w - 12, 11), 3, (255, 59, 48), -1)
            cv2.putText(frame, "REC", (self.w - 36, 14), cv2.FONT_HERSHEY_DUPLEX, 0.25, (255, 59, 48), 1, cv2.LINE_AA)

        # Protected Telemetry Bar at Bottom
        cv2.putText(frame, "f/1.8 | 1/500s | ISO 400", (8, self.h - 6),
                    cv2.FONT_HERSHEY_DUPLEX, 0.25, (135, 148, 168), 1, cv2.LINE_AA)

        return frame
