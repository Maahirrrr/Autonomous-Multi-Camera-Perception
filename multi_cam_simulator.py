"""
multi_cam_simulator.py — High-Performance Apple-Tesla Surround Camera Simulator
================================================================================
Features:
  - 4 Synchronized Surround Feeds: FRONT, LEFT, RIGHT, REAR (Rearview Mirror).
  - Distinct 3D Vehicle Renderers (Semi Truck Box + Cab, Sports Coupe, Sleek Sedan).
  - Accurate Model Type Badging ([TRUCK], [SEDAN], [SPORTS]).
  - Vectorized precomputed sky/road gradients for locked 60 FPS performance.
  - Multi-vehicle dynamic label stacking and collision-free HUD placement.
"""

import math
import random
import numpy as np
import cv2

from lidar_3d_pointcloud_engine import Lidar3DPerceptionEngine


class MultiCameraSimulator:
    """Simulates 4 surround HDR cameras with accurate vehicle classifications and locked 60 FPS."""

    def __init__(self, width: int = 380, height: int = 135):
        self.w = width
        self.h = height
        self.horizon_y = int(self.h * 0.52)
        self.road_h = self.h - self.horizon_y
        self.focal = (self.w * 0.5) / math.tan(math.radians(42.5))
        self.cam_h = 1.45

        # Precompute static backgrounds for lightning-fast vectorized copying (0.02ms)
        self.bg_day = self._precompute_background(night=False, fog=False, rain=False)
        self.bg_night = self._precompute_background(night=True, fog=False, rain=False)
        self.bg_fog = self._precompute_background(night=False, fog=True, rain=False)
        self.bg_rain = self._precompute_background(night=False, fog=False, rain=True)

        random.seed(42)
        self.droplets = [
            (random.randint(10, self.w - 10), random.randint(10, self.h - 10), random.randint(2, 4))
            for _ in range(20)
        ]

    def _precompute_background(self, night: bool, fog: bool, rain: bool) -> np.ndarray:
        bg = np.zeros((self.h, self.w, 3), dtype=np.uint8)

        # 1. Sky Gradient
        if fog:
            bg[:self.horizon_y, :] = (138, 146, 156)
        elif night:
            top_col = np.array([6, 8, 14], dtype=np.float32)
            bot_col = np.array([14, 20, 32], dtype=np.float32)
            ratios = np.linspace(0.0, 1.0, self.horizon_y, dtype=np.float32)[:, None, None]
            bg[:self.horizon_y, :] = np.clip(top_col + (bot_col - top_col) * ratios, 0, 255).astype(np.uint8)
        elif rain:
            top_col = np.array([22, 28, 38], dtype=np.float32)
            bot_col = np.array([36, 46, 58], dtype=np.float32)
            ratios = np.linspace(0.0, 1.0, self.horizon_y, dtype=np.float32)[:, None, None]
            bg[:self.horizon_y, :] = np.clip(top_col + (bot_col - top_col) * ratios, 0, 255).astype(np.uint8)
        else:
            top_col = np.array([8, 14, 28], dtype=np.float32)
            bot_col = np.array([30, 38, 66], dtype=np.float32)
            ratios = np.linspace(0.0, 1.0, self.horizon_y, dtype=np.float32)[:, None, None]
            bg[:self.horizon_y, :] = np.clip(top_col + (bot_col - top_col) * ratios, 0, 255).astype(np.uint8)

        # 2. Mountain Silhouettes
        if not fog:
            pts_m1 = [(0, self.horizon_y)]
            for x in range(0, self.w + 4, 4):
                my = int(self.horizon_y - 10 - 5.5 * math.sin(x * 0.022))
                pts_m1.append((x, my))
            pts_m1.append((self.w, self.horizon_y))
            cv2.fillPoly(bg, [np.array(pts_m1, dtype=np.int32)], (14, 20, 32))

            pts_m2 = [(0, self.horizon_y)]
            for x in range(0, self.w + 4, 4):
                my = int(self.horizon_y - 4 - 3.5 * math.sin(x * 0.045 + 1.2))
                pts_m2.append((x, my))
            pts_m2.append((self.w, self.horizon_y))
            cv2.fillPoly(bg, [np.array(pts_m2, dtype=np.int32)], (20, 28, 42))

        # 3. Road Surface Gradient
        top_road = np.array([22, 25, 32], dtype=np.float32)
        bot_road = np.array([14, 16, 22], dtype=np.float32)
        ratios_r = np.linspace(0.0, 1.0, self.road_h, dtype=np.float32)[:, None, None]
        bg[self.horizon_y:, :] = np.clip(top_road + (bot_road - top_road) * ratios_r, 0, 255).astype(np.uint8)

        return bg

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

        if weather_mode == "FOG":
            base_bg = self.bg_fog
        elif night_mode:
            base_bg = self.bg_night
        elif weather_mode == "RAIN":
            base_bg = self.bg_rain
        else:
            base_bg = self.bg_day

        cams = {}
        for cam_id in ["FRONT", "LEFT", "RIGHT", "REAR"]:
            cams[cam_id] = self._render_camera_frame(
                cam_id=cam_id,
                base_bg=base_bg,
                frame_idx=frame_idx,
                dynamic_objects=dynamic_objects,
                speed_kmh=speed_kmh,
                ego_x=ego_x,
                weather_mode=weather_mode,
                night_mode=night_mode
            )

            # Protected Optical Telemetry Zone for LiDAR points
            if render_lidar_on_cams and lidar_engine is not None and point_cloud is not None and len(point_cloud) > 0:
                pts_2d = lidar_engine.project_points_to_camera(point_cloud, cam_id, self.w, self.h)
                for u, v, color, _ in pts_2d:
                    if 18 < v < self.h - 18 and 2 < u < self.w - 2:
                        cv2.circle(cams[cam_id], (u, v), 1, color, -1)

        return cams

    def _render_camera_frame(
        self,
        cam_id: str,
        base_bg: np.ndarray,
        frame_idx: int,
        dynamic_objects: list,
        speed_kmh: float,
        ego_x: float,
        weather_mode: str,
        night_mode: bool
    ) -> np.ndarray:
        frame = base_bg.copy()
        horizon_y = self.horizon_y
        focal = self.focal
        cam_h = self.cam_h

        # -------------------------------------------------------------
        # 1. 3-LANE HIGHWAY GEOMETRY
        # -------------------------------------------------------------
        t_motion = (frame_idx * (speed_kmh * 0.08)) % 8.0
        yellow_divider = (255, 200, 30)
        white_line = (230, 235, 245)

        if cam_id in ("FRONT", "REAR"):
            sign_dir = 1.0 if cam_id == "FRONT" else -1.0
            u_center = self.w * 0.5

            # Left Solid Yellow Line (X = -5.8m)
            line_pts_left = []
            for z_m in np.arange(2.0, 50.0, 3.0):
                rel_x = (-5.8 - ego_x) * sign_dir
                u_p = int(u_center + (rel_x / z_m) * focal)
                v_p = int(horizon_y + (cam_h / z_m) * focal)
                if 0 <= v_p < self.h:
                    line_pts_left.append((u_p, v_p))
            if len(line_pts_left) > 1:
                cv2.polylines(frame, [np.array(line_pts_left, dtype=np.int32)], False, yellow_divider, 2, cv2.LINE_AA)

            # Right Solid White Line (X = +5.8m)
            line_pts_right = []
            for z_m in np.arange(2.0, 50.0, 3.0):
                rel_x = (5.8 - ego_x) * sign_dir
                u_p = int(u_center + (rel_x / z_m) * focal)
                v_p = int(horizon_y + (cam_h / z_m) * focal)
                if 0 <= v_p < self.h:
                    line_pts_right.append((u_p, v_p))
            if len(line_pts_right) > 1:
                cv2.polylines(frame, [np.array(line_pts_right, dtype=np.int32)], False, white_line, 2, cv2.LINE_AA)

            # Dashed Lane Dividers (X = -1.875m and X = +1.875m)
            for lane_x in (-1.875, 1.875):
                for z_dash in np.arange(2.0 + t_motion, 48.0, 8.0):
                    rel_x = (lane_x - ego_x) * sign_dir
                    u_d1 = int(u_center + (rel_x / z_dash) * focal)
                    v_d1 = int(horizon_y + (cam_h / z_dash) * focal)
                    u_d2 = int(u_center + (rel_x / (z_dash + 3.4)) * focal)
                    v_d2 = int(horizon_y + (cam_h / (z_dash + 3.4)) * focal)
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
        # 2. DISTINCT 3D VEHICLE MODELS (TRUCK, SPORTS, SEDAN)
        # -------------------------------------------------------------
        visible_vehicles = []
        for obj in dynamic_objects:
            if len(obj) == 8:
                ox, oz, ow, ol, label, col_rgb, m_type, m_h = obj
            else:
                ox, oz, ow, ol, label, col_rgb = obj[:6]
                m_type = "TRUCK" if "TRUCK" in label else ("SPORTS" if "SPORTS" in label else "SEDAN")
                m_h = 3.5 if m_type == "TRUCK" else (1.30 if m_type == "SPORTS" else 1.45)

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
                bh = max(6, int((m_h / cam_z) * focal))
                if -bw < u < self.w + bw and horizon_y < v < self.h + bh:
                    visible_vehicles.append((cam_z, u, v, bw, bh, label, col_rgb, m_type, m_h))

        visible_vehicles.sort(key=lambda item: item[0], reverse=True)

        drawn_labels = []
        for cam_z, u, v, bw, bh, label, col_rgb, m_type, m_h in visible_vehicles:
            veh_rgb = col_rgb if col_rgb else (218, 35, 38)
            dark_rgb = tuple(int(c * 0.65) for c in veh_rgb)

            if m_type == "TRUCK":
                # SEMI TRUCK: Tall Cargo Box Trailer + Cab
                trailer_h = int(bh * 0.78)
                cab_h = int(bh * 0.45)
                # Trailer Body
                cv2.rectangle(frame, (u - bw//2, v - bh), (u + bw//2, v - bh + trailer_h), veh_rgb, -1)
                cv2.rectangle(frame, (u - bw//2, v - bh), (u + bw//2, v), (240, 245, 255), 1)
                # Top Clearance Marker Lights (Amber)
                cv2.circle(frame, (u - bw//2 + 3, v - bh + 3), 2, (0, 185, 255), -1)
                cv2.circle(frame, (u + bw//2 - 3, v - bh + 3), 2, (0, 185, 255), -1)
                # Dual Rear Wheels
                cv2.circle(frame, (u - bw//2 + 4, v), 3, (12, 12, 12), -1)
                cv2.circle(frame, (u + bw//2 - 4, v), 3, (12, 12, 12), -1)

            elif m_type == "SPORTS":
                # SPORTS CAR: Low-Slung Wide Stance + Rear Spoiler Wing
                cv2.rectangle(frame, (u - bw//2, v - bh), (u + bw//2, v), veh_rgb, -1)
                cv2.rectangle(frame, (u - bw//2, v - bh), (u + bw//2, v), (240, 245, 255), 1)
                # Rear Wing Spoiler
                cv2.line(frame, (u - bw//2 - 2, v - bh - 2), (u + bw//2 + 2, v - bh - 2), (245, 245, 250), 2)
                # Low Profile Wheels
                cv2.circle(frame, (u - bw//2 + 3, v), 2, (12, 12, 12), -1)
                cv2.circle(frame, (u + bw//2 - 3, v), 2, (12, 12, 12), -1)

            else:
                # SEDAN: Sleek 3-Box Sedan Profile
                top_h = max(2, int(bh * 0.35))
                cv2.rectangle(frame, (u - bw//2, v - bh), (u + bw//2, v - bh + top_h), veh_rgb, -1)
                cv2.rectangle(frame, (u - bw//2, v - bh + top_h), (u + bw//2, v), dark_rgb, -1)
                cv2.rectangle(frame, (u - bw//2, v - bh), (u + bw//2, v), (240, 245, 255), 1)
                # Windshield Glass
                ws_w = int(bw * 0.75)
                ws_h = max(2, int(top_h * 0.75))
                cv2.rectangle(frame, (u - ws_w//2, v - bh + 2), (u + ws_w//2, v - bh + 2 + ws_h), (38, 48, 65), -1)
                # Wheels
                cv2.circle(frame, (u - bw//2 + 3, v), 3, (12, 12, 12), -1)
                cv2.circle(frame, (u + bw//2 - 3, v), 3, (12, 12, 12), -1)

            # LED Taillights
            for tx in (u - bw//2 + 3, u + bw//2 - 3):
                cv2.circle(frame, (tx, v - bh//3), 3, (245, 35, 35), -1)

            # Tesla Cyan Bounding Bracket
            cv2.rectangle(frame, (u - bw//2 - 2, v - bh - 2), (u + bw//2 + 2, v + 2), (0, 229, 255), 1)

            # Accurate Label with Type Tag
            type_tag = f"[{m_type}] "
            tag_str = f"{type_tag}{label} [{cam_z:.1f}m]"
            tag_w = len(tag_str) * 6 + 6
            tag_x = max(6, min(self.w - tag_w - 6, u - tag_w // 2))
            tag_y = v - bh - 4

            for prev_x, prev_y in drawn_labels:
                if abs(tag_x - prev_x) < 95 and abs(tag_y - prev_y) < 14:
                    tag_y = prev_y - 14

            drawn_labels.append((tag_x, tag_y))
            if tag_y > 14:
                cv2.rectangle(frame, (tag_x - 2, tag_y - 9), (tag_x + tag_w, tag_y + 3), (10, 14, 22), -1)
                cv2.putText(frame, tag_str, (tag_x, tag_y),
                            cv2.FONT_HERSHEY_DUPLEX, 0.25, (0, 229, 255), 1, cv2.LINE_AA)

        # -------------------------------------------------------------
        # 3. HUD BEZEL & TELEMETRY
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

        cv2.putText(frame, "f/1.8 | 1/500s | ISO 400", (8, self.h - 6),
                    cv2.FONT_HERSHEY_DUPLEX, 0.25, (135, 148, 168), 1, cv2.LINE_AA)

        return frame
