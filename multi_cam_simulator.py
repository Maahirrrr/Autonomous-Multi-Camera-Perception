"""
multi_cam_simulator.py — Real-Time 4-Camera Surround Simulator with Cinematic Rendering
========================================================================================
Features:
  - 4 Surround Viewports: FRONT (85°), REAR (85°), LEFT (85°), RIGHT (85°).
  - Cinematic Sky Gradients & 3-Layered Mountain Silhouettes at Horizon.
  - Textured Road Surface with Wet Sheen Specular Band & Bright Lane Markings.
  - 3D-Style Shaded Vehicles: Top & Side Faces, Windshield Specular Highlights, Concentric Headlights & Wheel Wells.
  - Camera HUD: Directional Color-Coded Badges, Blinking Recording Dot, Telemetry ("f=35mm | ISO 800 | 1/500s") & Corner Vignette.
  - Projected 3D LiDAR Laser Point Cloud Overlays with Depth Heatmap Colorization.
"""

import math
import random
import numpy as np
import cv2

from lidar_3d_pointcloud_engine import Lidar3DPerceptionEngine


class MultiCameraSimulator:
    """Simulates 4 synchronized surround HDR cameras with cinematic visual rendering."""

    def __init__(self, width: int = 300, height: int = 160):
        self.w = width
        self.h = height

        # Seeded random noise for asphalt texture
        np.random.seed(42)
        self.asphalt_noise = np.random.randint(-4, 1, (self.h, self.w), dtype=np.int8)

        # Rain droplet particles for lens (rain mode)
        random.seed(42)
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
                weather_mode=weather_mode,
                night_mode=night_mode
            )

            # Overlay 3D LiDAR point cloud returns onto camera image
            if render_lidar_on_cams and lidar_engine is not None and point_cloud is not None:
                pts_2d = lidar_engine.project_points_to_camera(point_cloud, cam_id, self.w, self.h)
                for u, v, color, _ in pts_2d:
                    # OpenCV uses BGR
                    bgr_col = (color[2], color[1], color[0])
                    cv2.circle(cams[cam_id], (u, v), 1, bgr_col, -1)

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
        horizon_y = int(self.h * 0.54)

        # -------------------------------------------------------------
        # 1. SKY GRADIENT (Per Camera Direction)
        # -------------------------------------------------------------
        if night_mode:
            top_color = (16, 10, 8)
            horizon_color = (35, 22, 14)
            for y in range(horizon_y):
                r = y / max(1, horizon_y)
                col = tuple(int(top_color[c] + (horizon_color[c] - top_color[c]) * r) for c in range(3))
                frame[y, :] = col
        elif weather_mode == "FOG":
            frame[:horizon_y, :] = (160, 150, 140)
        elif weather_mode == "RAIN":
            for y in range(horizon_y):
                r = y / max(1, horizon_y)
                frame[y, :] = (int(55 + r * 15), int(45 + r * 15), int(35 + r * 15))
        else:
            if cam_id in ("FRONT", "REAR"):
                # Top: RGB(12,18,35) -> Horizon: RGB(25,35,55) -> Glow: RGB(40,55,80)
                # In BGR: Top(35,18,12), Horizon(55,35,25), Glow(80,55,40)
                glow_start_y = max(0, horizon_y - 18)
                for y in range(glow_start_y):
                    r = y / max(1, glow_start_y)
                    frame[y, :] = (int(35 + r * 20), int(18 + r * 17), int(12 + r * 13))
                for y in range(glow_start_y, horizon_y):
                    r = (y - glow_start_y) / max(1, horizon_y - glow_start_y)
                    frame[y, :] = (int(55 + r * 25), int(35 + r * 20), int(25 + r * 15))
            else:
                # LEFT / RIGHT: Top RGB(8,14,28) -> Horizon RGB(20,30,50)
                # In BGR: Top(28,14,8), Horizon(50,30,20)
                for y in range(horizon_y):
                    r = y / max(1, horizon_y)
                    frame[y, :] = (int(28 + r * 22), int(14 + r * 16), int(8 + r * 12))

        # -------------------------------------------------------------
        # 2. LAYERED MOUNTAIN SILHOUETTES AT HORIZON
        # -------------------------------------------------------------
        if weather_mode != "FOG":
            # Layer 1 (Far): y = horizon_y - 12, fill RGB(18,25,40) -> BGR(40,25,18)
            pts_m1 = [(0, horizon_y)]
            for x in range(0, self.w + 4, 4):
                my = int(horizon_y - 12 - 7.0 * math.sin(x * 0.022))
                pts_m1.append((x, my))
            pts_m1.append((self.w, horizon_y))
            cv2.fillPoly(frame, [np.array(pts_m1, dtype=np.int32)], (40, 25, 18))

            # Layer 2 (Mid): y = horizon_y - 6, fill RGB(22,32,48) -> BGR(48,32,22) (more jagged)
            pts_m2 = [(0, horizon_y)]
            for x in range(0, self.w + 4, 4):
                my = int(horizon_y - 6 - 5.0 * math.sin(x * 0.045 + 1.5) - 3.0 * math.sin(x * 0.09))
                pts_m2.append((x, my))
            pts_m2.append((self.w, horizon_y))
            cv2.fillPoly(frame, [np.array(pts_m2, dtype=np.int32)], (48, 32, 22))

            # Layer 3 (Near): y = horizon_y, fill RGB(28,38,56) -> BGR(56,38,28)
            pts_m3 = [(0, horizon_y)]
            for x in range(0, self.w + 4, 4):
                my = int(horizon_y - 2.5 * math.sin(x * 0.035 + 0.8))
                pts_m3.append((x, my))
            pts_m3.append((self.w, horizon_y))
            cv2.fillPoly(frame, [np.array(pts_m3, dtype=np.int32)], (56, 38, 28))

        # -------------------------------------------------------------
        # 3. ROAD SURFACE & ASPHALT TEXTURE
        # -------------------------------------------------------------
        # Gradient: Top RGB(28,32,38) -> Bottom RGB(18,22,28) (BGR: Top(38,32,28) -> Bottom(28,22,18))
        road_h = self.h - horizon_y
        for y in range(horizon_y, self.h):
            r = (y - horizon_y) / max(1, road_h)
            b_val = int(38 - r * 10)
            g_val = int(32 - r * 10)
            r_val = int(28 - r * 10)
            frame[y, :] = (b_val, g_val, r_val)

            # Subtle asphalt texture: every 4th row, randomly darken pixels
            if (y % 4 == 0):
                row_noise = self.asphalt_noise[y, :]
                for x in range(0, self.w, 8):
                    frame[y, x:x+4] = np.clip(frame[y, x:x+4].astype(np.int16) + row_noise[x], 0, 255).astype(np.uint8)

        # Wet sheen: horizontal specular streak at 70% down road, 3px tall, RGB(60,80,100) alpha 0.35
        sheen_y = int(horizon_y + 0.70 * road_h)
        if sheen_y + 3 < self.h:
            sheen_overlay = frame[sheen_y:sheen_y+3, :].copy()
            # BGR for RGB(60,80,100) is (100,80,60)
            sheen_layer = np.full_like(sheen_overlay, (100, 80, 60))
            frame[sheen_y:sheen_y+3, :] = cv2.addWeighted(sheen_layer, 0.35, sheen_overlay, 0.65, 0)

        # -------------------------------------------------------------
        # 4. BRIGHT HIGH-VISIBILITY LANE MARKINGS
        # -------------------------------------------------------------
        t_motion = (frame_idx * (speed_kmh * 0.08)) % 8.0
        white_line_col = (230, 225, 220) # BGR for RGB(220,225,230)
        yellow_line_col = (0, 210, 255)  # BGR for RGB(255,210,0) - Left edge divider

        if cam_id in ("FRONT", "REAR"):
            # Solid Yellow Left Edge Divider: RGB(255,210,0)
            cv2.line(frame, (int(self.w * 0.5 - 28), horizon_y), (-45, self.h), yellow_line_col, 2, cv2.LINE_AA)
            # Solid White Right Edge: RGB(220,225,230)
            cv2.line(frame, (int(self.w * 0.5 + 28), horizon_y), (self.w + 45, self.h), white_line_col, 2, cv2.LINE_AA)

            # Center Dashed White Line
            for z_dash in np.arange(2.0 + t_motion, 45.0, 7.0):
                d_f1 = (self.h * 0.50) / max(0.5, z_dash)
                d_f2 = (self.h * 0.50) / max(0.5, z_dash + 2.6)
                y_d1 = int(horizon_y + d_f1 * 2.6)
                y_d2 = int(horizon_y + d_f2 * 2.6)
                if y_d1 < self.h and y_d2 > horizon_y:
                    cv2.line(frame, (int(self.w * 0.5), y_d1), (int(self.w * 0.5), y_d2), white_line_col, 2, cv2.LINE_AA)

        # -------------------------------------------------------------
        # 5. 3D-STYLE SHADED VEHICLE RENDERING
        # -------------------------------------------------------------
        for obj in dynamic_objects:
            ox, oz, ow, ol, label, col = obj
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
                focal = (self.w * 0.5) / math.tan(math.radians(42.5))
                u = int((self.w * 0.5) + (cam_x / cam_z) * focal)
                v = int(horizon_y + (1.30 / cam_z) * focal)
                bw = max(8, int((ow / cam_z) * focal))
                bh = max(6, int((1.65 / cam_z) * focal))

                if -bw < u < self.w + bw and horizon_y < v < self.h + bh:
                    # Vehicle Base RGB -> BGR
                    base_rgb = col if col else (200, 50, 50)
                    base_bgr = (base_rgb[2], base_rgb[1], base_rgb[0])
                    dark_bgr = (int(base_bgr[0] * 0.65), int(base_bgr[1] * 0.65), int(base_bgr[2] * 0.65))

                    top_h = max(2, int(bh * 0.30))
                    bot_h = bh - top_h

                    # Stacked 3D-Shaded Box
                    # Top Face (100% brightness)
                    cv2.rectangle(frame, (u - bw//2, v - bh), (u + bw//2, v - bh + top_h), base_bgr, -1)
                    # Bottom/Side Face (65% shadow brightness)
                    cv2.rectangle(frame, (u - bw//2, v - bh + top_h), (u + bw//2, v), dark_bgr, -1)
                    cv2.rectangle(frame, (u - bw//2, v - bh), (u + bw//2, v), (255, 255, 255), 1)

                    # Windshield: RGB(50,65,85) -> BGR(85,65,50)
                    ws_w = int(bw * 0.75)
                    ws_h = max(2, int(top_h * 0.75))
                    ws_x = u - ws_w // 2
                    ws_y = v - bh + 2
                    cv2.rectangle(frame, (ws_x, ws_y), (ws_x + ws_w, ws_y + ws_h), (85, 65, 50), -1)
                    # 2-pixel white specular highlight at top-left corner
                    cv2.line(frame, (ws_x + 1, ws_y + 1), (ws_x + 3, ws_y + 1), (255, 255, 255), 1)

                    # Wheel wells: 2 small dark ellipses at bottom corners
                    cv2.ellipse(frame, (u - bw//2 + 3, v), (3, 2), 0, 0, 360, (15, 15, 15), -1)
                    cv2.ellipse(frame, (u + bw//2 - 3, v), (3, 2), 0, 0, 360, (15, 15, 15), -1)

                    # Concentric Headlights / Taillights
                    if cam_id == "FRONT" and oz < 0: # Oncoming front headlights
                        for hx in (u - bw//2 + 4, u + bw//2 - 4):
                            # Outer: r=8, RGB(255,255,200) -> BGR(200,255,255)
                            cv2.circle(frame, (hx, v - bh//3), 7, (200, 255, 255), 1)
                            # Mid: r=5, RGB(255,255,220) -> BGR(220,255,255)
                            cv2.circle(frame, (hx, v - bh//3), 4, (220, 255, 255), -1)
                            # Inner: r=2, RGB(255,255,255)
                            cv2.circle(frame, (hx, v - bh//3), 2, (255, 255, 255), -1)
                    else: # Red Taillights
                        for tx in (u - bw//2 + 3, u + bw//2 - 3):
                            cv2.circle(frame, (tx, v - bh//3), 3, (40, 40, 255), -1)
                            cv2.circle(frame, (tx, v - bh//3), 6, (20, 20, 180), 1)

                    # 2D Bounding Box Bracket & Tag
                    cv2.rectangle(frame, (u - bw//2 - 2, v - bh - 2), (u + bw//2 + 2, v + 2), (0, 255, 180), 1)
                    cv2.putText(frame, f"{label} [{cam_z:.1f}m]", (u - bw//2, v - bh - 4),
                                cv2.FONT_HERSHEY_DUPLEX, 0.28, (0, 255, 180), 1, cv2.LINE_AA)

        # -------------------------------------------------------------
        # 6. RAIN & FOG ATMOSPHERIC LAYERS
        # -------------------------------------------------------------
        if weather_mode == "RAIN":
            for _ in range(35):
                rx = random.randint(0, self.w)
                ry = random.randint(0, self.h)
                cv2.line(frame, (rx, ry), (rx - 2, ry + 6), (180, 200, 220), 1)
            for dx, dy, dr in self.droplets:
                cv2.circle(frame, (dx, dy), dr, (220, 235, 255), 1)
                cv2.circle(frame, (dx - 1, dy - 1), max(1, dr - 2), (120, 180, 255), -1)
        elif weather_mode == "FOG":
            fog_overlay = np.full_like(frame, 150)
            cv2.addWeighted(fog_overlay, 0.42, frame, 0.58, 0, frame)

        # -------------------------------------------------------------
        # 7. CORNER VIGNETTING (Semi-Transparent Dark Triangles)
        # -------------------------------------------------------------
        vig_size = 18
        vig_pts = [
            [(0, 0), (vig_size, 0), (0, vig_size)],
            [(self.w, 0), (self.w - vig_size, 0), (self.w, vig_size)],
            [(0, self.h), (vig_size, self.h), (0, self.h - vig_size)],
            [(self.w, self.h), (self.w - vig_size, self.h), (self.w, self.h - vig_size)]
        ]
        vig_mask = frame.copy()
        for poly in vig_pts:
            cv2.fillPoly(vig_mask, [np.array(poly, dtype=np.int32)], (0, 0, 0))
        cv2.addWeighted(vig_mask, 0.25, frame, 0.75, 0, frame)

        # -------------------------------------------------------------
        # 8. DIRECTIONAL COLOR-CODED CAMERA HUD & BADGES
        # -------------------------------------------------------------
        # FRONT=RGB(0,180,255)->BGR(255,180,0), REAR=RGB(255,100,0)->BGR(0,100,255)
        # LEFT=RGB(0,255,140)->BGR(140,255,0), RIGHT=RGB(200,0,255)->BGR(255,0,200)
        badge_bgr = {
            "FRONT": (255, 180, 0),
            "REAR":  (0, 100, 255),
            "LEFT":  (140, 255, 0),
            "RIGHT": (255, 0, 200)
        }.get(cam_id, (255, 180, 0))

        cv2.rectangle(frame, (0, 0), (self.w - 1, self.h - 1), badge_bgr, 1)

        # Top-Left Camera Tag
        cv2.rectangle(frame, (4, 4), (68, 18), (14, 18, 26), -1)
        cv2.rectangle(frame, (4, 4), (68, 18), badge_bgr, 1)
        cv2.putText(frame, f"CAM: {cam_id}", (7, 14), cv2.FONT_HERSHEY_DUPLEX, 0.28, badge_bgr, 1, cv2.LINE_AA)

        # Blinking 4px Recording Dot (Blinks every 30 frames)
        if (frame_idx % 60 < 30):
            cv2.circle(frame, (self.w - 12, 11), 4, (0, 0, 255), -1) # Red BGR
            cv2.putText(frame, "REC", (self.w - 38, 14), cv2.FONT_HERSHEY_DUPLEX, 0.26, (0, 0, 255), 1, cv2.LINE_AA)

        # Bottom Optical Telemetry Bar
        cv2.putText(frame, "f=35mm | ISO 800 | 1/500s", (8, self.h - 6),
                    cv2.FONT_HERSHEY_DUPLEX, 0.26, (140, 160, 180), 1, cv2.LINE_AA)

        return frame
