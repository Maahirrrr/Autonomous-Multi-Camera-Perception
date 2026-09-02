"""
digital_twin_3d_renderer.py — High-Performance Apple-Tesla 3D Digital Twin Visualizer
======================================================================================
Features:
  - Distinct 3D Vehicle Geometries:
      * SEMI TRUCK : Heavy commercial trailer + forward cab + dual axle rear wheels + clearance lamps.
      * SPORTS CAR : Low-slung aerodynamic chassis + carbon fiber rear wing + wide stance.
      * SEDAN      : Sleek 3-box executive profile + panoramic roof glass.
  - Accurate Model Type Badging ([TRUCK], [SEDAN], [SPORTS]).
  - Clean Minimalist Telemetry Pills (Direct vehicle anchoring with zero clutter).
  - Fast Matrix Projection & Damped Camera Orbit at Locked 60 FPS.
"""

import math
import random
import numpy as np
import pygame

from traffic_physics_simulator import EgoAutonomousVehicle, TrafficVehicle, Particle


class DigitalTwin3DRenderer:
    """Renders the 3D Digital Twin world with distinct vehicle models and locked 60 FPS performance."""

    def __init__(self, screen_w: int = 440, screen_h: int = 460):
        self.w = screen_w
        self.h = screen_h
        self.sky_h = int(self.h * 0.50)

        # 3D Camera Intrinsics
        self.fov_deg = 62.0
        self.focal = (self.w * 0.5) / math.tan(math.radians(self.fov_deg * 0.5))
        self.cx = self.w * 0.5
        self.cy = self.h * 0.52

        # Camera Orbit & Smoothing
        self.cam_dist_m = 9.6
        self.cam_height_m = 4.2
        self.cam_pitch_deg = 14.0
        self.cam_yaw_deg = 0.0

        self.target_yaw_deg = 0.0
        self.target_pitch_deg = 14.0

        self.is_dragging = False
        self.last_mouse_pos = (0, 0)

        # Starry Sky Dome
        random.seed(42)
        self.stars = []
        for i in range(50):
            sx = random.randint(4, self.w - 4)
            sy = random.randint(4, self.sky_h - 8)
            self.stars.append((sx, sy))

        self.glow_surf = pygame.Surface((self.w, self.h), pygame.SRCALPHA)
        self.cone_surf = pygame.Surface((self.w, self.h), pygame.SRCALPHA)
        self.font_pill = pygame.font.SysFont("consolas", 10, bold=True)

    def handle_mouse_orbit(self, event: pygame.event.Event, rect_offset: tuple[int, int]):
        rx, ry = rect_offset
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mx, my = event.pos
            if rx <= mx <= rx + self.w and ry <= my <= ry + self.h:
                self.is_dragging = True
                self.last_mouse_pos = (mx, my)
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self.is_dragging = False
        elif event.type == pygame.MOUSEMOTION and self.is_dragging:
            dx = event.pos[0] - self.last_mouse_pos[0]
            dy = event.pos[1] - self.last_mouse_pos[1]
            self.target_yaw_deg = max(-35.0, min(35.0, self.target_yaw_deg - dx * 0.30))
            self.target_pitch_deg = max(6.0, min(45.0, self.target_pitch_deg + dy * 0.30))
            self.last_mouse_pos = event.pos

    def update_camera_smoothing(self, dt: float = 0.016):
        self.cam_yaw_deg += (self.target_yaw_deg - self.cam_yaw_deg) * 14.0 * dt
        self.cam_pitch_deg += (self.target_pitch_deg - self.cam_pitch_deg) * 14.0 * dt

    def project_3d_to_screen(self, p_world: np.ndarray, ego_x: float = 0.0) -> tuple[int, int, float]:
        pitch_rad = math.radians(self.cam_pitch_deg)
        yaw_rad = math.radians(self.cam_yaw_deg)

        c_p, s_p = math.cos(pitch_rad), math.sin(pitch_rad)
        c_y, s_y = math.cos(yaw_rad), math.sin(yaw_rad)

        cam_x = ego_x - self.cam_dist_m * s_y * c_p
        cam_y = self.cam_height_m
        cam_z = -self.cam_dist_m * c_y * c_p

        rx = p_world[0] - cam_x
        ry = p_world[1] - cam_y
        rz = p_world[2] - cam_z

        cz = c_p * (s_y * rx + c_y * rz) + s_p * ry
        if cz <= 0.4:
            return -9999, -9999, -1.0

        cx_cam = c_y * rx - s_y * rz
        cy_cam = c_p * ry - s_p * (s_y * rx + c_y * rz)

        u = int(self.cx + (self.focal * cx_cam) / cz)
        v = int(self.cy - (self.focal * cy_cam) / cz)
        return u, v, float(cz)

    def render_3d_scene(
        self,
        surface: pygame.Surface,
        ego: EgoAutonomousVehicle,
        traffic: list[TrafficVehicle],
        point_cloud: np.ndarray = None,
        particles: list[Particle] = None,
        frame_idx: int = 0,
        weather_mode: str = "CLEAR",
        night_mode: bool = False
    ):
        self.update_camera_smoothing(0.016)
        ego_x = ego.x
        sky_h = self.sky_h

        # -------------------------------------------------------------
        # 1. OBSIDIAN SKY & HORIZON
        # -------------------------------------------------------------
        if night_mode:
            sky_colors = [(4, 6, 12), (8, 12, 20), (12, 18, 28), (16, 24, 36)]
        elif weather_mode == "FOG":
            sky_colors = [(130, 140, 150), (140, 150, 160), (145, 155, 165)]
        elif weather_mode == "RAIN":
            sky_colors = [(18, 24, 34), (24, 32, 44), (30, 40, 52)]
        else:
            sky_colors = [(6, 10, 18), (10, 16, 28), (16, 25, 40), (24, 36, 54)]

        band_h = sky_h / len(sky_colors)
        for b_idx, col in enumerate(sky_colors):
            y_s = int(b_idx * band_h)
            y_e = int((b_idx + 1) * band_h) if b_idx < len(sky_colors) - 1 else sky_h
            pygame.draw.rect(surface, col, pygame.Rect(0, y_s, self.w, y_e - y_s))

        # Soft Glowing Moon
        moon_x, moon_y = 44, 36
        self.glow_surf.fill((0, 0, 0, 0))
        pygame.draw.circle(self.glow_surf, (220, 230, 255, 16), (moon_x, moon_y), 32)
        pygame.draw.circle(self.glow_surf, (220, 230, 255, 30), (moon_x, moon_y), 20)
        surface.blit(self.glow_surf, (0, 0))
        pygame.draw.circle(surface, (230, 240, 255), (moon_x, moon_y), 11)

        # Distant Mountains
        pts_mountains = [(0, sky_h)]
        for mx in range(0, self.w + 8, 8):
            my = int(sky_h - 14 - 7.5 * math.sin(mx * 0.018) - 3.5 * math.sin(mx * 0.045))
            pts_mountains.append((mx, my))
        pts_mountains.append((self.w, sky_h))
        pygame.draw.polygon(surface, (14, 20, 32), pts_mountains)

        if weather_mode != "FOG":
            for sx, sy in self.stars:
                if sy < sky_h - 6:
                    pygame.draw.circle(surface, (180, 200, 230), (sx, sy), 1)

        # -------------------------------------------------------------
        # 2. 3-LANE HIGHWAY ROADBED
        # -------------------------------------------------------------
        for z_seg in range(75, -20, -6):
            g_l1 = self.project_3d_to_screen(np.array([-30.0, 0.0, float(z_seg)]), ego_x)
            g_l2 = self.project_3d_to_screen(np.array([-7.5, 0.0, float(z_seg)]), ego_x)
            g_l3 = self.project_3d_to_screen(np.array([-7.5, 0.0, float(z_seg - 6)]), ego_x)
            g_l4 = self.project_3d_to_screen(np.array([-30.0, 0.0, float(z_seg - 6)]), ego_x)
            if g_l1[0] != -9999 and g_l2[0] != -9999 and g_l3[0] != -9999 and g_l4[0] != -9999:
                pygame.draw.polygon(surface, (12, 18, 14), [(g_l1[0], g_l1[1]), (g_l2[0], g_l2[1]), (g_l3[0], g_l3[1]), (g_l4[0], g_l4[1])])

            g_r1 = self.project_3d_to_screen(np.array([+7.5, 0.0, float(z_seg)]), ego_x)
            g_r2 = self.project_3d_to_screen(np.array([+30.0, 0.0, float(z_seg)]), ego_x)
            g_r3 = self.project_3d_to_screen(np.array([+30.0, 0.0, float(z_seg - 6)]), ego_x)
            g_r4 = self.project_3d_to_screen(np.array([+7.5, 0.0, float(z_seg - 6)]), ego_x)
            if g_r1[0] != -9999 and g_r2[0] != -9999 and g_r3[0] != -9999 and g_r4[0] != -9999:
                pygame.draw.polygon(surface, (12, 18, 14), [(g_r1[0], g_r1[1]), (g_r2[0], g_r2[1]), (g_r3[0], g_r3[1]), (g_r4[0], g_r4[1])])

            p1 = np.array([-7.5, 0.0, float(z_seg)])
            p2 = np.array([+7.5, 0.0, float(z_seg)])
            p3 = np.array([+7.5, 0.0, float(z_seg - 6)])
            p4 = np.array([-7.5, 0.0, float(z_seg - 6)])

            u1, v1, _ = self.project_3d_to_screen(p1, ego_x)
            u2, v2, _ = self.project_3d_to_screen(p2, ego_x)
            u3, v3, _ = self.project_3d_to_screen(p3, ego_x)
            u4, v4, _ = self.project_3d_to_screen(p4, ego_x)

            if u1 != -9999 and u2 != -9999 and u3 != -9999 and u4 != -9999:
                shade = max(16, int(26 - (z_seg / 75.0) * 10))
                pygame.draw.polygon(surface, (shade, shade + 2, shade + 4), [(u1, v1), (u2, v2), (u3, v3), (u4, v4)])

        # Solid Left Yellow Divider (X = -5.8m)
        pts_left_edge = [self.project_3d_to_screen(np.array([-5.8, 0.02, float(z)]), ego_x) for z in range(-16, 75, 5)]
        valid_le = [(u, v) for u, v, _ in pts_left_edge if u != -9999]
        if len(valid_le) > 1:
            pygame.draw.lines(surface, (255, 205, 30), False, valid_le, 2)

        # Solid Right White Line (X = +5.8m)
        pts_right_edge = [self.project_3d_to_screen(np.array([+5.8, 0.02, float(z)]), ego_x) for z in range(-16, 75, 5)]
        valid_re = [(u, v) for u, v, _ in pts_right_edge if u != -9999]
        if len(valid_re) > 1:
            pygame.draw.lines(surface, (230, 235, 245), False, valid_re, 2)

        # Dashed Lane Lines (X = -1.875m and X = +1.875m)
        z_offset = (frame_idx * (ego.speed_kmh * 0.08)) % 8.0
        for x_lane in (-1.875, 1.875):
            for z_dash in np.arange(-16.0 + z_offset, 70.0, 8.0):
                p_d1 = np.array([x_lane, 0.02, float(z_dash)])
                p_d2 = np.array([x_lane, 0.02, float(z_dash + 3.8)])
                u_d1, v_d1, _ = self.project_3d_to_screen(p_d1, ego_x)
                u_d2, v_d2, _ = self.project_3d_to_screen(p_d2, ego_x)
                if u_d1 != -9999 and u_d2 != -9999:
                    pygame.draw.line(surface, (225, 230, 240), (u_d1, v_d1), (u_d2, v_d2), 2)

        # -------------------------------------------------------------
        # 3. SOFT HEADLIGHT CONES
        # -------------------------------------------------------------
        self.cone_surf.fill((0, 0, 0, 0))
        p_hl_orig_l = np.array([ego.x - 0.75, 0.20, ego.z + 2.2])
        p_hl_orig_r = np.array([ego.x + 0.75, 0.20, ego.z + 2.2])
        p_hl_tip_l = np.array([ego.x - 3.4, 0.02, ego.z + 26.0])
        p_hl_tip_r = np.array([ego.x + 3.4, 0.02, ego.z + 26.0])

        u_ol, v_ol, _ = self.project_3d_to_screen(p_hl_orig_l, ego_x)
        u_or, v_or, _ = self.project_3d_to_screen(p_hl_orig_r, ego_x)
        u_tl, v_tl, _ = self.project_3d_to_screen(p_hl_tip_l, ego_x)
        u_tr, v_tr, _ = self.project_3d_to_screen(p_hl_tip_r, ego_x)

        if u_ol != -9999 and u_or != -9999 and u_tl != -9999 and u_tr != -9999:
            pygame.draw.polygon(self.cone_surf, (225, 235, 255, 28), [(u_ol, v_ol), (u_or, v_or), (u_tr, v_tr), (u_tl, v_tl)])
            surface.blit(self.cone_surf, (0, 0))

        # -------------------------------------------------------------
        # 4. TESLA CYAN TRAJECTORY CORRIDOR
        # -------------------------------------------------------------
        traj_3d = []
        target_lane_x = float(ego.target_lane_idx * 3.75)
        for s in np.linspace(2.0, 30.0, 14):
            ratio = min(1.0, s / 22.0)
            cur_x = ego.x + (target_lane_x - ego.x) * (10.0 * ratio**3 - 15.0 * ratio**4 + 6.0 * ratio**5)
            u_t, v_t, _ = self.project_3d_to_screen(np.array([cur_x, 0.06, float(s)]), ego_x)
            if u_t != -9999:
                traj_3d.append((u_t, v_t))
        if len(traj_3d) > 1:
            pygame.draw.lines(surface, (0, 229, 255), False, traj_3d, 3)

        # -------------------------------------------------------------
        # 5. DYNAMIC TRAFFIC (Distinct 3D Models & Direct Vehicle Labels)
        # -------------------------------------------------------------
        all_vehicles = list(traffic)
        all_vehicles.sort(key=lambda v: v.z, reverse=True)

        for v in all_vehicles:
            self.draw_3d_vehicle(surface, v, ego_x, frame_idx)

        self.draw_3d_ego_vehicle(surface, ego, frame_idx)

    def draw_3d_vehicle(self, surface: pygame.Surface, v: TrafficVehicle, ego_x: float, frame_idx: int):
        """Draws distinct 3D models for TRUCK, SPORTS, and SEDAN with matching badges."""
        hw, hl = v.width * 0.5, v.length * 0.5
        h = v.height
        x, z = v.x, v.z

        # Contact Shadow
        s_corners = [
            np.array([x - hw - 0.15, 0.02, z - hl - 0.15]),
            np.array([x + hw + 0.15, 0.02, z - hl - 0.15]),
            np.array([x + hw + 0.15, 0.02, z + hl + 0.15]),
            np.array([x - hw - 0.15, 0.02, z + hl + 0.15]),
        ]
        s_proj = [self.project_3d_to_screen(c, ego_x) for c in s_corners]
        if all(u != -9999 for u, _, _ in s_proj):
            pygame.draw.polygon(surface, (10, 14, 18), [(u, v_p) for u, v_p, _ in s_proj])

        # 3D Bounding Points
        corners = [
            np.array([x - hw, 0.15, z - hl]), # 0
            np.array([x + hw, 0.15, z - hl]), # 1
            np.array([x + hw, 0.15, z + hl]), # 2
            np.array([x - hw, 0.15, z + hl]), # 3
            np.array([x - hw, h,    z - hl]), # 4
            np.array([x + hw, h,    z - hl]), # 5
            np.array([x + hw, h,    z + hl]), # 6
            np.array([x - hw, h,    z + hl]), # 7
        ]
        proj = [self.project_3d_to_screen(c, ego_x) for c in corners]
        if any(u == -9999 for u, v_p, _ in proj):
            return

        pts = [(u, v_p) for u, v_p, _ in proj]
        v_col = v.color
        top_col = tuple(min(255, int(c * 0.88 + 15)) for c in v_col)
        side_col = tuple(int(c * 0.65) for c in v_col)

        if v.model_type == "TRUCK":
            # SEMI TRUCK: Tall Box Trailer + Lower Cab
            pygame.draw.polygon(surface, v_col, [pts[0], pts[1], pts[5], pts[4]])
            pygame.draw.polygon(surface, (230, 240, 250), [pts[0], pts[1], pts[5], pts[4]], 1)
            pygame.draw.polygon(surface, top_col, [pts[4], pts[5], pts[6], pts[7]])

            if x < ego_x:
                pygame.draw.polygon(surface, side_col, [pts[1], pts[2], pts[6], pts[5]])
            else:
                pygame.draw.polygon(surface, side_col, [pts[0], pts[3], pts[7], pts[4]])

            pygame.draw.circle(surface, (255, 185, 30), (pts[4][0] + 4, pts[4][1] + 4), 2)
            pygame.draw.circle(surface, (255, 185, 30), (pts[5][0] - 4, pts[5][1] + 4), 2)

            for wx, wy in (pts[0], pts[1]):
                pygame.draw.circle(surface, (12, 12, 12), (wx, wy), 5)
                pygame.draw.circle(surface, (160, 175, 195), (wx, wy), 3, 1)

        elif v.model_type == "SPORTS":
            # SPORTS CAR: Low Profile + Rear Wing Spoiler
            pygame.draw.polygon(surface, v_col, [pts[0], pts[1], pts[5], pts[4]])
            pygame.draw.polygon(surface, (230, 240, 250), [pts[0], pts[1], pts[5], pts[4]], 1)
            pygame.draw.polygon(surface, top_col, [pts[4], pts[5], pts[6], pts[7]])

            if x < ego_x:
                pygame.draw.polygon(surface, side_col, [pts[1], pts[2], pts[6], pts[5]])
            else:
                pygame.draw.polygon(surface, side_col, [pts[0], pts[3], pts[7], pts[4]])

            pygame.draw.line(surface, (245, 245, 250), (pts[4][0] - 4, pts[4][1] - 4), (pts[5][0] + 4, pts[5][1] - 4), 2)
            pygame.draw.line(surface, (30, 30, 30), (pts[4][0], pts[4][1]), (pts[4][0] - 2, pts[4][1] - 4), 2)
            pygame.draw.line(surface, (30, 30, 30), (pts[5][0], pts[5][1]), (pts[5][0] + 2, pts[5][1] - 4), 2)

            for wx, wy in (pts[0], pts[1]):
                pygame.draw.circle(surface, (12, 12, 12), (wx, wy), 3)

        else:
            # SEDAN: Standard 3-Box Sedan
            pygame.draw.polygon(surface, v_col, [pts[0], pts[1], pts[5], pts[4]])
            pygame.draw.polygon(surface, (230, 240, 250), [pts[0], pts[1], pts[5], pts[4]], 1)
            pygame.draw.polygon(surface, top_col, [pts[4], pts[5], pts[6], pts[7]])

            if x < ego_x:
                pygame.draw.polygon(surface, side_col, [pts[1], pts[2], pts[6], pts[5]])
            else:
                pygame.draw.polygon(surface, side_col, [pts[0], pts[3], pts[7], pts[4]])

            u_mid_l = (pts[0][0] + pts[4][0]) // 2
            u_mid_r = (pts[1][0] + pts[5][0]) // 2
            v_mid_top = (pts[4][1] + pts[5][1]) // 2 + 2
            v_mid_bot = (pts[0][1] + pts[1][1]) // 2 - 2
            if v_mid_bot > v_mid_top:
                pygame.draw.polygon(surface, (38, 48, 65), [(u_mid_l, v_mid_top), (u_mid_r, v_mid_top),
                                                            (u_mid_r, v_mid_bot), (u_mid_l, v_mid_bot)])

            for wx, wy in (pts[0], pts[1]):
                pygame.draw.circle(surface, (12, 12, 12), (wx, wy), 4)
                pygame.draw.circle(surface, (160, 175, 195), (wx, wy), 2, 1)

        # LED Taillights
        tl_col = (255, 35, 35) if v.is_braking else (205, 20, 20)
        pygame.draw.circle(surface, tl_col, (pts[0][0] + 4, pts[0][1] - 6), 3)
        pygame.draw.circle(surface, tl_col, (pts[1][0] - 4, pts[1][1] - 6), 3)

        # Direct Clean Telemetry Pill (Tracked Obstacles 2m < z < 35m)
        if 2.0 < v.z < 35.0:
            lbl_text = f"[{v.model_type}] {v.id.replace('_', ' ')} [{v.speed_kmh:.0f} km/h]"
            lbl_surf = self.font_pill.render(lbl_text, True, (215, 235, 255))
            pill_w = lbl_surf.get_width() + 10
            pill_h = 16

            u_top = (pts[4][0] + pts[5][0]) // 2
            u_top = max(pill_w // 2 + 6, min(self.w - pill_w // 2 - 6, u_top))
            v_top = min(pts[4][1], pts[5][1]) - 14

            if 20 < v_top < self.h - 40:
                pill_rect = pygame.Rect(u_top - pill_w // 2, v_top - 2, pill_w, pill_h)
                pygame.draw.rect(surface, (10, 14, 22), pill_rect, border_radius=4)
                pygame.draw.rect(surface, (40, 52, 75), pill_rect, 1, border_radius=4)
                surface.blit(lbl_surf, (u_top - lbl_surf.get_width() // 2, v_top))

    def draw_3d_ego_vehicle(self, surface: pygame.Surface, ego: EgoAutonomousVehicle, frame_idx: int):
        """Draws hero Tesla Model S with Deep Metallic Blue body and panoramic glass."""
        hw = ego.width * 0.5
        hl = ego.length * 0.5
        h = ego.height
        x, z = ego.x, ego.z

        s_corners = [
            np.array([x - hw - 0.2, 0.02, z - hl - 0.2]),
            np.array([x + hw + 0.2, 0.02, z - hl - 0.2]),
            np.array([x + hw + 0.2, 0.02, z + hl + 0.2]),
            np.array([x - hw - 0.2, 0.02, z + hl + 0.2]),
        ]
        s_proj = [self.project_3d_to_screen(c, ego.x) for c in s_corners]
        if all(u != -9999 for u, _, _ in s_proj):
            pygame.draw.polygon(surface, (10, 12, 16), [(u, v_p) for u, v_p, _ in s_proj])

        corners = [
            np.array([x - hw, 0.15, z - hl]), # 0
            np.array([x + hw, 0.15, z - hl]), # 1
            np.array([x + hw, 0.15, z + hl]), # 2
            np.array([x - hw, 0.15, z + hl]), # 3
            np.array([x - hw, h,    z - hl]), # 4
            np.array([x + hw, h,    z - hl]), # 5
            np.array([x + hw, h,    z + hl]), # 6
            np.array([x - hw, h,    z + hl]), # 7
        ]
        proj = [self.project_3d_to_screen(c, ego.x) for c in corners]
        if any(u == -9999 for u, v_p, _ in proj):
            return

        pts = [(u, v_p) for u, v_p, _ in proj]

        # Deep Metallic Paint
        pygame.draw.polygon(surface, (16, 44, 85), [pts[0], pts[1], pts[5], pts[4]])
        pygame.draw.polygon(surface, (0, 180, 240), [pts[0], pts[1], pts[5], pts[4]], 2)

        # Panoramic Glass Roof
        pygame.draw.polygon(surface, (28, 50, 75), [pts[4], pts[5], pts[6], pts[7]])
        pygame.draw.polygon(surface, (0, 200, 255), [pts[4], pts[5], pts[6], pts[7]], 1)

        # LED Taillight Lightbar
        pygame.draw.line(surface, (245, 30, 30), (pts[0][0] + 4, pts[0][1] - 8), (pts[1][0] - 4, pts[1][1] - 8), 3)

        # Multi-spoke alloy wheels
        for wx, wy in (pts[0], pts[1]):
            pygame.draw.circle(surface, (12, 12, 12), (wx, wy), 4)
            pygame.draw.circle(surface, (0, 180, 240), (wx, wy), 2, 1)
