"""
digital_twin_3d_renderer.py — Realistic 3D Digital Twin Highway & Vehicle World Visualizer
==========================================================================================
Features:
  - Realistic 3-Lane Highway with Solid Yellow Left Divider & Perspective Dashed Lane Lines.
  - Realistic Automotive Body Shading, Window Glints, Alloy Wheels & Soft Road Shadows.
  - Atmospheric 6-Band Twilight Gradient, Mountain Silhouettes & Dynamic Star Field.
  - Forward Volumetric Headlight Illumination Cones & LED Taillight Bloom.
  - Modern Minimalist Glassmorphic UI Aesthetics.
"""

import math
import random
import numpy as np
import pygame

from traffic_physics_simulator import EgoAutonomousVehicle, TrafficVehicle, Particle


class DigitalTwin3DRenderer:
    """Renders the realistic 3D Digital Twin world with 3-lane highway and authentic vehicle models."""

    def __init__(self, screen_w: int = 440, screen_h: int = 460):
        self.w = screen_w
        self.h = screen_h

        # 3D Camera Intrinsics
        self.fov_deg = 62.0
        self.focal = (self.w * 0.5) / math.tan(math.radians(self.fov_deg * 0.5))
        self.cx = self.w * 0.5
        self.cy = self.h * 0.52

        # Camera Extrinsics (Chase Orbit)
        self.cam_dist_m = 9.6
        self.cam_height_m = 4.2
        self.cam_pitch_deg = 14.0
        self.cam_yaw_deg = 0.0

        # Mouse Drag Orbiting
        self.is_dragging = False
        self.last_mouse_pos = (0, 0)

        # Seeded stars
        random.seed(42)
        sky_limit = int(self.h * 0.50)
        self.stars = []
        for i in range(60):
            sx = random.randint(4, self.w - 4)
            sy = random.randint(4, sky_limit - 8)
            self.stars.append((sx, sy, (i % 6 == 0)))

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
            self.cam_yaw_deg = max(-35.0, min(35.0, self.cam_yaw_deg - dx * 0.30))
            self.cam_pitch_deg = max(6.0, min(45.0, self.cam_pitch_deg + dy * 0.30))
            self.last_mouse_pos = event.pos

    def project_3d_to_screen(self, p_world: np.ndarray, ego_x: float = 0.0) -> tuple[int, int, float]:
        pitch_rad = math.radians(self.cam_pitch_deg)
        yaw_rad = math.radians(self.cam_yaw_deg)

        cam_pos = np.array([
            ego_x - self.cam_dist_m * math.sin(yaw_rad) * math.cos(pitch_rad),
            self.cam_height_m,
            -self.cam_dist_m * math.cos(yaw_rad) * math.cos(pitch_rad)
        ])

        p_rel = p_world - cam_pos

        c_p, s_p = math.cos(pitch_rad), math.sin(pitch_rad)
        c_y, s_y = math.cos(yaw_rad), math.sin(yaw_rad)

        R_cam = np.array([
            [c_y, 0, -s_y],
            [-s_p * s_y, c_p, -s_p * c_y],
            [c_p * s_y, s_p, c_p * c_y]
        ])

        p_cam = R_cam @ p_rel

        if p_cam[2] <= 0.4:
            return -9999, -9999, -1.0

        u = int(self.cx + (self.focal * p_cam[0]) / p_cam[2])
        v = int(self.cy - (self.focal * p_cam[1]) / p_cam[2])
        return u, v, float(p_cam[2])

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
        ego_x = ego.x
        sky_h = int(self.h * 0.50)

        # -------------------------------------------------------------
        # 1. ATMOSPHERIC SKY & HORIZON
        # -------------------------------------------------------------
        if night_mode:
            sky_colors = [(6, 10, 18), (10, 16, 28), (16, 24, 38), (22, 32, 48), (18, 26, 40)]
        elif weather_mode == "FOG":
            sky_colors = [(135, 145, 155), (145, 155, 165), (150, 160, 170), (140, 150, 160), (130, 140, 150)]
        elif weather_mode == "RAIN":
            sky_colors = [(26, 32, 44), (32, 40, 54), (38, 48, 62), (45, 55, 70), (35, 45, 58)]
        else:
            # Twilight Sky Gradient
            sky_colors = [(8, 14, 26), (14, 22, 38), (20, 32, 50), (32, 46, 68), (24, 36, 54)]

        band_h = sky_h / len(sky_colors)
        for b_idx, col in enumerate(sky_colors):
            y_s = int(b_idx * band_h)
            y_e = int((b_idx + 1) * band_h) if b_idx < len(sky_colors) - 1 else sky_h
            pygame.draw.rect(surface, col, pygame.Rect(0, y_s, self.w, y_e - y_s))

        # Moon with soft glow
        moon_x, moon_y = 44, 36
        glow_surf = pygame.Surface((self.w, self.h), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, (220, 230, 255, 15), (moon_x, moon_y), 32)
        pygame.draw.circle(glow_surf, (220, 230, 255, 30), (moon_x, moon_y), 20)
        surface.blit(glow_surf, (0, 0))
        pygame.draw.circle(surface, (225, 235, 255), (moon_x, moon_y), 12)

        # Distant Mountain Silhouettes
        pts_mountains = [(0, sky_h)]
        for mx in range(0, self.w + 10, 10):
            my = int(sky_h - 14 - 8.0 * math.sin(mx * 0.018) - 4.0 * math.sin(mx * 0.045))
            pts_mountains.append((mx, my))
        pts_mountains.append((self.w, sky_h))
        pygame.draw.polygon(surface, (18, 26, 40), pts_mountains)

        # Stars
        if weather_mode != "FOG":
            for sx, sy, is_cross in self.stars:
                if sy < sky_h - 6:
                    pygame.draw.circle(surface, (190, 205, 230), (sx, sy), 1)

        # -------------------------------------------------------------
        # 2. 3D HIGHWAY ASPHALT & 3 FULL LANES
        # -------------------------------------------------------------
        # Roadbed Asphalt: X = -7.5m to +7.5m
        for z_seg in range(80, -25, -5):
            # Left Grass Shoulder (X = -35m to -7.5m)
            g_l1 = self.project_3d_to_screen(np.array([-35.0, 0.0, float(z_seg)]), ego_x)
            g_l2 = self.project_3d_to_screen(np.array([-7.5, 0.0, float(z_seg)]), ego_x)
            g_l3 = self.project_3d_to_screen(np.array([-7.5, 0.0, float(z_seg - 5)]), ego_x)
            g_l4 = self.project_3d_to_screen(np.array([-35.0, 0.0, float(z_seg - 5)]), ego_x)
            if g_l1[0] != -9999 and g_l2[0] != -9999 and g_l3[0] != -9999 and g_l4[0] != -9999:
                pygame.draw.polygon(surface, (16, 24, 18), [(g_l1[0], g_l1[1]), (g_l2[0], g_l2[1]), (g_l3[0], g_l3[1]), (g_l4[0], g_l4[1])])

            # Right Grass Shoulder (X = +7.5m to +35m)
            g_r1 = self.project_3d_to_screen(np.array([+7.5, 0.0, float(z_seg)]), ego_x)
            g_r2 = self.project_3d_to_screen(np.array([+35.0, 0.0, float(z_seg)]), ego_x)
            g_r3 = self.project_3d_to_screen(np.array([+35.0, 0.0, float(z_seg - 5)]), ego_x)
            g_r4 = self.project_3d_to_screen(np.array([+7.5, 0.0, float(z_seg - 5)]), ego_x)
            if g_r1[0] != -9999 and g_r2[0] != -9999 and g_r3[0] != -9999 and g_r4[0] != -9999:
                pygame.draw.polygon(surface, (16, 24, 18), [(g_r1[0], g_r1[1]), (g_r2[0], g_r2[1]), (g_r3[0], g_r3[1]), (g_r4[0], g_r4[1])])

            # 3D Highway Asphalt Surface
            p1 = np.array([-7.5, 0.0, float(z_seg)])
            p2 = np.array([+7.5, 0.0, float(z_seg)])
            p3 = np.array([+7.5, 0.0, float(z_seg - 5)])
            p4 = np.array([-7.5, 0.0, float(z_seg - 5)])

            u1, v1, _ = self.project_3d_to_screen(p1, ego_x)
            u2, v2, _ = self.project_3d_to_screen(p2, ego_x)
            u3, v3, _ = self.project_3d_to_screen(p3, ego_x)
            u4, v4, _ = self.project_3d_to_screen(p4, ego_x)

            if u1 != -9999 and u2 != -9999 and u3 != -9999 and u4 != -9999:
                shade = max(18, int(30 - (z_seg / 80.0) * 12))
                pygame.draw.polygon(surface, (shade, shade + 3, shade + 6), [(u1, v1), (u2, v2), (u3, v3), (u4, v4)])

        # Vanishing Point Ambient Glow
        vp_surf = pygame.Surface((self.w, self.h), pygame.SRCALPHA)
        pygame.draw.ellipse(vp_surf, (0, 120, 255, 18), pygame.Rect(int(self.w * 0.5 - 50), sky_h - 8, 100, 16))
        surface.blit(vp_surf, (0, 0))

        # Solid Left Yellow Divider (X = -5.8m)
        pts_left_edge = [self.project_3d_to_screen(np.array([-5.8, 0.02, float(z)]), ego_x) for z in range(-20, 80, 4)]
        valid_le = [(u, v) for u, v, _ in pts_left_edge if u != -9999]
        if len(valid_le) > 1:
            pygame.draw.lines(surface, (255, 205, 30), False, valid_le, 2)

        # Solid Right White Edge Line (X = +5.8m)
        pts_right_edge = [self.project_3d_to_screen(np.array([+5.8, 0.02, float(z)]), ego_x) for z in range(-20, 80, 4)]
        valid_re = [(u, v) for u, v, _ in pts_right_edge if u != -9999]
        if len(valid_re) > 1:
            pygame.draw.lines(surface, (230, 235, 245), False, valid_re, 2)

        # Dashed Lane Dividers (X = -1.875m and X = +1.875m) -> CREATES 3 DISTINCT LANES
        z_offset = (frame_idx * (ego.speed_kmh * 0.08)) % 8.0
        for x_lane in (-1.875, 1.875):
            for z_dash in np.arange(-18.0 + z_offset, 75.0, 8.0):
                p_d1 = np.array([x_lane, 0.02, float(z_dash)])
                p_d2 = np.array([x_lane, 0.02, float(z_dash + 3.8)])
                u_d1, v_d1, _ = self.project_3d_to_screen(p_d1, ego_x)
                u_d2, v_d2, _ = self.project_3d_to_screen(p_d2, ego_x)
                if u_d1 != -9999 and u_d2 != -9999:
                    pygame.draw.line(surface, (225, 230, 240), (u_d1, v_d1), (u_d2, v_d2), 2)

        # -------------------------------------------------------------
        # 3. EGO VEHICLE HEADLIGHT ILLUMINATION CONES
        # -------------------------------------------------------------
        cone_surf = pygame.Surface((self.w, self.h), pygame.SRCALPHA)
        p_hl_orig_l = np.array([ego.x - 0.75, 0.20, ego.z + 2.2])
        p_hl_orig_r = np.array([ego.x + 0.75, 0.20, ego.z + 2.2])
        p_hl_tip_l = np.array([ego.x - 3.8, 0.02, ego.z + 28.0])
        p_hl_tip_r = np.array([ego.x + 3.8, 0.02, ego.z + 28.0])

        u_ol, v_ol, _ = self.project_3d_to_screen(p_hl_orig_l, ego_x)
        u_or, v_or, _ = self.project_3d_to_screen(p_hl_orig_r, ego_x)
        u_tl, v_tl, _ = self.project_3d_to_screen(p_hl_tip_l, ego_x)
        u_tr, v_tr, _ = self.project_3d_to_screen(p_hl_tip_r, ego_x)

        if u_ol != -9999 and u_or != -9999 and u_tl != -9999 and u_tr != -9999:
            pygame.draw.polygon(cone_surf, (220, 235, 255, 38), [(u_ol, v_ol), (u_or, v_or), (u_tr, v_tr), (u_tl, v_tl)])
            surface.blit(cone_surf, (0, 0))

        # -------------------------------------------------------------
        # 4. PHYSICS PARTICLES
        # -------------------------------------------------------------
        if particles:
            for p in particles:
                u_p, v_p, _ = self.project_3d_to_screen(np.array([p.x, p.y, p.z]), ego_x)
                if u_p != -9999 and 0 <= u_p < self.w and 0 <= v_p < self.h:
                    if p.p_type == "SPARK":
                        pygame.draw.circle(surface, p.color, (u_p, v_p), max(1, int(p.size)))
                    elif p.p_type == "SMOKE":
                        smoke_surf = pygame.Surface((int(p.size * 2), int(p.size * 2)), pygame.SRCALPHA)
                        pygame.draw.circle(smoke_surf, (220, 230, 240, 45), (int(p.size), int(p.size)), int(p.size))
                        surface.blit(smoke_surf, (u_p - int(p.size), v_p - int(p.size)))
                    else: # Exhaust
                        pygame.draw.circle(surface, p.color, (u_p, v_p), max(1, int(p.size * 0.75)))

        # -------------------------------------------------------------
        # 5. SMOOTH CLOTHOID OVERTAKE TRAJECTORY CORRIDOR
        # -------------------------------------------------------------
        if ego.state in ("CHECK_OVERTAKE", "LANE_CHANGE_LEFT", "OVERTAKING", "LANE_CHANGE_RIGHT"):
            traj_3d = []
            target_lane_x = float(ego.target_lane_idx * 3.75)
            for s in np.linspace(2.0, 32.0, 18):
                ratio = min(1.0, s / 24.0)
                cur_x = ego.x + (target_lane_x - ego.x) * (10.0 * ratio**3 - 15.0 * ratio**4 + 6.0 * ratio**5)
                u_t, v_t, _ = self.project_3d_to_screen(np.array([cur_x, 0.08, float(s)]), ego_x)
                if u_t != -9999:
                    traj_3d.append((u_t, v_t))
            if len(traj_3d) > 1:
                pygame.draw.lines(surface, (0, 230, 180), False, traj_3d, 3)

        # -------------------------------------------------------------
        # 6. DYNAMIC VEHICLES WITH REALISTIC SHADING & SOFT SHADOWS
        # -------------------------------------------------------------
        all_vehicles = list(traffic)
        all_vehicles.sort(key=lambda v: v.z, reverse=True)

        for v in all_vehicles:
            self.draw_3d_vehicle(surface, v, ego_x, frame_idx)

        # Draw 3D Ego Tesla
        self.draw_3d_ego_vehicle(surface, ego, frame_idx)

        # -------------------------------------------------------------
        # 7. DEPTH FOG BAND AT HORIZON
        # -------------------------------------------------------------
        fog_surf = pygame.Surface((self.w, 32), pygame.SRCALPHA)
        for y_fog in range(32):
            alpha_val = int(120 * ((32 - y_fog) / 32.0))
            pygame.draw.line(fog_surf, (14, 20, 32, alpha_val), (0, y_fog), (self.w, y_fog))
        surface.blit(fog_surf, (0, sky_h - 16))

    def draw_3d_vehicle(self, surface: pygame.Surface, v: TrafficVehicle, ego_x: float, frame_idx: int):
        """Draws dynamic traffic vehicle with realistic automotive shading and soft ground shadow."""
        hw, hl = v.width * 0.5, v.length * 0.5
        h = v.height
        x, z = v.x, v.z

        # 1. Soft Ground Contact Shadow
        s_corners = [
            np.array([x - hw - 0.15, 0.02, z - hl - 0.15]),
            np.array([x + hw + 0.15, 0.02, z - hl - 0.15]),
            np.array([x + hw + 0.15, 0.02, z + hl + 0.15]),
            np.array([x - hw - 0.15, 0.02, z + hl + 0.15]),
        ]
        s_proj = [self.project_3d_to_screen(c, ego_x) for c in s_corners]
        if all(u != -9999 for u, _, _ in s_proj):
            shadow_surf = pygame.Surface((self.w, self.h), pygame.SRCALPHA)
            pygame.draw.polygon(shadow_surf, (0, 0, 0, 85), [(u, v_p) for u, v_p, _ in s_proj])
            surface.blit(shadow_surf, (0, 0))

        # 2. 3D Body Bounding Box
        corners = [
            np.array([x - hw, 0.15, z - hl]), # 0 Rear-Left
            np.array([x + hw, 0.15, z - hl]), # 1 Rear-Right
            np.array([x + hw, 0.15, z + hl]), # 2 Front-Right
            np.array([x - hw, 0.15, z + hl]), # 3 Front-Left
            np.array([x - hw, h,    z - hl]), # 4 Rear-Left Top
            np.array([x + hw, h,    z - hl]), # 5 Rear-Right Top
            np.array([x + hw, h,    z + hl]), # 6 Front-Right Top
            np.array([x - hw, h,    z + hl]), # 7 Front-Left Top
        ]
        proj = [self.project_3d_to_screen(c, ego_x) for c in corners]
        if any(u == -9999 for u, v_p, _ in proj):
            return

        pts = [(u, v_p) for u, v_p, _ in proj]
        v_col = v.color # Exact RGB

        # Rear/Front Face
        pygame.draw.polygon(surface, v_col, [pts[0], pts[1], pts[5], pts[4]])
        pygame.draw.polygon(surface, (240, 245, 255), [pts[0], pts[1], pts[5], pts[4]], 1)

        # Top Face (88% brightness)
        top_col = tuple(int(c * 0.88) for c in v_col)
        pygame.draw.polygon(surface, top_col, [pts[4], pts[5], pts[6], pts[7]])

        # Side Face (65% shadow)
        side_col = tuple(int(c * 0.65) for c in v_col)
        if x < ego_x:
            pygame.draw.polygon(surface, side_col, [pts[1], pts[2], pts[6], pts[5]])
        else:
            pygame.draw.polygon(surface, side_col, [pts[0], pts[3], pts[7], pts[4]])

        # Tinted Window Band
        u_mid_l = (pts[0][0] + pts[4][0]) // 2
        u_mid_r = (pts[1][0] + pts[5][0]) // 2
        v_mid_top = (pts[4][1] + pts[5][1]) // 2 + 2
        v_mid_bot = (pts[0][1] + pts[1][1]) // 2 - 2
        if v_mid_bot > v_mid_top:
            pygame.draw.polygon(surface, (40, 52, 70), [(u_mid_l, v_mid_top), (u_mid_r, v_mid_top),
                                                        (u_mid_r, v_mid_bot), (u_mid_l, v_mid_bot)])

        # Alloy Wheels
        for wx, wy in (pts[0], pts[1]):
            pygame.draw.circle(surface, (15, 15, 15), (wx, wy), 4)
            pygame.draw.circle(surface, (60, 75, 95), (wx, wy), 2)

        # LED Taillights
        tl_col = (255, 30, 30) if v.is_braking else (210, 20, 20)
        pygame.draw.circle(surface, tl_col, (pts[0][0] + 4, pts[0][1] - 6), 3)
        pygame.draw.circle(surface, tl_col, (pts[1][0] - 4, pts[1][1] - 6), 3)

        # Minimalist Label Tag
        u_top = (pts[4][0] + pts[5][0]) // 2
        v_top = min(pts[4][1], pts[5][1]) - 10
        lbl_surf = pygame.font.SysFont("segoeui", 10, bold=True).render(f"{v.id} [{v.speed_kmh:.0f} km/h]", True, (180, 220, 245))
        surface.blit(lbl_surf, (u_top - lbl_surf.get_width() // 2, v_top - 4))

    def draw_3d_ego_vehicle(self, surface: pygame.Surface, ego: EgoAutonomousVehicle, frame_idx: int):
        """Draws hero 3D Tesla vehicle with metallic blue body and glass roof."""
        hw = ego.width * 0.5
        hl = ego.length * 0.5
        h = ego.height
        x, z = ego.x, ego.z

        # Soft Ground Shadow
        s_corners = [
            np.array([x - hw - 0.2, 0.02, z - hl - 0.2]),
            np.array([x + hw + 0.2, 0.02, z - hl - 0.2]),
            np.array([x + hw + 0.2, 0.02, z + hl + 0.2]),
            np.array([x - hw - 0.2, 0.02, z + hl + 0.2]),
        ]
        s_proj = [self.project_3d_to_screen(c, ego.x) for c in s_corners]
        if all(u != -9999 for u, _, _ in s_proj):
            shadow_surf = pygame.Surface((self.w, self.h), pygame.SRCALPHA)
            pygame.draw.polygon(shadow_surf, (0, 0, 0, 110), [(u, v_p) for u, v_p, _ in s_proj])
            surface.blit(shadow_surf, (0, 0))

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

        # Metallic Deep Blue Body
        pygame.draw.polygon(surface, (20, 50, 95), [pts[0], pts[1], pts[5], pts[4]])
        pygame.draw.polygon(surface, (0, 190, 245), [pts[0], pts[1], pts[5], pts[4]], 2)

        # Glass Panoramic Roof
        pygame.draw.polygon(surface, (32, 58, 85), [pts[4], pts[5], pts[6], pts[7]])
        pygame.draw.polygon(surface, (0, 210, 255), [pts[4], pts[5], pts[6], pts[7]], 1)

        # LED Taillight Lightbar
        pygame.draw.line(surface, (255, 35, 35), (pts[0][0] + 4, pts[0][1] - 8), (pts[1][0] - 4, pts[1][1] - 8), 3)

        # Wheels
        for wx, wy in (pts[0], pts[1]):
            pygame.draw.circle(surface, (15, 15, 15), (wx, wy), 4)
            pygame.draw.circle(surface, (0, 190, 245), (wx, wy), 2)
