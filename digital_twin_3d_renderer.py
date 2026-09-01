"""
digital_twin_3d_renderer.py — Cinematic 3D Digital Twin World & Vehicle Visualizer
===================================================================================
Features:
  - Atmospheric 6-Band Sky Gradient, Moon with 3 Glow Rings & 80 Seeded Stars with Crosses.
  - 3D Perspective Roadbed with Grid Lines, Rumble Strips & Distant Vanishing Point Glow.
  - High-Fidelity 3D Vehicle Models: 3-Face Shaded Blocks, Roof Antennas, Window Rows & Alloy Wheels.
  - Ego Headlight Volumetric Cones with Gaussian Falloff & Traffic Taillight Glow Ellipses.
  - Multi-Layered Depth Fog Bands at Horizon.
"""

import math
import random
import numpy as np
import pygame

from traffic_physics_simulator import EgoAutonomousVehicle, TrafficVehicle, Particle


class DigitalTwin3DRenderer:
    """
    Renders the cinematic 3D Digital Twin highway, lighting, atmosphere, and vehicles.
    """

    def __init__(self, screen_w: int = 440, screen_h: int = 480):
        self.w = screen_w
        self.h = screen_h

        # 3D Camera Intrinsics
        self.fov_deg = 64.0
        self.focal = (self.w * 0.5) / math.tan(math.radians(self.fov_deg * 0.5))
        self.cx = self.w * 0.5
        self.cy = self.h * 0.53

        # Camera Extrinsics (Chase Orbit)
        self.cam_dist_m = 9.8
        self.cam_height_m = 4.3
        self.cam_pitch_deg = 14.5
        self.cam_yaw_deg = 0.0

        # Mouse Drag Orbiting
        self.is_dragging = False
        self.last_mouse_pos = (0, 0)

        # Generate 80 seeded star positions (seed=42)
        random.seed(42)
        sky_limit = int(self.h * 0.52)
        self.stars = []
        for i in range(80):
            sx = random.randint(4, self.w - 4)
            sy = random.randint(4, sky_limit - 8)
            is_cross = (i % 5 == 0)
            self.stars.append((sx, sy, is_cross))

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
            self.cam_yaw_deg = max(-40.0, min(40.0, self.cam_yaw_deg - dx * 0.35))
            self.cam_pitch_deg = max(5.0, min(50.0, self.cam_pitch_deg + dy * 0.35))
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
        sky_h = int(self.h * 0.52)

        # -------------------------------------------------------------
        # 1. 6-BAND SKY GRADIENT
        # -------------------------------------------------------------
        # Band 0 (top): RGB(4,8,20), Band 1: RGB(8,14,30), Band 2: RGB(12,20,40)
        # Band 3: RGB(18,28,50), Band 4: RGB(28,40,62), Band 5 (edge): RGB(22,32,48)
        sky_bands = [
            (4, 8, 20),
            (8, 14, 30),
            (12, 20, 40),
            (18, 28, 50),
            (28, 40, 62),
            (22, 32, 48)
        ]
        band_h = sky_h / len(sky_bands)
        for b_idx, col in enumerate(sky_bands):
            y_start = int(b_idx * band_h)
            y_end = int((b_idx + 1) * band_h) if b_idx < len(sky_bands) - 1 else sky_h
            pygame.draw.rect(surface, col, pygame.Rect(0, y_start, self.w, y_end - y_start))

        # -------------------------------------------------------------
        # 2. MOON / SUN WITH 3 GLOW RINGS
        # -------------------------------------------------------------
        moon_cx, moon_cy = 48, 40
        # 3 Glow Rings: radius 20/28/38, decreasing opacity 0.15/0.08/0.04
        glow_surf = pygame.Surface((self.w, self.h), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, (220, 230, 255, 10), (moon_cx, moon_cy), 38)
        pygame.draw.circle(glow_surf, (220, 230, 255, 20), (moon_cx, moon_cy), 28)
        pygame.draw.circle(glow_surf, (220, 230, 255, 38), (moon_cx, moon_cy), 20)
        surface.blit(glow_surf, (0, 0))
        # Solid Moon Core: radius 14px, RGB(220,230,255)
        pygame.draw.circle(surface, (220, 230, 255), (moon_cx, moon_cy), 14)

        # -------------------------------------------------------------
        # 3. 80 SEEDED STARS WITH CROSS HIGHLIGHTS
        # -------------------------------------------------------------
        for sx, sy, is_cross in self.stars:
            if sy < sky_h - 4:
                if is_cross:
                    pygame.draw.circle(surface, (200, 210, 255), (sx, sy), 2)
                    pygame.draw.line(surface, (200, 210, 255), (sx - 2, sy), (sx + 2, sy), 1)
                    pygame.draw.line(surface, (200, 210, 255), (sx, sy - 2), (sx, sy + 2), 1)
                else:
                    pygame.draw.circle(surface, (200, 210, 255), (sx, sy), 1)

        # -------------------------------------------------------------
        # 4. ROADBED & PERSPECTIVE ASPHALT GRID
        # -------------------------------------------------------------
        # Base Asphalt: RGB(20,24,30)
        for z_seg in range(80, -25, -5):
            # Left Grass Shoulder (X = -35m to -7.5m)
            g_l1 = self.project_3d_to_screen(np.array([-35.0, 0.0, float(z_seg)]), ego_x)
            g_l2 = self.project_3d_to_screen(np.array([-7.5, 0.0, float(z_seg)]), ego_x)
            g_l3 = self.project_3d_to_screen(np.array([-7.5, 0.0, float(z_seg - 5)]), ego_x)
            g_l4 = self.project_3d_to_screen(np.array([-35.0, 0.0, float(z_seg - 5)]), ego_x)
            if g_l1[0] != -9999 and g_l2[0] != -9999 and g_l3[0] != -9999 and g_l4[0] != -9999:
                pygame.draw.polygon(surface, (18, 28, 20), [(g_l1[0], g_l1[1]), (g_l2[0], g_l2[1]), (g_l3[0], g_l3[1]), (g_l4[0], g_l4[1])])

            # Right Grass Shoulder (X = +7.5m to +35m)
            g_r1 = self.project_3d_to_screen(np.array([+7.5, 0.0, float(z_seg)]), ego_x)
            g_r2 = self.project_3d_to_screen(np.array([+35.0, 0.0, float(z_seg)]), ego_x)
            g_r3 = self.project_3d_to_screen(np.array([+35.0, 0.0, float(z_seg - 5)]), ego_x)
            g_r4 = self.project_3d_to_screen(np.array([+7.5, 0.0, float(z_seg - 5)]), ego_x)
            if g_r1[0] != -9999 and g_r2[0] != -9999 and g_r3[0] != -9999 and g_r4[0] != -9999:
                pygame.draw.polygon(surface, (18, 28, 20), [(g_r1[0], g_r1[1]), (g_r2[0], g_r2[1]), (g_r3[0], g_r3[1]), (g_r4[0], g_r4[1])])

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
                pygame.draw.polygon(surface, (20, 24, 30), [(u1, v1), (u2, v2), (u3, v3), (u4, v4)])

        # Vanishing Point Road Glow Ellipse: RGB(0,100,255) alpha 0.08, radius 120x20
        vp_surf = pygame.Surface((self.w, self.h), pygame.SRCALPHA)
        pygame.draw.ellipse(vp_surf, (0, 100, 255, 20), pygame.Rect(int(self.w * 0.5 - 60), sky_h - 10, 120, 20))
        surface.blit(vp_surf, (0, 0))

        # Horizontal Grid Lines Every 8 Meters (RGB(28,32,38))
        for z_grid in range(-16, 75, 8):
            gp1 = self.project_3d_to_screen(np.array([-7.5, 0.01, float(z_grid)]), ego_x)
            gp2 = self.project_3d_to_screen(np.array([+7.5, 0.01, float(z_grid)]), ego_x)
            if gp1[0] != -9999 and gp2[0] != -9999:
                t_thick = 2 if z_grid < 10 else 1
                pygame.draw.line(surface, (28, 32, 38), (gp1[0], gp1[1]), (gp2[0], gp2[1]), t_thick)

        # Road Edge Rumble Strips: alternating RGB(40,50,60) and RGB(60,70,80) 5px wide repeating every 12px
        for z_rumble in np.arange(-20.0, 75.0, 3.0):
            r_col = (60, 70, 80) if (int(z_rumble // 3) % 2 == 0) else (40, 50, 60)
            # Left Strip (X = -5.8m)
            rp_l1 = self.project_3d_to_screen(np.array([-6.0, 0.02, z_rumble]), ego_x)
            rp_l2 = self.project_3d_to_screen(np.array([-5.6, 0.02, z_rumble + 2.5]), ego_x)
            if rp_l1[0] != -9999 and rp_l2[0] != -9999:
                pygame.draw.line(surface, r_col, (rp_l1[0], rp_l1[1]), (rp_l2[0], rp_l2[1]), 3)

            # Right Strip (X = +5.8m)
            rp_r1 = self.project_3d_to_screen(np.array([+5.6, 0.02, z_rumble]), ego_x)
            rp_r2 = self.project_3d_to_screen(np.array([+6.0, 0.02, z_rumble + 2.5]), ego_x)
            if rp_r1[0] != -9999 and rp_r2[0] != -9999:
                pygame.draw.line(surface, r_col, (rp_r1[0], rp_r1[1]), (rp_r2[0], rp_r2[1]), 3)

        # Solid Edge Lines: Yellow Left (RGB(255,210,0)), White Right (RGB(220,225,230))
        pts_left_edge = [self.project_3d_to_screen(np.array([-5.8, 0.02, float(z)]), ego_x) for z in range(-20, 80, 4)]
        valid_le = [(u, v) for u, v, _ in pts_left_edge if u != -9999]
        if len(valid_le) > 1:
            pygame.draw.lines(surface, (255, 210, 0), False, valid_le, 2)

        pts_right_edge = [self.project_3d_to_screen(np.array([+5.8, 0.02, float(z)]), ego_x) for z in range(-20, 80, 4)]
        valid_re = [(u, v) for u, v, _ in pts_right_edge if u != -9999]
        if len(valid_re) > 1:
            pygame.draw.lines(surface, (220, 225, 230), False, valid_re, 2)

        # Dashed Lane Lines (X = -1.875m and X = +1.875m)
        z_offset = (frame_idx * (ego.speed_kmh * 0.08)) % 8.0
        for x_lane in (-1.875, 1.875):
            for z_dash in np.arange(-18.0 + z_offset, 75.0, 8.0):
                p_d1 = np.array([x_lane, 0.02, float(z_dash)])
                p_d2 = np.array([x_lane, 0.02, float(z_dash + 3.8)])
                u_d1, v_d1, _ = self.project_3d_to_screen(p_d1, ego_x)
                u_d2, v_d2, _ = self.project_3d_to_screen(p_d2, ego_x)
                if u_d1 != -9999 and u_d2 != -9999:
                    pygame.draw.line(surface, (220, 225, 230), (u_d1, v_d1), (u_d2, v_d2), 2)

        # -------------------------------------------------------------
        # 5. EGO VEHICLE HEADLIGHT VOLUMETRIC CONES (Two Trapezoids)
        # -------------------------------------------------------------
        # Each cone: spread 25 deg each side, length 120px ahead, alpha lerp 0.18 -> 0.0
        cone_surf = pygame.Surface((self.w, self.h), pygame.SRCALPHA)
        p_hl_orig_l = np.array([ego.x - 0.75, 0.20, ego.z + 2.2])
        p_hl_orig_r = np.array([ego.x + 0.75, 0.20, ego.z + 2.2])
        p_hl_tip_l1 = np.array([ego.x - 4.5, 0.02, ego.z + 26.0])
        p_hl_tip_l2 = np.array([ego.x - 0.2, 0.02, ego.z + 26.0])
        p_hl_tip_r1 = np.array([ego.x + 0.2, 0.02, ego.z + 26.0])
        p_hl_tip_r2 = np.array([ego.x + 4.5, 0.02, ego.z + 26.0])

        u_ol, v_ol, _ = self.project_3d_to_screen(p_hl_orig_l, ego_x)
        u_tl1, v_tl1, _ = self.project_3d_to_screen(p_hl_tip_l1, ego_x)
        u_tl2, v_tl2, _ = self.project_3d_to_screen(p_hl_tip_l2, ego_x)
        if u_ol != -9999 and u_tl1 != -9999 and u_tl2 != -9999:
            pygame.draw.polygon(cone_surf, (200, 220, 255, 45), [(u_ol, v_ol), (u_tl1, v_tl1), (u_tl2, v_tl2)])

        u_or, v_or, _ = self.project_3d_to_screen(p_hl_orig_r, ego_x)
        u_tr1, v_tr1, _ = self.project_3d_to_screen(p_hl_tip_r1, ego_x)
        u_tr2, v_tr2, _ = self.project_3d_to_screen(p_hl_tip_r2, ego_x)
        if u_or != -9999 and u_tr1 != -9999 and u_tr2 != -9999:
            pygame.draw.polygon(cone_surf, (200, 220, 255, 45), [(u_or, v_or), (u_tr1, v_tr1), (u_tr2, v_tr2)])

        surface.blit(cone_surf, (0, 0))

        # -------------------------------------------------------------
        # 6. PHYSICS PARTICLES
        # -------------------------------------------------------------
        if particles:
            for p in particles:
                u_p, v_p, _ = self.project_3d_to_screen(np.array([p.x, p.y, p.z]), ego_x)
                if u_p != -9999 and 0 <= u_p < self.w and 0 <= v_p < self.h:
                    if p.p_type == "SPARK":
                        pygame.draw.circle(surface, p.color, (u_p, v_p), max(1, int(p.size)))
                    elif p.p_type == "SMOKE":
                        smoke_surf = pygame.Surface((int(p.size * 2), int(p.size * 2)), pygame.SRCALPHA)
                        pygame.draw.circle(smoke_surf, (220, 230, 240, 50), (int(p.size), int(p.size)), int(p.size))
                        surface.blit(smoke_surf, (u_p - int(p.size), v_p - int(p.size)))
                    else: # Exhaust
                        pygame.draw.circle(surface, p.color, (u_p, v_p), max(1, int(p.size * 0.75)))

        # -------------------------------------------------------------
        # 7. 3D CLOTHOID OVERTAKE TRAJECTORY CORRIDOR
        # -------------------------------------------------------------
        if ego.state in ("CHECK_OVERTAKE", "LANE_CHANGE_LEFT", "OVERTAKING", "LANE_CHANGE_RIGHT"):
            traj_3d = []
            target_lane_x = float(ego.target_lane_idx * 3.75)
            for s in np.linspace(2.0, 34.0, 20):
                ratio = min(1.0, s / 24.0)
                cur_x = ego.x + (target_lane_x - ego.x) * (10.0 * ratio**3 - 15.0 * ratio**4 + 6.0 * ratio**5)
                u_t, v_t, _ = self.project_3d_to_screen(np.array([cur_x, 0.08, float(s)]), ego_x)
                if u_t != -9999:
                    traj_3d.append((u_t, v_t))
            if len(traj_3d) > 1:
                pygame.draw.lines(surface, (0, 255, 180), False, traj_3d, 3)

        # -------------------------------------------------------------
        # 8. 3D SHADED VEHICLES (3 Faces, Antenna, Windows, Alloy Wheels)
        # -------------------------------------------------------------
        all_vehicles = list(traffic)
        all_vehicles.sort(key=lambda v: v.z, reverse=True)

        for v in all_vehicles:
            self.draw_3d_vehicle(surface, v, ego_x, frame_idx)

        # Draw 3D Ego Vehicle
        self.draw_3d_ego_vehicle(surface, ego, frame_idx)

        # -------------------------------------------------------------
        # 9. 3 HORIZONTAL DEPTH FOG BANDS NEAR HORIZON
        # -------------------------------------------------------------
        fog_surf = pygame.Surface((self.w, 40), pygame.SRCALPHA)
        # Band near horizon: RGB(12,18,30) alpha 0.0 -> 0.7 (bottom to top)
        for y_fog in range(40):
            alpha_val = int(140 * ((40 - y_fog) / 40.0))
            pygame.draw.line(fog_surf, (12, 18, 30, alpha_val), (0, y_fog), (self.w, y_fog))
        surface.blit(fog_surf, (0, sky_h - 20))

    def draw_3d_vehicle(self, surface: pygame.Surface, v: TrafficVehicle, ego_x: float, frame_idx: int):
        """Draws a 3-face 3D shaded vehicle block with antenna, window row, and alloy wheels."""
        hw, hl = v.width * 0.5, v.length * 0.5
        h = v.height
        x, z = v.x, v.z

        # Project 3D bounding box corners
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
        v_col = v.color

        # 1. FRONT/REAR FACE (Facing Viewer) — 100% Brightness
        pygame.draw.polygon(surface, v_col, [pts[0], pts[1], pts[5], pts[4]])
        pygame.draw.polygon(surface, (255, 255, 255), [pts[0], pts[1], pts[5], pts[4]], 1)

        # 2. TOP FACE — Parallelogram, vehicle_color * 0.85
        top_col = tuple(int(c * 0.85) for c in v_col)
        pygame.draw.polygon(surface, top_col, [pts[4], pts[5], pts[6], pts[7]])

        # 3. SIDE FACE (Right) — Parallelogram, vehicle_color * 0.60
        side_col = tuple(int(c * 0.60) for c in v_col)
        if x < ego_x:
            pygame.draw.polygon(surface, side_col, [pts[1], pts[2], pts[6], pts[5]])
        else:
            pygame.draw.polygon(surface, side_col, [pts[0], pts[3], pts[7], pts[4]])

        # 4. ROOF ANTENNA: 2px vertical line at center-top
        top_cx = (pts[4][0] + pts[5][0] + pts[6][0] + pts[7][0]) // 4
        top_cy = min(pts[4][1], pts[5][1], pts[6][1], pts[7][1])
        pygame.draw.line(surface, (200, 210, 220), (top_cx, top_cy), (top_cx, top_cy - 7), 2)

        # 5. WINDOW ROW: Across middle third of vehicle (RGB(45,60,80))
        u_mid_l = (pts[0][0] + pts[4][0]) // 2
        u_mid_r = (pts[1][0] + pts[5][0]) // 2
        v_mid_top = (pts[4][1] + pts[5][1]) // 2 + 2
        v_mid_bot = (pts[0][1] + pts[1][1]) // 2 - 2
        if v_mid_bot > v_mid_top:
            pygame.draw.polygon(surface, (45, 60, 80), [(u_mid_l, v_mid_top), (u_mid_r, v_mid_top),
                                                        (u_mid_r, v_mid_bot), (u_mid_l, v_mid_bot)])

        # 6. ALLOY WHEELS: 4 circles at bottom corners (radius 4px RGB(15,15,15) with RGB(50,50,60) rim inside radius 2px)
        for wx, wy in (pts[0], pts[1]):
            pygame.draw.circle(surface, (15, 15, 15), (wx, wy), 4)
            pygame.draw.circle(surface, (50, 50, 60), (wx, wy), 2)

        # 7. RED TAILLIGHT GLOW ELLIPSES (RGB(255,30,0) alpha 0.4, size 8x4px)
        tl_surf = pygame.Surface((16, 10), pygame.SRCALPHA)
        pygame.draw.ellipse(tl_surf, (255, 30, 0, 100), pygame.Rect(0, 0, 16, 10))
        surface.blit(tl_surf, (pts[0][0] - 8, pts[0][1] - 12))
        surface.blit(tl_surf, (pts[1][0] - 8, pts[1][1] - 12))
        pygame.draw.circle(surface, (255, 40, 40), (pts[0][0], pts[0][1] - 8), 3)
        pygame.draw.circle(surface, (255, 40, 40), (pts[1][0], pts[1][1] - 8), 3)

        # 8. 3D Label Tag
        u_top = (pts[4][0] + pts[5][0]) // 2
        v_top = min(pts[4][1], pts[5][1]) - 10
        lbl_surf = pygame.font.SysFont("segoeui", 10, bold=True).render(f"{v.id} [{v.speed_kmh:.0f} km/h]", True, (0, 255, 180) if "LEAD" in v.id else (255, 210, 0))
        surface.blit(lbl_surf, (u_top - lbl_surf.get_width() // 2, v_top - 6))

    def draw_3d_ego_vehicle(self, surface: pygame.Surface, ego: EgoAutonomousVehicle, frame_idx: int):
        """Draws the hero 3D Ego vehicle with shaded bodywork and glass roof."""
        hw = ego.width * 0.5
        hl = ego.length * 0.5
        h = ego.height
        x, z = ego.x, ego.z

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

        # Metallic Blue Bodywork
        pygame.draw.polygon(surface, (18, 55, 95), [pts[0], pts[1], pts[5], pts[4]])
        pygame.draw.polygon(surface, (0, 200, 255), [pts[0], pts[1], pts[5], pts[4]], 2)

        # Glass Roof
        pygame.draw.polygon(surface, (30, 60, 85), [pts[4], pts[5], pts[6], pts[7]])
        pygame.draw.polygon(surface, (0, 230, 255), [pts[4], pts[5], pts[6], pts[7]], 2)

        # LED Taillight Lightbar
        pygame.draw.line(surface, (255, 30, 30), (pts[0][0] + 4, pts[0][1] - 8), (pts[1][0] - 4, pts[1][1] - 8), 3)

        # Wheels
        for wx, wy in (pts[0], pts[1]):
            pygame.draw.circle(surface, (15, 15, 15), (wx, wy), 4)
            pygame.draw.circle(surface, (0, 200, 255), (wx, wy), 2)
