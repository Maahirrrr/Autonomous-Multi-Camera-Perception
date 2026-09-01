"""
digital_twin_3d_renderer.py — Cinematic 3D Digital Twin World & Particle Visualizer
===================================================================================
Features:
  - 3D Perspective Highway with Multi-Lane Asphalt, Grass Shoulders & Dynamic Sky Dome.
  - Day / Dusk / Night Lighting Modes with Volumetric Headlight Illumination Cones.
  - Integrated Physics Particles: Tire smoke on braking, sparks on guardrails & exhaust plumes.
  - 3D V2X Concentric Radio Pulse Waves radiating from the lead connected vehicle.
  - 3D Vehicle Suspension Dynamics: Real-Time Pitch (accel/brake) & Roll (cornering tilt).
"""

import math
import random
import numpy as np
import pygame

from traffic_physics_simulator import EgoAutonomousVehicle, TrafficVehicle, Particle


class DigitalTwin3DRenderer:
    """
    Renders the cinematic 3D Digital Twin highway, particles, V2X radio rings, and HUD ribbons.
    """

    def __init__(self, screen_w: int = 440, screen_h: int = 515):
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
        t = frame_idx * 0.035
        ego_x = ego.x

        # 1. Sky Gradient & Horizon
        sky_h = int(self.h * 0.52)
        if night_mode:
            # Deep Starry Night
            surface.fill((8, 10, 16), pygame.Rect(0, 0, self.w, sky_h))
            pygame.draw.polygon(surface, (15, 20, 30), [(0, sky_h), (60, sky_h - 22), (140, sky_h - 12), (240, sky_h - 26), (self.w, sky_h)])
        elif weather_mode == "FOG":
            # Dense Fog Haze
            surface.fill((130, 140, 150), pygame.Rect(0, 0, self.w, sky_h))
        elif weather_mode == "RAIN":
            # Stormy Overcast Sky
            for y in range(sky_h):
                ratio = y / float(max(1, sky_h))
                surface.fill((int(30 + ratio*20), int(38 + ratio*20), int(48 + ratio*20)), pygame.Rect(0, y, self.w, 1))
        else:
            # Sunset Horizon Glow
            for y in range(sky_h):
                ratio = y / float(max(1, sky_h))
                b = int(48 + ratio * 22)
                g = int(28 + ratio * 32)
                r = int(22 + ratio * 42)
                pygame.draw.line(surface, (r, g, b), (0, y), (self.w, y))
            pygame.draw.polygon(surface, (28, 38, 52), [(0, sky_h), (60, sky_h - 26), (130, sky_h - 14), (210, sky_h - 32), (320, sky_h - 16), (self.w, sky_h)])

        # 2. 3D Highway Asphalt Surface & Grass Shoulders
        for z_seg in range(80, -25, -5):
            # Left Grass Shoulder (X = -35m to -7.5m)
            g_l1 = self.project_3d_to_screen(np.array([-35.0, 0.0, float(z_seg)]), ego_x)
            g_l2 = self.project_3d_to_screen(np.array([-7.5, 0.0, float(z_seg)]), ego_x)
            g_l3 = self.project_3d_to_screen(np.array([-7.5, 0.0, float(z_seg - 5)]), ego_x)
            g_l4 = self.project_3d_to_screen(np.array([-35.0, 0.0, float(z_seg - 5)]), ego_x)
            if g_l1[0] != -9999 and g_l2[0] != -9999 and g_l3[0] != -9999 and g_l4[0] != -9999:
                g_shade = max(14, int((32 - (z_seg / 80.0) * 16) * (0.5 if night_mode else 1.0)))
                pygame.draw.polygon(surface, (int(g_shade * 0.65), g_shade + 10, int(g_shade * 0.55)), [(g_l1[0], g_l1[1]), (g_l2[0], g_l2[1]), (g_l3[0], g_l3[1]), (g_l4[0], g_l4[1])])

            # Right Grass Shoulder (X = +7.5m to +35m)
            g_r1 = self.project_3d_to_screen(np.array([+7.5, 0.0, float(z_seg)]), ego_x)
            g_r2 = self.project_3d_to_screen(np.array([+35.0, 0.0, float(z_seg)]), ego_x)
            g_r3 = self.project_3d_to_screen(np.array([+35.0, 0.0, float(z_seg - 5)]), ego_x)
            g_r4 = self.project_3d_to_screen(np.array([+7.5, 0.0, float(z_seg - 5)]), ego_x)
            if g_r1[0] != -9999 and g_r2[0] != -9999 and g_r3[0] != -9999 and g_r4[0] != -9999:
                g_shade = max(14, int((32 - (z_seg / 80.0) * 16) * (0.5 if night_mode else 1.0)))
                pygame.draw.polygon(surface, (int(g_shade * 0.65), g_shade + 10, int(g_shade * 0.55)), [(g_r1[0], g_r1[1]), (g_r2[0], g_r2[1]), (g_r3[0], g_r3[1]), (g_r4[0], g_r4[1])])

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
                shade = max(16, int((42 - (z_seg / 80.0) * 18) * (0.45 if night_mode else 1.0)))
                if weather_mode == "RAIN":
                    shade += 6 # Wet specular gloss
                pygame.draw.polygon(surface, (shade, shade + 2, shade + 5), [(u1, v1), (u2, v2), (u3, v3), (u4, v4)])

        # 3. 3D Highway Lane Dividers & Solid Edge Lines
        pts_left_edge = [self.project_3d_to_screen(np.array([-5.8, 0.02, float(z)]), ego_x) for z in range(-20, 80, 4)]
        valid_le = [(u, v) for u, v, _ in pts_left_edge if u != -9999]
        if len(valid_le) > 1:
            pygame.draw.lines(surface, (0, 215, 255), False, valid_le, 2)

        pts_right_edge = [self.project_3d_to_screen(np.array([+5.8, 0.02, float(z)]), ego_x) for z in range(-20, 80, 4)]
        valid_re = [(u, v) for u, v, _ in pts_right_edge if u != -9999]
        if len(valid_re) > 1:
            pygame.draw.lines(surface, (230, 235, 240), False, valid_re, 2)

        # Dashed Lane Line 1 (X = -1.875m)
        z_offset = (frame_idx * (ego.speed_kmh * 0.08)) % 8.0
        for z_dash in np.arange(-18.0 + z_offset, 75.0, 8.0):
            p_d1 = np.array([-1.875, 0.02, float(z_dash)])
            p_d2 = np.array([-1.875, 0.02, float(z_dash + 3.8)])
            u_d1, v_d1, _ = self.project_3d_to_screen(p_d1, ego_x)
            u_d2, v_d2, _ = self.project_3d_to_screen(p_d2, ego_x)
            if u_d1 != -9999 and u_d2 != -9999:
                pygame.draw.line(surface, (200, 210, 220), (u_d1, v_d1), (u_d2, v_d2), 2)

        # Dashed Lane Line 2 (X = +1.875m)
        for z_dash in np.arange(-18.0 + z_offset, 75.0, 8.0):
            p_d1 = np.array([+1.875, 0.02, float(z_dash)])
            p_d2 = np.array([+1.875, 0.02, float(z_dash + 3.8)])
            u_d1, v_d1, _ = self.project_3d_to_screen(p_d1, ego_x)
            u_d2, v_d2, _ = self.project_3d_to_screen(p_d2, ego_x)
            if u_d1 != -9999 and u_d2 != -9999:
                pygame.draw.line(surface, (200, 210, 220), (u_d1, v_d1), (u_d2, v_d2), 2)

        # 4. 3D V2X Pulsing Concentric Radio Wave Rings (Lead Vehicle)
        lead_car = next((v for v in traffic if abs(v.x - ego.x) < 2.0 and v.z > 0), None)
        if lead_car:
            pulse_radius = (frame_idx * 0.45) % 18.0
            r_pts = []
            for ang_deg in range(0, 360, 24):
                rad = math.radians(ang_deg)
                px = lead_car.x + pulse_radius * math.sin(rad)
                pz = lead_car.z + pulse_radius * math.cos(rad)
                u_r, v_r, _ = self.project_3d_to_screen(np.array([px, 0.05, pz]), ego_x)
                if u_r != -9999:
                    r_pts.append((u_r, v_r))
            if len(r_pts) > 6:
                alpha_val = max(0, int(255 * (1.0 - pulse_radius / 18.0)))
                pygame.draw.lines(surface, (0, 255, 220), True, r_pts, 2)

        # 5. Physics Particles (Tire smoke, sparks, exhaust)
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

        # 6. 3D LiDAR Point Cloud Returns
        if point_cloud is not None and len(point_cloud) > 0:
            for pt in point_cloud[::3]:
                px, py, pz, intensity, rng = pt
                u_pt, v_pt, _ = self.project_3d_to_screen(np.array([px + ego_x, py, pz]), ego_x)
                if u_pt != -9999 and 0 <= u_pt < self.w and 0 <= v_pt < self.h:
                    if py >= 0.28:
                        col = (0, 255, 255) if rng > 14.0 else (0, 180, 255)
                        pygame.draw.circle(surface, col, (u_pt, v_pt), 2)
                    else:
                        pygame.draw.circle(surface, (0, int(160 * intensity), int(80 * intensity)), (u_pt, v_pt), 1)

        # 7. Animated Neon Clothoid Trajectory Ribbon
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
                pygame.draw.lines(surface, (0, 255, 180), False, traj_3d, 4)
                p_idx = int((frame_idx * 1.2) % len(traj_3d))
                pu, pv = traj_3d[p_idx]
                pygame.draw.circle(surface, (255, 255, 255), (pu, pv), 5)
                pygame.draw.circle(surface, (0, 255, 180), (pu, pv), 3)

        # 8. Render 3D Surrounding Vehicles
        all_vehicles = list(traffic)
        all_vehicles.sort(key=lambda v: v.z, reverse=True)

        for v in all_vehicles:
            self.draw_3d_vehicle(surface, v, ego_x, frame_idx, night_mode=night_mode)

        # 9. Render 3D Ego Tesla Vehicle
        self.draw_3d_ego_vehicle(surface, ego, frame_idx, night_mode=night_mode)

    def draw_3d_vehicle(
        self,
        surface: pygame.Surface,
        v: TrafficVehicle,
        ego_x: float,
        frame_idx: int,
        night_mode: bool = False
    ):
        hw, hl = v.width * 0.5, v.length * 0.5
        h = v.height
        x, z = v.x, v.z

        sin_r = math.sin(math.radians(v.roll_deg))
        sin_p = math.sin(math.radians(v.pitch_deg))

        corners_3d = [
            np.array([x - hw, 0.15 + hw * sin_r - hl * sin_p, z - hl]), # 0
            np.array([x + hw, 0.15 - hw * sin_r - hl * sin_p, z - hl]), # 1
            np.array([x + hw, 0.15 - hw * sin_r + hl * sin_p, z + hl]), # 2
            np.array([x - hw, 0.15 + hw * sin_r + hl * sin_p, z + hl]), # 3
            np.array([x - hw, h    + hw * sin_r - hl * sin_p, z - hl]), # 4
            np.array([x + hw, h    - hw * sin_r - hl * sin_p, z - hl]), # 5
            np.array([x + hw, h    - hw * sin_r + hl * sin_p, z + hl]), # 6
            np.array([x - hw, h    + hw * sin_r + hl * sin_p, z + hl]), # 7
        ]

        proj = [self.project_3d_to_screen(c, ego_x) for c in corners_3d]
        if any(u == -9999 for u, v_p, _ in proj):
            return

        pts = [(u, v_p) for u, v_p, _ in proj]

        # Vehicle Body
        v_col = tuple(max(10, int(c * (0.6 if night_mode else 1.0))) for c in v.color)
        pygame.draw.polygon(surface, v_col, [pts[0], pts[1], pts[5], pts[4]])
        pygame.draw.polygon(surface, (255, 255, 255), [pts[0], pts[1], pts[5], pts[4]], 1)

        top_color = (min(255, v_col[0] + 30), min(255, v_col[1] + 30), min(255, v_col[2] + 30))
        pygame.draw.polygon(surface, top_color, [pts[4], pts[5], pts[6], pts[7]])

        if x < ego_x:
            pygame.draw.polygon(surface, (int(v_col[0]*0.8), int(v_col[1]*0.8), int(v_col[2]*0.8)), [pts[1], pts[2], pts[6], pts[5]])
        else:
            pygame.draw.polygon(surface, (int(v_col[0]*0.8), int(v_col[1]*0.8), int(v_col[2]*0.8)), [pts[0], pts[3], pts[7], pts[4]])

        # Rear Red LED Taillights / Brake Lights with Glow
        u_rl = (pts[0][0] + pts[4][0]) // 2 + 3
        v_rl = (pts[0][1] + pts[4][1]) // 2
        u_rr = (pts[1][0] + pts[5][0]) // 2 - 3
        v_rr = (pts[1][1] + pts[5][1]) // 2

        tail_col = (255, 40, 40) if v.is_braking else (180, 0, 0)
        tail_r = 5 if v.is_braking else 3
        pygame.draw.circle(surface, tail_col, (u_rl, v_rl), tail_r)
        pygame.draw.circle(surface, tail_col, (u_rr, v_rr), tail_r)

        if night_mode or v.is_braking:
            glow_surf = pygame.Surface((20, 20), pygame.SRCALPHA)
            pygame.draw.circle(glow_surf, (255, 20, 20, 90), (10, 10), 9)
            surface.blit(glow_surf, (u_rl - 10, v_rl - 10))
            surface.blit(glow_surf, (u_rr - 10, v_rr - 10))

        # Amber Blinkers
        is_flash = (frame_idx % 24) < 12
        if v.blinker == "LEFT" and is_flash:
            pygame.draw.circle(surface, (255, 180, 0), (u_rl - 4, v_rl), 5)
        elif v.blinker == "RIGHT" and is_flash:
            pygame.draw.circle(surface, (255, 180, 0), (u_rr + 4, v_rr), 5)

        # 3D Label Tag
        u_top = (pts[4][0] + pts[5][0]) // 2
        v_top = min(pts[4][1], pts[5][1]) - 8
        lbl_text = f"{v.id} [{v.speed_kmh:.0f} km/h]"
        lbl_surf = pygame.font.SysFont("segoeui", 10, bold=True).render(lbl_text, True, (0, 255, 180) if "LEAD" in v.id else (255, 210, 0))
        surface.blit(lbl_surf, (u_top - lbl_surf.get_width() // 2, v_top - 6))

    def draw_3d_ego_vehicle(self, surface: pygame.Surface, ego: EgoAutonomousVehicle, frame_idx: int, night_mode: bool = False):
        hw = ego.width * 0.5
        hl = ego.length * 0.5
        h = ego.height
        x = ego.x
        z = ego.z

        sin_r = math.sin(math.radians(ego.roll_deg))

        corners_3d = [
            np.array([x - hw, 0.15 + hw * sin_r, z - hl]), # 0
            np.array([x + hw, 0.15 - hw * sin_r, z - hl]), # 1
            np.array([x + hw, 0.15 - hw * sin_r, z + hl]), # 2
            np.array([x - hw, 0.15 + hw * sin_r, z + hl]), # 3
            np.array([x - hw, h    + hw * sin_r, z - hl]), # 4
            np.array([x + hw, h    - hw * sin_r, z - hl]), # 5
            np.array([x + hw, h    - hw * sin_r, z + hl]), # 6
            np.array([x - hw, h    + hw * sin_r, z + hl]), # 7
        ]

        proj = [self.project_3d_to_screen(c, ego.x) for c in corners_3d]
        if any(u == -9999 for u, v_p, _ in proj):
            return

        pts = [(u, v_p) for u, v_p, _ in proj]

        # Headlight Light Cones Casting Forward onto Road
        p_hl_left = np.array([x - hw - 2.8, 0.02, z + 24.0])
        p_hl_right = np.array([x + hw + 2.8, 0.02, z + 24.0])
        u_hl1, v_hl1, _ = self.project_3d_to_screen(p_hl_left, ego.x)
        u_hl2, v_hl2, _ = self.project_3d_to_screen(p_hl_right, ego.x)
        if u_hl1 != -9999 and u_hl2 != -9999:
            hl_poly = [pts[3], pts[2], (u_hl2, v_hl2), (u_hl1, v_hl1)]
            hl_surf = pygame.Surface((self.w, self.h), pygame.SRCALPHA)
            alpha_hl = 70 if night_mode else 35
            pygame.draw.polygon(hl_surf, (220, 240, 255, alpha_hl), hl_poly)
            surface.blit(hl_surf, (0, 0))

        # 3D Metallic Blue Body
        pygame.draw.polygon(surface, (18, 45, 85), [pts[0], pts[1], pts[5], pts[4]])
        pygame.draw.polygon(surface, (0, 200, 255), [pts[0], pts[1], pts[5], pts[4]], 2)

        # Glass Panoramic Roof
        pygame.draw.polygon(surface, (30, 50, 75), [pts[4], pts[5], pts[6], pts[7]])
        pygame.draw.polygon(surface, (0, 230, 255), [pts[4], pts[5], pts[6], pts[7]], 2)

        # Taillight LED Lightbar
        u_tl1 = (pts[0][0] + pts[4][0]) // 2
        u_tl2 = (pts[1][0] + pts[5][0]) // 2
        v_tl = (pts[0][1] + pts[4][1]) // 2
        pygame.draw.line(surface, (255, 30, 30), (u_tl1, v_tl), (u_tl2, v_tl), 3)

        # Amber Turn Signal Blinkers
        is_flash = (frame_idx % 24) < 12
        if ego.blinker == "LEFT" and is_flash:
            pygame.draw.circle(surface, (255, 180, 0), (u_tl1 - 6, v_tl), 6)
            pygame.draw.circle(surface, (255, 255, 255), (u_tl1 - 6, v_tl), 3)
        elif ego.blinker == "RIGHT" and is_flash:
            pygame.draw.circle(surface, (255, 180, 0), (u_tl2 + 6, v_tl), 6)
            pygame.draw.circle(surface, (255, 255, 255), (u_tl2 + 6, v_tl), 3)

        # Autopilot State Tag
        state_tag = f"L4 AUTOPILOT: {ego.state}"
        s_surf = pygame.font.SysFont("segoeui", 10, bold=True).render(state_tag, True, (0, 255, 180))
        surface.blit(s_surf, (self.w // 2 - s_surf.get_width() // 2, min(pts[4][1], pts[5][1]) - 16))
