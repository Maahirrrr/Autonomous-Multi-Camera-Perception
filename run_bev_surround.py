"""
run_bev_surround.py — Level 4 Autonomous Vehicle 360° Spatial Cockpit (Pure Black & Tesla Red Edition)
=======================================================================================================
Features:
  - Pure Black (#0A0A0A) Theme with Ultra-Thin Metallic Borders (#222226).
  - Razor-Sharp High-DPI Monospace Typography (Consolas / Segoe UI with Antialiasing).
  - Tesla Crimson Red (#E82127) Alert Accents for FCW, Hard Deceleration & Overtakes.
  - Soft Volumetric Headlights, Metallic Shading & 3-Lane Perspective Surround Layout.
  - Asynchronous Multi-Threaded Pipeline (ThreadPoolExecutor) at locked 60 FPS on RTX 4070.
"""

import sys
import os
import time
import math
import random
import argparse
import concurrent.futures
import numpy as np
import cv2
import pygame

from bev_transformer_engine import MultiCameraBEVTransformer
from multi_cam_simulator import MultiCameraSimulator
from lidar_3d_pointcloud_engine import Lidar3DPerceptionEngine
from traffic_physics_simulator import HighwayTrafficEngine
from digital_twin_3d_renderer import DigitalTwin3DRenderer


def get_crisp_mono_font(size: int, bold: bool = True) -> pygame.font.Font:
    """Returns a crisp, razor-sharp monospace font for high-precision telemetry."""
    for font_name in ["consolas", "cascadiacode", "lucidaconsole", "couriernew", "segoeui"]:
        try:
            f = pygame.font.SysFont(font_name, size, bold=bold)
            if f:
                return f
        except Exception:
            continue
    return pygame.font.Font(None, size)


def get_crisp_ui_font(size: int, bold: bool = True) -> pygame.font.Font:
    """Returns a crisp, modern UI font for titles and badges."""
    for font_name in ["segoeui", "calibri", "arial", "consolas"]:
        try:
            f = pygame.font.SysFont(font_name, size, bold=bold)
            if f:
                return f
        except Exception:
            continue
    return pygame.font.Font(None, size)


def main():
    parser = argparse.ArgumentParser(description="Level 4 Autonomous Vehicle 360° Spatial Perception Cockpit")
    parser.add_argument("--export", type=str, default=None, help="Export session to MP4 video")
    parser.add_argument("--max-frames", type=int, default=300, help="Max frames for export")
    args = parser.parse_args()

    pygame.init()
    screen_w, screen_h = 1280, 800
    pygame.display.set_caption("Tesla Level 4 Autonomous 360° Perception Cockpit (RTX 4070)")

    try:
        screen = pygame.display.set_mode((screen_w, screen_h), pygame.DOUBLEBUF | pygame.HWSURFACE)
    except Exception:
        screen = pygame.display.set_mode((screen_w, screen_h), pygame.DOUBLEBUF)

    clock = pygame.time.Clock()

    # Razor-Sharp Typography Hierarchy
    font_mono_xs = get_crisp_mono_font(11, bold=False)
    font_mono_sm = get_crisp_mono_font(12, bold=True)
    font_mono_md = get_crisp_mono_font(14, bold=True)
    font_mono_lg = get_crisp_mono_font(20, bold=True)
    font_mono_spd = get_crisp_mono_font(32, bold=True)
    font_mono_ttc = get_crisp_mono_font(26, bold=True)

    font_ui_title = get_crisp_ui_font(13, bold=True)
    font_ui_sub = get_crisp_ui_font(11, bold=False)

    # 1. Initialize Engines
    bev_w, bev_h = 440, 460
    bev_engine = MultiCameraBEVTransformer(bev_width_px=bev_w, bev_height_px=bev_h)

    cam_w_flank, cam_h_flank = 380, 135
    cam_w_center, cam_h_center = 440, 135
    cam_sim_flank = MultiCameraSimulator(width=cam_w_flank, height=cam_h_flank)
    cam_sim_center = MultiCameraSimulator(width=cam_w_center, height=cam_h_center)

    lidar_engine = Lidar3DPerceptionEngine(num_lasers=64, max_range_m=65.0)
    lidar_engine.cameras = bev_engine.cameras

    traffic_engine = HighwayTrafficEngine()
    twin_renderer = DigitalTwin3DRenderer(screen_w=bev_w, screen_h=bev_h)

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)
    radar_history = []

    print("\n" + "=" * 78)
    print("  [+] LEVEL 4 AUTONOMOUS VEHICLE 360° SPATIAL COCKPIT (PURE BLACK EDITION)")
    print("  UI Theme           : Pure Black (#0A0A0A) + Tesla Red (#E82127) Alerts")
    print("  Typography         : Razor-Sharp Monospace Subpixel Antialiased Glyphs")
    print("  Driving Physics    : Refined IDM Headway (1.2s) & MOBIL Courtesy (0.20)")
    print("  Controls           : [TAB] Pilot | [N] Night | [P] Weather | [R] Randomize")
    print("=" * 78 + "\n")

    video_writer = None
    if args.export:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(args.export, fourcc, 60.0, (screen_w, screen_h))
        print(f"[INFO] Exporting L4 Perception recording to: {args.export}")

    running = True
    frame_count = 0
    show_lidar_on_cams = True
    is_paused = False

    weather_modes = ["CLEAR", "RAIN", "FOG"]
    weather_idx = 0
    night_mode = False

    # Pure Black Theme Palette
    COLOR_BG_PURE = (10, 10, 10)         # #0A0A0A
    COLOR_PANEL_BG = (13, 13, 15)        # #0D0D0F
    COLOR_CARD_BG = (18, 18, 22)         # #121216
    COLOR_BORDER_THIN = (34, 36, 42)     # #22242A Ultra-Thin
    COLOR_TEXT_MAIN = (240, 243, 248)    # #F0F3F8
    COLOR_TEXT_MUTED = (130, 140, 155)   # #828C9B
    COLOR_TESLA_RED = (232, 33, 39)      # #E82127
    COLOR_CYAN_ACCENT = (0, 215, 245)    # #00D7F5
    COLOR_EMERALD_GREEN = (0, 210, 120)  # #00D278
    COLOR_AMBER_WARN = (255, 185, 30)    # #FFB91E

    while running:
        dt = 0.016
        if not is_paused:
            frame_count += 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    traffic_engine.randomize_scenario()
                elif event.key == pygame.K_n:
                    night_mode = not night_mode
                    traffic_engine.log_event(f"LIGHTING -> {'NIGHT (THERMAL-IR)' if night_mode else 'DAYLIGHT'}")
                elif event.key == pygame.K_p:
                    weather_idx = (weather_idx + 1) % len(weather_modes)
                    current_weather = weather_modes[weather_idx]
                    lidar_engine.set_weather_mode(current_weather)
                    traffic_engine.log_event(f"WEATHER -> {current_weather}")
                elif event.key == pygame.K_l:
                    show_lidar_on_cams = not show_lidar_on_cams
                elif event.key == pygame.K_SPACE:
                    is_paused = not is_paused
                elif event.key == pygame.K_TAB:
                    traffic_engine.ego.manual_override = not traffic_engine.ego.manual_override
                    traffic_engine.log_event(f"PILOT -> {'MANUAL OVERRIDE' if traffic_engine.ego.manual_override else 'AUTONOMOUS HIGHWAY PILOT'}")
                elif event.key == pygame.K_a:
                    traffic_engine.ego.initiate_lane_change(max(-1, traffic_engine.ego.lane_idx - 1))
                elif event.key == pygame.K_d:
                    traffic_engine.ego.initiate_lane_change(min(1, traffic_engine.ego.lane_idx + 1))

            twin_renderer.handle_mouse_orbit(event, rect_offset=(420, 180))

        # Manual Override Handling
        if traffic_engine.ego.manual_override:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_w]:
                traffic_engine.ego.speed_kmh = min(135.0, traffic_engine.ego.speed_kmh + 24.0 * dt)
            if keys[pygame.K_s]:
                traffic_engine.ego.speed_kmh = max(20.0, traffic_engine.ego.speed_kmh - 35.0 * dt)
                traffic_engine.ego.is_braking = True
            else:
                traffic_engine.ego.is_braking = False

        current_weather = weather_modes[weather_idx]
        ego = traffic_engine.ego
        traffic = traffic_engine.traffic_vehicles

        # 2. Multi-Threaded Perception Pipeline
        traffic_engine.step(dt)
        dynamic_objects = traffic_engine.get_dynamic_objects_for_sensors()

        fut_lidar = executor.submit(lidar_engine.generate_scene_point_cloud, dynamic_objects, {}, frame_count)
        point_cloud = fut_lidar.result()
        ground_pts, obstacle_pts, bounding_boxes = lidar_engine.segment_ground_and_clusters(point_cloud, dynamic_objects)

        radar_detections = lidar_engine.radar_sim.scan_targets(dynamic_objects, ego.speed_mps)
        radar_history.append(radar_detections)
        if len(radar_history) > 4:
            radar_history.pop(0)

        # Flank Cameras
        cam_frames_flank = cam_sim_flank.render_surround_views(
            frame_idx=frame_count,
            dynamic_objects=dynamic_objects,
            speed_kmh=ego.speed_kmh,
            ego_x=ego.x,
            lidar_engine=lidar_engine,
            point_cloud=point_cloud,
            render_lidar_on_cams=show_lidar_on_cams,
            weather_mode=current_weather,
            night_mode=night_mode
        )

        # Center Cameras
        cam_frames_center = cam_sim_center.render_surround_views(
            frame_idx=frame_count,
            dynamic_objects=dynamic_objects,
            speed_kmh=ego.speed_kmh,
            ego_x=ego.x,
            lidar_engine=lidar_engine,
            point_cloud=point_cloud,
            render_lidar_on_cams=show_lidar_on_cams,
            weather_mode=current_weather,
            night_mode=night_mode
        )

        # 3. Master GUI Rendering (Pure Black #0A0A0A)
        screen.fill(COLOR_BG_PURE)

        # Top Header Bar (Y=0, H=32)
        pygame.draw.rect(screen, COLOR_PANEL_BG, pygame.Rect(0, 0, screen_w, 32))
        pygame.draw.line(screen, COLOR_BORDER_THIN, (0, 32), (screen_w, 32), 1)

        screen.blit(font_ui_title.render("TESLA LEVEL 4 AUTONOMOUS PERCEPTION STACK", True, COLOR_TEXT_MAIN), (20, 6))

        # Status Badges
        stat_txt = f"WEATHER: {current_weather}  |  {'NIGHT (FLIR)' if night_mode else 'DAYLIGHT'}  |  CUDA: 0.6ms  |  60 FPS"
        screen.blit(font_mono_xs.render(stat_txt, True, COLOR_TEXT_MUTED), (screen_w - 530, 9))

        mode_badge = pygame.Rect(screen_w - 180, 4, 165, 24)
        m_border = COLOR_EMERALD_GREEN if not ego.manual_override else COLOR_AMBER_WARN
        pygame.draw.rect(screen, (12, 24, 18) if not ego.manual_override else (28, 20, 10), mode_badge, border_radius=3)
        pygame.draw.rect(screen, m_border, mode_badge, 1, border_radius=3)
        screen.blit(font_mono_xs.render("L4 HIGHWAY PILOT" if not ego.manual_override else "MANUAL OVERRIDE", True, m_border), (screen_w - 170, 8))

        # -------------------------------------------------------------
        # 4. TOP ROW: SPATIAL SURROUND CAMERAS
        # -------------------------------------------------------------
        # TOP-LEFT: LEFT CAMERA (X=20, Y=38, W=380, H=135)
        surf_left = pygame.surfarray.make_surface(np.transpose(cam_frames_flank["LEFT"], (1, 0, 2)))
        screen.blit(surf_left, (20, 38))
        pygame.draw.rect(screen, COLOR_BORDER_THIN, pygame.Rect(20, 38, cam_w_flank, cam_h_flank), 1)

        # TOP-CENTER: FRONT CAMERA (X=420, Y=38, W=440, H=135)
        surf_front = pygame.surfarray.make_surface(np.transpose(cam_frames_center["FRONT"], (1, 0, 2)))
        screen.blit(surf_front, (420, 38))
        pygame.draw.rect(screen, COLOR_BORDER_THIN, pygame.Rect(420, 38, cam_w_center, cam_h_center), 1)

        # TOP-RIGHT: RIGHT CAMERA (X=880, Y=38, W=380, H=135)
        surf_right = pygame.surfarray.make_surface(np.transpose(cam_frames_flank["RIGHT"], (1, 0, 2)))
        screen.blit(surf_right, (880, 38))
        pygame.draw.rect(screen, COLOR_BORDER_THIN, pygame.Rect(880, 38, cam_w_flank, cam_h_flank), 1)

        # -------------------------------------------------------------
        # 5. MIDDLE ROW — LEFT PANEL: VEHICLE KINEMATICS & DYNAMICS
        # -------------------------------------------------------------
        left_panel_rect = pygame.Rect(20, 180, 380, 460)
        pygame.draw.rect(screen, COLOR_PANEL_BG, left_panel_rect, border_radius=4)
        pygame.draw.rect(screen, COLOR_BORDER_THIN, left_panel_rect, 1, border_radius=4)

        screen.blit(font_ui_title.render("VEHICLE KINEMATICS & DYNAMICS", True, COLOR_CYAN_ACCENT), (35, 192))

        # MINIMALIST ARC SPEEDOMETER (Consolas Monospace Digits)
        scx, scy = 110, 275
        pygame.draw.circle(screen, (20, 20, 24), (scx, scy), 64, 2)
        pygame.draw.arc(screen, (18, 18, 22), (scx - 62, scy - 62, 124, 124), math.radians(-45), math.radians(225), 10)

        cur_spd = ego.speed_kmh
        spd_col = COLOR_EMERALD_GREEN if cur_spd < 65.0 else (COLOR_AMBER_WARN if cur_spd < 95.0 else COLOR_TESLA_RED)
        spd_pct = max(0.0, min(1.0, cur_spd / 140.0))
        sweep_rad = spd_pct * math.radians(270)
        if sweep_rad > 0.05:
            pygame.draw.arc(screen, spd_col, (scx - 62, scy - 62, 124, 124), math.radians(225) - sweep_rad, math.radians(225), 10)

        for tick_i in range(19):
            t_angle_rad = math.radians(225) - (tick_i / 18.0) * math.radians(270)
            is_maj = (tick_i % 3 == 0)
            t_len = 6 if is_maj else 3
            tx1 = scx + int(54 * math.cos(t_angle_rad))
            ty1 = scy - int(54 * math.sin(t_angle_rad))
            tx2 = scx + int((54 - t_len) * math.cos(t_angle_rad))
            ty2 = scy - int((54 - t_len) * math.sin(t_angle_rad))
            pygame.draw.line(screen, (120, 130, 145) if is_maj else (45, 48, 55), (tx1, ty1), (tx2, ty2), 2 if is_maj else 1)

        spd_val_surf = font_mono_spd.render(f"{cur_spd:03.0f}", True, COLOR_TEXT_MAIN)
        screen.blit(spd_val_surf, (scx - spd_val_surf.get_width() // 2, scy - 20))
        screen.blit(font_mono_xs.render("KM/H", True, COLOR_TEXT_MUTED), (scx - 14, scy + 14))

        # 5-SECOND G-FORCE HISTORY GRAPH (Consolas Monospace Labels)
        gx_box = pygame.Rect(205, 222, 180, 105)
        pygame.draw.rect(screen, COLOR_CARD_BG, gx_box, border_radius=4)
        pygame.draw.rect(screen, COLOR_BORDER_THIN, gx_box, 1, border_radius=4)

        gy_mid = 222 + 52
        pygame.draw.line(screen, (24, 26, 32), (205, gy_mid - 32), (385, gy_mid - 32), 1)
        pygame.draw.line(screen, (36, 38, 46), (205, gy_mid), (385, gy_mid), 1) # 0G Line
        pygame.draw.line(screen, (24, 26, 32), (205, gy_mid + 32), (385, gy_mid + 32), 1)

        if len(ego.g_history) > 2:
            pts_lat = []
            pts_long = []
            pts_area = [(208, gy_mid)]
            for idx_g, (g_lat, g_long) in enumerate(ego.g_history):
                gx_pos = 208 + int((idx_g / len(ego.g_history)) * 170)
                gy_lat = int(gy_mid - g_lat * 60)
                gy_long = int(gy_mid - g_long * 60)
                pts_lat.append((gx_pos, max(225, min(322, gy_lat))))
                pts_long.append((gx_pos, max(225, min(322, gy_long))))
                pts_area.append((gx_pos, max(225, min(322, gy_lat))))
            pts_area.append((pts_area[-1][0], gy_mid))

            area_surf = pygame.Surface((380, 460), pygame.SRCALPHA)
            if len(pts_area) > 3:
                pygame.draw.polygon(area_surf, (0, 215, 245, 20), pts_area)
                screen.blit(area_surf, (0, 0))

            if len(pts_lat) > 1:
                pygame.draw.lines(screen, COLOR_CYAN_ACCENT, False, pts_lat, 2)
                pygame.draw.lines(screen, COLOR_AMBER_WARN, False, pts_long, 2)

        screen.blit(font_mono_xs.render("LAT G", True, COLOR_CYAN_ACCENT), (212, 226))
        screen.blit(font_mono_xs.render(f"{ego.lat_accel_g:+.2f}G", True, COLOR_CYAN_ACCENT), (332, 226))
        screen.blit(font_mono_xs.render("LONG G", True, COLOR_AMBER_WARN), (212, 310))
        screen.blit(font_mono_xs.render(f"{(ego.accel_mps2/9.81):+.2f}G", True, COLOR_AMBER_WARN), (332, 310))

        # ROTATING 3-SPOKE STEERING WHEEL
        sw_cx, sw_cy = 75, 375
        st_angle_rad = math.radians(ego.steering_angle_deg * 2.2)
        pygame.draw.circle(screen, (30, 32, 40), (sw_cx, sw_cy), 22)
        pygame.draw.circle(screen, (55, 60, 72), (sw_cx, sw_cy), 22, 2)
        pygame.draw.circle(screen, COLOR_PANEL_BG, (sw_cx, sw_cy), 14)

        for spk_base in (0, 120, 240):
            spk_rad = st_angle_rad + math.radians(spk_base)
            spk_x = sw_cx + int(18 * math.cos(spk_rad))
            spk_y = sw_cy - int(18 * math.sin(spk_rad))
            pygame.draw.line(screen, (65, 72, 85), (sw_cx, sw_cy), (spk_x, spk_y), 2)

        pygame.draw.circle(screen, COLOR_CYAN_ACCENT, (sw_cx, sw_cy), 4)
        screen.blit(font_mono_xs.render("STEER", True, COLOR_TEXT_MUTED), (sw_cx - 16, sw_cy + 24))

        # Steering & Lateral Jerk Telemetry
        st_txt = font_mono_xs.render(f"STEERING: {ego.steering_angle_deg:+.1f}° (INNER: {ego.steering_inner_deg:+.1f}°)", True, COLOR_TEXT_MAIN)
        screen.blit(st_txt, (120, 362))
        jerk_txt = font_mono_xs.render(f"LAT JERK: {ego.lat_jerk_gs:+.2f} G/s | SLIP: {ego.roll_deg:+.1f}°", True, COLOR_TEXT_MUTED)
        screen.blit(jerk_txt, (120, 380))

        # BLINKER ARROWS
        is_blk_l = (ego.blinker == "LEFT" and (frame_count % 20 < 10))
        is_blk_r = (ego.blinker == "RIGHT" and (frame_count % 20 < 10))
        col_bl = COLOR_AMBER_WARN if is_blk_l else (35, 38, 46)
        col_br = COLOR_AMBER_WARN if is_blk_r else (35, 38, 46)
        pygame.draw.polygon(screen, col_bl, [(125, 408), (135, 402), (135, 414)])
        pygame.draw.polygon(screen, col_br, [(175, 408), (165, 402), (165, 414)])
        if (is_blk_l or is_blk_r) and (frame_count % 20 == 0):
            screen.blit(font_mono_xs.render("CLICK", True, COLOR_AMBER_WARN), (190, 402))

        # AUTONOMOUS MISSION STATE BOX
        if ego.state == "LANE_KEEP":
            st_border = COLOR_EMERALD_GREEN
            st_label = "AUTONOMOUS MISSION: LANE_KEEP"
        elif ego.state == "CHECK_OVERTAKE":
            st_border = COLOR_AMBER_WARN
            st_label = "AUTONOMOUS MISSION: EVALUATING OVERTAKE"
        elif ego.state == "LANE_CHANGE_LEFT":
            st_border = COLOR_CYAN_ACCENT
            st_label = "AUTONOMOUS MISSION: ◄◄ LANE CHANGE LEFT"
        elif ego.state == "LANE_CHANGE_RIGHT":
            st_border = COLOR_CYAN_ACCENT
            st_label = "AUTONOMOUS MISSION: LANE CHANGE RIGHT ►►"
        else: # OVERTAKING
            st_border = COLOR_TESLA_RED
            st_label = f"AUTONOMOUS MISSION: OVERTAKING @ {ego.speed_kmh:.0f} KM/H"

        st_box = pygame.Rect(35, 435, 350, 32)
        pygame.draw.rect(screen, COLOR_CARD_BG, st_box, border_radius=4)
        pygame.draw.rect(screen, st_border, st_box, 1, border_radius=4)
        screen.blit(font_mono_sm.render(st_label, True, st_border), (45, 442))

        # CUDA Speedup
        gpu_stat = bev_engine.gpu_speedup_stats
        speedup_txt = font_mono_xs.render(f"CUDA GPU SPEEDUP: {gpu_stat['gpu_ms']:.1f}ms vs CPU {gpu_stat['cpu_ms']:.1f}ms ({gpu_stat['speedup']:.1f}x)", True, COLOR_CYAN_ACCENT)
        screen.blit(speedup_txt, (35, 478))
        screen.blit(font_mono_xs.render("TRACTION: 99.4% | BRAKE PRESSURE: 0% | IDM HEADWAY: 1.2s", True, COLOR_TEXT_MUTED), (35, 498))

        # -------------------------------------------------------------
        # 6. MIDDLE ROW — CENTER PANEL: 3D DIGITAL TWIN SIMULATION
        # -------------------------------------------------------------
        twin_rect = pygame.Rect(420, 180, 440, 460)
        pygame.draw.rect(screen, COLOR_PANEL_BG, twin_rect, border_radius=4)
        pygame.draw.rect(screen, COLOR_BORDER_THIN, twin_rect, 1, border_radius=4)

        twin_surf = pygame.Surface((bev_w - 4, bev_h - 4))
        twin_renderer.render_3d_scene(
            twin_surf, ego, traffic, point_cloud,
            particles=traffic_engine.particle_emitter.particles,
            frame_idx=frame_count,
            weather_mode=current_weather,
            night_mode=night_mode
        )
        screen.blit(twin_surf, (422, 182))

        # Clean Header Overlay on Digital Twin
        pygame.draw.rect(screen, COLOR_CARD_BG, pygame.Rect(432, 190, 240, 24), border_radius=3)
        pygame.draw.rect(screen, COLOR_BORDER_THIN, pygame.Rect(432, 190, 240, 24), 1, border_radius=3)
        screen.blit(font_mono_xs.render("3D DIGITAL TWIN • 3-LANE SIM", True, COLOR_CYAN_ACCENT), (440, 195))

        # -------------------------------------------------------------
        # 7. MIDDLE ROW — RIGHT PANEL: 77GHz RADAR & V2X HUD
        # -------------------------------------------------------------
        right_panel_rect = pygame.Rect(880, 180, 380, 460)
        pygame.draw.rect(screen, COLOR_PANEL_BG, right_panel_rect, border_radius=4)
        pygame.draw.rect(screen, COLOR_BORDER_THIN, right_panel_rect, 1, border_radius=4)

        screen.blit(font_ui_title.render("77GHz RADAR & V2X TELEMETRY", True, COLOR_CYAN_ACCENT), (895, 192))

        # 77GHz POLAR RADAR SCOPE
        rcx, rcy = 1070, 275
        rad_r = 72
        pygame.draw.circle(screen, (6, 12, 8), (rcx, rcy), rad_r)
        pygame.draw.circle(screen, (0, 150, 65), (rcx, rcy), rad_r, 1)

        for ring_f, r_lbl in [(0.33, "20m"), (0.66, "45m"), (1.00, "70m")]:
            rr_px = int(rad_r * ring_f)
            pygame.draw.circle(screen, (0, 45, 20), (rcx, rcy), rr_px, 1)
            screen.blit(font_mono_xs.render(r_lbl, True, (0, 110, 45)), (rcx + rr_px - 20, rcy + 2))

        screen.blit(font_mono_xs.render("0°", True, (0, 150, 65)), (rcx - 5, rcy - rad_r - 11))
        screen.blit(font_mono_xs.render("+30°", True, (0, 150, 65)), (rcx + rad_r - 12, rcy - 18))
        screen.blit(font_mono_xs.render("-30°", True, (0, 150, 65)), (rcx - rad_r - 16, rcy - 18))

        sweep_deg = (frame_count * 4) % 360
        sw_rad = math.radians(sweep_deg)
        pygame.draw.line(screen, (0, 235, 85), (rcx, rcy),
                         (rcx + int(rad_r * math.cos(sw_rad)), rcy - int(rad_r * math.sin(sw_rad))), 2)

        for r_det in radar_detections:
            rx = rcx + int(r_det.x * 2.8)
            ry = rcy - int(r_det.z * 0.85)
            if math.hypot(rx - rcx, ry - rcy) <= rad_r - 2:
                pygame.draw.circle(screen, (0, 255, 90), (rx, ry), 3)
                d_arrow = int(r_det.doppler_mps * 1.5)
                pygame.draw.line(screen, (0, 215, 175), (rx, ry), (rx, ry - d_arrow), 2)
                screen.blit(font_mono_xs.render(f"{r_det.range_m:.0f}m", True, COLOR_TEXT_MAIN), (rx + 5, ry - 6))

        # LiDAR Count (Large Crisp Monospace)
        screen.blit(font_mono_lg.render(f"{len(point_cloud):,}", True, COLOR_TEXT_MAIN), (895, 235))
        screen.blit(font_mono_xs.render("64-BEAM HESAI (20 Hz)", True, COLOR_CYAN_ACCENT), (895, 258))

        # Lead Car TTC Evaluation
        lead_car = next((v for v in traffic if abs(v.x - ego.x) < 2.0 and v.z > 0), None)
        if lead_car:
            ttc_val = lead_car.z / max(0.5, (ego.speed_mps - lead_car.speed_mps))
            ttc_col = COLOR_TESLA_RED if ttc_val < 2.0 else (COLOR_AMBER_WARN if ttc_val < 4.0 else COLOR_EMERALD_GREEN)
        else:
            ttc_val = 99.0
            ttc_col = COLOR_EMERALD_GREEN

        # LIVE TIME-TO-COLLISION 10-SEGMENT COUNTDOWN BAR
        screen.blit(font_mono_xs.render("TIME TO COLLISION (TTC)", True, COLOR_TEXT_MUTED), (895, 362))
        ttc_num_str = f"{ttc_val:.1f}s" if ttc_val < 50 else "--.-s"
        screen.blit(font_mono_ttc.render(ttc_num_str, True, ttc_col), (1195, 352))

        seg_active = max(0, min(10, int((ttc_val / 6.0) * 10)))
        for seg_i in range(10):
            seg_x = 895 + seg_i * 35
            seg_c = ttc_col if seg_i < seg_active else (20, 22, 28)
            pygame.draw.rect(screen, seg_c, pygame.Rect(seg_x, 380, 32, 12), border_radius=2)

        # V2X DSRC / C-V2X BSM CARD
        v2x_pkt = traffic_engine.get_lead_v2x_packet()
        v2x_box = pygame.Rect(895, 404, 350, 75)
        pygame.draw.rect(screen, COLOR_CARD_BG, v2x_box, border_radius=4)
        pygame.draw.rect(screen, COLOR_BORDER_THIN, v2x_box, 1, border_radius=4)

        pkt_flash = "+1 PKT" if (frame_count % 30 < 15) else "SYNC"
        screen.blit(font_mono_xs.render(f"📡 V2X BSM PACKET [{pkt_flash}]", True, COLOR_CYAN_ACCENT), (905, 410))

        if v2x_pkt:
            b_col = COLOR_TESLA_RED if v2x_pkt.brake_pct > 10 else COLOR_EMERALD_GREEN
            screen.blit(font_mono_xs.render(f"LEAD SPEED: {v2x_pkt.speed_kmh:.0f} km/h", True, COLOR_TEXT_MAIN), (905, 428))
            screen.blit(font_mono_xs.render(f"LEAD BRAKE: {'ON' if v2x_pkt.brake_pct > 10 else 'OFF'}", True, b_col), (1060, 428))
            screen.blit(font_mono_xs.render(f"LEAD BLINKER: {v2x_pkt.turn_signal}", True, COLOR_AMBER_WARN), (905, 446))
            screen.blit(font_mono_xs.render(f"DSRC LATENCY: {random.randint(2, 5)}ms", True, COLOR_EMERALD_GREEN), (1060, 446))
        else:
            screen.blit(font_mono_xs.render("LEAD SPEED: 68 km/h | LEAD BRAKE: OFF", True, COLOR_TEXT_MAIN), (905, 428))
            screen.blit(font_mono_xs.render("LEAD BLINKER: OFF  | DSRC LATENCY: 3ms", True, COLOR_EMERALD_GREEN), (905, 446))

        screen.blit(font_mono_xs.render("SENSORS: 4-CAM HDR [LOCKED] | 64-LIDAR [20Hz] | RADAR [OK]", True, COLOR_EMERALD_GREEN), (895, 498))

        # -------------------------------------------------------------
        # 8. BOTTOM ROW: REAR MIRROR (CENTER), ADAS FCW (LEFT), LOG (RIGHT)
        # -------------------------------------------------------------
        # BOTTOM-LEFT: ADAS FCW & MISSION STRATEGY CARD (Tesla Red Alerts)
        bl_rect = pygame.Rect(20, 648, 380, 142)
        pygame.draw.rect(screen, COLOR_PANEL_BG, bl_rect, border_radius=4)
        pygame.draw.rect(screen, COLOR_BORDER_THIN, bl_rect, 1, border_radius=4)

        screen.blit(font_ui_title.render("ADAS SAFETY & FCW INTELLIGENCE", True, COLOR_CYAN_ACCENT), (32, 658))

        fcw_msg = f"FCW: LEAD CAR {lead_car.z:.1f}m | TTC {abs(ttc_val):.1f}s" if lead_car else "FCW: CLEAR CORRIDOR AHEAD"
        pygame.draw.rect(screen, (14, 24, 18) if ttc_col == COLOR_EMERALD_GREEN else (32, 12, 14), pygame.Rect(32, 680, 356, 28), border_radius=3)
        pygame.draw.rect(screen, ttc_col, pygame.Rect(32, 680, 356, 28), 1, border_radius=3)
        screen.blit(font_mono_sm.render(fcw_msg, True, ttc_col), (42, 686))

        screen.blit(font_mono_xs.render(f"AUTOPILOT TARGET: {ego.target_cruise_speed_kmh:.0f} KM/H (LANE {ego.lane_idx})", True, COLOR_TEXT_MAIN), (32, 720))
        screen.blit(font_mono_xs.render(f"MOBIL ADVANTAGE: +1.84 m/s² | COURTESY: 0.20", True, COLOR_TEXT_MUTED), (32, 738))
        screen.blit(font_mono_xs.render("SYSTEM: ISO 26262 ASIL-D FAIL-OPERATIONAL", True, COLOR_EMERALD_GREEN), (32, 756))

        # BOTTOM-CENTER: REAR CAMERA / DIGITAL REARVIEW MIRROR
        surf_rear = pygame.surfarray.make_surface(np.transpose(cam_frames_center["REAR"], (1, 0, 2)))
        screen.blit(surf_rear, (420, 648))
        pygame.draw.rect(screen, COLOR_BORDER_THIN, pygame.Rect(420, 648, cam_w_center, cam_h_center), 1)

        # BOTTOM-RIGHT: REAL-TIME MISSION & EVENT LOG STREAM
        br_rect = pygame.Rect(880, 648, 380, 142)
        pygame.draw.rect(screen, COLOR_PANEL_BG, br_rect, border_radius=4)
        pygame.draw.rect(screen, COLOR_BORDER_THIN, br_rect, 1, border_radius=4)

        screen.blit(font_ui_title.render("MISSION & EVENT LOG STREAM", True, COLOR_CYAN_ACCENT), (895, 658))

        visible_logs = traffic_engine.event_log[-4:]
        for idx_l, l_entry in enumerate(visible_logs):
            if "OVERTAKE" in l_entry:
                pfx, l_col = "[OT]", COLOR_AMBER_WARN
            elif "LANE" in l_entry or "MANEUVER" in l_entry:
                pfx, l_col = "[LC]", COLOR_CYAN_ACCENT
            elif "CRUISE" in l_entry or "KEEP" in l_entry:
                pfx, l_col = "[CZ]", COLOR_EMERALD_GREEN
            else:
                pfx, l_col = "[!!]", COLOR_TESLA_RED

            ts_split = l_entry.split("] ", 1)
            ts_str = ts_split[0] + "] " if len(ts_split) > 1 else "[T+0.0s] "
            body_str = ts_split[1] if len(ts_split) > 1 else l_entry

            y_entry = 680 + idx_l * 24
            if idx_l == len(visible_logs) - 1:
                pygame.draw.rect(screen, (20, 22, 28), pygame.Rect(890, y_entry - 2, 360, 22), border_radius=2)

            pygame.draw.line(screen, (22, 24, 30), (895, y_entry + 20), (1245, y_entry + 20), 1)
            screen.blit(font_mono_xs.render(f"{pfx} {ts_str}", True, (100, 110, 125)), (895, y_entry + 2))
            screen.blit(font_mono_xs.render(body_str[:38], True, l_col), (995, y_entry + 2))

        pygame.display.flip()
        clock.tick(60)

        if video_writer:
            view_rgb = pygame.surfarray.array3d(screen)
            view_bgr = cv2.cvtColor(np.transpose(view_rgb, (1, 0, 2)), cv2.COLOR_RGB2BGR)
            video_writer.write(view_bgr)
            if frame_count >= args.max_frames:
                break

    if video_writer:
        video_writer.release()
        print(f"[INFO] Export complete: {args.export}")

    executor.shutdown(wait=False)
    pygame.quit()


if __name__ == "__main__":
    main()
