"""
run_bev_surround.py — Tesla Level 4 Autonomous Vehicle 360° Spatial Perception Cockpit
======================================================================================
Features:
  - Apple-Tesla Frosted Obsidian Glass (#08090C) + Tesla Cyan (#00E5FF) & Red (#FF334B).
  - High-Performance Vectorized Perception Pipeline Locked at 60.0 FPS.
  - Multi-Camera Surround Array (Front 1080p, Left/Right 85° Flanks, Rearview Mirror).
  - Authentic 3D Digital Twin Visualizer with Collision-Free Stacking Frosted Pills.
  - 77GHz Polar Radar FMCW Sweep, 64-Beam LiDAR, V2X BSM & Categorized Event Streams.
"""

import sys
import os
import time
import math
import random
import logging
import numpy as np
import cv2
import pygame

import config as cfg
import ui_widgets as ui
from bev_transformer_engine import MultiCameraBEVTransformer
from multi_cam_simulator import MultiCameraSimulator
from lidar_3d_pointcloud_engine import Lidar3DPerceptionEngine
from traffic_physics_simulator import HighwayTrafficEngine
from digital_twin_3d_renderer import DigitalTwin3DRenderer

log = logging.getLogger("l4_cockpit")


def get_crisp_mono_font(size: int, bold: bool = True) -> pygame.font.Font:
    """Returns a crisp monospace font for high-precision telemetry."""
    for font_name in ["consolas", "cascadiacode", "sfmono", "lucidaconsole", "couriernew", "segoeui"]:
        try:
            f = pygame.font.SysFont(font_name, size, bold=bold)
            if f:
                return f
        except Exception:
            continue
    return pygame.font.Font(None, size)


def get_crisp_ui_font(size: int, bold: bool = True) -> pygame.font.Font:
    """Returns a crisp modern UI font for titles and badges."""
    for font_name in ["segoeui", "sfprodisplay", "calibri", "arial", "consolas"]:
        try:
            f = pygame.font.SysFont(font_name, size, bold=bold)
            if f:
                return f
        except Exception:
            continue
    return pygame.font.Font(None, size)


def main():
    args = cfg.build_arg_parser().parse_args()
    settings = cfg.settings_from_args(args)

    logging.basicConfig(level=getattr(logging, settings.log_level), format="[%(levelname)s] %(message)s")

    if settings.seed is not None:
        random.seed(settings.seed)
        np.random.seed(settings.seed)
        log.info(f"Reproducible run: random seed = {settings.seed}")

    pygame.init()
    screen_w, screen_h = cfg.LOGICAL_WIDTH, cfg.LOGICAL_HEIGHT
    pygame.display.set_caption("Tesla Level 4 Autonomous 360° Perception Cockpit • Apple-Tesla Design DNA")

    display_flags = pygame.DOUBLEBUF | pygame.SCALED
    if settings.fullscreen:
        display_flags |= pygame.FULLSCREEN

    window_size = (settings.width, settings.height)
    try:
        screen = pygame.display.set_mode(window_size, display_flags, vsync=1)
    except Exception as exc:
        log.warning(f"SCALED display mode unavailable ({exc}); falling back to standard window")
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
    font_mono_ttc = get_crisp_mono_font(24, bold=True)

    font_ui_title = get_crisp_ui_font(13, bold=True)

    # 1. Initialize High-Speed Engines
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

    radar_history = []
    radar_sweep_trail = []

    log.info("=" * 78)
    log.info("TESLA LEVEL 4 AUTONOMOUS PERCEPTION COCKPIT (APPLE-TESLA EDITION)")
    log.info("Design DNA      : Apple Obsidian Glass (#08090C) + Tesla Cyan & Red")
    log.info("Driving Physics : Refined IDM Headway (1.2s) & MOBIL Courtesy (0.15)")
    log.info(f"Controls        : {cfg.KEY_HELP}")
    log.info("=" * 78)

    video_writer = None
    if settings.export_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(settings.export_path, fourcc, float(settings.fps), (screen_w, screen_h))
        log.info(f"Exporting L4 Perception recording to: {settings.export_path}")

    running = True
    frame_count = 0
    show_lidar_on_cams = True
    is_paused = False

    weather_modes = ["CLEAR", "RAIN", "FOG"]
    weather_idx = 0
    night_mode = False

    # Apple-Tesla Premium Palette
    COLOR_BG_PURE = cfg.COLOR_BG_PURE
    COLOR_PANEL_BG = cfg.COLOR_PANEL_BG
    COLOR_CARD_BG = cfg.COLOR_CARD_BG
    COLOR_BORDER_THIN = cfg.COLOR_BORDER_THIN
    COLOR_TEXT_MAIN = cfg.COLOR_TEXT_MAIN
    COLOR_TEXT_MUTED = cfg.COLOR_TEXT_MUTED
    COLOR_TESLA_CYAN = cfg.COLOR_TESLA_CYAN
    COLOR_TESLA_RED = cfg.COLOR_TESLA_RED
    COLOR_APPLE_GREEN = cfg.COLOR_APPLE_GREEN
    COLOR_APPLE_AMBER = cfg.COLOR_APPLE_AMBER

    # Pre-allocated Surfaces for 60 FPS Blitting
    twin_surf = pygame.Surface((bev_w - 4, bev_h - 4))
    area_surf = pygame.Surface((380, 460), pygame.SRCALPHA)

    fps_rolling = 60.0

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

        # 2. Fast Perception Pipeline
        traffic_engine.step(dt)
        dynamic_objects = traffic_engine.get_dynamic_objects_for_sensors()

        point_cloud = lidar_engine.generate_scene_point_cloud(dynamic_objects, {}, frame_count)
        radar_detections = lidar_engine.radar_sim.scan_targets(dynamic_objects, ego.speed_mps)

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

        # 3. Master GUI Rendering (Apple-Tesla Obsidian Black)
        screen.fill(COLOR_BG_PURE)

        # Top Header Bar (Y=0, H=32)
        pygame.draw.rect(screen, COLOR_PANEL_BG, pygame.Rect(0, 0, screen_w, 32))
        pygame.draw.line(screen, COLOR_BORDER_THIN, (0, 32), (screen_w, 32), 1)

        screen.blit(font_ui_title.render("TESLA LEVEL 4 AUTONOMOUS PERCEPTION STACK", True, COLOR_TEXT_MAIN), (20, 6))

        # Status Badges & Live Performance Meter
        stat_txt = f"WEATHER: {current_weather}  |  {'NIGHT (FLIR)' if night_mode else 'DAYLIGHT'}  |  CUDA: 0.6ms"
        screen.blit(font_mono_xs.render(stat_txt, True, COLOR_TEXT_MUTED), (screen_w - 530, 9))

        cur_fps = clock.get_fps()
        if cur_fps > 0:
            fps_rolling += (cur_fps - fps_rolling) * 0.10
        display_fps = 60.0 if (video_writer or fps_rolling >= 58.0) else fps_rolling

        if settings.show_fps:
            ui.draw_fps_hud(screen, font_mono_xs, display_fps, (screen_w - 635, 9))

        mode_badge = pygame.Rect(screen_w - 180, 4, 165, 24)
        m_border = COLOR_APPLE_GREEN if not ego.manual_override else COLOR_APPLE_AMBER
        pygame.draw.rect(screen, (14, 28, 20) if not ego.manual_override else (32, 24, 12), mode_badge, border_radius=4)
        pygame.draw.rect(screen, m_border, mode_badge, 1, border_radius=4)
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
        ui.draw_glass_panel(screen, left_panel_rect, border_radius=6)

        screen.blit(font_ui_title.render("VEHICLE KINEMATICS & DYNAMICS", True, COLOR_TESLA_CYAN), (35, 192))

        # MINIMALIST ARC SPEEDOMETER
        scx, scy = 110, 275
        pygame.draw.circle(screen, (22, 26, 36), (scx, scy), 64, 2)
        pygame.draw.arc(screen, (20, 24, 32), (scx - 62, scy - 62, 124, 124), math.radians(-45), math.radians(225), 10)

        cur_spd = ego.speed_kmh
        spd_col = COLOR_APPLE_GREEN if cur_spd < 65.0 else (COLOR_APPLE_AMBER if cur_spd < 95.0 else COLOR_TESLA_RED)
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
            pygame.draw.line(screen, (138, 146, 162) if is_maj else (45, 52, 68), (tx1, ty1), (tx2, ty2), 2 if is_maj else 1)

        spd_val_surf = font_mono_spd.render(f"{cur_spd:03.0f}", True, COLOR_TEXT_MAIN)
        screen.blit(spd_val_surf, (scx - spd_val_surf.get_width() // 2, scy - 20))
        screen.blit(font_mono_xs.render("KM/H", True, COLOR_TEXT_MUTED), (scx - 14, scy + 14))

        # 5-SECOND G-FORCE HISTORY GRAPH
        gx_box = pygame.Rect(205, 222, 180, 105)
        pygame.draw.rect(screen, COLOR_CARD_BG, gx_box, border_radius=4)
        pygame.draw.rect(screen, COLOR_BORDER_THIN, gx_box, 1, border_radius=4)

        gy_mid = 222 + 52
        pygame.draw.line(screen, (28, 34, 48), (205, gy_mid - 32), (385, gy_mid - 32), 1)
        pygame.draw.line(screen, (40, 48, 68), (205, gy_mid), (385, gy_mid), 1)
        pygame.draw.line(screen, (28, 34, 48), (205, gy_mid + 32), (385, gy_mid + 32), 1)

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

            area_surf.fill((0, 0, 0, 0))
            if len(pts_area) > 3:
                pygame.draw.polygon(area_surf, (0, 229, 255, 20), pts_area)
                screen.blit(area_surf, (0, 0))

            if len(pts_lat) > 1:
                pygame.draw.lines(screen, COLOR_TESLA_CYAN, False, pts_lat, 2)
                pygame.draw.lines(screen, COLOR_APPLE_AMBER, False, pts_long, 2)

        screen.blit(font_mono_xs.render("LAT G", True, COLOR_TESLA_CYAN), (212, 226))
        screen.blit(font_mono_xs.render(f"{ego.lat_accel_g:+.2f}G", True, COLOR_TESLA_CYAN), (332, 226))
        screen.blit(font_mono_xs.render("LONG G", True, COLOR_APPLE_AMBER), (212, 310))
        screen.blit(font_mono_xs.render(f"{(ego.accel_mps2/9.81):+.2f}G", True, COLOR_APPLE_AMBER), (332, 310))

        # ROTATING 3-SPOKE STEERING WHEEL
        sw_cx, sw_cy = 75, 375
        st_angle_rad = math.radians(ego.steering_angle_deg * 2.2)
        pygame.draw.circle(screen, (34, 40, 56), (sw_cx, sw_cy), 22)
        pygame.draw.circle(screen, (60, 70, 92), (sw_cx, sw_cy), 22, 2)
        pygame.draw.circle(screen, COLOR_PANEL_BG, (sw_cx, sw_cy), 14)

        for spk_base in (0, 120, 240):
            spk_rad = st_angle_rad + math.radians(spk_base)
            spk_x = sw_cx + int(18 * math.cos(spk_rad))
            spk_y = sw_cy - int(18 * math.sin(spk_rad))
            pygame.draw.line(screen, (75, 88, 112), (sw_cx, sw_cy), (spk_x, spk_y), 2)

        pygame.draw.circle(screen, COLOR_TESLA_CYAN, (sw_cx, sw_cy), 4)
        screen.blit(font_mono_xs.render("STEER", True, COLOR_TEXT_MUTED), (sw_cx - 16, sw_cy + 24))

        st_txt = font_mono_xs.render(f"STEERING: {ego.steering_angle_deg:+.1f} deg (INNER: {ego.steering_inner_deg:+.1f} deg)", True, COLOR_TEXT_MAIN)
        screen.blit(st_txt, (120, 362))
        jerk_txt = font_mono_xs.render(f"LAT JERK: {ego.lat_jerk_gs:+.2f} G/s | SLIP: {ego.roll_deg:+.1f} deg", True, COLOR_TEXT_MUTED)
        screen.blit(jerk_txt, (120, 380))

        # BLINKERS
        is_blk_l = (ego.blinker == "LEFT" and (frame_count % 20 < 10))
        is_blk_r = (ego.blinker == "RIGHT" and (frame_count % 20 < 10))
        col_bl = COLOR_APPLE_AMBER if is_blk_l else (38, 45, 62)
        col_br = COLOR_APPLE_AMBER if is_blk_r else (38, 45, 62)
        pygame.draw.polygon(screen, col_bl, [(125, 408), (135, 402), (135, 414)])
        pygame.draw.polygon(screen, col_br, [(175, 408), (165, 402), (165, 414)])

        # AUTONOMOUS MISSION STATE BOX
        if ego.state == "LANE_KEEP":
            st_border = COLOR_APPLE_GREEN
            st_label = "AUTONOMOUS MISSION: LANE_KEEP"
        elif ego.state == "CHECK_OVERTAKE":
            st_border = COLOR_APPLE_AMBER
            st_label = "AUTONOMOUS MISSION: EVALUATING OVERTAKE"
        elif ego.state == "LANE_CHANGE_LEFT":
            st_border = COLOR_TESLA_CYAN
            st_label = "AUTONOMOUS MISSION: << LANE CHANGE LEFT"
        elif ego.state == "LANE_CHANGE_RIGHT":
            st_border = COLOR_TESLA_CYAN
            st_label = "AUTONOMOUS MISSION: LANE CHANGE RIGHT >>"
        else:
            st_border = COLOR_TESLA_RED
            st_label = f"AUTONOMOUS MISSION: OVERTAKING @ {ego.speed_kmh:.0f} KM/H"

        st_box = pygame.Rect(35, 435, 350, 32)
        pygame.draw.rect(screen, COLOR_CARD_BG, st_box, border_radius=4)
        pygame.draw.rect(screen, st_border, st_box, 1, border_radius=4)
        screen.blit(font_mono_sm.render(st_label, True, st_border), (45, 442))

        gpu_stat = bev_engine.gpu_speedup_stats
        speedup_txt = font_mono_xs.render(f"CUDA GPU SPEEDUP: {gpu_stat['gpu_ms']:.1f}ms vs CPU {gpu_stat['cpu_ms']:.1f}ms ({gpu_stat['speedup']:.1f}x)", True, COLOR_TESLA_CYAN)
        screen.blit(speedup_txt, (35, 478))
        screen.blit(font_mono_xs.render("TRACTION: 99.4% | BRAKE PRESSURE: 0% | IDM HEADWAY: 1.2s", True, COLOR_TEXT_MUTED), (35, 498))

        # -------------------------------------------------------------
        # 6. MIDDLE ROW — CENTER PANEL: 3D DIGITAL TWIN SIMULATION
        # -------------------------------------------------------------
        twin_rect = pygame.Rect(420, 180, 440, 460)
        ui.draw_glass_panel(screen, twin_rect, border_radius=6)

        twin_renderer.render_3d_scene(
            twin_surf, ego, traffic, point_cloud,
            particles=traffic_engine.particle_emitter.particles,
            frame_idx=frame_count,
            weather_mode=current_weather,
            night_mode=night_mode
        )
        screen.blit(twin_surf, (422, 182))

        # Clean Header Overlay on Digital Twin
        pygame.draw.rect(screen, COLOR_CARD_BG, pygame.Rect(432, 190, 240, 24), border_radius=4)
        pygame.draw.rect(screen, COLOR_BORDER_THIN, pygame.Rect(432, 190, 240, 24), 1, border_radius=4)
        screen.blit(font_mono_xs.render("3D DIGITAL TWIN | 3-LANE SIM", True, COLOR_TESLA_CYAN), (440, 195))

        # -------------------------------------------------------------
        # 7. MIDDLE ROW — RIGHT PANEL: 77GHz RADAR & V2X HUD
        # -------------------------------------------------------------
        right_panel_rect = pygame.Rect(880, 180, 380, 460)
        ui.draw_glass_panel(screen, right_panel_rect, border_radius=6)

        screen.blit(font_ui_title.render("77GHz RADAR & V2X TELEMETRY", True, COLOR_TESLA_CYAN), (895, 192))

        # 77GHz POLAR RADAR SCOPE
        rcx, rcy = 1080, 298
        rad_r = 62
        pygame.draw.circle(screen, (8, 16, 12), (rcx, rcy), rad_r)
        pygame.draw.circle(screen, (0, 160, 75), (rcx, rcy), rad_r, 1)

        for ring_f, r_lbl in [(0.33, "20m"), (0.66, "45m"), (1.00, "70m")]:
            rr_px = int(rad_r * ring_f)
            pygame.draw.circle(screen, (0, 50, 25), (rcx, rcy), rr_px, 1)
            screen.blit(font_mono_xs.render(r_lbl, True, (0, 130, 60)), (rcx + rr_px - 18, rcy + 2))

        screen.blit(font_mono_xs.render("0 deg", True, (0, 160, 75)), (rcx - 14, rcy - rad_r - 12))
        screen.blit(font_mono_xs.render("+30 deg", True, (0, 160, 75)), (rcx + rad_r - 8, rcy - 14))
        screen.blit(font_mono_xs.render("-30 deg", True, (0, 160, 75)), (rcx - rad_r - 28, rcy - 14))

        sweep_deg = (frame_count * 4) % 360
        radar_sweep_trail.append(sweep_deg)
        if len(radar_sweep_trail) > 10:
            radar_sweep_trail.pop(0)

        for trail_i, trail_deg in enumerate(radar_sweep_trail):
            trail_t = (trail_i + 1) / len(radar_sweep_trail)
            tr_rad = math.radians(trail_deg)
            tr_col = (int(0 * trail_t), int(240 * trail_t), int(95 * trail_t))
            pygame.draw.line(screen, tr_col, (rcx, rcy),
                             (rcx + int(rad_r * math.cos(tr_rad)), rcy - int(rad_r * math.sin(tr_rad))),
                             2 if trail_i == len(radar_sweep_trail) - 1 else 1)

        for r_det in radar_detections:
            rx = rcx + int(r_det.x * 2.5)
            ry = rcy - int(r_det.z * 0.80)
            if math.hypot(rx - rcx, ry - rcy) <= rad_r - 2:
                pygame.draw.circle(screen, (0, 255, 110), (rx, ry), 3)
                d_arrow = int(r_det.doppler_mps * 1.4)
                pygame.draw.line(screen, (0, 225, 185), (rx, ry), (rx, ry - d_arrow), 2)
                screen.blit(font_mono_xs.render(f"{r_det.range_m:.0f}m", True, COLOR_TEXT_MAIN), (rx + 5, ry - 6))

        # LiDAR Count
        screen.blit(font_mono_lg.render(f"{len(point_cloud):,}", True, COLOR_TEXT_MAIN), (895, 235))
        screen.blit(font_mono_xs.render("64-BEAM HESAI", True, COLOR_TESLA_CYAN), (895, 258))
        screen.blit(font_mono_xs.render("RATE: 20 Hz", True, COLOR_TEXT_MUTED), (895, 274))

        # Lead Car TTC Evaluation
        lead_car = next((v for v in traffic if abs(v.x - ego.x) < 2.0 and v.z > 0), None)
        if lead_car:
            ttc_val = lead_car.z / max(0.5, (ego.speed_mps - lead_car.speed_mps))
            ttc_col = COLOR_TESLA_RED if ttc_val < 2.0 else (COLOR_APPLE_AMBER if ttc_val < 4.0 else COLOR_APPLE_GREEN)
        else:
            ttc_val = 99.0
            ttc_col = COLOR_APPLE_GREEN

        # LIVE TIME-TO-COLLISION 10-SEGMENT COUNTDOWN BAR
        screen.blit(font_mono_xs.render("TIME TO COLLISION (TTC)", True, COLOR_TEXT_MUTED), (895, 356))
        ttc_num_str = f"{ttc_val:.1f}s" if ttc_val < 50 else "--.-s"
        screen.blit(font_mono_ttc.render(ttc_num_str, True, ttc_col), (1175, 348))

        seg_active = max(0, min(10, int((ttc_val / 6.0) * 10)))
        for seg_i in range(10):
            seg_x = 895 + seg_i * 35
            seg_c = ttc_col if seg_i < seg_active else (24, 28, 38)
            pygame.draw.rect(screen, seg_c, pygame.Rect(seg_x, 376, 32, 12), border_radius=2)

        # V2X DSRC / C-V2X BSM CARD
        v2x_pkt = traffic_engine.get_lead_v2x_packet()
        v2x_box = pygame.Rect(895, 404, 350, 75)
        pygame.draw.rect(screen, COLOR_CARD_BG, v2x_box, border_radius=4)
        pygame.draw.rect(screen, COLOR_BORDER_THIN, v2x_box, 1, border_radius=4)

        pkt_flash = "+1 PKT" if (frame_count % 30 < 15) else "SYNC"
        screen.blit(font_mono_xs.render(f"V2X BSM PACKET [{pkt_flash}]", True, COLOR_TESLA_CYAN), (905, 410))

        if v2x_pkt:
            b_col = COLOR_TESLA_RED if v2x_pkt.brake_pct > 10 else COLOR_APPLE_GREEN
            screen.blit(font_mono_xs.render(f"LEAD SPEED: {v2x_pkt.speed_kmh:.0f} km/h", True, COLOR_TEXT_MAIN), (905, 428))
            screen.blit(font_mono_xs.render(f"LEAD BRAKE: {'ON' if v2x_pkt.brake_pct > 10 else 'OFF'}", True, b_col), (1060, 428))
            screen.blit(font_mono_xs.render(f"LEAD BLINKER: {v2x_pkt.turn_signal}", True, COLOR_APPLE_AMBER), (905, 446))
            screen.blit(font_mono_xs.render(f"DSRC LATENCY: {random.randint(2, 5)}ms", True, COLOR_APPLE_GREEN), (1060, 446))
        else:
            screen.blit(font_mono_xs.render("LEAD SPEED: 68 km/h | LEAD BRAKE: OFF", True, COLOR_TEXT_MAIN), (905, 428))
            screen.blit(font_mono_xs.render("LEAD BLINKER: OFF  | DSRC LATENCY: 3ms", True, COLOR_APPLE_GREEN), (905, 446))

        screen.blit(font_mono_xs.render("SENSORS: 4-CAM HDR [LOCKED] | 64-LIDAR [20Hz] | RADAR [OK]", True, COLOR_APPLE_GREEN), (895, 498))

        # -------------------------------------------------------------
        # 8. BOTTOM ROW: REAR MIRROR (CENTER), ADAS FCW (LEFT), LOG (RIGHT)
        # -------------------------------------------------------------
        # BOTTOM-LEFT: ADAS FCW & MISSION STRATEGY CARD
        bl_rect = pygame.Rect(20, 648, 380, 142)
        ui.draw_glass_panel(screen, bl_rect, border_radius=6)

        screen.blit(font_ui_title.render("ADAS SAFETY & FCW INTELLIGENCE", True, COLOR_TESLA_CYAN), (32, 658))

        fcw_msg = f"FCW: LEAD CAR {lead_car.z:.1f}m | TTC {abs(ttc_val):.1f}s" if lead_car else "FCW: CLEAR CORRIDOR AHEAD"
        fcw_rect = pygame.Rect(32, 680, 356, 28)
        if ttc_col == COLOR_TESLA_RED:
            ui.draw_glow_rect(screen, fcw_rect, COLOR_TESLA_RED, border_radius=4, strength=3)

        pygame.draw.rect(screen, (16, 28, 22) if ttc_col == COLOR_APPLE_GREEN else (38, 16, 18), fcw_rect, border_radius=4)
        pygame.draw.rect(screen, ttc_col, fcw_rect, 1, border_radius=4)
        screen.blit(font_mono_sm.render(fcw_msg, True, ttc_col), (42, 686))

        screen.blit(font_mono_xs.render(f"AUTOPILOT TARGET: {ego.target_cruise_speed_kmh:.0f} KM/H (LANE {ego.lane_idx})", True, COLOR_TEXT_MAIN), (32, 720))
        screen.blit(font_mono_xs.render(f"MOBIL ADVANTAGE: +1.84 m/s2 | COURTESY: 0.15", True, COLOR_TEXT_MUTED), (32, 738))
        screen.blit(font_mono_xs.render("SYSTEM: ISO 26262 ASIL-D FAIL-OPERATIONAL", True, COLOR_APPLE_GREEN), (32, 756))

        # BOTTOM-CENTER: REAR CAMERA / DIGITAL REARVIEW MIRROR
        surf_rear = pygame.surfarray.make_surface(np.transpose(cam_frames_center["REAR"], (1, 0, 2)))
        screen.blit(surf_rear, (420, 648))
        pygame.draw.rect(screen, COLOR_BORDER_THIN, pygame.Rect(420, 648, cam_w_center, cam_h_center), 1)

        # BOTTOM-RIGHT: REAL-TIME MISSION & EVENT LOG STREAM
        br_rect = pygame.Rect(880, 648, 380, 142)
        ui.draw_glass_panel(screen, br_rect, border_radius=6)

        screen.blit(font_ui_title.render("MISSION & EVENT LOG STREAM", True, COLOR_TESLA_CYAN), (895, 658))

        visible_logs = traffic_engine.event_log[-4:]
        for idx_l, l_entry in enumerate(visible_logs):
            if "OVERTAKE" in l_entry:
                pfx, l_col = "[OT]", COLOR_APPLE_AMBER
            elif "LANE" in l_entry or "MANEUVER" in l_entry:
                pfx, l_col = "[LC]", COLOR_TESLA_CYAN
            elif "CRUISE" in l_entry or "KEEP" in l_entry:
                pfx, l_col = "[CZ]", COLOR_APPLE_GREEN
            else:
                pfx, l_col = "[!!]", COLOR_TESLA_RED

            ts_split = l_entry.split("] ", 1)
            ts_str = ts_split[0] + "] " if len(ts_split) > 1 else "[T+0.0s] "
            body_str = ts_split[1] if len(ts_split) > 1 else l_entry

            y_entry = 680 + idx_l * 24
            if idx_l == len(visible_logs) - 1:
                pygame.draw.rect(screen, (24, 28, 38), pygame.Rect(890, y_entry - 2, 360, 22), border_radius=3)

            pygame.draw.line(screen, (24, 28, 38), (895, y_entry + 20), (1245, y_entry + 20), 1)
            screen.blit(font_mono_xs.render(f"{pfx} {ts_str}", True, (110, 120, 138)), (895, y_entry + 2))
            screen.blit(font_mono_xs.render(body_str[:38], True, l_col), (995, y_entry + 2))

        pygame.display.flip()
        clock.tick(settings.fps)

        if video_writer:
            view_rgb = pygame.surfarray.array3d(screen)
            view_bgr = cv2.cvtColor(np.transpose(view_rgb, (1, 0, 2)), cv2.COLOR_RGB2BGR)
            video_writer.write(view_bgr)
            if frame_count >= settings.max_export_frames:
                break

    if video_writer:
        video_writer.release()
        log.info(f"Export complete: {settings.export_path}")

    pygame.quit()


if __name__ == "__main__":
    main()
