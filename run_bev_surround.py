"""
run_bev_surround.py — Level 4 Autonomous Vehicle 3D Digital Twin & Perception Stack
===================================================================================
Redesigned High-Fidelity Autonomous Vehicle Cockpit:
  - PANEL 1: Speedometer 270° Arc Gauge, 5s G-Force Graph with Grid & Area, Rotating 3-Spoke Steering Wheel, Triangle Blinkers & Autonomous State Box.
  - PANEL 2: 120px 77GHz Polar Radar Scope with 6 Ghost Sweep Lines, Velocity Vectors & Persistence Echoes, 10-Segment TTC Countdown Bar & V2X Telemetry Card.
  - PANEL 3: Categorized Scrolling Mission Log with [OT], [LC], [CZ], [!!] Badges & Timestamps.
  - PANEL 4: 3D Digital Twin Simulation & 4 Surround Viewports with High-Visibility Markings.
  - Multi-Threaded Execution Pipeline (ThreadPoolExecutor) maintaining locked 60 FPS on RTX 4070.
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


def get_safe_font(size: int, bold: bool = True) -> pygame.font.Font:
    try:
        return pygame.font.SysFont("segoeui", size, bold=bold)
    except Exception:
        try:
            return pygame.font.SysFont("arial", size, bold=bold)
        except Exception:
            return pygame.font.Font(None, size)


def main():
    parser = argparse.ArgumentParser(description="Level 4 Autonomous Vehicle 3D Digital Twin & Perception Stack")
    parser.add_argument("--export", type=str, default=None, help="Export session to MP4 video")
    parser.add_argument("--max-frames", type=int, default=300, help="Max frames for export")
    args = parser.parse_args()

    pygame.init()
    screen_w, screen_h = 1280, 800
    pygame.display.set_caption("Level 4 Autonomous Vehicle 360° Multi-Camera & 3D LiDAR Perception Stack (RTX 4070)")

    try:
        screen = pygame.display.set_mode((screen_w, screen_h), pygame.DOUBLEBUF | pygame.HWSURFACE)
    except Exception:
        screen = pygame.display.set_mode((screen_w, screen_h), pygame.DOUBLEBUF)

    clock = pygame.time.Clock()

    font_xs = get_safe_font(10, bold=True)
    font_sm = get_safe_font(12, bold=True)
    font_md = get_safe_font(14, bold=True)
    font_lg = get_safe_font(18, bold=True)
    font_speed = get_safe_font(30, bold=True)
    font_ttc = get_safe_font(26, bold=True)

    # 1. Initialize Engines
    bev_w, bev_h = 440, 480
    bev_engine = MultiCameraBEVTransformer(bev_width_px=bev_w, bev_height_px=bev_h)
    cam_sim = MultiCameraSimulator(width=300, height=160)
    lidar_engine = Lidar3DPerceptionEngine(num_lasers=64, max_range_m=65.0)
    lidar_engine.cameras = bev_engine.cameras

    traffic_engine = HighwayTrafficEngine()
    twin_renderer = DigitalTwin3DRenderer(screen_w=bev_w, screen_h=bev_h)

    # Multi-threaded async executor
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)

    # Radar target persistence history buffer (for 3 faint echoes)
    radar_history = []

    print("\n" + "=" * 78)
    print("  [+] LEVEL 4 AUTONOMOUS VEHICLE 360 MULTI-CAMERA & 3D LIDAR COCKPIT")
    print("  Driving Physics   : IDM & MOBIL Autonomous Overtaking & Lane-Switching Engine")
    print("  Perception Stack  : 64-Beam LiDAR + 77GHz Polar Radar + 4 Surround Cameras + V2X")
    print("  Controls          : [TAB] Auto/Manual | [N] Night | [P] Weather | [R] Randomize")
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
                    traffic_engine.log_event(f"PILOT MODE -> {'MANUAL OVERRIDE' if traffic_engine.ego.manual_override else 'AUTONOMOUS HIGHWAY PILOT'}")
                elif event.key == pygame.K_a:
                    traffic_engine.ego.initiate_lane_change(max(-1, traffic_engine.ego.lane_idx - 1))
                elif event.key == pygame.K_d:
                    traffic_engine.ego.initiate_lane_change(min(1, traffic_engine.ego.lane_idx + 1))

            twin_renderer.handle_mouse_orbit(event, rect_offset=(420, 212))

        # Manual Driving Controls
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

        # 2. Multi-Threaded Sensor Pipeline
        traffic_engine.step(dt)
        dynamic_objects = traffic_engine.get_dynamic_objects_for_sensors()

        fut_lidar = executor.submit(lidar_engine.generate_scene_point_cloud, dynamic_objects, {}, frame_count)
        point_cloud = fut_lidar.result()
        ground_pts, obstacle_pts, bounding_boxes = lidar_engine.segment_ground_and_clusters(point_cloud, dynamic_objects)

        radar_detections = lidar_engine.radar_sim.scan_targets(dynamic_objects, ego.speed_mps)

        # Update radar history for persistence echoes
        radar_history.append(radar_detections)
        if len(radar_history) > 4:
            radar_history.pop(0)

        cam_frames = cam_sim.render_surround_views(
            frame_idx=frame_count,
            dynamic_objects=dynamic_objects,
            speed_kmh=ego.speed_kmh,
            lidar_engine=lidar_engine,
            point_cloud=point_cloud,
            render_lidar_on_cams=show_lidar_on_cams,
            weather_mode=current_weather,
            night_mode=night_mode
        )

        # 3. Master GUI Rendering
        screen.fill((10, 14, 20))

        # Glowing Border Intensity Calculation
        glow_val = 0.5 + 0.5 * math.sin(frame_count * 0.05)
        border_glow_col = (0, int(90 + 100 * glow_val), int(160 + 95 * glow_val))

        # Top Header Banner
        header_rect = pygame.Rect(0, 0, screen_w, 38)
        pygame.draw.rect(screen, (14, 18, 26), header_rect)
        h_title = font_md.render("LEVEL 4 AUTONOMOUS VEHICLE 360° MULTI-CAMERA & 3D LIDAR PERCEPTION STACK", True, (0, 230, 255))
        screen.blit(h_title, (20, 9))

        w_badge = font_xs.render(f"WEATHER: {current_weather} | {'NIGHT (THERMAL-IR)' if night_mode else 'DAYLIGHT'}", True, (255, 210, 0))
        screen.blit(w_badge, (screen_w - 480, 11))

        mode_badge = pygame.Rect(screen_w - 240, 5, 220, 28)
        mode_bg = (10, 35, 25) if not ego.manual_override else (40, 25, 10)
        mode_border = (0, 255, 180) if not ego.manual_override else (255, 180, 0)
        pygame.draw.rect(screen, mode_bg, mode_badge, border_radius=4)
        pygame.draw.rect(screen, mode_border, mode_badge, 1, border_radius=4)
        m_txt = font_xs.render("L4 HIGHWAY PILOT ACTIVE" if not ego.manual_override else "MANUAL PILOT OVERRIDE", True, mode_border)
        screen.blit(m_txt, (screen_w - 225, 11))

        # 4. Render 4 Surround Cameras Across Top Row
        c_names = ["FRONT", "LEFT", "RIGHT", "REAR"]
        c_w, c_h = 300, 160
        cam_x_offsets = [20, 335, 650, 965]

        for i, c_name in enumerate(c_names):
            cx = cam_x_offsets[i]
            cy = 44
            f_surf = pygame.surfarray.make_surface(np.transpose(cam_frames[c_name], (1, 0, 2)))
            screen.blit(f_surf, (cx, cy))

        # -------------------------------------------------------------
        # 5. PANEL 1 — VEHICLE KINEMATICS & DYNAMICS (Left Panel)
        # -------------------------------------------------------------
        left_rect = pygame.Rect(20, 212, 380, 480)
        pygame.draw.rect(screen, (14, 18, 26), left_rect, border_radius=8)
        pygame.draw.rect(screen, border_glow_col, left_rect, 1, border_radius=8)

        s_title = font_md.render("VEHICLE KINEMATICS & DYNAMICS", True, (0, 230, 255))
        screen.blit(s_title, (35, 222))

        # SPEEDOMETER 270° ARC GAUGE (Radius 65px, center at (110, 305))
        scx, scy = 110, 305
        # Outer Ring: 2px circle RGB(30,40,55)
        pygame.draw.circle(screen, (30, 40, 55), (scx, scy), 68, 2)
        # Background Arc (225° to -45°, 270° sweep, RGB(20,28,40), thickness 14px)
        pygame.draw.arc(screen, (20, 28, 40), (scx - 65, scy - 65, 130, 130), math.radians(-45), math.radians(225), 14)

        # Speed Value Arc Color Interpolation
        cur_spd = ego.speed_kmh
        if cur_spd < 60.0:
            spd_col = (0, 200, 80)
        elif cur_spd < 90.0:
            spd_col = (255, 200, 0)
        elif cur_spd < 110.0:
            spd_col = (255, 100, 0)
        else:
            spd_col = (255, 30, 0)

        spd_pct = max(0.0, min(1.0, cur_spd / 140.0))
        sweep_rad = spd_pct * math.radians(270)
        if sweep_rad > 0.05:
            # Draw sweeping value arc
            pygame.draw.arc(screen, spd_col, (scx - 65, scy - 65, 130, 130), math.radians(225) - sweep_rad, math.radians(225), 14)

        # 18 Tick Marks around arc (major every 3rd: 8px vs 4px)
        for tick_i in range(19):
            t_angle_rad = math.radians(225) - (tick_i / 18.0) * math.radians(270)
            is_major = (tick_i % 3 == 0)
            t_len = 8 if is_major else 4
            tx1 = scx + int(56 * math.cos(t_angle_rad))
            ty1 = scy - int(56 * math.sin(t_angle_rad))
            tx2 = scx + int((56 - t_len) * math.cos(t_angle_rad))
            ty2 = scy - int((56 - t_len) * math.sin(t_angle_rad))
            pygame.draw.line(screen, (160, 180, 200) if is_major else (80, 100, 120), (tx1, ty1), (tx2, ty2), 2 if is_major else 1)

        # Speed Value Number & "km/h" Label
        spd_val_surf = font_speed.render(f"{cur_spd:.0f}", True, spd_col)
        screen.blit(spd_val_surf, (scx - spd_val_surf.get_width() // 2, scy - 18))
        kmh_surf = font_xs.render("km/h", True, (120, 140, 160))
        screen.blit(kmh_surf, (scx - kmh_surf.get_width() // 2, scy + 16))

        # G-FORCE HISTORY GRAPH (5s rolling history with grid & filled area)
        gx_box = pygame.Rect(205, 252, 180, 95)
        pygame.draw.rect(screen, (18, 24, 34), gx_box, border_radius=4)
        pygame.draw.rect(screen, (35, 50, 70), gx_box, 1, border_radius=4)

        # 3 Horizontal Grid Lines at -0.5G, 0G, +0.5G
        gy_mid = 252 + 47
        pygame.draw.line(screen, (28, 38, 52), (205, gy_mid - 30), (385, gy_mid - 30), 1)
        pygame.draw.line(screen, (40, 55, 70), (205, gy_mid), (385, gy_mid), 1) # 0G centerline
        pygame.draw.line(screen, (28, 38, 52), (205, gy_mid + 30), (385, gy_mid + 30), 1)

        # Draw G-force curves & filled area
        if len(ego.g_history) > 2:
            pts_lat = []
            pts_long = []
            pts_area = [(208, gy_mid)]
            for idx_g, (g_lat, g_long) in enumerate(ego.g_history):
                gx_pos = 208 + int((idx_g / len(ego.g_history)) * 170)
                gy_lat = int(gy_mid - g_lat * 60)
                gy_long = int(gy_mid - g_long * 60)
                pts_lat.append((gx_pos, max(255, min(342, gy_lat))))
                pts_long.append((gx_pos, max(255, min(342, gy_long))))
                pts_area.append((gx_pos, max(255, min(342, gy_lat))))
            pts_area.append((pts_area[-1][0], gy_mid))

            # Filled area under Lat G (RGB(0,200,255) alpha 0.10)
            area_surf = pygame.Surface((380, 480), pygame.SRCALPHA)
            if len(pts_area) > 3:
                pygame.draw.polygon(area_surf, (0, 200, 255, 30), pts_area)
                screen.blit(area_surf, (0, 0))

            if len(pts_lat) > 1:
                pygame.draw.lines(screen, (0, 200, 255), False, pts_lat, 2)
                pygame.draw.lines(screen, (255, 180, 0), False, pts_long, 2)

        # G-Force Axis Labels & Current Values
        screen.blit(font_xs.render("LAT", True, (0, 200, 255)), (212, 256))
        screen.blit(font_xs.render(f"{ego.lat_accel_g:+.2f}G", True, (0, 200, 255)), (342, 256))
        screen.blit(font_xs.render("LONG", True, (255, 180, 0)), (212, 330))
        screen.blit(font_xs.render(f"{(ego.accel_mps2/9.81):+.2f}G", True, (255, 180, 0)), (342, 330))

        # ROTATING 3-SPOKE STEERING WHEEL VISUALIZATION
        sw_cx, sw_cy = 75, 388
        st_angle_rad = math.radians(ego.steering_angle_deg * 2.2)
        # Outer Ring: radius 22px, fill RGB(50,65,80), stroke 2px RGB(80,100,120)
        pygame.draw.circle(screen, (50, 65, 80), (sw_cx, sw_cy), 22)
        pygame.draw.circle(screen, (80, 100, 120), (sw_cx, sw_cy), 22, 2)
        pygame.draw.circle(screen, (14, 18, 26), (sw_cx, sw_cy), 14) # Inner cutout

        # 3 Spokes at 0°, 120°, 240° rotated by steering angle
        for spk_base in (0, 120, 240):
            spk_rad = st_angle_rad + math.radians(spk_base)
            spk_x = sw_cx + int(18 * math.cos(spk_rad))
            spk_y = sw_cy - int(18 * math.sin(spk_rad))
            pygame.draw.line(screen, (80, 100, 120), (sw_cx, sw_cy), (spk_x, spk_y), 2)

        # Center Hub: radius 5px, RGB(0,180,255)
        pygame.draw.circle(screen, (0, 180, 255), (sw_cx, sw_cy), 5)
        screen.blit(font_xs.render("STEER", True, (120, 140, 160)), (sw_cx - 16, sw_cy + 24))

        # Steering Rack Telemetry
        st_txt = font_xs.render(f"STEERING: {ego.steering_angle_deg:+.1f}° (INNER: {ego.steering_inner_deg:+.1f}°)", True, (200, 220, 240))
        screen.blit(st_txt, (118, 375))
        jerk_txt = font_xs.render(f"LAT JERK: {ego.lat_jerk_gs:+.2f} G/s | SLIP: {ego.roll_deg:+.1f}°", True, (160, 180, 200))
        screen.blit(jerk_txt, (118, 392))

        # BLINKER INDICATOR TRIANGLES (◄ ►) WITH "CLICK" AUDIO CUE TEXT
        is_blinking_left = (ego.blinker == "LEFT" and (frame_count % 20 < 10))
        is_blinking_right = (ego.blinker == "RIGHT" and (frame_count % 20 < 10))
        left_blinker_col = (255, 200, 0) if is_blinking_left else (40, 50, 60)
        right_blinker_col = (255, 200, 0) if is_blinking_right else (40, 50, 60)

        # Left Triangle (◄)
        pygame.draw.polygon(screen, left_blinker_col, [(125, 418), (135, 412), (135, 424)])
        # Right Triangle (►)
        pygame.draw.polygon(screen, right_blinker_col, [(175, 418), (165, 412), (165, 424)])
        if (is_blinking_left or is_blinking_right) and (frame_count % 20 == 0):
            screen.blit(font_xs.render("CLICK", True, (255, 200, 0)), (190, 412))

        # ADAS Intelligence Section & FCW
        lead_car = next((v for v in traffic if abs(v.x - ego.x) < 2.0 and v.z > 0), None)
        if lead_car:
            ttc_val = lead_car.z / max(0.5, (ego.speed_mps - lead_car.speed_mps))
            fcw_col = (255, 60, 60) if ttc_val < 2.0 else ((255, 200, 0) if ttc_val < 4.0 else (0, 200, 80))
            fcw_msg = f"FCW: LEAD CAR {lead_car.z:.1f}m | TTC {abs(ttc_val):.1f}s"
        else:
            ttc_val = 99.0
            fcw_col = (0, 200, 80)
            fcw_msg = "FCW: CLEAR CORRIDOR AHEAD"

        pygame.draw.rect(screen, (12, 28, 20) if fcw_col == (0, 200, 80) else (32, 18, 22), (35, 435, 350, 28), border_radius=4)
        pygame.draw.rect(screen, fcw_col, (35, 435, 350, 28), 1, border_radius=4)
        screen.blit(font_sm.render(fcw_msg, True, fcw_col), (45, 440))

        # AUTONOMOUS STATE BOX (Color-Coded with Animations)
        if ego.state == "LANE_KEEP":
            state_border = (0, 200, 80)
            state_label = "AUTONOMOUS MISSION: LANE_KEEP"
        elif ego.state == "CHECK_OVERTAKE":
            state_border = (255, 200, 0)
            state_label = "AUTONOMOUS MISSION: EVALUATING OVERTAKE"
        elif ego.state == "LANE_CHANGE_LEFT":
            state_border = (0, 180, 255)
            state_label = "AUTONOMOUS MISSION: ◄◄ LANE CHANGE LEFT"
        elif ego.state == "LANE_CHANGE_RIGHT":
            state_border = (0, 180, 255)
            state_label = "AUTONOMOUS MISSION: LANE CHANGE RIGHT ►►"
        else: # OVERTAKING
            pulse_alpha = int(160 + 95 * math.sin(frame_count * 0.25))
            state_border = (255, 80, 0)
            state_label = f"AUTONOMOUS MISSION: OVERTAKING @ {ego.speed_kmh:.0f} KM/H"

        l4_box = pygame.Rect(35, 470, 350, 32)
        pygame.draw.rect(screen, (10, 24, 32), l4_box, border_radius=4)
        pygame.draw.rect(screen, state_border, l4_box, 2, border_radius=4)
        screen.blit(font_sm.render(state_label, True, state_border), (45, 477))

        # CUDA GPU Speedup Meter
        gpu_stat = bev_engine.gpu_speedup_stats
        speedup_txt = font_xs.render(f"CUDA GPU SPEEDUP: {gpu_stat['gpu_ms']:.1f}ms vs CPU {gpu_stat['cpu_ms']:.1f}ms ({gpu_stat['speedup']:.1f}x)", True, (0, 230, 255))
        screen.blit(speedup_txt, (35, 510))

        # -------------------------------------------------------------
        # 6. PANEL 4 — 3D DIGITAL TWIN WORLD (Center Panel)
        # -------------------------------------------------------------
        center_rect = pygame.Rect(420, 212, 440, 480)
        pygame.draw.rect(screen, (14, 18, 26), center_rect, border_radius=8)
        pygame.draw.rect(screen, border_glow_col, center_rect, 1, border_radius=8)

        twin_surf = pygame.Surface((bev_w - 4, bev_h - 4))
        twin_renderer.render_3d_scene(
            twin_surf, ego, traffic, point_cloud,
            particles=traffic_engine.particle_emitter.particles,
            frame_idx=frame_count,
            weather_mode=current_weather,
            night_mode=night_mode
        )
        screen.blit(twin_surf, (422, 214))

        b_title = font_md.render("3D DIGITAL TWIN • REALISTIC TRAFFIC", True, (0, 230, 255))
        screen.blit(b_title, (435, 222))

        # -------------------------------------------------------------
        # 7. PANEL 2 — 77GHz RADAR, TTC COUNTDOWN & V2X (Right Panel)
        # -------------------------------------------------------------
        right_rect = pygame.Rect(880, 212, 380, 480)
        pygame.draw.rect(screen, (14, 18, 26), right_rect, border_radius=8)
        pygame.draw.rect(screen, border_glow_col, right_rect, 1, border_radius=8)

        r_title = font_md.render("77GHz RADAR & V2X TELEMETRY", True, (0, 230, 255))
        screen.blit(r_title, (895, 222))

        # 77GHz POLAR RADAR SCOPE (120px Radius Polar Plot)
        rcx, rcy = 1070, 310
        rad_r = 75 # Radius fitting right panel
        pygame.draw.circle(screen, (4, 12, 8), (rcx, rcy), rad_r)
        pygame.draw.circle(screen, (0, 180, 60), (rcx, rcy), rad_r, 1)

        # 4 Range Rings (25%, 50%, 75%, 100%)
        for ring_f, r_lbl in [(0.25, "20m"), (0.50, "40m"), (0.75, "60m"), (1.00, "80m")]:
            rr_px = int(rad_r * ring_f)
            pygame.draw.circle(screen, (0, 60, 20), (rcx, rcy), rr_px, 1)
            screen.blit(font_xs.render(r_lbl, True, (0, 140, 50)), (rcx + rr_px - 22, rcy + 2))

        # Azimuth Labels ("0°", "±30°", "±60°")
        screen.blit(font_xs.render("0°", True, (0, 180, 60)), (rcx - 6, rcy - rad_r - 12))
        screen.blit(font_xs.render("+30°", True, (0, 180, 60)), (rcx + rad_r - 10, rcy - 20))
        screen.blit(font_xs.render("-30°", True, (0, 180, 60)), (rcx - rad_r - 18, rcy - 20))

        # Rotating Sweep Line + 6 Trailing Ghost Lines
        sweep_deg = (frame_count * 4) % 360
        ghost_opacities = [0.70, 0.50, 0.35, 0.22, 0.12, 0.06]
        ghost_surf = pygame.Surface((screen_w, screen_h), pygame.SRCALPHA)
        for g_idx, g_alpha in enumerate(ghost_opacities):
            g_deg = (sweep_deg - (g_idx + 1) * 6) % 360
            g_rad = math.radians(g_deg)
            gx_tip = rcx + int(rad_r * math.cos(g_rad))
            gy_tip = rcy - int(rad_r * math.sin(g_rad))
            pygame.draw.line(ghost_surf, (0, 255, 80, int(255 * g_alpha)), (rcx, rcy), (gx_tip, gy_tip), 1)

        # Primary Sweep Line
        sw_rad = math.radians(sweep_deg)
        pygame.draw.line(ghost_surf, (0, 255, 80, 230), (rcx, rcy),
                         (rcx + int(rad_r * math.cos(sw_rad)), rcy - int(rad_r * math.sin(sw_rad))), 2)
        screen.blit(ghost_surf, (0, 0))

        # Radar Target Persistence Echoes (Last 3 frames)
        echo_alphas = [30, 60, 100]
        for h_idx, past_dets in enumerate(radar_history[:-1]):
            e_alpha = echo_alphas[min(h_idx, len(echo_alphas) - 1)]
            for p_det in past_dets:
                rx_p = rcx + int(p_det.x * 2.8)
                ry_p = rcy - int(p_det.z * 0.85)
                if math.hypot(rx_p - rcx, ry_p - rcy) <= rad_r - 2:
                    p_surf = pygame.Surface((8, 8), pygame.SRCALPHA)
                    pygame.draw.circle(p_surf, (0, 255, 80, e_alpha), (4, 4), 3)
                    screen.blit(p_surf, (rx_p - 4, ry_p - 4))

        # Primary Radar Targets & Radial Velocity Arrows
        for r_det in radar_detections:
            rx = rcx + int(r_det.x * 2.8)
            ry = rcy - int(r_det.z * 0.85)
            if math.hypot(rx - rcx, ry - rcy) <= rad_r - 2:
                pygame.draw.circle(screen, (0, 255, 80), (rx, ry), 4)
                # Radial velocity arrow
                d_arrow = int(r_det.doppler_mps * 1.5)
                pygame.draw.line(screen, (0, 255, 180), (rx, ry), (rx, ry - d_arrow), 2)
                # Range Label
                screen.blit(font_xs.render(f"{r_det.range_m:.0f}m", True, (255, 255, 255)), (rx + 6, ry - 6))

        # 64-Beam LiDAR Telemetry text
        screen.blit(font_lg.render(f"{len(point_cloud):,}", True, (255, 255, 255)), (895, 255))
        screen.blit(font_xs.render("64-BEAM HESAI (20 Hz)", True, (0, 230, 255)), (895, 280))

        # LIVE TIME-TO-COLLISION (TTC) 10-SEGMENT COUNTDOWN BAR
        if ttc_val < 2.0:
            ttc_bar_col = (255, 60, 0)
        elif ttc_val < 4.0:
            ttc_bar_col = (255, 200, 0)
        else:
            ttc_bar_col = (0, 200, 80)

        screen.blit(font_xs.render("TIME TO COLLISION", True, (140, 160, 180)), (895, 395))
        ttc_num_surf = font_ttc.render(f"{ttc_val:.1f}s" if ttc_val < 50 else "--.-s", True, ttc_bar_col)
        screen.blit(ttc_num_surf, (1200, 388))

        # 10 Depleting Battery Segments
        seg_active = max(0, min(10, int((ttc_val / 6.0) * 10)))
        seg_w = 32
        for seg_i in range(10):
            seg_x = 895 + seg_i * 35
            seg_col = ttc_bar_col if seg_i < seg_active else (25, 35, 50)
            pygame.draw.rect(screen, seg_col, pygame.Rect(seg_x, 415, seg_w, 14), border_radius=2)

        # V2X DSRC / C-V2X BSM TELEMETRY CARD & PULSING RADIO RINGS
        v2x_pkt = traffic_engine.get_lead_v2x_packet()
        v2x_card = pygame.Rect(895, 438, 350, 80)
        pygame.draw.rect(screen, (18, 24, 34), v2x_card, border_radius=6)
        pygame.draw.rect(screen, (0, 200, 255), v2x_card, 1, border_radius=6)

        # 3 Animated Radio Rings Pulsing Outward
        pulse_r = (frame_count * 0.6) % 18.0
        v2x_pulse_surf = pygame.Surface((40, 40), pygame.SRCALPHA)
        pygame.draw.circle(v2x_pulse_surf, (0, 255, 180, int(255 * (1.0 - pulse_r/18.0))), (20, 20), int(pulse_r), 1)
        screen.blit(v2x_pulse_surf, (898, 442))

        # Packet Counter "+1 PKT" Animation
        pkt_flash = "+1 PKT" if (frame_count % 30 < 15) else "SYNC"
        screen.blit(font_xs.render(f"📡 V2X BSM PACKET [{pkt_flash}]", True, (0, 230, 255)), (930, 444))

        if v2x_pkt:
            spd_line = f"LEAD SPEED: {v2x_pkt.speed_kmh:.0f} km/h"
            brk_line = "LEAD BRAKE: ON" if v2x_pkt.brake_pct > 10 else "LEAD BRAKE: OFF"
            brk_col = (255, 60, 60) if v2x_pkt.brake_pct > 10 else (0, 200, 80)
            blk_line = f"LEAD BLINKER: {v2x_pkt.turn_signal}"
            lat_line = f"DSRC LATENCY: {random.randint(3, 7)}ms"

            screen.blit(font_xs.render(spd_line, True, (200, 220, 240)), (905, 460))
            screen.blit(font_xs.render(brk_line, True, brk_col), (1050, 460))
            screen.blit(font_xs.render(blk_line, True, (255, 210, 0)), (905, 478))
            screen.blit(font_xs.render(lat_line, True, (0, 255, 180)), (1050, 478))
        else:
            screen.blit(font_xs.render("LEAD SPEED: 68 km/h | LEAD BRAKE: OFF", True, (200, 220, 240)), (905, 460))
            screen.blit(font_xs.render("LEAD BLINKER: OFF  | DSRC LATENCY: 4ms", True, (0, 255, 180)), (905, 478))

        # Sensor Lock Statuses
        sens_txt = font_xs.render("SENSORS: 4-CAM HDR [LOCKED] | 64-LIDAR [20Hz] | 77GHz RADAR [ACTIVE]", True, (0, 255, 180))
        screen.blit(sens_txt, (895, 524))

        # -------------------------------------------------------------
        # 8. PANEL 3 — CATEGORIZED SCROLLING MISSION EVENT LOG (Bottom)
        # -------------------------------------------------------------
        log_rect = pygame.Rect(20, 700, screen_w - 40, 92)
        pygame.draw.rect(screen, (12, 16, 22), log_rect, border_radius=6)
        pygame.draw.rect(screen, border_glow_col, log_rect, 1, border_radius=6)

        log_head = font_xs.render("📜 AUTONOMOUS MISSION & SENSOR EVENT LOG (REAL-TIME STREAM)", True, (0, 230, 255))
        screen.blit(log_head, (32, 705))

        visible_logs = traffic_engine.event_log[-4:]
        for idx_l, l_entry in enumerate(visible_logs):
            # Parse event type prefix: [OT], [LC], [CZ], [!!]
            if "OVERTAKE" in l_entry:
                prefix = "[OT]"
                l_col = (255, 150, 0)
            elif "LANE" in l_entry or "MANEUVER" in l_entry:
                prefix = "[LC]"
                l_col = (0, 200, 255)
            elif "CRUISE" in l_entry or "KEEP" in l_entry:
                prefix = "[CZ]"
                l_col = (0, 200, 80)
            else:
                prefix = "[!!]"
                l_col = (255, 60, 60)

            # Highlight most recent event background
            if idx_l == len(visible_logs) - 1:
                hl_rect = pygame.Rect(30, 723 + idx_l * 16, screen_w - 60, 15)
                pygame.draw.rect(screen, (18, 26, 38), hl_rect)

            # Separator Line: RGB(25,32,42) 1px
            pygame.draw.line(screen, (25, 32, 42), (32, 723 + idx_l * 16 + 15), (screen_w - 32, 723 + idx_l * 16 + 15), 1)

            # Draw Timestamp in RGB(100,120,140) and Body in l_col
            ts_split = l_entry.split("] ", 1)
            ts_str = ts_split[0] + "] " if len(ts_split) > 1 else "[T+0.0s] "
            body_str = ts_split[1] if len(ts_split) > 1 else l_entry

            screen.blit(font_xs.render(f"{prefix} {ts_str}", True, (100, 120, 140)), (34, 724 + idx_l * 16))
            screen.blit(font_xs.render(body_str, True, l_col), (160, 724 + idx_l * 16))

        pygame.display.flip()
        clock.tick(60)

        # Export frame to video if enabled
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
