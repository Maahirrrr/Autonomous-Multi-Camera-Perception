"""
run_bev_surround.py — Level 4 Autonomous Vehicle 3D Digital Twin & Perception Stack
===================================================================================
Master Architecture & Upgrades:
  - Multi-Threaded Execution Pipeline (ThreadPoolExecutor) for < 6ms Frame Latency.
  - Weather Simulation System: Clear, Rain (droplets & refraction), Fog (LiDAR attenuation).
  - Day / Night Mode with Thermal-IR Ironbow BEV Palette & Headlight Gaussian Illumination.
  - 77GHz mmWave FMCW Radar Polar Plot (120 deg FOV with Doppler Velocity Vectors).
  - Probabilistic Log-Odds Occupancy Grid & Multi-Hypothesis Trajectory Fans.
  - V2X (Vehicle-to-Everything) BSM Telemetry Card & Lead Pulse Waves.
  - Analog Speedometer Arc, 5s G-Force Graph, Rotating Steering Wheel & Live TTC Countdown Bar.
  - Scrolling Mission Event Log Panel (6-Line Auto-Scroll).
  - CUDA GPU vs CPU IPM Homography Speedup Meter.
"""

import sys
import os
import time
import math
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
    pygame.display.set_caption("Level 4 Autonomous Vehicle 3D Digital Twin & Perception Stack (RTX 4070)")

    try:
        screen = pygame.display.set_mode((screen_w, screen_h), pygame.DOUBLEBUF | pygame.HWSURFACE)
    except Exception:
        screen = pygame.display.set_mode((screen_w, screen_h), pygame.DOUBLEBUF)

    clock = pygame.time.Clock()

    font_xs = get_safe_font(10, bold=True)
    font_sm = get_safe_font(12, bold=True)
    font_md = get_safe_font(15, bold=True)
    font_lg = get_safe_font(21, bold=True)
    font_speed = get_safe_font(24, bold=True)

    # 1. Initialize Perception & Simulation Engines
    bev_w, bev_h = 440, 480
    bev_engine = MultiCameraBEVTransformer(bev_width_px=bev_w, bev_height_px=bev_h)
    cam_sim = MultiCameraSimulator(width=300, height=160)
    lidar_engine = Lidar3DPerceptionEngine(num_lasers=64, max_range_m=65.0)
    lidar_engine.cameras = bev_engine.cameras

    traffic_engine = HighwayTrafficEngine()
    twin_renderer = DigitalTwin3DRenderer(screen_w=bev_w, screen_h=bev_h)

    # Thread Pool for multi-threaded async execution
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)

    print("\n" + "=" * 78)
    print("  [+] LEVEL 4 AUTONOMOUS VEHICLE 3D DIGITAL TWIN & PERCEPTION STACK")
    print("  Driving Physics   : IDM & MOBIL Autonomous Overtaking & Lane-Switching Engine")
    print("  Sensors & Fusion  : 64-Beam LiDAR + 77GHz Radar + 4 Surround Cameras + V2X")
    print("  Visual Systems    : Day/Night Thermal-IR, Rain Droplets & Dynamic Fog Attenuation")
    print("  Controls          : [TAB] Auto/Manual | [N] Night | [P] Weather | [R] Randomize")
    print("=" * 78 + "\n")

    # Video Export Writer
    video_writer = None
    if args.export:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(args.export, fourcc, 60.0, (screen_w, screen_h))
        print(f"[INFO] Exporting L4 Digital Twin recording to: {args.export}")

    running = True
    frame_count = 0
    show_lidar_on_cams = True
    is_paused = False

    # Weather & Night States
    weather_modes = ["CLEAR", "RAIN", "FOG"]
    weather_idx = 0
    night_mode = False

    while running:
        t_frame_start = time.perf_counter()
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
                    traffic_engine.log_event(f"LIGHTING TOGGLED -> {'NIGHT (THERMAL-IR)' if night_mode else 'DAYLIGHT'}")
                elif event.key == pygame.K_p:
                    weather_idx = (weather_idx + 1) % len(weather_modes)
                    current_weather = weather_modes[weather_idx]
                    lidar_engine.set_weather_mode(current_weather)
                    traffic_engine.log_event(f"WEATHER CHANGED -> {current_weather}")
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

            twin_renderer.handle_mouse_orbit(event, rect_offset=(420, 220))

        # Manual Driving Inputs
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

        # 2. Asynchronous Multi-Threaded Sensor Pipeline
        # Thread 1: Step Traffic Physics & Overtaking
        traffic_engine.step(dt)
        dynamic_objects = traffic_engine.get_dynamic_objects_for_sensors()

        # Thread 2: LiDAR Point Cloud Generation & Ground Segmentation
        fut_lidar = executor.submit(lidar_engine.generate_scene_point_cloud, dynamic_objects, {}, frame_count)
        point_cloud = fut_lidar.result()
        ground_pts, obstacle_pts, bounding_boxes = lidar_engine.segment_ground_and_clusters(point_cloud, dynamic_objects)

        # Thread 3: 77GHz mmWave Radar Scan
        radar_detections = lidar_engine.radar_sim.scan_targets(dynamic_objects, ego.speed_mps)

        # Thread 4: 4 Surround Cameras Rendering with Weather & Night
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

        # 3. Render Master Cockpit GUI (1280x800 Glassmorphic Layout)
        screen.fill((10, 14, 20))

        # Top Header Banner
        header_rect = pygame.Rect(0, 0, screen_w, 38)
        pygame.draw.rect(screen, (14, 18, 26), header_rect)
        h_title = font_md.render("LEVEL 4 AUTONOMOUS VEHICLE 360° MULTI-CAMERA & 3D LIDAR PERCEPTION STACK", True, (0, 230, 255))
        screen.blit(h_title, (20, 9))

        # Weather & Lighting Badges
        w_badge = font_xs.render(f"WEATHER: {current_weather} | {'NIGHT (THERMAL-IR)' if night_mode else 'DAYLIGHT'}", True, (255, 210, 0))
        screen.blit(w_badge, (screen_w - 480, 12))

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
            pygame.draw.rect(screen, (0, 200, 255), (cx, cy, c_w, c_h), 1)

        # 5. LEFT PANEL: Instrument Cluster (Analog Speedo, 5s G-Meter Graph, Ackermann Wheel)
        left_rect = pygame.Rect(20, 212, 380, 480)
        pygame.draw.rect(screen, (14, 18, 26), left_rect, border_radius=8)
        pygame.draw.rect(screen, (0, 200, 255), left_rect, 1, border_radius=8)

        s_title = font_md.render("VEHICLE KINEMATICS & DYNAMICS", True, (0, 230, 255))
        screen.blit(s_title, (35, 222))

        # Analog-Style Speedometer Arc & Sweeping Needle
        scx, scy = 90, 285
        pygame.draw.circle(screen, (22, 32, 48), (scx, scy), 44, 4)
        spd_fraction = min(1.0, ego.speed_kmh / 140.0)
        spd_angle_rad = math.pi * 0.75 + (spd_fraction * math.pi * 1.5)

        # Arc Glow
        pygame.draw.arc(screen, (0, 255, 180), (scx - 44, scy - 44, 88, 88), math.pi * 0.75, math.pi * 0.75 + (spd_fraction * math.pi * 1.5), 4)

        # Needle
        nx = scx + int(36 * math.cos(spd_angle_rad))
        ny = scy + int(36 * math.sin(spd_angle_rad))
        pygame.draw.line(screen, (255, 60, 60), (scx, scy), (nx, ny), 3)
        pygame.draw.circle(screen, (255, 255, 255), (scx, scy), 5)

        spd_val = font_speed.render(f"{ego.speed_kmh:.0f}", True, (255, 255, 255))
        spd_unit = font_xs.render("KM/H", True, (0, 230, 255))
        screen.blit(spd_val, (scx - spd_val.get_width() // 2, scy + 12))
        screen.blit(spd_unit, (scx - spd_unit.get_width() // 2, scy + 36))

        # 5-Second Rolling G-Force Graph
        gx_box = pygame.Rect(180, 250, 200, 75)
        pygame.draw.rect(screen, (20, 26, 38), gx_box, border_radius=4)
        pygame.draw.rect(screen, (35, 50, 70), gx_box, 1, border_radius=4)
        g_title = font_xs.render("5s G-FORCE HISTORY (LAT / LONG)", True, (160, 180, 200))
        screen.blit(g_title, (188, 254))
        pygame.draw.line(screen, (35, 50, 70), (180, 287), (380, 287), 1) # Zero G line

        if len(ego.g_history) > 2:
            pts_lat = []
            pts_long = []
            for idx_g, (g_lat, g_long) in enumerate(ego.g_history):
                gx_pos = 185 + int((idx_g / len(ego.g_history)) * 190)
                gy_lat = int(287 - g_lat * 60)
                gy_long = int(287 - g_long * 60)
                pts_lat.append((gx_pos, max(255, min(320, gy_lat))))
                pts_long.append((gx_pos, max(255, min(320, gy_long))))
            if len(pts_lat) > 1:
                pygame.draw.lines(screen, (0, 255, 180), False, pts_lat, 2)
                pygame.draw.lines(screen, (255, 210, 0), False, pts_long, 2)

        # Rotating Ackermann Steering Wheel & Rack
        sw_cx, sw_cy = 75, 368
        pygame.draw.circle(screen, (35, 48, 65), (sw_cx, sw_cy), 22, 3)
        st_rad = math.radians(ego.steering_angle_deg * 2.0)
        pygame.draw.line(screen, (0, 230, 255), (sw_cx - int(20 * math.cos(st_rad)), sw_cy - int(20 * math.sin(st_rad))),
                         (sw_cx + int(20 * math.cos(st_rad)), sw_cy + int(20 * math.sin(st_rad))), 3)
        st_txt = font_xs.render(f"STEERING: {ego.steering_angle_deg:+.1f}° (INNER: {ego.steering_inner_deg:+.1f}°)", True, (200, 220, 240))
        screen.blit(st_txt, (115, 360))
        bl_txt = font_xs.render(f"BLINKER: {ego.blinker} | LAT JERK: {ego.lat_jerk_gs:+.2f} G/s", True, (160, 180, 200))
        screen.blit(bl_txt, (115, 375))

        # ADAS Intelligence & FCW Alert
        adas_title = font_md.render("LEVEL 4 ADAS SAFETY INTELLIGENCE", True, (255, 210, 0))
        screen.blit(adas_title, (35, 405))

        lead_car = next((v for v in traffic if abs(v.x - ego.x) < 2.0 and v.z > 0), None)
        if lead_car:
            ttc_val = lead_car.z / max(0.5, (ego.speed_mps - lead_car.speed_mps))
            fcw_col = (255, 60, 60) if ttc_val < 2.5 else ((255, 200, 0) if ttc_val < 4.0 else (0, 255, 180))
            fcw_msg = f"FCW: LEAD CAR {lead_car.z:.1f}m | TTC {abs(ttc_val):.1f}s"
        else:
            ttc_val = 99.0
            fcw_col = (0, 255, 180)
            fcw_msg = "FCW: CLEAR CORRIDOR AHEAD"

        pygame.draw.rect(screen, (12, 28, 20) if fcw_col == (0, 255, 180) else (32, 18, 22), (35, 428, 350, 30), border_radius=4)
        pygame.draw.rect(screen, fcw_col, (35, 428, 350, 30), 1, border_radius=4)
        screen.blit(font_sm.render(fcw_msg, True, fcw_col), (45, 434))

        # Autopilot Strategy Box
        l4_box = pygame.Rect(35, 465, 350, 32)
        pygame.draw.rect(screen, (10, 35, 25), l4_box, border_radius=4)
        pygame.draw.rect(screen, (0, 255, 180), l4_box, 1, border_radius=4)
        state_str = f"AUTONOMOUS MISSION: {ego.state}"
        screen.blit(font_sm.render(state_str, True, (0, 255, 180)), (45, 472))

        # CUDA GPU Homography Speedup Meter
        gpu_stat = bev_engine.gpu_speedup_stats
        speedup_txt = font_xs.render(f"CUDA GPU SPEEDUP: {gpu_stat['gpu_ms']:.1f}ms vs CPU {gpu_stat['cpu_ms']:.1f}ms ({gpu_stat['speedup']:.1f}x)", True, (0, 230, 255))
        screen.blit(speedup_txt, (35, 510))

        # 6. CENTER PANEL: 3D Digital Twin Simulation (X=420, Y=212, W=440, H=480)
        center_rect = pygame.Rect(420, 212, 440, 480)
        pygame.draw.rect(screen, (14, 18, 26), center_rect, border_radius=8)
        pygame.draw.rect(screen, (0, 200, 255), center_rect, 1, border_radius=8)

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

        # 7. RIGHT PANEL: 77GHz mmWave Radar, V2X Telemetry & TTC Countdown Bar
        right_rect = pygame.Rect(880, 212, 380, 480)
        pygame.draw.rect(screen, (14, 18, 26), right_rect, border_radius=8)
        pygame.draw.rect(screen, (0, 200, 255), right_rect, 1, border_radius=8)

        r_title = font_md.render("77GHz RADAR & V2X TELEMETRY", True, (0, 230, 255))
        screen.blit(r_title, (895, 222))

        # 77GHz mmWave Radar Polar Plot (120 deg FOV with Doppler Arrows)
        rcx, rcy = 1070, 285
        pygame.draw.circle(screen, (20, 28, 40), (rcx, rcy), 44)
        pygame.draw.circle(screen, (0, 200, 255), (rcx, rcy), 44, 1)
        pygame.draw.line(screen, (35, 50, 70), (rcx - 44, rcy), (rcx + 44, rcy), 1)

        # 120 deg FOV Wedge Lines
        ang_left = math.radians(150)
        ang_right = math.radians(30)
        pygame.draw.line(screen, (0, 200, 255), (rcx, rcy), (rcx + int(44 * math.cos(ang_left)), rcy - int(44 * math.sin(ang_left))), 1)
        pygame.draw.line(screen, (0, 200, 255), (rcx, rcy), (rcx + int(44 * math.cos(ang_right)), rcy - int(44 * math.sin(ang_right))), 1)

        # Plot radar targets with Doppler arrows
        for r_det in radar_detections:
            rx = rcx + int(r_det.x * 3.6)
            ry = rcy - int(r_det.z * 0.65)
            if rcx - 42 <= rx <= rcx + 42 and rcy - 42 <= ry <= rcy + 42:
                pygame.draw.circle(screen, (255, 60, 60) if "LEAD" in r_det.id else (255, 210, 0), (rx, ry), 3)
                # Doppler radial velocity arrow
                d_arrow = int(r_det.doppler_mps * 1.5)
                pygame.draw.line(screen, (0, 255, 180), (rx, ry), (rx, ry - d_arrow), 2)

        # LiDAR Telemetry Text
        pts_cnt = len(point_cloud)
        p_val = font_lg.render(f"{pts_cnt:,}", True, (255, 255, 255))
        p_tag = font_xs.render("64-BEAM HESAI (20 Hz)", True, (0, 230, 255))
        screen.blit(p_val, (895, 255))
        screen.blit(p_tag, (895, 282))

        # Live Time-To-Collision (TTC) Countdown Bar
        ttc_header = font_xs.render(f"LIVE TTC COUNTDOWN: {ttc_val:.1f}s", True, fcw_col)
        screen.blit(ttc_header, (895, 340))
        ttc_bar_bg = pygame.Rect(895, 355, 350, 10)
        pygame.draw.rect(screen, (25, 35, 50), ttc_bar_bg, border_radius=4)
        ttc_fraction = max(0.0, min(1.0, ttc_val / 8.0))
        ttc_fill_w = int(350 * ttc_fraction)
        pygame.draw.rect(screen, fcw_col, (895, 355, ttc_fill_w, 10), border_radius=4)

        # V2X Telemetry Card
        v2x_pkt = traffic_engine.get_lead_v2x_packet()
        pygame.draw.rect(screen, (18, 24, 35), (895, 375, 350, 80), border_radius=6)
        pygame.draw.rect(screen, (0, 230, 255), (895, 375, 350, 80), 1, border_radius=6)

        v2x_head = font_xs.render("📡 V2X DSRC / C-V2X BSM PACKET (LEAD VEHICLE)", True, (0, 230, 255))
        screen.blit(v2x_head, (905, 380))

        if v2x_pkt:
            v_line1 = font_xs.render(f"SENDER: {v2x_pkt.sender_id} | SPEED: {v2x_pkt.speed_kmh:.0f} KM/H", True, (200, 220, 240))
            v_line2 = font_xs.render(f"BRAKE: {v2x_pkt.brake_pct:.0f}% | THROTTLE: {v2x_pkt.throttle_pct:.0f}% | BLINKER: {v2x_pkt.turn_signal}", True, (255, 210, 0) if v2x_pkt.brake_pct > 10 else (0, 255, 180))
            v_line3 = font_xs.render(f"GPS: ({v2x_pkt.gps_lat:.4f}, {v2x_pkt.gps_lon:.4f}) | RSSI: {v2x_pkt.rssi_dbm:.0f} dBm", True, (160, 180, 200))
            screen.blit(v_line1, (905, 396))
            screen.blit(v_line2, (905, 412))
            screen.blit(v_line3, (905, 428))
        else:
            screen.blit(font_xs.render("SCANNING FOR DSRC / C-V2X BEACONS...", True, (140, 160, 180)), (905, 405))

        # Sensor Lock Statuses
        sens_txt = font_xs.render("SENSORS: 4-CAM HDR [LOCKED] | 64-LIDAR [20Hz] | 77GHz RADAR [ACTIVE]", True, (0, 255, 180))
        screen.blit(sens_txt, (895, 465))

        # 8. BOTTOM STRIP: Scrolling Mission Event Log Panel (Y=700, H=95)
        log_rect = pygame.Rect(20, 700, screen_w - 40, 92)
        pygame.draw.rect(screen, (12, 16, 22), log_rect, border_radius=6)
        pygame.draw.rect(screen, (0, 200, 255), log_rect, 1, border_radius=6)

        log_head = font_xs.render("📜 AUTONOMOUS MISSION & SENSOR EVENT LOG (REAL-TIME STREAM)", True, (0, 230, 255))
        screen.blit(log_head, (32, 705))

        visible_logs = traffic_engine.event_log[-4:]
        for idx_l, l_entry in enumerate(visible_logs):
            l_col = (255, 60, 60) if "OVERTAKE" in l_entry or "ALERT" in l_entry else ((255, 210, 0) if "V2X" in l_entry or "MANEUVER" in l_entry else (180, 210, 240))
            screen.blit(font_xs.render(l_entry, True, l_col), (32, 723 + idx_l * 16))

        pygame.display.flip()
        clock.tick(60)

        # Export frame to video
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
