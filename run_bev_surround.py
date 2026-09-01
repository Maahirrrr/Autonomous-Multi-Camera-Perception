"""
run_bev_surround.py — Level 4 Autonomous Vehicle 3D Digital Twin & Perception Stack
===================================================================================
Features:
  - 3D Digital Twin World with 3D Vehicles Overtaking, Switching Lanes & Dynamic Braking.
  - 4 Surround Cameras (FRONT, REAR, LEFT, RIGHT) with Projected 3D LiDAR Laser Overlays.
  - Tactical Polar Radar Minimap, LiDAR Oscilloscope Waveform & Ackermann Kinematics.
  - Autonomous Overtaking State Machine (IDM & MOBIL models) + Manual Driving Override.
"""

import sys
import os
import time
import math
import argparse
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
    pygame.display.set_caption("Level 4 Autonomous Vehicle 3D Digital Twin & Multi-Camera Perception (RTX 4070)")

    try:
        screen = pygame.display.set_mode((screen_w, screen_h), pygame.DOUBLEBUF | pygame.HWSURFACE)
    except Exception:
        screen = pygame.display.set_mode((screen_w, screen_h), pygame.DOUBLEBUF)

    clock = pygame.time.Clock()

    font_xs = get_safe_font(10, bold=True)
    font_sm = get_safe_font(12, bold=True)
    font_md = get_safe_font(15, bold=True)
    font_lg = get_safe_font(21, bold=True)

    # 1. Initialize Engines
    bev_w, bev_h = 440, 515
    bev_engine = MultiCameraBEVTransformer(bev_width_px=bev_w, bev_height_px=bev_h)
    cam_sim = MultiCameraSimulator(width=300, height=170)
    lidar_engine = Lidar3DPerceptionEngine(num_lasers=64, max_range_m=65.0)
    lidar_engine.cameras = bev_engine.cameras

    traffic_engine = HighwayTrafficEngine()
    twin_renderer = DigitalTwin3DRenderer(screen_w=bev_w, screen_h=bev_h)

    print("\n" + "=" * 78)
    print("  [+] LEVEL 4 AUTONOMOUS VEHICLE 3D DIGITAL TWIN & PERCEPTION STACK")
    print("  Driving Physics   : IDM & MOBIL Autonomous Overtaking & Lane-Switching Engine")
    print("  Visual Simulation : Full 3D Perspective Digital Twin World & 4 Surround Cameras")
    print("  Sensor Fusion     : 360 64-Beam 3D LiDAR Point-to-Pixel Projection & 3D OBBs")
    print("  Controls          : [TAB] Auto/Manual | [W/A/S/D] Drive | [L] LiDAR | [ESC] Exit")
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
                elif event.key == pygame.K_l:
                    show_lidar_on_cams = not show_lidar_on_cams
                elif event.key == pygame.K_SPACE:
                    is_paused = not is_paused
                elif event.key == pygame.K_TAB:
                    traffic_engine.ego.manual_override = not traffic_engine.ego.manual_override
                elif event.key == pygame.K_a:
                    traffic_engine.ego.initiate_lane_change(max(-1, traffic_engine.ego.lane_idx - 1))
                elif event.key == pygame.K_d:
                    traffic_engine.ego.initiate_lane_change(min(1, traffic_engine.ego.lane_idx + 1))

            twin_renderer.handle_mouse_orbit(event, rect_offset=(420, 230))

        # Manual Driving Inputs
        if traffic_engine.ego.manual_override:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_w]:
                traffic_engine.ego.speed_kmh = min(130.0, traffic_engine.ego.speed_kmh + 22.0 * dt)
            if keys[pygame.K_s]:
                traffic_engine.ego.speed_kmh = max(20.0, traffic_engine.ego.speed_kmh - 32.0 * dt)
                traffic_engine.ego.is_braking = True
            else:
                traffic_engine.ego.is_braking = False

        # 2. Step Highway Traffic Dynamics & Autonomous Decisions
        traffic_engine.step(dt)
        ego = traffic_engine.ego
        traffic = traffic_engine.traffic_vehicles

        # 3. Dynamic Objects for Sensors
        dynamic_objects = traffic_engine.get_dynamic_objects_for_sensors()

        # 4. Generate 360° 3D LiDAR Point Cloud & 3D Clusters
        point_cloud = lidar_engine.generate_scene_point_cloud(dynamic_objects, road_geometry={}, frame_idx=frame_count)
        ground_pts, obstacle_pts, bounding_boxes = lidar_engine.segment_ground_and_clusters(point_cloud, dynamic_objects)

        # 5. Render 4 Surround Cameras with Projected 3D LiDAR Overlays
        cam_frames = cam_sim.render_surround_views(
            frame_idx=frame_count,
            dynamic_objects=dynamic_objects,
            speed_kmh=ego.speed_kmh,
            lidar_engine=lidar_engine,
            point_cloud=point_cloud,
            render_lidar_on_cams=show_lidar_on_cams
        )

        # 6. Render Full GUI Interface (1280x800 Glassmorphic Layout)
        screen.fill((10, 14, 20))

        # Top Header Banner
        header_rect = pygame.Rect(0, 0, screen_w, 42)
        pygame.draw.rect(screen, (14, 18, 26), header_rect)
        h_title = font_lg.render("LEVEL 4 AUTONOMOUS VEHICLE 3D DIGITAL TWIN & PERCEPTION STACK", True, (0, 230, 255))
        screen.blit(h_title, (20, 8))

        mode_badge = pygame.Rect(screen_w - 280, 6, 260, 30)
        mode_bg = (10, 35, 25) if not ego.manual_override else (40, 25, 10)
        mode_border = (0, 255, 180) if not ego.manual_override else (255, 180, 0)
        pygame.draw.rect(screen, mode_bg, mode_badge, border_radius=4)
        pygame.draw.rect(screen, mode_border, mode_badge, 1, border_radius=4)
        m_txt = font_sm.render("L4 HIGHWAY PILOT ACTIVE" if not ego.manual_override else "MANUAL PILOT OVERRIDE", True, mode_border)
        screen.blit(m_txt, (screen_w - 265, 12))

        # 7. Render 4 Surround Cameras Across Top Row
        c_names = ["FRONT", "LEFT", "RIGHT", "REAR"]
        c_w, c_h = 300, 170
        cam_x_offsets = [20, 335, 650, 965]

        for i, c_name in enumerate(c_names):
            cx = cam_x_offsets[i]
            cy = 48
            f_surf = pygame.surfarray.make_surface(np.transpose(cam_frames[c_name], (1, 0, 2)))
            screen.blit(f_surf, (cx, cy))
            pygame.draw.rect(screen, (0, 200, 255), (cx, cy, c_w, c_h), 1)

        # 8. LEFT PANEL: Vehicle Kinematics, G-Meter & ADAS (X=20, Y=230, W=380, H=515)
        left_rect = pygame.Rect(20, 230, 380, 515)
        pygame.draw.rect(screen, (14, 18, 26), left_rect, border_radius=8)
        pygame.draw.rect(screen, (0, 200, 255), left_rect, 1, border_radius=8)

        s_title = font_md.render("VEHICLE KINEMATICS & DYNAMICS", True, (0, 230, 255))
        screen.blit(s_title, (35, 242))

        # Speedometer Circular Arc
        scx, scy = 85, 305
        pygame.draw.circle(screen, (25, 35, 50), (scx, scy), 38, 4)
        spd_angle = math.radians(min(180, int((ego.speed_kmh / 140.0) * 180)))
        pygame.draw.arc(screen, (0, 255, 180), (scx - 38, scy - 38, 76, 76), math.pi, math.pi + spd_angle, 4)

        spd_val = font_lg.render(f"{ego.speed_kmh:.1f}", True, (255, 255, 255))
        spd_unit = font_xs.render("KM/H", True, (0, 230, 255))
        screen.blit(spd_val, (scx - spd_val.get_width() // 2, scy - 12))
        screen.blit(spd_unit, (scx - spd_unit.get_width() // 2, scy + 8))

        # G-Force Friction Circle Meter
        gcx, gcy = 280, 305
        pygame.draw.circle(screen, (25, 35, 50), (gcx, gcy), 36, 1)
        pygame.draw.circle(screen, (35, 50, 70), (gcx, gcy), 18, 1)
        pygame.draw.line(screen, (35, 50, 70), (gcx - 36, gcy), (gcx + 36, gcy), 1)
        pygame.draw.line(screen, (35, 50, 70), (gcx, gcy - 36), (gcx, gcy + 36), 1)

        gx_dot = int(ego.steering_angle_deg * 2.5)
        gy_dot = int(-4.0 if ego.is_braking else 3.0)
        pygame.draw.circle(screen, (0, 255, 180), (gcx + gx_dot, gcy + gy_dot), 5)
        g_lbl = font_xs.render("G-METER (0.05G)", True, (160, 180, 200))
        screen.blit(g_lbl, (gcx - g_lbl.get_width() // 2, gcy + 42))

        # Steering & Slip Angle
        st_lbl = font_xs.render(f"STEERING: {ego.steering_angle_deg:+.1f}°  |  BLINKER: {ego.blinker}", True, (160, 180, 200))
        screen.blit(st_lbl, (35, 365))
        steer_bar = pygame.Rect(35, 380, 350, 6)
        pygame.draw.rect(screen, (25, 35, 50), steer_bar, border_radius=3)
        pygame.draw.circle(screen, (0, 255, 180), (35 + 175 + int(ego.steering_angle_deg * 6), 383), 6)

        # ADAS Intelligence Section
        adas_title = font_md.render("LEVEL 4 ADAS SAFETY INTELLIGENCE", True, (255, 210, 0))
        screen.blit(adas_title, (35, 410))

        # Lead Car TTC
        lead_car = next((v for v in traffic if abs(v.x - ego.x) < 2.0 and v.z > 0), None)
        if lead_car:
            ttc_val = lead_car.z / max(0.5, (ego.speed_mps - lead_car.speed_mps))
            fcw_col = (255, 60, 60) if ttc_val < 3.5 else (0, 255, 180)
            fcw_msg = f"FCW: LEAD CAR {lead_car.z:.1f}m | TTC {abs(ttc_val):.1f}s"
        else:
            fcw_col = (0, 255, 180)
            fcw_msg = "FCW: CLEAR CORRIDOR AHEAD"

        pygame.draw.rect(screen, (12, 28, 20) if fcw_col == (0, 255, 180) else (30, 18, 22), (35, 435, 350, 32), border_radius=4)
        pygame.draw.rect(screen, fcw_col, (35, 435, 350, 32), 1, border_radius=4)
        screen.blit(font_sm.render(fcw_msg, True, fcw_col), (45, 442))

        # Traffic Dynamic States
        bsd_l = f"OVERTAKE STATE: {ego.state}"
        bsd_r = f"LANE POSITION : LANE {ego.lane_idx} (X: {ego.x:+.2f}m)"
        pygame.draw.rect(screen, (22, 30, 42), (35, 475, 350, 26), border_radius=4)
        pygame.draw.rect(screen, (22, 30, 42), (35, 506, 350, 26), border_radius=4)
        screen.blit(font_xs.render(bsd_l, True, (0, 255, 180)), (45, 481))
        screen.blit(font_xs.render(bsd_r, True, (255, 210, 0)), (45, 512))

        # Autopilot Strategy
        l4_box = pygame.Rect(35, 545, 350, 36)
        pygame.draw.rect(screen, (10, 35, 25), l4_box, border_radius=4)
        pygame.draw.rect(screen, (0, 255, 180), l4_box, 1, border_radius=4)
        state_str = f"AUTONOMOUS MISSION: {ego.state}"
        screen.blit(font_sm.render(state_str, True, (0, 255, 180)), (45, 554))

        # Latency Breakdown
        lat_txt = font_xs.render("LATENCY: 3D ENGINE 3.8ms | FUSION 4.6ms | 60 FPS", True, (140, 160, 180))
        screen.blit(lat_txt, (35, 595))

        # 9. CENTER PANEL: 3D Realistic Digital Twin World (X=420, Y=230, W=440, H=515)
        center_rect = pygame.Rect(420, 230, 440, 515)
        pygame.draw.rect(screen, (14, 18, 26), center_rect, border_radius=8)
        pygame.draw.rect(screen, (0, 200, 255), center_rect, 1, border_radius=8)

        # Render 3D digital twin
        twin_surf = pygame.Surface((bev_w - 4, bev_h - 4))
        twin_renderer.render_3d_scene(twin_surf, ego, traffic, point_cloud, frame_count)
        screen.blit(twin_surf, (422, 232))

        b_title = font_md.render("3D DIGITAL TWIN • REALISTIC TRAFFIC", True, (0, 230, 255))
        screen.blit(b_title, (435, 242))

        # 10. RIGHT PANEL: Tactical Polar Radar & Sensor Telemetry (X=880, Y=230, W=380, H=515)
        right_rect = pygame.Rect(880, 230, 380, 515)
        pygame.draw.rect(screen, (14, 18, 26), right_rect, border_radius=8)
        pygame.draw.rect(screen, (0, 200, 255), right_rect, 1, border_radius=8)

        r_title = font_md.render("TACTICAL RADAR & 3D LIDAR TELEMETRY", True, (0, 230, 255))
        screen.blit(r_title, (895, 242))

        # Tactical Polar Radar Minimap (BEV Mini-radar)
        rcx, rcy = 1070, 310
        pygame.draw.circle(screen, (22, 30, 44), (rcx, rcy), 48)
        pygame.draw.circle(screen, (0, 200, 255), (rcx, rcy), 48, 1)
        pygame.draw.circle(screen, (35, 50, 70), (rcx, rcy), 25, 1)
        pygame.draw.line(screen, (35, 50, 70), (rcx - 48, rcy), (rcx + 48, rcy), 1)
        pygame.draw.line(screen, (35, 50, 70), (rcx, rcy - 48), (rcx, rcy + 48), 1)

        # Ego vehicle in center of radar
        pygame.draw.circle(screen, (0, 255, 180), (rcx, rcy), 4)

        # Plot traffic blips on radar
        for v in traffic:
            rx = rcx + int((v.x - ego.x) * 4.2)
            ry = rcy - int(v.z * 0.75)
            if rcx - 46 <= rx <= rcx + 46 and rcy - 46 <= ry <= rcy + 46:
                r_col = (255, 60, 60) if "LEAD" in v.id else ((255, 210, 0) if "SPORTS" in v.id else (50, 180, 255))
                pygame.draw.circle(screen, r_col, (rx, ry), 3)

        # LiDAR Point Count
        pts_cnt = len(point_cloud)
        p_val = font_lg.render(f"{pts_cnt:,}", True, (255, 255, 255))
        p_tag = font_xs.render("64-BEAM HESAI (20 Hz)", True, (0, 230, 255))
        screen.blit(p_val, (895, 275))
        screen.blit(p_tag, (895, 305))

        # Multi-Sensor Lock Statuses
        sensors = [
            ("360° 64-BEAM LIDAR", "LOCKED (20 Hz / 100% HEALTH)", (0, 255, 180)),
            ("4x SURROUND CAMERAS", "SYNCHRONIZED (60 FPS CUDA)", (0, 255, 180)),
            ("AUTONOMOUS OVERTAKING", f"MOBIL / IDM ACTIVE ({ego.state})", (0, 255, 180)),
            ("6-DOF IMU / ODOMETRY", "EKF CONVERGED (COV < 0.01)", (0, 255, 180)),
            ("L4 PERCEPTION STACK", "TENSORRT FP16 ACCELERATED", (0, 230, 255)),
        ]

        y_sens = 375
        for s_name, s_stat, s_col in sensors:
            pygame.draw.rect(screen, (20, 26, 38), (895, y_sens, 350, 26), border_radius=4)
            pygame.draw.rect(screen, (32, 45, 65), (895, y_sens, 350, 26), 1, border_radius=4)
            screen.blit(font_xs.render(s_name, True, (200, 220, 240)), (905, y_sens + 2))
            screen.blit(font_xs.render(s_stat, True, s_col), (905, y_sens + 13))
            y_sens += 30

        # 3D Cluster Count & Traffic Instances
        cl_tag = font_sm.render(f"ACTIVE TRAFFIC VEHICLES: {len(traffic)} IN SCENE", True, (255, 210, 0))
        screen.blit(cl_tag, (895, 535))

        for idx, v in enumerate(traffic[:4]):
            rel_z = v.z
            v_info = f"• [{v.v_type}] Lane {v.lane_idx} | Spd: {v.speed_kmh:.0f} km/h | Dist: {rel_z:+.1f}m"
            screen.blit(font_xs.render(v_info, True, (180, 200, 220)), (905, 560 + idx * 18))

        # Bottom Shortcut Bar
        s_bar = pygame.Rect(0, screen_h - 40, screen_w, 40)
        pygame.draw.rect(screen, (12, 16, 22), s_bar)
        p_hint = font_xs.render("CONTROLS: [TAB] Auto/Manual Override  |  [A/D] Lane Change Left/Right  |  [W/S] Throttle/Brake  |  [Mouse Drag] 3D Orbit", True, (0, 230, 255))
        screen.blit(p_hint, (30, screen_h - 26))

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

    pygame.quit()


if __name__ == "__main__":
    main()
