"""
tests/test_cinematic_systems.py — Automated Unit Tests for Cinematic Upgrades
"""

import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from lidar_3d_pointcloud_engine import Lidar3DPerceptionEngine
from traffic_physics_simulator import HighwayTrafficEngine, ParticleEmitter, V2XPacket, TrajectoryPredictionFan
from bev_transformer_engine import ProbabilisticOccupancyGrid


def test_weather_fog_lidar_attenuation():
    engine = Lidar3DPerceptionEngine(num_lasers=64, max_range_m=65.0)

    # Test Clear
    engine.set_weather_mode("CLEAR")
    dynamic_objs = [(0.0, 30.0, 1.85, 4.7, "LEAD", (200, 0, 0))]
    pts_clear = engine.generate_scene_point_cloud(dynamic_objs)
    max_range_clear = np.max(pts_clear[:, 4])
    assert max_range_clear > 40.0, "Clear weather should detect beyond 40m"

    # Test Fog
    engine.set_weather_mode("FOG")
    pts_fog = engine.generate_scene_point_cloud(dynamic_objs)
    max_range_fog = np.max(pts_fog[:, 4])
    assert max_range_fog <= 30.0, "Fog weather must attenuate max LiDAR range below 30m"


def test_radar_77ghz_doppler_detection():
    engine = Lidar3DPerceptionEngine()
    dynamic_objs = [(0.0, 25.0, 1.85, 4.7, "LEAD_CAR_1", (200, 0, 0))]
    dets = engine.radar_sim.scan_targets(dynamic_objs, ego_speed_mps=24.0)

    assert len(dets) >= 1, "77GHz Radar must detect lead vehicle within 120 deg FOV"
    assert abs(dets[0].azimuth_deg) < 15.0, "Azimuth angle for lead car should be close to 0 deg"


def test_v2x_bsm_packet_generation():
    engine = HighwayTrafficEngine()
    engine.step(dt=0.1)
    v2x_pkt = engine.get_lead_v2x_packet()

    assert v2x_pkt is not None, "V2X packet should be broadcasted by lead vehicle"
    assert v2x_pkt.speed_kmh > 0.0, "V2X speed must be positive"


def test_particle_emitter_physics():
    emitter = ParticleEmitter()
    emitter.emit_tire_smoke(0.0, 0.0)
    emitter.emit_guardrail_sparks(5.5, 0.0)
    emitter.emit_exhaust(0.0, 0.2, 0.0, 20.0)

    assert len(emitter.particles) > 0, "Emitter must generate particles"
    emitter.update(dt=0.1)
    assert len(emitter.particles) > 0, "Particles should survive initial dt step"


def test_probabilistic_occupancy_grid():
    grid = ProbabilisticOccupancyGrid(grid_w_cells=110, grid_h_cells=130)
    dummy_pts = np.array([
        [0.0, 0.8, 15.0, 0.9, 15.0], # Obstacle point
        [0.0, 0.02, 10.0, 0.3, 10.0]  # Ground point
    ], dtype=np.float32)

    grid.update_with_points(dummy_pts, [])
    heatmap = grid.generate_heatmap_rgb(out_w=440, out_h=480)
    assert heatmap.shape == (480, 440, 3), "Heatmap must be 480x440 RGB"
