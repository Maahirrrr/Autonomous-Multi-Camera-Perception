"""
tests/test_level4_perception.py — Automated Unit Tests for Level 4 LiDAR + Camera Fusion Stack
"""

import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from lidar_3d_pointcloud_engine import Lidar3DPerceptionEngine
from bev_transformer_engine import MultiCameraBEVTransformer
from multi_cam_simulator import MultiCameraSimulator


def test_lidar_point_cloud_generation():
    engine = Lidar3DPerceptionEngine(num_lasers=64)
    dynamic_objects = [
        (0.0, 25.0, 1.85, 4.7, "LEAD CAR", (40, 60, 220)),
        (-3.75, 12.0, 2.05, 5.0, "LEFT SUV", (180, 120, 40)),
    ]
    pc = engine.generate_scene_point_cloud(dynamic_objects, road_geometry={}, frame_idx=0)
    assert len(pc) > 1000, "Point cloud should contain dense laser returns (>1000 pts)"
    assert pc.shape[1] == 5, "Point cloud must have 5 channels (x, y, z, intensity, range)"


def test_ground_ransac_and_clustering():
    engine = Lidar3DPerceptionEngine(num_lasers=64)
    dynamic_objects = [
        (0.0, 22.0, 1.85, 4.7, "LEAD CAR", (40, 60, 220)),
    ]
    pc = engine.generate_scene_point_cloud(dynamic_objects, road_geometry={}, frame_idx=0)
    ground_pts, obstacle_pts, boxes = engine.segment_ground_and_clusters(pc)

    assert len(ground_pts) > 0, "Ground points must be extracted"
    assert len(obstacle_pts) > 0, "Obstacle points must be extracted"
    assert len(boxes) >= 1, "At least one 3D bounding box cluster must be fitted"


def test_point_to_pixel_camera_projection():
    bev_engine = MultiCameraBEVTransformer()
    lidar_engine = Lidar3DPerceptionEngine(num_lasers=64)
    lidar_engine.cameras = bev_engine.cameras

    # Point directly in front of FRONT camera (X=0, Y=1.45, Z=15.0)
    test_pts = np.array([
        [0.0, 1.45, 15.0, 0.9, 15.0],
        [-3.75, 1.45, 12.0, 0.9, 12.0],
    ], dtype=np.float32)

    cam_spec = bev_engine.cameras["FRONT"]
    proj_pts = lidar_engine.project_lidar_to_camera_image(test_pts, cam_spec, img_w=300, img_h=170)

    assert len(proj_pts) >= 1, "Points in camera FOV must successfully project to pixel coordinates"
    u, v, rgb = proj_pts[0]
    assert 0 <= u < 300 and 0 <= v < 170, "Projected pixel coordinates must be within image bounds"


def test_bev_fusion_map_generation():
    bev_engine = MultiCameraBEVTransformer(bev_width_px=440, bev_height_px=390)
    cam_sim = MultiCameraSimulator(width=300, height=170)
    lidar_engine = Lidar3DPerceptionEngine(num_lasers=64)

    dynamic_objects = [(0.0, 20.0, 1.85, 4.7, "LEAD CAR", (40, 60, 220))]
    pc = lidar_engine.generate_scene_point_cloud(dynamic_objects, {}, 0)
    _, _, boxes = lidar_engine.segment_ground_and_clusters(pc)
    cam_frames = cam_sim.render_surround_views(0, dynamic_objects)

    bev_map = bev_engine.generate_surround_bev_map(cam_frames, pc, boxes, render_lidar=True)
    assert bev_map.shape == (390, 440, 3), "BEV map shape must match target dimensions"
