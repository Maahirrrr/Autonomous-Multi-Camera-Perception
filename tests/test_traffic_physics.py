"""
tests/test_traffic_physics.py — Automated Unit Tests for IDM, MOBIL & Autonomous Overtaking
"""

import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from traffic_physics_simulator import HighwayTrafficEngine, TrafficVehicle, EgoAutonomousVehicle


def test_idm_car_following_deceleration():
    follower = TrafficVehicle("FOLLOWER", lane_idx=0, z_pos=0.0, speed_kmh=80.0)
    lead = TrafficVehicle("LEAD", lane_idx=0, z_pos=12.0, speed_kmh=50.0)

    # Follower should decelerate to avoid collision
    follower.update_physics(dt=0.1, lead_vehicle=lead)
    assert follower.is_braking or follower.speed_kmh < 80.0, "Follower must decelerate when approaching slower lead car"


def test_quintic_polynomial_lane_change():
    vehicle = TrafficVehicle("TEST_VEH", lane_idx=0, z_pos=0.0, speed_kmh=80.0)
    vehicle.initiate_lane_change(-1) # Move to left lane (-3.75m)

    assert vehicle.blinker == "LEFT"
    assert vehicle.lane_change_progress == 0.0

    # Step simulation across full duration
    for _ in range(30):
        vehicle.update_physics(dt=0.1)

    assert abs(vehicle.x - (-3.75)) < 0.05, f"Vehicle should reach target lane X=-3.75m (got {vehicle.x:.2f}m)"
    assert vehicle.lane_change_progress >= 1.0


def test_ego_autonomous_overtake_trigger():
    engine = HighwayTrafficEngine()
    engine.ego.state = "LANE_KEEP"
    engine.ego.speed_kmh = 85.0

    # Place slower lead car in front of Ego in Lane 0
    engine.traffic_vehicles = [
        TrafficVehicle("SLOW_LEAD", lane_idx=0, z_pos=18.0, speed_kmh=55.0)
    ]

    # Step engine
    engine.step(dt=0.1)

    assert engine.ego.state in ("CHECK_OVERTAKE", "LANE_CHANGE_LEFT"), "Ego must trigger overtake state when lead car is slow"
