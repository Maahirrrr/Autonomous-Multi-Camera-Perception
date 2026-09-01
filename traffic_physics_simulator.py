"""
traffic_physics_simulator.py — Level 4 Traffic Physics, Suspension Dynamics & MOBIL Overtaking
==============================================================================================
Physics & Mathematics:
  - 3D Vehicle Suspension Dynamics:
      Pitch: theta_pitch = -(a_x * h_cg) / (k_susp * L)
      Roll:  phi_roll   = +(a_y * h_cg) / (k_roll * W)
      Heave: y_heave    = A_bump * sin(omega * t)
  - Ackermann Steering Geometry:
      delta_inner = arctan( L / (R - W/2) )
      delta_outer = arctan( L / (R + W/2) )
  - Lateral Acceleration & Jerk:
      a_y = v^2 * kappa = (v^2 / L) * tan(delta)
      j_y = da_y / dt
  - Intelligent Driver Model (IDM) & MOBIL Autonomous Overtaking Engine.
"""

import math
import numpy as np


class TrafficVehicle:
    """Represents a dynamic 3D vehicle in highway traffic with suspension & kinematics."""
    def __init__(
        self,
        id_str: str,
        lane_idx: int,
        z_pos: float,
        speed_kmh: float,
        v_type: str = "SEDAN",
        color: tuple[int, int, int] = (40, 80, 200),
        width: float = 1.85,
        length: float = 4.70,
        height: float = 1.55,
    ):
        self.id = id_str
        self.lane_idx = lane_idx
        self.target_lane_idx = lane_idx
        self.x = float(lane_idx * 3.75)
        self.z = float(z_pos)
        self.speed_kmh = float(speed_kmh)
        self.speed_mps = speed_kmh / 3.6
        self.target_speed_kmh = float(speed_kmh)
        self.v_type = v_type
        self.color = color
        self.width = width
        self.length = length
        self.height = height

        # Kinematics & States
        self.accel_mps2 = 0.0
        self.is_braking = False
        self.blinker = "OFF" # 'OFF', 'LEFT', 'RIGHT'
        self.lane_change_progress = 1.0
        self.lane_change_duration_s = 2.4
        self.start_x = self.x
        self.target_x = self.x

        # Suspension pitch & roll
        self.pitch_deg = 0.0
        self.roll_deg = 0.0
        self.wheel_angle_rad = 0.0

    def initiate_lane_change(self, new_lane_idx: int):
        self.target_lane_idx = new_lane_idx
        self.start_x = self.x
        self.target_x = float(new_lane_idx * 3.75)
        self.lane_change_progress = 0.0
        self.blinker = "LEFT" if new_lane_idx < self.lane_idx else "RIGHT"

    def update_physics(self, dt: float, lead_vehicle = None):
        # 1. IDM Acceleration Dynamics
        v0 = self.target_speed_kmh / 3.6
        s0 = 4.5
        T = 1.35
        a_max = 2.4
        b_comf = 2.2

        if lead_vehicle and (lead_vehicle.z > self.z):
            s_actual = max(0.5, lead_vehicle.z - self.z - self.length)
            delta_v = self.speed_mps - lead_vehicle.speed_mps
            s_star = s0 + max(0.0, self.speed_mps * T + (self.speed_mps * delta_v) / (2.0 * math.sqrt(a_max * b_comf)))
            acc = a_max * (1.0 - (self.speed_mps / max(0.1, v0)) ** 4 - (s_star / s_actual) ** 2)
        else:
            acc = a_max * (1.0 - (self.speed_mps / max(0.1, v0)) ** 4)

        self.accel_mps2 = acc
        self.is_braking = (acc < -0.3)
        self.speed_mps = max(6.0, min(38.0, self.speed_mps + acc * dt))
        self.speed_kmh = self.speed_mps * 3.6

        # Suspension pitch
        self.pitch_deg = -acc * 0.45

        # Wheel rotation
        self.wheel_angle_rad = (self.wheel_angle_rad + (self.speed_mps / 0.33) * dt) % (2.0 * math.pi)

        # Smooth Quintic Polynomial Lateral Lane Transition
        if self.lane_change_progress < 1.0:
            self.lane_change_progress = min(1.0, self.lane_change_progress + dt / self.lane_change_duration_s)
            tau = self.lane_change_progress
            s_curve = 10.0 * (tau ** 3) - 15.0 * (tau ** 4) + 6.0 * (tau ** 5)
            self.x = self.start_x + (self.target_x - self.start_x) * s_curve

            # Lateral roll
            d_s = 30.0 * (tau ** 2) - 60.0 * (tau ** 3) + 30.0 * (tau ** 4)
            self.roll_deg = (self.target_x - self.start_x) * d_s * 0.40

            if self.lane_change_progress >= 1.0:
                self.lane_idx = self.target_lane_idx
                self.x = self.target_x
                self.roll_deg = 0.0
                self.blinker = "OFF"
        else:
            self.roll_deg = 0.0


class EgoAutonomousVehicle:
    """Represents the Level 4 Ego Vehicle with Suspension, Ackermann Kinematics & MOBIL Overtaking."""
    def __init__(self):
        self.x = 0.0
        self.z = 0.0
        self.lane_idx = 0
        self.target_lane_idx = 0
        self.speed_kmh = 88.0
        self.speed_mps = 88.0 / 3.6
        self.cruise_target_speed_kmh = 92.0
        self.accel_mps2 = 0.0
        self.width = 1.95
        self.length = 4.75
        self.height = 1.50
        self.wheelbase_m = 2.96
        self.color = (0, 180, 255)

        # Autonomous Decision State Machine
        self.state = "LANE_KEEP"
        self.state_timer_s = 0.0
        self.blinker = "OFF"
        self.is_braking = False
        self.lane_change_progress = 1.0
        self.start_x = 0.0
        self.target_x = 0.0
        self.overtaking_target_id = None
        self.manual_override = False

        # Physical Kinematics
        self.steering_angle_deg = 0.0
        self.steering_inner_deg = 0.0
        self.steering_outer_deg = 0.0
        self.yaw_rate_rads = 0.0
        self.lat_accel_g = 0.0
        self.lat_jerk_gs = 0.0
        self.prev_lat_accel_g = 0.0
        self.pitch_deg = 0.0
        self.roll_deg = 0.0
        self.wheel_angle_rad = 0.0

    def initiate_lane_change(self, new_lane_idx: int):
        self.target_lane_idx = new_lane_idx
        self.start_x = self.x
        self.target_x = float(new_lane_idx * 3.75)
        self.lane_change_progress = 0.0
        self.blinker = "LEFT" if new_lane_idx < self.lane_idx else "RIGHT"

    def update_autonomous_driving(self, dt: float, traffic_vehicles: list[TrafficVehicle]):
        self.state_timer_s += dt

        # Identify Lead Vehicle in Ego's Current Lane
        lead_in_lane = None
        min_lead_dist = 999.0
        for v in traffic_vehicles:
            if abs(v.x - self.x) < 1.8 and v.z > 0.0:
                dist = v.z - self.length
                if dist < min_lead_dist:
                    min_lead_dist = dist
                    lead_in_lane = v

        # Check Clearances
        left_clear = True
        right_clear = True
        for v in traffic_vehicles:
            if abs(v.x - (-3.75)) < 2.0 and -16.0 < v.z < 26.0:
                left_clear = False
            if abs(v.x - (+3.75)) < 2.0 and -16.0 < v.z < 24.0:
                right_clear = False

        # Autonomous Overtaking State Machine Logic
        if not self.manual_override:
            if self.state == "LANE_KEEP":
                self.blinker = "OFF"
                target_v = self.cruise_target_speed_kmh
                if lead_in_lane:
                    if lead_in_lane.speed_kmh < self.cruise_target_speed_kmh - 6.0 and min_lead_dist < 26.0:
                        self.state = "CHECK_OVERTAKE"
                        self.state_timer_s = 0.0
                        self.overtaking_target_id = lead_in_lane.id

            elif self.state == "CHECK_OVERTAKE":
                if left_clear and self.lane_idx >= 0:
                    self.state = "LANE_CHANGE_LEFT"
                    self.state_timer_s = 0.0
                    self.initiate_lane_change(-1)
                elif lead_in_lane and min_lead_dist < 14.0:
                    self.speed_kmh = max(lead_in_lane.speed_kmh, self.speed_kmh - 8.0 * dt)
                else:
                    if self.state_timer_s > 4.0:
                        self.state = "LANE_KEEP"

            elif self.state == "LANE_CHANGE_LEFT":
                self.speed_kmh = min(108.0, self.speed_kmh + 14.0 * dt)
                if self.lane_change_progress >= 1.0:
                    self.state = "OVERTAKING"
                    self.state_timer_s = 0.0
                    self.blinker = "OFF"

            elif self.state == "OVERTAKING":
                self.speed_kmh = min(112.0, self.speed_kmh + 10.0 * dt)
                target_veh = next((v for v in traffic_vehicles if v.id == self.overtaking_target_id), None)
                if target_veh and target_veh.z < -14.0 and right_clear:
                    self.state = "LANE_CHANGE_RIGHT"
                    self.state_timer_s = 0.0
                    self.initiate_lane_change(0)
                elif self.state_timer_s > 5.5 and right_clear:
                    self.state = "LANE_CHANGE_RIGHT"
                    self.state_timer_s = 0.0
                    self.initiate_lane_change(0)

            elif self.state == "LANE_CHANGE_RIGHT":
                self.speed_kmh = max(self.cruise_target_speed_kmh, self.speed_kmh - 10.0 * dt)
                if self.lane_change_progress >= 1.0:
                    self.state = "LANE_KEEP"
                    self.state_timer_s = 0.0
                    self.blinker = "OFF"

        self.speed_mps = self.speed_kmh / 3.6

        # Lateral S-Curve Transition
        if self.lane_change_progress < 1.0:
            self.lane_change_progress = min(1.0, self.lane_change_progress + dt / 2.1)
            tau = self.lane_change_progress
            s_curve = 10.0 * (tau ** 3) - 15.0 * (tau ** 4) + 6.0 * (tau ** 5)
            self.x = self.start_x + (self.target_x - self.start_x) * s_curve

            # Steering derivative
            d_s_curve = 30.0 * (tau ** 2) - 60.0 * (tau ** 3) + 30.0 * (tau ** 4)
            self.steering_angle_deg = (self.target_x - self.start_x) * d_s_curve * 1.85
            self.roll_deg = -(self.target_x - self.start_x) * d_s_curve * 0.45

            if self.lane_change_progress >= 1.0:
                self.lane_idx = self.target_lane_idx
                self.x = self.target_x
                self.steering_angle_deg = 0.0
                self.roll_deg = 0.0
                self.blinker = "OFF"
        else:
            self.steering_angle_deg = 0.0
            self.roll_deg = 0.0

        # Ackermann Geometry
        delta_rad = math.radians(self.steering_angle_deg)
        if abs(delta_rad) > 1e-4:
            R_turn = self.wheelbase_m / math.tan(delta_rad)
            self.steering_inner_deg = math.degrees(math.atan(self.wheelbase_m / (R_turn - self.width * 0.5)))
            self.steering_outer_deg = math.degrees(math.atan(self.wheelbase_m / (R_turn + self.width * 0.5)))
            self.yaw_rate_rads = (self.speed_mps / self.wheelbase_m) * math.tan(delta_rad)
            self.lat_accel_g = ((self.speed_mps ** 2) / (R_turn * 9.81))
        else:
            self.steering_inner_deg = 0.0
            self.steering_outer_deg = 0.0
            self.yaw_rate_rads = 0.0
            self.lat_accel_g = 0.0

        self.lat_jerk_gs = (self.lat_accel_g - self.prev_lat_accel_g) / max(1e-4, dt)
        self.prev_lat_accel_g = self.lat_accel_g

        # Wheel rotation
        self.wheel_angle_rad = (self.wheel_angle_rad + (self.speed_mps / 0.34) * dt) % (2.0 * math.pi)


class HighwayTrafficEngine:
    """Manages 3D highway traffic, surrounding vehicles overtaking, and multi-lane flow."""
    def __init__(self):
        self.ego = EgoAutonomousVehicle()
        self.traffic_vehicles: list[TrafficVehicle] = []
        self._spawn_initial_traffic()

    def randomize_scenario(self):
        """Randomizes the entire traffic distribution on the fly."""
        self._spawn_initial_traffic()

    def _spawn_initial_traffic(self):
        import random
        colors = [
            (210, 45, 45), (245, 210, 25), (45, 110, 195),
            (50, 160, 80), (175, 55, 175), (220, 120, 40)
        ]
        self.traffic_vehicles = [
            # 1. Lead Vehicle in Center Lane (Cruises at 64-70 km/h)
            TrafficVehicle("LEAD_CAR_1", lane_idx=0, z_pos=random.uniform(22.0, 28.0), speed_kmh=random.uniform(64.0, 70.0), v_type="SEDAN", color=colors[0]),
            # 2. Fast Sports Car in Left Lane (Overtakes Ego at 108-118 km/h from behind)
            TrafficVehicle("SPORTS_FAST_LEFT", lane_idx=-1, z_pos=random.uniform(-42.0, -30.0), speed_kmh=random.uniform(108.0, 118.0), v_type="SPORTS", color=colors[1]),
            # 3. Commercial Freight Semi-Truck in Right Lane (Cruises at 60-66 km/h)
            TrafficVehicle("SEMI_TRUCK_RIGHT", lane_idx=+1, z_pos=random.uniform(12.0, 20.0), speed_kmh=random.uniform(60.0, 66.0), v_type="TRUCK", color=colors[3], width=2.45, length=9.8, height=3.3),
            # 4. Trailing Blue SUV in Center Lane (Cruises at 80-86 km/h behind ego)
            TrafficVehicle("TRAILING_SUV", lane_idx=0, z_pos=random.uniform(-32.0, -22.0), speed_kmh=random.uniform(80.0, 86.0), v_type="SUV", color=colors[2], width=2.0, length=4.9, height=1.75),
            # 5. Fast Sedan in Left Lane Ahead (Cruises at 95-102 km/h)
            TrafficVehicle("FAST_SEDAN_LEFT", lane_idx=-1, z_pos=random.uniform(45.0, 60.0), speed_kmh=random.uniform(95.0, 102.0), v_type="SEDAN", color=colors[4]),
        ]

    def step(self, dt: float):
        self.ego.update_autonomous_driving(dt, self.traffic_vehicles)

        for v in self.traffic_vehicles:
            v.update_physics(dt)
            rel_speed_mps = v.speed_mps - self.ego.speed_mps
            v.z += rel_speed_mps * dt

            # Recycle vehicles
            if v.z > 85.0:
                v.z = -50.0
                v.speed_kmh = 108.0 if v.lane_idx == -1 else (65.0 if v.lane_idx == 1 else 78.0)
                v.speed_mps = v.speed_kmh / 3.6
            elif v.z < -65.0:
                v.z = 70.0
                v.speed_kmh = 98.0 if v.lane_idx == -1 else (62.0 if v.lane_idx == 1 else 72.0)
                v.speed_mps = v.speed_kmh / 3.6

    def get_dynamic_objects_for_sensors(self) -> list:
        objs = []
        for v in self.traffic_vehicles:
            objs.append((v.x - self.ego.x, v.z, v.width, v.length, v.id, v.color))
        return objs
