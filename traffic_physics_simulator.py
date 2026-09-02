"""
traffic_physics_simulator.py — Real-Time Highway Traffic & Driving Dynamics Engine
===================================================================================
Physics Models & Algorithms:
  - Intelligent Driver Model (IDM) for Longitudinal Car-Following Dynamics.
  - MOBIL (Minimizing Overall Braking Induced by Lane changes) Lane-Switching.
  - Quintic Polynomial (5th-Order Spline) Smooth Trajectory Planner for Lane Changes.
  - Two-Track Ackermann Steering Kinematics.
  - Particle Physics Emitter: Tire Smoke, Sparks & Exhaust Plumes.
  - V2X (Vehicle-to-Everything) BSM Telemetry Broadcasting.
  - Multi-Hypothesis Trajectory Prediction Fans (H0, H1, H2).
  - 5-Second Rolling G-Force History Buffer.
"""

import math
import random
from dataclasses import dataclass, field
import numpy as np


@dataclass
class Particle:
    x: float
    y: float
    z: float
    vx: float
    vy: float
    vz: float
    size: float
    alpha: float
    color: tuple
    life: float
    max_life: float
    p_type: str # 'SMOKE', 'SPARK', 'EXHAUST'


class ParticleEmitter:
    """Physics-based particle emitter for tire smoke, guardrail sparks, and exhaust plumes."""
    def __init__(self):
        self.particles: list[Particle] = []

    def emit_tire_smoke(self, x: float, z: float, count: int = 4):
        for _ in range(count):
            self.particles.append(Particle(
                x=x + random.uniform(-0.4, 0.4),
                y=0.10,
                z=z + random.uniform(-0.3, 0.3),
                vx=random.uniform(-0.6, 0.6),
                vy=random.uniform(0.3, 0.8),
                vz=random.uniform(-1.5, -0.5),
                size=random.uniform(3.0, 7.0),
                alpha=0.85,
                color=(220, 225, 235),
                life=0.0,
                max_life=random.uniform(0.6, 1.2),
                p_type="SMOKE"
            ))

    def emit_guardrail_sparks(self, x: float, z: float, count: int = 6):
        for _ in range(count):
            self.particles.append(Particle(
                x=x + random.uniform(-0.1, 0.1),
                y=random.uniform(0.2, 0.6),
                z=z + random.uniform(-0.2, 0.2),
                vx=random.uniform(-1.5, 1.5),
                vy=random.uniform(1.0, 3.5),
                vz=random.uniform(-3.0, -1.0),
                size=random.uniform(1.5, 3.5),
                alpha=1.0,
                color=(255, random.randint(180, 240), 20),
                life=0.0,
                max_life=random.uniform(0.25, 0.55),
                p_type="SPARK"
            ))

    def emit_exhaust(self, x: float, y: float, z: float, speed_kmh: float):
        if random.random() < 0.35:
            self.particles.append(Particle(
                x=x + random.uniform(-0.25, 0.25),
                y=y,
                z=z - 0.2,
                vx=random.uniform(-0.15, 0.15),
                vy=random.uniform(0.1, 0.3),
                vz=-speed_kmh * 0.05,
                size=random.uniform(2.0, 4.0),
                alpha=0.45,
                color=(160, 170, 180),
                life=0.0,
                max_life=random.uniform(0.4, 0.8),
                p_type="EXHAUST"
            ))

    def update(self, dt: float):
        alive = []
        for p in self.particles:
            p.life += dt
            if p.life < p.max_life:
                p.x += p.vx * dt
                p.y += p.vy * dt
                p.z += p.vz * dt
                if p.p_type == "SPARK":
                    p.vy -= 9.81 * dt # Gravity on sparks
                elif p.p_type == "SMOKE":
                    p.size += 4.0 * dt
                p.alpha = max(0.0, 1.0 - (p.life / p.max_life))
                alive.append(p)
        self.particles = alive


@dataclass
class V2XPacket:
    """SAE J2735 Basic Safety Message (BSM) Data Structure."""
    sender_id: str
    speed_kmh: float
    brake_pct: float
    throttle_pct: float
    turn_signal: str # 'OFF', 'LEFT', 'RIGHT'
    gps_lat: float
    gps_lon: float
    timestamp_ms: float
    rssi_dbm: float


@dataclass
class TrajectoryPredictionFan:
    """Multi-hypothesis 3-second trajectory prediction fan."""
    h0_straight: list[tuple[float, float]] = field(default_factory=list)
    h1_left: list[tuple[float, float]] = field(default_factory=list)
    h2_right: list[tuple[float, float]] = field(default_factory=list)
    prob_h0: float = 0.70
    prob_h1: float = 0.15
    prob_h2: float = 0.15


class TrafficVehicle:
    """Surrounding dynamic traffic participant with IDM car-following and MOBIL lane changing."""

    def __init__(
        self,
        veh_id: str,
        lane_idx: int = 0, # -1: Fast/Left, 0: Cruising/Center, 1: Slow/Right
        z_pos: float = 0.0,
        speed_kmh: float = 75.0,
        target_speed_kmh: float = 80.0,
        color: tuple[int, int, int] = (215, 35, 35),
        width: float = 1.90,
        length: float = 4.75,
        height: float = 1.55,
        model_type: str = "SEDAN"
    ):
        self.id = veh_id
        self.lane_idx = lane_idx
        self.target_lane_idx = lane_idx
        self.x = float(lane_idx * 3.75)
        self.z = z_pos
        self.speed_kmh = speed_kmh
        self.target_speed_kmh = target_speed_kmh
        self.color = color
        self.width = width
        self.length = length
        self.height = height
        self.model_type = model_type

        # IDM Model Parameters
        self.v0 = target_speed_kmh / 3.6
        self.T = 1.4 # Safe time headway (s)
        self.a_max = 2.2 # Max comfortable acceleration (m/s^2)
        self.b_comf = 2.8 # Max comfortable braking (m/s^2)
        self.s0 = 3.5 # Minimum standstill gap (m)
        self.delta_idm = 4.0

        # Dynamics & Suspension Tilt
        self.accel = 0.0
        self.is_braking = False
        self.blinker = "OFF" # 'OFF', 'LEFT', 'RIGHT'
        self.pitch_deg = 0.0
        self.roll_deg = 0.0

        # Lane Change Spline (Quintic Polynomial)
        self.is_changing_lane = False
        self.lc_progress = 0.0
        self.lc_duration = 3.0
        self.lc_start_x = self.x
        self.lc_target_x = self.x

        # V2X Telemetry Packet
        self.v2x_packet = None

        # Trajectory Prediction Fan
        self.prediction_fan = []

    @property
    def speed_mps(self) -> float:
        return self.speed_kmh / 3.6

    def compute_idm_accel(self, leader_dist: float, leader_speed_mps: float) -> float:
        v = max(0.1, self.speed_mps)
        if leader_dist is not None and leader_dist > 0.0:
            s = max(0.5, leader_dist - self.length)
            delta_v = v - leader_speed_mps
            s_star = self.s0 + max(0.0, v * self.T + (v * delta_v) / (2.0 * math.sqrt(self.a_max * self.b_comf)))
            accel = self.a_max * (1.0 - (v / self.v0) ** self.delta_idm - (s_star / s) ** 2)
        else:
            accel = self.a_max * (1.0 - (v / self.v0) ** self.delta_idm)
        return float(np.clip(accel, -7.0, self.a_max))

    def update_prediction_fan(self):
        """Calculates 3-second multi-hypothesis trajectory prediction fan."""
        dt_fan = 0.35
        steps = 8
        cur_v = max(5.0, self.speed_mps)

        # H0: Nominal lane-keep
        pts_h0 = []
        for step in range(1, steps + 1):
            t_s = step * dt_fan
            pts_h0.append((self.x, self.z + cur_v * t_s))

        # H1: Left lane change hypothesis
        pts_h1 = []
        target_left_x = self.x - 3.75
        for step in range(1, steps + 1):
            t_s = step * dt_fan
            ratio = min(1.0, t_s / 2.5)
            x_step = self.x + (target_left_x - self.x) * (10.0 * ratio**3 - 15.0 * ratio**4 + 6.0 * ratio**5)
            pts_h1.append((x_step, self.z + cur_v * t_s))

        # H2: Right lane change hypothesis
        pts_h2 = []
        target_right_x = self.x + 3.75
        for step in range(1, steps + 1):
            t_s = step * dt_fan
            ratio = min(1.0, t_s / 2.5)
            x_step = self.x + (target_right_x - self.x) * (10.0 * ratio**3 - 15.0 * ratio**4 + 6.0 * ratio**5)
            pts_h2.append((x_step, self.z + cur_v * t_s))

        # Probabilistic weights based on blinker intent
        if self.blinker == "LEFT":
            p0, p1, p2 = 0.20, 0.70, 0.10
        elif self.blinker == "RIGHT":
            p0, p1, p2 = 0.20, 0.10, 0.70
        else:
            p0, p1, p2 = 0.70, 0.15, 0.15

        self.prediction_fan = [
            {"label": "H0_KEEP", "prob": p0, "points": pts_h0, "color": (0, 255, 180)},
            {"label": "H1_LEFT", "prob": p1, "points": pts_h1, "color": (0, 200, 255)},
            {"label": "H2_RIGHT", "prob": p2, "points": pts_h2, "color": (255, 210, 0)},
        ]

    def update_v2x_telemetry(self, timestamp_ms: float = 0.0, time_ms: float = 0.0):
        """Simulates SAE J2735 BSM packet broadcasting."""
        ts = timestamp_ms if timestamp_ms != 0.0 else time_ms
        brake_val = 80.0 if self.is_braking else 0.0
        throttle_val = max(0.0, min(100.0, (self.accel / self.a_max) * 100.0)) if not self.is_braking else 0.0

        # Approx GPS based on highway position
        gps_lat = 37.7749 + (self.z * 0.000009)
        gps_lon = -122.4194 + (self.x * 0.000009)
        dist_to_ego = math.hypot(self.x, self.z)
        rssi = -45.0 - (dist_to_ego * 0.6)

        self.v2x_packet = V2XPacket(
            sender_id=self.id,
            speed_kmh=self.speed_kmh,
            brake_pct=brake_val,
            throttle_pct=throttle_val,
            turn_signal=self.blinker,
            gps_lat=gps_lat,
            gps_lon=gps_lon,
            timestamp_ms=ts,
            rssi_dbm=rssi
        )

    @property
    def lane_change_progress(self) -> float:
        return self.lc_progress

    @lane_change_progress.setter
    def lane_change_progress(self, val: float):
        self.lc_progress = val

    def initiate_lane_change(self, target_lane: int):
        target_lane = int(np.clip(target_lane, -1, 1))
        if target_lane == self.lane_idx:
            return
        self.target_lane_idx = target_lane
        self.lc_start_x = self.x
        self.lc_target_x = float(target_lane * 3.75)
        self.lc_progress = 0.0
        self.is_changing_lane = True
        self.blinker = "LEFT" if target_lane < self.lane_idx else "RIGHT"

    def update_physics(self, dt: float, lead_vehicle: "TrafficVehicle" = None):
        leader_dist = (lead_vehicle.z - self.z) if lead_vehicle else None
        leader_spd = lead_vehicle.speed_mps if lead_vehicle else None
        self.step(dt, leader_dist, leader_spd)

    def step(self, dt: float, leader_dist: float = None, leader_speed_mps: float = None):
        # 1. Update Longitudinal Velocity & Acceleration
        self.accel = self.compute_idm_accel(leader_dist, leader_speed_mps)
        new_speed_mps = max(5.0, self.speed_mps + self.accel * dt)
        self.speed_kmh = new_speed_mps * 3.6
        self.is_braking = self.accel < -1.2

        # 2. Update Suspension Dynamics (Pitch & Roll)
        target_pitch = np.clip(self.accel * -0.9, -3.5, 3.5)
        self.pitch_deg += (target_pitch - self.pitch_deg) * 6.0 * dt

        # 3. Step Smooth Quintic Polynomial Lane Change
        if self.is_changing_lane:
            self.lc_progress += dt / self.lc_duration
            if self.lc_progress >= 0.9999:
                self.lc_progress = 1.0
            t = min(1.0, self.lc_progress)
            poly_val = 10.0 * (t ** 3) - 15.0 * (t ** 4) + 6.0 * (t ** 5)
            self.x = self.lc_start_x + (self.lc_target_x - self.lc_start_x) * poly_val

            lat_vel = ((self.lc_target_x - self.lc_start_x) / self.lc_duration) * (30.0 * (t**2) - 60.0 * (t**3) + 30.0 * (t**4))
            self.roll_deg = float(np.clip(-lat_vel * 1.8, -4.5, 4.5))

            if t >= 1.0:
                self.x = self.lc_target_x
                self.lane_idx = self.target_lane_idx
                self.is_changing_lane = False
                self.blinker = "OFF"
                self.roll_deg = 0.0
        else:
            self.roll_deg *= 0.85

        # 4. Update Prediction Fans & V2X Telemetry
        self.update_prediction_fan()
        self.update_v2x_telemetry(time_ms=0.0)


class EgoAutonomousVehicle:
    """Ego Tesla Level 4 Autonomous Highway Pilot with Quintic Polynomials & Ackermann Steering."""

    def __init__(self):
        self.lane_idx = 0 # Center Cruising Lane
        self.target_lane_idx = 0
        self.x = 0.0
        self.z = 0.0
        self.speed_kmh = 88.0
        self.target_cruise_speed_kmh = 105.0
        self.overtake_cruise_speed_kmh = 115.0

        # Geometry & Kinematics (Tesla Model S)
        self.width = 1.96
        self.length = 4.97
        self.height = 1.45
        self.wheelbase_m = 2.96
        self.track_width_m = 1.66
        self.mass_kg = 2100.0

        # State Machine: 'LANE_KEEP', 'CHECK_OVERTAKE', 'LANE_CHANGE_LEFT', 'OVERTAKING', 'LANE_CHANGE_RIGHT'
        self.state = "LANE_KEEP"
        self.manual_override = False

        # Ackermann Steering Angles
        self.steering_angle_deg = 0.0
        self.steering_inner_deg = 0.0
        self.steering_outer_deg = 0.0
        self.yaw_rate_rads = 0.0
        self.lat_accel_g = 0.0
        self.lat_jerk_gs = 0.0
        self.last_lat_accel_g = 0.0

        # 5-Second Rolling G-Force History Buffer: (lat_g, long_g)
        self.g_history: list[tuple[float, float]] = [(0.0, 0.0)] * 150

        # Suspension & Blinkers
        self.pitch_deg = 0.0
        self.roll_deg = 0.0
        self.blinker = "OFF"
        self.is_braking = False
        self.accel_mps2 = 0.0

        # Quintic Trajectory Parameters
        self.lc_progress = 0.0
        self.lc_duration = 3.2
        self.lc_start_x = 0.0
        self.lc_target_x = 0.0

        # Overtaking Memory
        self.overtake_target_id = None
        self.overtake_start_z = 0.0

    @property
    def speed_mps(self) -> float:
        return self.speed_kmh / 3.6

    def compute_ackermann_angles(self, delta_rad: float):
        if abs(delta_rad) < 1e-4:
            self.steering_inner_deg = 0.0
            self.steering_outer_deg = 0.0
            return

        R = self.wheelbase_m / math.tan(abs(delta_rad))
        w = self.track_width_m
        delta_i = math.atan(self.wheelbase_m / (R - w * 0.5))
        delta_o = math.atan(self.wheelbase_m / (R + w * 0.5))

        sign = 1.0 if delta_rad > 0 else -1.0
        self.steering_inner_deg = math.degrees(delta_i) * sign
        self.steering_outer_deg = math.degrees(delta_o) * sign

    def initiate_lane_change(self, target_lane: int):
        target_lane = int(np.clip(target_lane, -1, 1))
        if target_lane == self.lane_idx:
            return

        self.target_lane_idx = target_lane
        self.lc_start_x = self.x
        self.lc_target_x = float(target_lane * 3.75)
        self.lc_progress = 0.0
        self.is_changing_lane = True
        self.blinker = "LEFT" if target_lane < self.lane_idx else "RIGHT"

        if target_lane < self.lane_idx:
            self.state = "LANE_CHANGE_LEFT"
        else:
            self.state = "LANE_CHANGE_RIGHT"

    def step(self, dt: float):
        # 1. Update 5th-Order Quintic Polynomial Lane Change Trajectory
        if self.state in ("LANE_CHANGE_LEFT", "LANE_CHANGE_RIGHT"):
            self.lc_progress += dt / self.lc_duration
            t = min(1.0, self.lc_progress)

            # Quintic polynomial: s(t) = 10t^3 - 15t^4 + 6t^5
            s = 10.0 * (t ** 3) - 15.0 * (t ** 4) + 6.0 * (t ** 5)
            s_dot = (30.0 * (t ** 2) - 60.0 * (t ** 3) + 30.0 * (t ** 4)) / self.lc_duration
            s_ddot = (60.0 * t - 180.0 * (t ** 2) + 120.0 * (t ** 3)) / (self.lc_duration ** 2)

            dx = self.lc_target_x - self.lc_start_x
            self.x = self.lc_start_x + dx * s
            lat_vel = dx * s_dot
            lat_accel = dx * s_ddot

            # Calculate Steering Angle from Curvature
            v = max(5.0, self.speed_mps)
            curvature = lat_accel / (v ** 2)
            delta_rad = math.atan(self.wheelbase_m * curvature)
            self.steering_angle_deg = float(np.clip(math.degrees(delta_rad) * 4.5, -35.0, 35.0))
            self.compute_ackermann_angles(delta_rad)

            self.lat_accel_g = lat_accel / 9.81
            self.lat_jerk_gs = (self.lat_accel_g - self.last_lat_accel_g) / max(0.001, dt)
            self.last_lat_accel_g = self.lat_accel_g

            self.roll_deg = float(np.clip(-self.lat_accel_g * 4.5, -4.5, 4.5))

            if t >= 1.0:
                self.x = self.lc_target_x
                self.lane_idx = self.target_lane_idx
                self.steering_angle_deg = 0.0
                self.compute_ackermann_angles(0.0)
                self.roll_deg = 0.0
                self.blinker = "OFF"

                if self.state == "LANE_CHANGE_LEFT":
                    self.state = "OVERTAKING"
                elif self.state == "LANE_CHANGE_RIGHT":
                    self.state = "LANE_KEEP"
                    self.overtake_target_id = None
        else:
            self.steering_angle_deg *= 0.85
            self.compute_ackermann_angles(math.radians(self.steering_angle_deg))
            self.roll_deg *= 0.85
            self.lat_accel_g *= 0.85
            self.lat_jerk_gs = 0.0

        # Update 5-second Rolling G-Force Buffer
        long_g = self.accel_mps2 / 9.81
        self.g_history.append((self.lat_accel_g, long_g))
        if len(self.g_history) > 150:
            self.g_history.pop(0)


class HighwayTrafficEngine:
    """Master Traffic Engine orchestrating IDM, MOBIL autonomous overtaking, and particle generation."""

    def __init__(self):
        self.ego = EgoAutonomousVehicle()
        self.traffic_vehicles: list[TrafficVehicle] = []
        self.particle_emitter = ParticleEmitter()
        self.event_log: list[str] = []
        self.sim_time_s = 0.0

        self.randomize_scenario()
        self.log_event("360 LIDAR + 4-CAM SENSOR FUSION ACTIVE")
        self.log_event("HIGHWAY PILOT ENGAGED — CRUISING AT 88 KM/H")

    def log_event(self, msg: str):
        timestamp = f"[T+{self.sim_time_s:.1f}s]"
        self.event_log.append(f"{timestamp} {msg}")
        if len(self.event_log) > 16:
            self.event_log.pop(0)

    def randomize_scenario(self):
        """Spawns realistic traffic participants with authentic automotive paint colors."""
        self.traffic_vehicles.clear()

        # Authentic Automotive Paint Palettes (RGB)
        # Lead Car: Tesla Crimson Red (215, 35, 35)
        lead_car = TrafficVehicle(
            veh_id="LEAD_CAR_1",
            lane_idx=0,
            z_pos=22.0,
            speed_kmh=68.0,
            target_speed_kmh=70.0,
            color=(215, 35, 35),
            model_type="SEDAN"
        )

        # Fast Sedan: Pearl White (240, 242, 245)
        fast_sedan = TrafficVehicle(
            veh_id="FAST_SEDAN_LEFT",
            lane_idx=-1,
            z_pos=38.0,
            speed_kmh=102.0,
            target_speed_kmh=105.0,
            color=(240, 242, 245),
            model_type="SEDAN"
        )

        # Semi Truck: Midnight Forest Emerald (30, 140, 75)
        semi_truck = TrafficVehicle(
            veh_id="SEMI_TRUCK_RIGHT",
            lane_idx=1,
            z_pos=16.0,
            speed_kmh=62.0,
            target_speed_kmh=65.0,
            color=(30, 140, 75),
            width=2.55,
            length=11.2,
            height=3.4,
            model_type="TRUCK"
        )

        # Trailing Sports Car: Sunset Metallic Amber (245, 175, 25)
        rear_sports = TrafficVehicle(
            veh_id="SPORTS_REAR_LEFT",
            lane_idx=-1,
            z_pos=-24.0,
            speed_kmh=98.0,
            target_speed_kmh=100.0,
            color=(245, 175, 25),
            model_type="SPORTS"
        )

        self.traffic_vehicles.extend([lead_car, fast_sedan, semi_truck, rear_sports])
        self.log_event("SCENARIO INITIALIZED: 4 DYNAMIC VEHICLES SPAWNED")

    def step(self, dt: float):
        self.sim_time_s += dt

        # 1. Step Ego Vehicle Decision Logic (IDM + MOBIL Autonomous Overtaking)
        self._step_ego_l4_autopilot(dt)

        # 2. Step Surrounding Traffic Vehicles
        for v in self.traffic_vehicles:
            # Relative longitudinal movement with respect to Ego
            rel_v_mps = v.speed_mps - self.ego.speed_mps
            v.z += rel_v_mps * dt

            # Wrap-around bounds for continuous dynamic traffic
            if v.z > 85.0:
                v.z = -35.0
                v.speed_kmh = random.uniform(85.0, 105.0)
            elif v.z < -45.0:
                v.z = 65.0
                v.speed_kmh = random.uniform(65.0, 75.0)

            v.step(dt)

            # Emit exhaust particles
            self.particle_emitter.emit_exhaust(v.x, 0.25, v.z - v.length * 0.5, v.speed_kmh)

        # 3. Particle Emitters for Ego Hard Braking or Guardrail Proximity
        if self.ego.accel_mps2 < -2.0 or self.ego.is_braking:
            self.particle_emitter.emit_tire_smoke(self.ego.x - 0.7, self.ego.z - 2.0, count=2)
            self.particle_emitter.emit_tire_smoke(self.ego.x + 0.7, self.ego.z - 2.0, count=2)

        if abs(self.ego.x) > 5.2: # Near guardrail
            self.particle_emitter.emit_guardrail_sparks(self.ego.x, self.ego.z, count=4)

        self.particle_emitter.update(dt)

    def _step_ego_l4_autopilot(self, dt: float):
        if self.ego.manual_override:
            self.ego.step(dt)
            return

        # Find closest lead vehicle in same lane
        same_lane_leads = [
            v for v in self.traffic_vehicles
            if abs(v.x - self.ego.x) < 2.0 and v.z > 0.0
        ]
        same_lane_leads.sort(key=lambda v: v.z)
        lead_car = same_lane_leads[0] if same_lane_leads else None

        # Level 4 Overtake State Machine
        if self.ego.state == "LANE_KEEP":
            if lead_car and lead_car.z < 26.0 and lead_car.speed_kmh < self.ego.target_cruise_speed_kmh - 8.0:
                self.ego.state = "CHECK_OVERTAKE"
                self.ego.overtake_target_id = lead_car.id
                self.log_event(f"OVERTAKE INITIATED — TARGET: {lead_car.id}")
            else:
                # Cruise with IDM following lead vehicle
                if lead_car:
                    dist = lead_car.z
                    rel_speed = lead_car.speed_mps
                else:
                    dist = None
                    rel_speed = None

                accel = self._compute_idm_accel(self.ego.speed_mps, self.ego.target_cruise_speed_kmh / 3.6, dist, rel_speed)
                self.ego.accel_mps2 = accel
                self.ego.speed_kmh = max(20.0, min(120.0, (self.ego.speed_mps + accel * dt) * 3.6))

        elif self.ego.state == "CHECK_OVERTAKE":
            # Check safety in Left Fast Lane (Lane -1)
            left_lane_traffic = [v for v in self.traffic_vehicles if abs(v.x - (-3.75)) < 1.8]
            is_left_safe = True
            for v in left_lane_traffic:
                # If vehicle in left lane is within [-15m, +25m], lane change is unsafe
                if -15.0 < v.z < 25.0:
                    is_left_safe = False
                    break

            if is_left_safe and self.ego.lane_idx > -1:
                self.ego.initiate_lane_change(self.ego.lane_idx - 1)
                self.log_event("MANEUVERING TO FAST LANE (-1) WITH LEFT BLINKER")
            else:
                # Fallback to following lead car safely
                accel = self._compute_idm_accel(self.ego.speed_mps, 70.0 / 3.6, lead_car.z if lead_car else 30.0, lead_car.speed_mps if lead_car else 19.0)
                self.ego.accel_mps2 = accel
                self.ego.speed_kmh = max(20.0, (self.ego.speed_mps + accel * dt) * 3.6)

        elif self.ego.state == "OVERTAKING":
            # Accelerate past the overtaken target in the fast lane
            target_veh = next((v for v in self.traffic_vehicles if v.id == self.ego.overtake_target_id), None)
            self.ego.speed_kmh = min(self.ego.overtake_cruise_speed_kmh, self.ego.speed_kmh + 18.0 * dt)

            # Once target is > 14m behind ego, merge back to center cruising lane (Lane 0)
            if target_veh and target_veh.z < -14.0:
                # Check safety in center lane
                center_traffic = [v for v in self.traffic_vehicles if abs(v.x - 0.0) < 1.8]
                is_center_safe = all(not (-10.0 < v.z < 18.0) for v in center_traffic)
                if is_center_safe:
                    self.ego.initiate_lane_change(0)
                    self.log_event("MERGING SAFELY BACK TO CENTER CRUISING LANE")

        self.ego.step(dt)

    def _compute_idm_accel(self, v: float, v0: float, s: float = None, v_lead: float = None) -> float:
        a_max = 2.2
        b_comf = 2.8
        T = 1.4
        s0 = 4.0
        delta = 4.0

        if s is not None and s > 0.0:
            delta_v = v - (v_lead if v_lead is not None else v)
            s_star = s0 + max(0.0, v * T + (v * delta_v) / (2.0 * math.sqrt(a_max * b_comf)))
            accel = a_max * (1.0 - (v / v0) ** delta - (s_star / s) ** 2)
        else:
            accel = a_max * (1.0 - (v / v0) ** delta)
        return float(np.clip(accel, -6.0, a_max))

    def get_lead_v2x_packet(self) -> V2XPacket:
        lead_car = next((v for v in self.traffic_vehicles if abs(v.x - self.ego.x) < 2.0 and v.z > 0), None)
        if lead_car:
            return lead_car.v2x_packet
        return None

    def get_dynamic_objects_for_sensors(self) -> list:
        objs = []
        for v in self.traffic_vehicles:
            # Relative to Ego
            rel_x = v.x - self.ego.x
            rel_z = v.z
            objs.append((rel_x, rel_z, v.width, v.length, v.id, v.color))
        return objs
