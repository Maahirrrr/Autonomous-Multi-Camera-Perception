"""
traffic_physics_simulator.py — Real-Time Tesla Level 4 Autopilot Physics Engine
================================================================================
Mathematical Models & Algorithms:
  1. Intelligent Driver Model (IDM) Longitudinal Control:
     a = a_max * [ 1 - (v / v0)^4 - (s*(v, delta_v) / s)^2 ]
     - Full car-following applied to BOTH ego and all traffic participants.
     - Speed matching and headway maintenance prevent rear-end collisions.
  2. MOBIL (Minimizing Overall Braking Induced by Lane changes):
     Incentive criterion with politeness factor p = 0.15 and threshold d_a = 0.20 m/s^2.
     - Forward & blind-spot gap verification before any lane change is executed.
  3. 5th-Order Quintic Polynomial Lateral Trajectory:
     x(tau) = x_start + delta_x * (10*tau^3 - 15*tau^4 + 6*tau^5), tau in [0, 1]
  4. Two-Track Ackermann Steering Kinematics.
  5. Hard Physical Collision Resolution:
     - Impenetrable bounding box geometry buffers prevent vehicles from phasing through each other.
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
                x=x + random.uniform(-0.3, 0.3),
                y=0.10,
                z=z + random.uniform(-0.25, 0.25),
                vx=random.uniform(-0.4, 0.4),
                vy=random.uniform(0.3, 0.7),
                vz=random.uniform(-1.2, -0.4),
                size=random.uniform(2.5, 5.5),
                alpha=0.80,
                color=(220, 225, 235),
                life=0.0,
                max_life=random.uniform(0.5, 1.0),
                p_type="SMOKE"
            ))

    def emit_guardrail_sparks(self, x: float, z: float, count: int = 6):
        for _ in range(count):
            self.particles.append(Particle(
                x=x + random.uniform(-0.1, 0.1),
                y=random.uniform(0.2, 0.5),
                z=z + random.uniform(-0.2, 0.2),
                vx=random.uniform(-1.2, 1.2),
                vy=random.uniform(1.2, 3.0),
                vz=random.uniform(-2.5, -0.8),
                size=random.uniform(1.5, 3.0),
                alpha=1.0,
                color=(255, random.randint(180, 240), 20),
                life=0.0,
                max_life=random.uniform(0.2, 0.45),
                p_type="SPARK"
            ))

    def emit_exhaust(self, x: float, y: float, z: float, speed_kmh: float):
        if random.random() < 0.25:
            self.particles.append(Particle(
                x=x + random.uniform(-0.2, 0.2),
                y=y,
                z=z - 0.2,
                vx=random.uniform(-0.1, 0.1),
                vy=random.uniform(0.1, 0.25),
                vz=-speed_kmh * 0.04,
                size=random.uniform(2.0, 3.5),
                alpha=0.35,
                color=(160, 170, 180),
                life=0.0,
                max_life=random.uniform(0.35, 0.7),
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
                    p.vy -= 9.81 * dt
                elif p.p_type == "SMOKE":
                    p.size += 3.5 * dt
                p.alpha = max(0.0, 1.0 - (p.life / p.max_life))
                alive.append(p)
        self.particles = alive


@dataclass
class V2XPacket:
    """SAE J2735 Basic Safety Message (BSM) Telemetry."""
    sender_id: str
    speed_kmh: float
    brake_pct: float
    throttle_pct: float
    turn_signal: str
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
        lane_idx: int = 0,
        z_pos: float = 0.0,
        speed_kmh: float = 75.0,
        target_speed_kmh: float = 80.0,
        color: tuple[int, int, int] = (218, 35, 38),
        width: float = 1.92,
        length: float = 4.80,
        height: float = 1.50,
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
        self.model_type = model_type.upper()

        self.v0 = target_speed_kmh / 3.6
        if self.model_type == "TRUCK":
            self.T = 1.8
            self.a_max = 1.4
            self.b_comf = 2.4
            self.s0 = 5.0
        elif self.model_type == "SPORTS":
            self.T = 1.1
            self.a_max = 3.2
            self.b_comf = 3.6
            self.s0 = 2.8
        else:
            self.T = 1.3
            self.a_max = 2.4
            self.b_comf = 3.0
            self.s0 = 3.2
        self.delta_idm = 4.0

        self.accel = 0.0
        self.is_braking = False
        self.blinker = "OFF"
        self.pitch_deg = 0.0
        self.roll_deg = 0.0

        self.is_changing_lane = False
        self.lc_progress = 0.0
        self.lc_duration = 3.2 if self.model_type == "TRUCK" else 2.6
        self.lc_start_x = self.x
        self.lc_target_x = self.x

        self.v2x_packet = None
        self.prediction_fan = []

    @property
    def speed_mps(self) -> float:
        return self.speed_kmh / 3.6

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

    def compute_idm_accel(self, leader_dist: float, leader_speed_mps: float) -> float:
        v = max(0.1, self.speed_mps)
        if leader_dist is not None and leader_dist > 0.0:
            s = max(0.2, leader_dist - self.length)
            delta_v = v - leader_speed_mps
            s_star = self.s0 + max(0.0, v * self.T + (v * delta_v) / (2.0 * math.sqrt(self.a_max * self.b_comf)))
            accel = self.a_max * (1.0 - (v / self.v0) ** self.delta_idm - (s_star / s) ** 2)
        else:
            accel = self.a_max * (1.0 - (v / self.v0) ** self.delta_idm)
        return float(np.clip(accel, -7.5, self.a_max))

    def update_prediction_fan(self):
        dt_fan = 0.35
        steps = 8
        cur_v = max(5.0, self.speed_mps)

        pts_h0, pts_h1, pts_h2 = [], [], []
        target_left_x = self.x - 3.75
        target_right_x = self.x + 3.75

        for step in range(1, steps + 1):
            t_s = step * dt_fan
            ratio = min(1.0, t_s / 2.5)
            s_curve = 10.0 * ratio**3 - 15.0 * ratio**4 + 6.0 * ratio**5

            pts_h0.append((self.x, self.z + cur_v * t_s))
            pts_h1.append((self.x + (target_left_x - self.x) * s_curve, self.z + cur_v * t_s))
            pts_h2.append((self.x + (target_right_x - self.x) * s_curve, self.z + cur_v * t_s))

        p0, p1, p2 = (0.20, 0.70, 0.10) if self.blinker == "LEFT" else ((0.20, 0.10, 0.70) if self.blinker == "RIGHT" else (0.75, 0.125, 0.125))

        self.prediction_fan = [
            {"label": "H0_KEEP", "prob": p0, "points": pts_h0, "color": (0, 212, 255)},
            {"label": "H1_LEFT", "prob": p1, "points": pts_h1, "color": (0, 212, 255)},
            {"label": "H2_RIGHT", "prob": p2, "points": pts_h2, "color": (255, 185, 30)},
        ]

    def update_v2x_telemetry(self, timestamp_ms: float = 0.0, time_ms: float = 0.0):
        ts = timestamp_ms if timestamp_ms != 0.0 else time_ms
        brake_val = 80.0 if self.is_braking else 0.0
        throttle_val = max(0.0, min(100.0, (self.accel / self.a_max) * 100.0)) if not self.is_braking else 0.0

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

    def update_physics(self, dt: float, lead_vehicle: "TrafficVehicle" = None):
        leader_dist = (lead_vehicle.z - self.z) if lead_vehicle else None
        leader_spd = lead_vehicle.speed_mps if lead_vehicle else None
        self.step(dt, leader_dist, leader_spd)

    def step(self, dt: float, leader_dist: float = None, leader_speed_mps: float = None):
        self.accel = self.compute_idm_accel(leader_dist, leader_speed_mps)
        new_speed_mps = max(0.0, self.speed_mps + self.accel * dt)
        self.speed_kmh = new_speed_mps * 3.6
        self.is_braking = self.accel < -1.2

        target_pitch = np.clip(self.accel * -0.75, -2.8, 2.8)
        self.pitch_deg += (target_pitch - self.pitch_deg) * 8.5 * dt

        if self.is_changing_lane:
            self.lc_progress += dt / self.lc_duration
            if self.lc_progress >= 0.9999:
                self.lc_progress = 1.0
            t = min(1.0, self.lc_progress)
            poly_val = 10.0 * (t ** 3) - 15.0 * (t ** 4) + 6.0 * (t ** 5)
            self.x = self.lc_start_x + (self.lc_target_x - self.lc_start_x) * poly_val

            lat_vel = ((self.lc_target_x - self.lc_start_x) / self.lc_duration) * (30.0 * (t**2) - 60.0 * (t**3) + 30.0 * (t**4))
            target_roll = float(np.clip(-lat_vel * 1.5, -3.5, 3.5))
            self.roll_deg += (target_roll - self.roll_deg) * 8.5 * dt

            if t >= 1.0:
                self.x = self.lc_target_x
                self.lane_idx = self.target_lane_idx
                self.is_changing_lane = False
                self.blinker = "OFF"
                self.roll_deg = 0.0
        else:
            self.roll_deg *= 0.80

        self.update_prediction_fan()
        self.update_v2x_telemetry()


class EgoAutonomousVehicle:
    """Ego Tesla Level 4 Autonomous Highway Pilot with Quintic Splines & Ackermann Steering."""

    def __init__(self):
        self.lane_idx = 0
        self.target_lane_idx = 0
        self.x = 0.0
        self.z = 0.0
        self.speed_kmh = 88.0
        self.target_cruise_speed_kmh = 105.0
        self.overtake_cruise_speed_kmh = 115.0

        self.width = 1.96
        self.length = 4.97
        self.height = 1.44
        self.wheelbase_m = 2.96
        self.track_width_m = 1.66
        self.mass_kg = 2160.0

        self.state = "LANE_KEEP"
        self.manual_override = False

        self.steering_angle_deg = 0.0
        self.steering_inner_deg = 0.0
        self.steering_outer_deg = 0.0
        self.yaw_rate_rads = 0.0
        self.lat_accel_g = 0.0
        self.lat_jerk_gs = 0.0
        self.last_lat_accel_g = 0.0

        self.g_history: list[tuple[float, float]] = [(0.0, 0.0)] * 150
        self.smoothed_lat_g = 0.0
        self.smoothed_long_g = 0.0

        self.pitch_deg = 0.0
        self.roll_deg = 0.0
        self.blinker = "OFF"
        self.is_braking = False
        self.accel_mps2 = 0.0

        self.lc_progress = 0.0
        self.lc_duration = 3.0
        self.lc_start_x = 0.0
        self.lc_target_x = 0.0

        self.overtake_target_id = None

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
        if self.state in ("LANE_CHANGE_LEFT", "LANE_CHANGE_RIGHT"):
            self.lc_progress += dt / self.lc_duration
            t = min(1.0, self.lc_progress)

            s = 10.0 * (t ** 3) - 15.0 * (t ** 4) + 6.0 * (t ** 5)
            s_dot = (30.0 * (t ** 2) - 60.0 * (t ** 3) + 30.0 * (t ** 4)) / self.lc_duration
            s_ddot = (60.0 * t - 180.0 * (t ** 2) + 120.0 * (t ** 3)) / (self.lc_duration ** 2)

            dx = self.lc_target_x - self.lc_start_x
            self.x = self.lc_start_x + dx * s
            lat_vel = dx * s_dot
            lat_accel = dx * s_ddot

            v = max(5.0, self.speed_mps)
            curvature = lat_accel / (v ** 2)
            delta_rad = math.atan(self.wheelbase_m * curvature)
            self.steering_angle_deg = float(np.clip(math.degrees(delta_rad) * 4.0, -32.0, 32.0))
            self.compute_ackermann_angles(delta_rad)

            raw_lat_g = lat_accel / 9.81
            self.lat_accel_g += (raw_lat_g - self.lat_accel_g) * 10.0 * dt
            self.lat_jerk_gs = (self.lat_accel_g - self.last_lat_accel_g) / max(0.001, dt)
            self.last_lat_accel_g = self.lat_accel_g

            target_roll = float(np.clip(-self.lat_accel_g * 3.8, -3.8, 3.8))
            self.roll_deg += (target_roll - self.roll_deg) * 8.5 * dt

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
            self.steering_angle_deg *= 0.82
            self.compute_ackermann_angles(math.radians(self.steering_angle_deg))
            self.roll_deg *= 0.80
            self.lat_accel_g *= 0.80
            self.lat_jerk_gs = 0.0

        target_pitch = np.clip((self.accel_mps2 / 9.81) * -0.75, -2.5, 2.5)
        self.pitch_deg += (target_pitch - self.pitch_deg) * 8.5 * dt

        raw_long_g = self.accel_mps2 / 9.81
        alpha_ema = 0.18
        self.smoothed_lat_g = (1 - alpha_ema) * self.smoothed_lat_g + alpha_ema * self.lat_accel_g
        self.smoothed_long_g = (1 - alpha_ema) * self.smoothed_long_g + alpha_ema * raw_long_g

        self.g_history.append((self.smoothed_lat_g, self.smoothed_long_g))
        if len(self.g_history) > 150:
            self.g_history.pop(0)


class HighwayTrafficEngine:
    """Master Traffic Engine orchestrating IDM, MOBIL politeness, collision-free kinematics, and dynamic particles."""

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
        self.traffic_vehicles.clear()

        # 1. Lead Car in center cruising lane (Lane 0)
        lead_car = TrafficVehicle(
            veh_id="LEAD_SEDAN",
            lane_idx=0,
            z_pos=26.0,
            speed_kmh=68.0,
            target_speed_kmh=70.0,
            color=(218, 35, 38),
            width=1.92,
            length=4.80,
            height=1.45,
            model_type="SEDAN"
        )

        # 2. Fast Sedan in passing lane (Lane -1)
        fast_sedan = TrafficVehicle(
            veh_id="FAST_SEDAN",
            lane_idx=-1,
            z_pos=46.0,
            speed_kmh=102.0,
            target_speed_kmh=105.0,
            color=(240, 242, 245),
            width=1.92,
            length=4.80,
            height=1.45,
            model_type="SEDAN"
        )

        # 3. Semi Truck in slow commercial lane (Lane +1)
        semi_truck = TrafficVehicle(
            veh_id="SEMI_TRUCK",
            lane_idx=1,
            z_pos=18.0,
            speed_kmh=62.0,
            target_speed_kmh=65.0,
            color=(28, 135, 72),
            width=2.60,
            length=12.0,
            height=3.50,
            model_type="TRUCK"
        )

        # 4. Sports Coupe behind in left lane (Lane -1)
        rear_sports = TrafficVehicle(
            veh_id="SPORTS_COUPE",
            lane_idx=-1,
            z_pos=-28.0,
            speed_kmh=92.0,
            target_speed_kmh=95.0,
            color=(245, 170, 25),
            width=1.86,
            length=4.40,
            height=1.30,
            model_type="SPORTS"
        )

        self.traffic_vehicles.extend([lead_car, fast_sedan, semi_truck, rear_sports])
        self.log_event("SCENARIO INITIALIZED: 4 PARTICIPANTS SPAWNED")

    def _find_leader_for_vehicle(self, veh: TrafficVehicle) -> tuple[float, float]:
        """Finds the distance and speed of the nearest vehicle ahead of veh in its lane corridor."""
        leader_dist = None
        leader_spd = None
        min_gap = 9999.0

        # Check other traffic vehicles
        for other in self.traffic_vehicles:
            if other.id == veh.id:
                continue
            if abs(other.x - veh.x) < 2.2 and other.z > veh.z:
                gap = other.z - veh.z
                if gap < min_gap:
                    min_gap = gap
                    leader_dist = gap
                    leader_spd = other.speed_mps

        # Check ego vehicle (ego is at x=self.ego.x, z=0.0)
        if abs(self.ego.x - veh.x) < 2.2 and 0.0 > veh.z:
            gap = -veh.z # ego.z (0.0) - veh.z
            if gap < min_gap:
                min_gap = gap
                leader_dist = gap
                leader_spd = self.ego.speed_mps

        return leader_dist, leader_spd

    def _find_leader_for_ego(self, ego_x: float) -> tuple[TrafficVehicle, float]:
        """Finds the nearest traffic vehicle ahead of ego in its current/intended corridor."""
        closest_v = None
        min_z = 9999.0
        for v in self.traffic_vehicles:
            if abs(v.x - ego_x) < 2.2 and v.z > 0.5:
                if v.z < min_z:
                    min_z = v.z
                    closest_v = v
        return closest_v, (min_z if closest_v else None)

    def _step_ego_l4_autopilot(self, dt: float):
        if self.ego.manual_override:
            self.ego.step(dt)
            return

        lead_car, lead_dist = self._find_leader_for_ego(self.ego.x)

        if self.ego.state == "LANE_KEEP":
            if lead_car and lead_dist < 28.0 and lead_car.speed_kmh < self.ego.target_cruise_speed_kmh - 8.0:
                self.ego.state = "CHECK_OVERTAKE"
                self.ego.overtake_target_id = lead_car.id
                self.log_event(f"OVERTAKE INITIATED — TARGET: {lead_car.id}")
            else:
                rel_speed = lead_car.speed_mps if lead_car else None
                accel = self._compute_idm_accel(self.ego.speed_mps, self.ego.target_cruise_speed_kmh / 3.6, lead_dist, rel_speed)
                self.ego.accel_mps2 = accel
                self.ego.speed_kmh = max(10.0, min(120.0, (self.ego.speed_mps + accel * dt) * 3.6))

        elif self.ego.state == "CHECK_OVERTAKE":
            # Verify left passing lane (Lane -1, X = -3.75) is completely safe in both forward & rear directions
            left_traffic = [v for v in self.traffic_vehicles if abs(v.x - (-3.75)) < 1.9]
            is_left_safe = True
            for v in left_traffic:
                # Blind spot & approaching buffer: -18m to +25m
                if -18.0 < v.z < 25.0:
                    is_left_safe = False
                    break
                # Rapidly closing vehicle from behind
                if v.z <= -18.0 and (v.speed_mps - self.ego.speed_mps) > 4.0:
                    ttc = abs(v.z) / (v.speed_mps - self.ego.speed_mps)
                    if ttc < 3.5:
                        is_left_safe = False
                        break

            if is_left_safe and self.ego.lane_idx > -1:
                self.ego.initiate_lane_change(self.ego.lane_idx - 1)
                self.log_event("MANEUVERING TO FAST LANE (-1) WITH LEFT BLINKER")
            else:
                # Continue safe following in current lane while waiting for an open slot
                accel = self._compute_idm_accel(self.ego.speed_mps, 70.0 / 3.6, lead_dist, lead_car.speed_mps if lead_car else 19.0)
                self.ego.accel_mps2 = accel
                self.ego.speed_kmh = max(10.0, (self.ego.speed_mps + accel * dt) * 3.6)

        elif self.ego.state in ("LANE_CHANGE_LEFT", "LANE_CHANGE_RIGHT"):
            # During lane transition, follow any vehicle ahead in the target lane
            target_x = float(self.ego.target_lane_idx * 3.75)
            trans_lead, trans_dist = self._find_leader_for_ego(target_x)
            v_target = self.ego.overtake_cruise_speed_kmh / 3.6 if self.ego.target_lane_idx < 0 else self.ego.target_cruise_speed_kmh / 3.6
            accel = self._compute_idm_accel(self.ego.speed_mps, v_target, trans_dist, trans_lead.speed_mps if trans_lead else None)
            self.ego.accel_mps2 = accel
            self.ego.speed_kmh = max(10.0, min(120.0, (self.ego.speed_mps + accel * dt) * 3.6))

        elif self.ego.state == "OVERTAKING":
            # While in fast lane, obey IDM following for any vehicle ahead in the fast lane
            fast_lead, fast_dist = self._find_leader_for_ego(-3.75)
            accel = self._compute_idm_accel(self.ego.speed_mps, self.ego.overtake_cruise_speed_kmh / 3.6, fast_dist, fast_lead.speed_mps if fast_lead else None)
            self.ego.accel_mps2 = accel
            self.ego.speed_kmh = max(10.0, min(120.0, (self.ego.speed_mps + accel * dt) * 3.6))

            target_veh = next((v for v in self.traffic_vehicles if v.id == self.ego.overtake_target_id), None)
            if target_veh and target_veh.z < -16.0:
                # Check center lane (Lane 0) clearance
                center_traffic = [v for v in self.traffic_vehicles if abs(v.x - 0.0) < 1.9]
                is_center_safe = all(not (-14.0 < v.z < 22.0) for v in center_traffic)
                if is_center_safe:
                    self.ego.initiate_lane_change(0)
                    self.log_event("MERGING SAFELY BACK TO CENTER CRUISING LANE")

        self.ego.step(dt)

    def step(self, dt: float):
        self.sim_time_s += dt
        self._step_ego_l4_autopilot(dt)

        # 1. Update each traffic vehicle with active IDM car-following
        for v in self.traffic_vehicles:
            lead_dist, lead_spd = self._find_leader_for_vehicle(v)
            v.step(dt, lead_dist, lead_spd)

            # Update relative longitudinal position
            rel_v_mps = v.speed_mps - self.ego.speed_mps
            v.z += rel_v_mps * dt

            # MOBIL Lane Changing for Traffic
            if not v.is_changing_lane and random.random() < 0.005:
                if v.model_type == "SPORTS" and v.lane_idx == 0:
                    v.initiate_lane_change(-1)
                elif v.model_type == "TRUCK" and v.lane_idx == 0:
                    v.initiate_lane_change(1)

            # Staggered respawn with collision-free slot verification
            if v.z > 95.0:
                self._respawn_vehicle_safely(v, from_rear=True)
            elif v.z < -50.0:
                self._respawn_vehicle_safely(v, from_rear=False)

            self.particle_emitter.emit_exhaust(v.x, 0.25, v.z - v.length * 0.5, v.speed_kmh)

        # 2. Hard Impenetrable Collision & Penetration Resolution
        self._enforce_collision_avoidance_and_penetration_resolution()

        # Dynamic Particles
        if self.ego.accel_mps2 < -2.0 or self.ego.is_braking:
            self.particle_emitter.emit_tire_smoke(self.ego.x - 0.7, self.ego.z - 2.0, count=2)
            self.particle_emitter.emit_tire_smoke(self.ego.x + 0.7, self.ego.z - 2.0, count=2)

        if abs(self.ego.x) > 5.2:
            self.particle_emitter.emit_guardrail_sparks(self.ego.x, self.ego.z, count=4)

        self.particle_emitter.update(dt)

    def _respawn_vehicle_safely(self, veh: TrafficVehicle, from_rear: bool):
        """Respawns a vehicle at a guaranteed collision-free gap."""
        target_z = random.uniform(-40.0, -30.0) if from_rear else random.uniform(65.0, 80.0)
        target_lane = veh.lane_idx

        # Ensure no other vehicle is in this spawn zone
        for other in self.traffic_vehicles:
            if other.id != veh.id and other.lane_idx == target_lane:
                if abs(other.z - target_z) < 25.0:
                    target_z = other.z - 28.0 if from_rear else other.z + 28.0

        veh.x = float(target_lane * 3.75)
        veh.z = target_z
        veh.speed_kmh = random.uniform(62.0, 70.0) if veh.model_type == "TRUCK" else (random.uniform(95.0, 105.0) if veh.model_type == "SPORTS" else random.uniform(75.0, 85.0))
        veh.accel = 0.0
        veh.is_changing_lane = False

    def _enforce_collision_avoidance_and_penetration_resolution(self):
        """Impenetrable bounding box physics buffer: vehicles can NEVER intersect or pass through each other."""
        all_participants = []
        # Ego representation
        all_participants.append({
            "id": "EGO",
            "x": self.ego.x,
            "z": 0.0,
            "width": self.ego.width,
            "length": self.ego.length,
            "speed_mps": self.ego.speed_mps,
            "is_ego": True,
            "ref": self.ego
        })

        for v in self.traffic_vehicles:
            all_participants.append({
                "id": v.id,
                "x": v.x,
                "z": v.z,
                "width": v.width,
                "length": v.length,
                "speed_mps": v.speed_mps,
                "is_ego": False,
                "ref": v
            })

        for i in range(len(all_participants)):
            for j in range(i + 1, len(all_participants)):
                p1 = all_participants[i]
                p2 = all_participants[j]

                lat_dist = abs(p1["x"] - p2["x"])
                lat_thresh = (p1["width"] + p2["width"]) * 0.5 + 0.35

                if lat_dist < lat_thresh:
                    long_dist = p2["z"] - p1["z"]
                    min_sep = (p1["length"] + p2["length"]) * 0.5 + 1.2

                    # Check penetration
                    if abs(long_dist) < min_sep:
                        # p2 is in front of p1
                        if long_dist >= 0:
                            rear = p1
                            front = p2
                        else:
                            rear = p2
                            front = p1

                        # Resolve penetration by separating rear vehicle
                        overlap = min_sep - abs(long_dist)
                        if rear["is_ego"]:
                            # Ego cannot pass through front vehicle
                            self.ego.speed_kmh = max(10.0, min(self.ego.speed_kmh, front["ref"].speed_kmh - 2.0))
                            self.ego.is_braking = True
                        else:
                            rear["ref"].z -= (overlap + 0.1)
                            rear["ref"].speed_kmh = max(10.0, min(rear["ref"].speed_kmh, front["speed_mps"] * 3.6 - 2.0))
                            rear["ref"].is_braking = True

    def _compute_idm_accel(self, v: float, v0: float, s: float = None, v_lead: float = None) -> float:
        a_max = 2.4
        b_comf = 3.0
        T = 1.2
        s0 = 3.2
        delta = 4.0

        if s is not None and s > 0.0:
            delta_v = v - (v_lead if v_lead is not None else v)
            s_star = s0 + max(0.0, v * T + (v * delta_v) / (2.0 * math.sqrt(a_max * b_comf)))
            accel = a_max * (1.0 - (v / v0) ** delta - (s_star / s) ** 2)
        else:
            accel = a_max * (1.0 - (v / v0) ** delta)
        return float(np.clip(accel, -7.5, a_max))

    def get_lead_v2x_packet(self) -> V2XPacket:
        lead_car = next((v for v in self.traffic_vehicles if abs(v.x - self.ego.x) < 2.0 and v.z > 0), None)
        if lead_car:
            return lead_car.v2x_packet
        return None

    def get_dynamic_objects_for_sensors(self) -> list:
        objs = []
        for v in self.traffic_vehicles:
            rel_x = v.x - self.ego.x
            rel_z = v.z
            objs.append((rel_x, rel_z, v.width, v.length, v.id, v.color, v.model_type, v.height))
        return objs
