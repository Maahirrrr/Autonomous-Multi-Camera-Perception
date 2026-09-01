"""
traffic_physics_simulator.py — Level 4 Traffic Physics, V2X Telemetry & Particle Dynamics
========================================================================================
Physics & Systems:
  - 3D Vehicle Suspension Dynamics:
      Pitch: theta_pitch = -(a_x * h_cg) / (k_susp * L)
      Roll:  phi_roll   = +(a_y * h_cg) / (k_roll * W)
  - Multi-Hypothesis Trajectory Prediction Fans (3-Second Horizon, Probabilities P(H_k)):
      H0: Nominal Lane Keep (P ~ 0.70)
      H1: Lane Change Left  (P ~ 0.15)
      H2: Lane Change Right (P ~ 0.15)
  - V2X (Vehicle-to-Everything) Basic Safety Message (BSM) Broadcast Simulation.
  - Physics-Driven Particle System:
      * Tire smoke on hard braking (a_x < -2.2 m/s^2)
      * Sparks on guardrail proximity (|x| > 5.4m)
      * Dual-exhaust smoke / heat particles trailing vehicles.
"""

import math
import random
import numpy as np


class Particle:
    """Represents a physics particle for tire smoke, exhaust, or sparks."""
    def __init__(self, x: float, y: float, z: float, vx: float, vy: float, vz: float,
                 p_type: str = "EXHAUST", lifetime_s: float = 0.8, color: tuple = (200, 200, 200), size: float = 3.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.vx = float(vx)
        self.vy = float(vy)
        self.vz = float(vz)
        self.p_type = p_type # 'EXHAUST', 'SMOKE', 'SPARK'
        self.life = float(lifetime_s)
        self.max_life = float(lifetime_s)
        self.color = color
        self.size = size

    def update(self, dt: float) -> bool:
        self.life -= dt
        if self.life <= 0:
            return False

        self.x += self.vx * dt
        self.y += self.vy * dt
        self.z += self.vz * dt

        if self.p_type == "SPARK":
            self.vy -= 9.81 * dt * 0.4 # Gravity on sparks
        elif self.p_type in ("SMOKE", "EXHAUST"):
            self.vy += 0.4 * dt # Buoyancy rising
            self.size += 1.8 * dt # Expansion

        return True


class ParticleEmitter:
    """Manages particle emission for smoke, sparks, and exhaust."""
    def __init__(self):
        self.particles: list[Particle] = []

    def emit_exhaust(self, x: float, y: float, z: float, ego_speed_mps: float):
        # Dual exhaust tips
        for offset_x in (-0.55, 0.55):
            vx = random.uniform(-0.15, 0.15)
            vy = random.uniform(0.05, 0.25)
            vz = -random.uniform(0.4, 1.2) - (ego_speed_mps * 0.2)
            shade = random.randint(140, 190)
            self.particles.append(
                Particle(x + offset_x, y + 0.25, z - 2.2, vx, vy, vz,
                         p_type="EXHAUST", lifetime_s=random.uniform(0.4, 0.7),
                         color=(shade, shade, shade), size=random.uniform(2.0, 4.0))
            )

    def emit_tire_smoke(self, x: float, z: float):
        for offset_x in (-0.8, 0.8):
            for offset_z in (-1.4, 1.4):
                vx = random.uniform(-0.5, 0.5)
                vy = random.uniform(0.4, 1.2)
                vz = random.uniform(-0.8, 0.2)
                self.particles.append(
                    Particle(x + offset_x, 0.12, z + offset_z, vx, vy, vz,
                             p_type="SMOKE", lifetime_s=random.uniform(0.6, 1.1),
                             color=(225, 230, 240), size=random.uniform(4.0, 8.0))
                )

    def emit_guardrail_sparks(self, x: float, z: float):
        for _ in range(3):
            vx = random.uniform(-2.0, 2.0)
            vy = random.uniform(1.0, 4.0)
            vz = random.uniform(-3.0, 1.0)
            self.particles.append(
                Particle(x, 0.45, z, vx, vy, vz,
                         p_type="SPARK", lifetime_s=random.uniform(0.2, 0.45),
                         color=(255, random.randint(180, 255), 40), size=random.uniform(2.0, 3.5))
            )

    def update(self, dt: float):
        self.particles = [p for p in self.particles if p.update(dt)]
        # Cap max particles for 60 FPS performance
        if len(self.particles) > 180:
            self.particles = self.particles[-180:]


class V2XPacket:
    """Represents a simulated SAE J2735 Basic Safety Message (BSM) V2X packet."""
    def __init__(self, sender_id: str, speed_kmh: float, brake_pct: float, throttle_pct: float,
                 turn_signal: str, lat_lon: tuple = (37.7749, -122.4194)):
        self.sender_id = sender_id
        self.speed_kmh = speed_kmh
        self.brake_pct = brake_pct
        self.throttle_pct = throttle_pct
        self.turn_signal = turn_signal
        self.gps_lat = lat_lon[0]
        self.gps_lon = lat_lon[1]
        self.transmission_state = "FORWARD_DRIVE"
        self.packet_loss = False
        self.rssi_dbm = -62.0


class TrajectoryPredictionFan:
    """Computes a 3-second multi-hypothesis trajectory prediction fan for dynamic obstacles."""
    @staticmethod
    def generate_predictions(x: float, z: float, vx: float, vz: float, blinker: str = "OFF",
                             time_horizon_s: float = 3.0, num_steps: int = 12) -> list[dict]:
        time_steps = np.linspace(0.25, time_horizon_s, num_steps)
        hypotheses = []

        # Hypothesis 0: Nominal Lane Keep
        pts_h0 = []
        for t in time_steps:
            pred_x = x + vx * t
            pred_z = z + vz * t
            pts_h0.append((float(pred_x), float(pred_z)))
        p_h0 = 0.70 if blinker == "OFF" else 0.20
        hypotheses.append({"name": "LANE_KEEP", "prob": p_h0, "points": pts_h0, "color": (0, 255, 180)})

        # Hypothesis 1: Left Lane Change
        pts_h1 = []
        for t in time_steps:
            lat_shift = -3.75 * min(1.0, (t / 2.0)**2)
            pred_x = x + lat_shift
            pred_z = z + vz * t
            pts_h1.append((float(pred_x), float(pred_z)))
        p_h1 = 0.70 if blinker == "LEFT" else 0.15
        hypotheses.append({"name": "LANE_LEFT", "prob": p_h1, "points": pts_h1, "color": (255, 210, 0)})

        # Hypothesis 2: Right Lane Change
        pts_h2 = []
        for t in time_steps:
            lat_shift = +3.75 * min(1.0, (t / 2.0)**2)
            pred_x = x + lat_shift
            pred_z = z + vz * t
            pts_h2.append((float(pred_x), float(pred_z)))
        p_h2 = 0.70 if blinker == "RIGHT" else 0.15
        hypotheses.append({"name": "LANE_RIGHT", "prob": p_h2, "points": pts_h2, "color": (255, 120, 0)})

        return hypotheses


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
        self.blinker = "OFF"
        self.lane_change_progress = 1.0
        self.lane_change_duration_s = 2.4
        self.start_x = self.x
        self.target_x = self.x

        # Suspension pitch & roll
        self.pitch_deg = 0.0
        self.roll_deg = 0.0
        self.wheel_angle_rad = 0.0

        # Trajectory Predictions & V2X
        self.prediction_fan: list[dict] = []
        self.v2x_packet: V2XPacket = None

    def initiate_lane_change(self, new_lane_idx: int):
        self.target_lane_idx = new_lane_idx
        self.start_x = self.x
        self.target_x = float(new_lane_idx * 3.75)
        self.lane_change_progress = 0.0
        self.blinker = "LEFT" if new_lane_idx < self.lane_idx else "RIGHT"

    def update_physics(self, dt: float, lead_vehicle = None):
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

        self.pitch_deg = -acc * 0.45
        self.wheel_angle_rad = (self.wheel_angle_rad + (self.speed_mps / 0.33) * dt) % (2.0 * math.pi)

        if self.lane_change_progress < 1.0:
            self.lane_change_progress = min(1.0, self.lane_change_progress + dt / self.lane_change_duration_s)
            tau = self.lane_change_progress
            s_curve = 10.0 * (tau ** 3) - 15.0 * (tau ** 4) + 6.0 * (tau ** 5)
            self.x = self.start_x + (self.target_x - self.start_x) * s_curve

            d_s = 30.0 * (tau ** 2) - 60.0 * (tau ** 3) + 30.0 * (tau ** 4)
            self.roll_deg = (self.target_x - self.start_x) * d_s * 0.40

            if self.lane_change_progress >= 1.0:
                self.lane_idx = self.target_lane_idx
                self.x = self.target_x
                self.roll_deg = 0.0
                self.blinker = "OFF"
        else:
            self.roll_deg = 0.0

        # Update Trajectory Predictions
        rel_vx = (self.target_x - self.start_x) / max(0.1, self.lane_change_duration_s) if self.lane_change_progress < 1.0 else 0.0
        self.prediction_fan = TrajectoryPredictionFan.generate_predictions(
            self.x, self.z, rel_vx, self.speed_mps, self.blinker
        )

        # Update V2X BSM Packet
        brake_pct = min(100.0, max(0.0, -acc * 35.0)) if self.is_braking else 0.0
        throttle_pct = min(100.0, max(0.0, acc * 40.0)) if not self.is_braking else 0.0
        self.v2x_packet = V2XPacket(
            sender_id=self.id,
            speed_kmh=self.speed_kmh,
            brake_pct=brake_pct,
            throttle_pct=throttle_pct,
            turn_signal=self.blinker
        )


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

        # G-Force History (Last 5 seconds)
        self.g_history: list[tuple[float, float]] = []

    def initiate_lane_change(self, new_lane_idx: int):
        self.target_lane_idx = new_lane_idx
        self.start_x = self.x
        self.target_x = float(new_lane_idx * 3.75)
        self.lane_change_progress = 0.0
        self.blinker = "LEFT" if new_lane_idx < self.lane_idx else "RIGHT"

    def update_autonomous_driving(self, dt: float, traffic_vehicles: list[TrafficVehicle]):
        self.state_timer_s += dt

        lead_in_lane = None
        min_lead_dist = 999.0
        for v in traffic_vehicles:
            if abs(v.x - self.x) < 1.8 and v.z > 0.0:
                dist = v.z - self.length
                if dist < min_lead_dist:
                    min_lead_dist = dist
                    lead_in_lane = v

        left_clear = True
        right_clear = True
        for v in traffic_vehicles:
            if abs(v.x - (-3.75)) < 2.0 and -16.0 < v.z < 26.0:
                left_clear = False
            if abs(v.x - (+3.75)) < 2.0 and -16.0 < v.z < 24.0:
                right_clear = False

        if not self.manual_override:
            if self.state == "LANE_KEEP":
                self.blinker = "OFF"
                target_v = self.cruise_target_speed_kmh
                if lead_in_lane:
                    if lead_in_lane.speed_kmh < self.cruise_target_speed_kmh - 6.0 and min_lead_dist < 26.0:
                        self.state = "CHECK_OVERTAKE"
                        self.state_timer_s = 0.0
                        self.overtaking_target_id = lead_in_lane.id
                else:
                    self.speed_kmh = min(self.cruise_target_speed_kmh, self.speed_kmh + 8.0 * dt)

            elif self.state == "CHECK_OVERTAKE":
                if left_clear and self.lane_idx >= 0:
                    self.state = "LANE_CHANGE_LEFT"
                    self.state_timer_s = 0.0
                    self.initiate_lane_change(-1)
                elif lead_in_lane and min_lead_dist < 14.0:
                    self.speed_kmh = max(lead_in_lane.speed_kmh, self.speed_kmh - 10.0 * dt)
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

        # Record G-force history
        long_g = self.accel_mps2 / 9.81
        self.g_history.append((self.lat_accel_g, long_g))
        if len(self.g_history) > 150: # 5 seconds at 30 fps sample
            self.g_history.pop(0)


class HighwayTrafficEngine:
    """Manages 3D highway traffic, particle emitters, V2X broadcast and multi-lane flow."""
    def __init__(self):
        self.ego = EgoAutonomousVehicle()
        self.traffic_vehicles: list[TrafficVehicle] = []
        self.particle_emitter = ParticleEmitter()
        self.event_log: list[str] = [
            "[T+0.0s] L4 PERCEPTION STACK INITIALIZED (RTX 4070 CUDA FP16)",
            "[T+0.2s] 360 LIDAR + 4-CAM SENSOR FUSION ACTIVE",
            "[T+0.5s] HIGHWAY PILOT ENGAGED — CRUISING AT 88 KM/H",
        ]
        self.sim_time_s = 0.0
        self._spawn_initial_traffic()

    def log_event(self, msg: str):
        entry = f"[T+{self.sim_time_s:.1f}s] {msg}"
        self.event_log.append(entry)
        if len(self.event_log) > 25:
            self.event_log.pop(0)

    def randomize_scenario(self):
        self._spawn_initial_traffic()
        self.log_event("TRAFFIC SCENARIO RE-SEEDED & RANDOMIZED")

    def _spawn_initial_traffic(self):
        colors = [
            (210, 45, 45), (245, 210, 25), (45, 110, 195),
            (50, 160, 80), (175, 55, 175), (220, 120, 40)
        ]
        self.traffic_vehicles = [
            TrafficVehicle("LEAD_CAR_1", lane_idx=0, z_pos=random.uniform(22.0, 28.0), speed_kmh=random.uniform(64.0, 70.0), v_type="SEDAN", color=colors[0]),
            TrafficVehicle("SPORTS_FAST_LEFT", lane_idx=-1, z_pos=random.uniform(-42.0, -30.0), speed_kmh=random.uniform(108.0, 118.0), v_type="SPORTS", color=colors[1]),
            TrafficVehicle("SEMI_TRUCK_RIGHT", lane_idx=+1, z_pos=random.uniform(12.0, 20.0), speed_kmh=random.uniform(60.0, 66.0), v_type="TRUCK", color=colors[3], width=2.45, length=9.8, height=3.3),
            TrafficVehicle("TRAILING_SUV", lane_idx=0, z_pos=random.uniform(-32.0, -22.0), speed_kmh=random.uniform(80.0, 86.0), v_type="SUV", color=colors[2], width=2.0, length=4.9, height=1.75),
            TrafficVehicle("FAST_SEDAN_LEFT", lane_idx=-1, z_pos=random.uniform(45.0, 60.0), speed_kmh=random.uniform(95.0, 102.0), v_type="SEDAN", color=colors[4]),
        ]

    def step(self, dt: float):
        self.sim_time_s += dt
        prev_state = self.ego.state

        self.ego.update_autonomous_driving(dt, self.traffic_vehicles)

        if self.ego.state != prev_state:
            if self.ego.state == "CHECK_OVERTAKE":
                self.log_event(f"OVERTAKE INITIATED — TARGET: {self.ego.overtaking_target_id}")
            elif self.ego.state == "LANE_CHANGE_LEFT":
                self.log_event("MANEUVERING TO FAST LANE (-1) WITH LEFT BLINKER")
            elif self.ego.state == "OVERTAKING":
                self.log_event(f"OVERTAKING IN FAST LANE @ {self.ego.speed_kmh:.0f} KM/H")
            elif self.ego.state == "LANE_CHANGE_RIGHT":
                self.log_event("MERGING SAFELY BACK TO CENTER CRUISING LANE")
            elif self.ego.state == "LANE_KEEP":
                self.log_event("LANE CHANGE COMPLETE — RETURNING TO CRUISE")

        # Emit particles for Ego
        if self.ego.is_braking or self.ego.accel_mps2 < -2.2:
            self.particle_emitter.emit_tire_smoke(self.ego.x, self.ego.z)

        if abs(self.ego.x) > 5.2:
            self.particle_emitter.emit_guardrail_sparks(self.ego.x, self.ego.z)

        # Update Traffic
        for v in self.traffic_vehicles:
            v.update_physics(dt)
            rel_speed_mps = v.speed_mps - self.ego.speed_mps
            v.z += rel_speed_mps * dt

            # Vehicle exhaust
            self.particle_emitter.emit_exhaust(v.x, 0.15, v.z, self.ego.speed_mps)

            # Recycle vehicles
            if v.z > 85.0:
                v.z = -50.0
                v.speed_kmh = 108.0 if v.lane_idx == -1 else (65.0 if v.lane_idx == 1 else 78.0)
                v.speed_mps = v.speed_kmh / 3.6
            elif v.z < -65.0:
                v.z = 70.0
                v.speed_kmh = 98.0 if v.lane_idx == -1 else (62.0 if v.lane_idx == 1 else 72.0)
                v.speed_mps = v.speed_kmh / 3.6

        # Update all particles
        self.particle_emitter.update(dt)

    def get_dynamic_objects_for_sensors(self) -> list:
        objs = []
        for v in self.traffic_vehicles:
            objs.append((v.x - self.ego.x, v.z, v.width, v.length, v.id, v.color))
        return objs

    def get_lead_v2x_packet(self) -> V2XPacket:
        """Returns V2X packet from closest lead vehicle."""
        lead = next((v for v in self.traffic_vehicles if abs(v.x - self.ego.x) < 2.0 and v.z > 0), None)
        if lead and lead.v2x_packet:
            return lead.v2x_packet
        return None
