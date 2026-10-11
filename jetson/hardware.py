"""Robot I/O: 12 STS3215 on one bus + the Betaflight F3 (IMU, battery current), converted to the
policy's conventions via the calibration in config.json. MockHardware runs everything without a robot."""
import math
import time

import numpy as np

from controller import JOINTS, Q_STAND, gravity_from_attitude, gyro_to_body, rad_to_ticks, ticks_to_rad


class HardwareFault(Exception):
    pass


class State(object):
    __slots__ = ("t", "q", "qd", "ang_vel", "gravity", "yaw_deg", "bus_errors")


class Hardware(object):
    def __init__(self, cfg):
        from msp import MSP
        from sts import STSBus
        self.cfg = cfg
        self.bus = STSBus(cfg["servo_port"], cfg.get("servo_baud", 1000000))
        self.f3 = MSP(cfg["f3_port"])
        self.ids = [cfg["joints"][j]["id"] for j in JOINTS]
        self.offset = np.array([cfg["joints"][j]["offset"] for j in JOINTS], dtype=float)
        self.dir = np.array([cfg["joints"][j]["dir"] for j in JOINTS], dtype=float)
        imu = cfg["imu"]
        self.gyro_axes, self.gyro_signs = imu["gyro_axes"], imu["gyro_signs"]
        self.roll_sign, self.pitch_sign = imu["roll_sign"], imu["pitch_sign"]
        self.fd_vel = cfg.get("joint_velocity_source", "servo") == "finite_difference"
        self.max_err = cfg["safety"]["max_consecutive_bus_errors"]
        self.q = None
        self.qd = np.zeros(12)
        self.miss = np.zeros(12, dtype=int)
        self.last_t = None
        self.imu_miss = 0
        self.ang_vel = np.zeros(3)
        self.gravity = np.array([0.0, 0.0, -1.0])
        self.yaw = 0.0
        self._health_i = 0
        self.health = {}          # id -> (V, C, A)
        self.battery = None       # (V, A, mAh) from the F3

    def check_servos(self):
        missing = [j for j, sid in zip(JOINTS, self.ids) if not self.bus.ping(sid)]
        if missing:
            raise HardwareFault("servos not answering: " + ", ".join(missing))

    def read(self):
        t = time.monotonic()
        raw = self.bus.sync_read_pos_speed(self.ids)
        q = np.zeros(12) if self.q is None else self.q.copy()
        qd = self.qd.copy()
        for i, sid in enumerate(self.ids):
            r = raw.get(sid)
            if r is None:
                self.miss[i] += 1
                continue
            self.miss[i] = 0
            q[i] = ticks_to_rad(r[0], self.offset[i], self.dir[i])
            qd[i] = self.dir[i] * r[1] * (2.0 * math.pi / 4096.0)
        if self.q is None and self.miss.any():
            raise HardwareFault("first servo read incomplete: " +
                                ", ".join(JOINTS[i] for i in np.nonzero(self.miss)[0]))
        if self.miss.max() >= self.max_err:
            raise HardwareFault("lost servo " + JOINTS[int(np.argmax(self.miss))])
        if self.fd_vel and self.q is not None and self.last_t is not None:
            qd = 0.5 * qd + 0.5 * (q - self.q) / max(t - self.last_t, 1e-3)   # smoothed finite difference
        self.q, self.qd, self.last_t = q, qd, t

        att, gyr = self.f3.attitude(), self.f3.gyro()
        if att is None or gyr is None:
            self.imu_miss += 1
            if self.imu_miss >= self.max_err:
                raise HardwareFault("lost the F3 (MSP)")
        else:
            self.imu_miss = 0
            self.gravity = gravity_from_attitude(self.roll_sign * att[0], self.pitch_sign * att[1])
            self.yaw = att[2]
            self.ang_vel = gyro_to_body(gyr, self.gyro_axes, self.gyro_signs)

        s = State()
        s.t, s.q, s.qd, s.ang_vel, s.gravity, s.yaw_deg, s.bus_errors = \
            t, q, qd, self.ang_vel.copy(), self.gravity.copy(), self.yaw, int(self.miss.max())
        return s

    def write(self, q_target):
        ticks = rad_to_ticks(q_target, self.offset, self.dir)
        self.bus.sync_write_positions({sid: ticks[i] for i, sid in enumerate(self.ids)})

    def torque(self, on):
        self.bus.torque(self.ids, on)

    def poll_health(self):
        """One servo per call (keeps the loop short) + the F3 battery every 12 calls."""
        sid = self.ids[self._health_i % 12]
        h = self.bus.read_health(sid)
        if h is not None:
            self.health[sid] = h
        if self._health_i % 12 == 0:
            b = self.f3.analog()
            if b is not None:
                self.battery = b
        self._health_i += 1

    def close(self):
        for d in (self.bus, self.f3):
            try:
                d.close()
            except Exception:
                pass


class MockHardware(Hardware):
    """Servos follow their targets with a first-order lag; IMU level. For testing without a robot."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.ids = [cfg["joints"][j]["id"] for j in JOINTS]
        self.q = Q_STAND * 0.3
        self.qd = np.zeros(12)
        self.target = self.q.copy()
        self.torque_on = False
        self.last_t = time.monotonic()
        self.health, self.battery = {}, (11.8, 1.5, 0)

    def check_servos(self):
        pass

    def read(self):
        t = time.monotonic()
        dt = min(t - self.last_t, 0.05)
        self.last_t = t
        if self.torque_on:
            qn = self.q + (self.target - self.q) * min(1.0, dt / 0.03)
            self.qd = (qn - self.q) / max(dt, 1e-3)
            self.q = qn
        s = State()
        s.t, s.q, s.qd = t, self.q.copy(), self.qd.copy()
        s.ang_vel, s.gravity, s.yaw_deg, s.bus_errors = np.zeros(3), np.array([0.0, 0.0, -1.0]), 0.0, 0
        return s

    def write(self, q_target):
        self.target = np.asarray(q_target, dtype=float).copy()

    def torque(self, on):
        self.torque_on = on

    def poll_health(self):
        self.health = {sid: (11.8, 35, 0.2) for sid in self.ids}

    def close(self):
        pass
