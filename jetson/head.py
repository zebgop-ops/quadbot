"""Neck + jaw: 2 x ZOSKAY DS3218 on the Jetson Nano's hardware PWM (pin 32 = pwmchip0/pwm0,
pin 33 = pwmchip0/pwm2; enable both with `sudo /opt/nvidia/jetson-io/jetson-io.py`).

Angles follow the CAD: neck + = nose down (range -60..55 deg), jaw + = mouth opens (0..90 deg).
pulse_us = center_us + dir * angle * us_per_deg    (DS3218 180 deg: 2000 us / 180 deg)
Moves are slew-limited: the head is ~150 g hanging 70 mm in front of the neck axis.
"""
import os
import time


class SysfsPWM(object):
    def __init__(self, chip, channel, period_ns=20000000):
        self.base = os.path.join(chip, "pwm%d" % channel)
        if not os.path.isdir(self.base):
            self._w(os.path.join(chip, "export"), channel)
            for _ in range(50):                    # udev needs a moment to fix permissions
                if os.access(os.path.join(self.base, "enable"), os.W_OK):
                    break
                time.sleep(0.02)
        self._w(os.path.join(self.base, "period"), period_ns)
        self.last = None

    @staticmethod
    def _w(path, value):
        with open(path, "w") as f:
            f.write(str(int(value)))

    def pulse_us(self, us):
        ns = int(us * 1000)
        if ns != self.last:
            self._w(os.path.join(self.base, "duty_cycle"), ns)
            if self.last is None:
                self._w(os.path.join(self.base, "enable"), 1)
            self.last = ns

    def off(self):
        """Stop the pulses: the servo goes limp."""
        try:
            self._w(os.path.join(self.base, "enable"), 0)
        except (IOError, OSError):
            pass
        self.last = None


class Joint(object):
    def __init__(self, pwm, c):
        self.pwm = pwm
        self.center, self.dir, self.k = c["center_us"], c["dir"], c["us_per_deg"]
        self.lo, self.hi = c["min_deg"], c["max_deg"]
        self.speed = c["max_speed_dps"]
        self.angle = c.get("rest_deg", 0.0)
        self.target = self.angle

    def pulse(self, deg):
        return min(2500.0, max(500.0, self.center + self.dir * deg * self.k))

    def update(self, dt):
        step = self.speed * dt
        self.angle += max(-step, min(step, self.target - self.angle))
        self.pwm.pulse_us(self.pulse(self.angle))

    def set(self, deg):
        self.target = min(self.hi, max(self.lo, deg))


class _NullPWM(object):
    def __init__(self):
        self.last = None

    def pulse_us(self, us):
        self.last = int(us * 1000)

    def off(self):
        self.last = None


class Head(object):
    """neck/jaw targets in degrees; call update(dt) every control step."""

    def __init__(self, cfg, mock=False):
        h = cfg["head"]
        self.cfg = h
        mk = (lambda ch: _NullPWM()) if mock else (lambda ch: SysfsPWM(h["pwm_chip"], ch))
        self.neck = Joint(mk(h["neck"]["channel"]), h["neck"])
        self.jaw = Joint(mk(h["jaw"]["channel"]), h["jaw"])
        self.neck_cmd = 0.0          # where the user wants the head (deg, + = nose down)
        self.chomp_t = None

    def chomp(self):
        self.chomp_t = 0.0

    def update(self, dt, body_pitch_down_deg=0.0, walking=False):
        """body_pitch_down_deg: body nose-down tilt from the IMU; with 'stabilize' the head stays
        level (gaze steady) by counter-rotating the neck. While walking the neck is kept within
        walk_neck_limit_deg (the policy was trained with the head fixed at neutral)."""
        neck = self.neck_cmd
        if self.cfg.get("stabilize", True):
            neck -= body_pitch_down_deg
        if walking:
            lim = self.cfg["walk_neck_limit_deg"]
            neck = max(-lim, min(lim, neck))
        self.neck.set(neck)
        if self.chomp_t is not None:        # open, then snap shut
            self.chomp_t += dt
            if self.chomp_t < 0.35:
                self.jaw.set(self.cfg["chomp_open_deg"])
            else:
                self.jaw.set(0.0)
                if self.chomp_t > 0.8:
                    self.chomp_t = None
        self.neck.update(dt)
        self.jaw.update(dt)

    def rest(self, seconds=1.0, hz=50):
        """Ease to neutral head / closed mouth."""
        self.neck_cmd, self.chomp_t = 0.0, None
        self.jaw.set(0.0)
        for _ in range(int(seconds * hz)):
            self.update(1.0 / hz)
            time.sleep(1.0 / hz)

    def off(self):
        self.neck.pwm.off()
        self.jaw.pwm.off()
