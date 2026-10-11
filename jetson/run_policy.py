#!/usr/bin/env python3
"""Run the trained 'cat' locomotion policy on the quadbot (Jetson Nano, 50 Hz).

  python3 run_policy.py                  # stand up, walk with the keyboard / gamepad
  python3 run_policy.py --stand-only     # stand up and hold the stand pose (no policy)
  python3 run_policy.py --mock           # no robot: fake servos + level IMU (test the software)

Keys:  w/s forward/back   a/d left/right   q/e turn left/right   space = stop   x or Ctrl-C = sit & quit
       r/f head up/down   g head level   c chomp        (--no-head: leave the neck and jaw alone)
The robot starts in whatever pose it is in, eases into the stand pose over 3 s, then hands over to
the policy. On exit it eases down to a sit and switches torque off. If it tips past max_tilt_deg,
a servo stops answering or overheats, or the battery sags, torque goes off immediately.
"""
import argparse
import csv
import os
import select
import struct
import sys
import threading
import time

import numpy as np

from controller import JOINTS, Q_SIT, Q_STAND, Controller, Policy, load_config, stand_limits
from hardware import Hardware, HardwareFault, MockHardware
from head import Head

HERE = os.path.dirname(os.path.abspath(__file__))


class Commands(object):
    """Velocity command from the keyboard (terminal) and/or a Linux joystick, slew-limited."""

    def __init__(self, cfg):
        self.c = cfg["commands"]
        self.js_cfg = cfg.get("joystick", {})
        self.target = np.zeros(3)
        self.value = np.zeros(3)
        self.quit = False
        self.lock = threading.Lock()
        self.js_axes = {}
        self.js_rest = {}
        self.neck = 0.0             # head command, deg (+ = nose down)
        self.neck_step = cfg.get("head", {}).get("neck_step_deg", 10)
        self.neck_range = (cfg.get("head", {}).get("neck", {}).get("min_deg", -55),
                           cfg.get("head", {}).get("neck", {}).get("max_deg", 50))
        self.chomp = False
        self.js_active = False

    def _clamp(self):
        for i, k in enumerate(("vx", "vy", "wz")):
            self.target[i] = min(max(self.target[i], self.c[k][0]), self.c[k][1])

    def key(self, ch):
        with self.lock:
            c = self.c
            if ch == "w": self.target[0] += c["vx_step"]
            elif ch == "s": self.target[0] -= c["vx_step"]
            elif ch == "a": self.target[1] += c["vy_step"]
            elif ch == "d": self.target[1] -= c["vy_step"]
            elif ch == "q": self.target[2] += c["wz_step"]
            elif ch == "e": self.target[2] -= c["wz_step"]
            elif ch == " ": self.target[:] = 0.0
            elif ch == "r": self.neck = max(self.neck_range[0], self.neck - self.neck_step)
            elif ch == "f": self.neck = min(self.neck_range[1], self.neck + self.neck_step)
            elif ch == "g": self.neck = 0.0
            elif ch == "c": self.chomp = True
            elif ch in ("x", "\x03"): self.quit = True
            self._clamp()

    def start_keyboard(self):
        if not sys.stdin.isatty():
            return None
        import termios
        import tty
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        tty.setcbreak(fd)

        def loop():
            while not self.quit:
                r, _, _ = select.select([sys.stdin], [], [], 0.1)
                if r:
                    self.key(sys.stdin.read(1))
        threading.Thread(target=loop, daemon=True).start()
        return lambda: termios.tcsetattr(fd, termios.TCSADRAIN, old)

    def start_joystick(self):
        dev = self.js_cfg.get("device")
        if not dev or not os.path.exists(dev):
            return
        js = self.js_cfg

        def loop():
            with open(dev, "rb") as f:
                while not self.quit:
                    ev = f.read(8)
                    if len(ev) < 8:
                        break
                    _t, value, typ, num = struct.unpack("<IhBB", ev)
                    v = value / 32767.0
                    if typ & 0x80:            # initial state on open: that is the axis' rest position
                        if typ & 0x02:
                            self.js_rest[num] = v
                        continue
                    if typ & 0x02:            # axis moved
                        self.js_axes[num] = v - self.js_rest.get(num, 0.0)
                        if abs(self.js_axes[num]) > 0.3:
                            self.js_active = True      # the stick only takes over once it is really moved
                        if self.js_active:
                            self._from_joystick(js)
                    elif typ & 0x01 and value:  # buttons
                        with self.lock:
                            if num == js.get("button_chomp", 1):
                                self.chomp = True
                            else:                  # any other button = stop
                                self.target[:] = 0.0
        threading.Thread(target=loop, daemon=True).start()
        print("joystick:", dev)

    def _from_joystick(self, js):
        def ax(name, inv):
            v = self.js_axes.get(js[name], 0.0) * (-1 if js.get(inv) else 1)
            return 0.0 if abs(v) < js.get("deadzone", 0.08) else v
        with self.lock:
            for i, (k, a, inv) in enumerate((("vx", "axis_vx", "invert_vx"), ("vy", "axis_vy", "invert_vy"),
                                             ("wz", "axis_wz", "invert_wz"))):
                v = ax(a, inv)
                self.target[i] = v * (self.c[k][1] if v > 0 else -self.c[k][0])
            if "axis_neck" in js:
                v = ax("axis_neck", "invert_neck")
                self.neck = v * (self.neck_range[1] if v > 0 else -self.neck_range[0])

    def update(self, dt):
        """Slew toward the target: linear speeds at `accel` m/s^2, yaw rate at 4x that in rad/s^2."""
        with self.lock:
            lim = np.array([self.c["accel"], self.c["accel"], 4.0 * self.c["accel"]]) * dt
            self.value += np.clip(self.target - self.value, -lim, lim)
            return self.value.copy()


def smoothstep(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)


def ramp(hw, q_from, q_to, seconds, hz, cmds=None, on_tick=None):
    """Ease the servo targets from q_from to q_to (reads state to keep the bus watchdogs happy)."""
    n = max(1, int(seconds * hz))
    t_next = time.monotonic()
    for k in range(n + 1):
        if cmds is not None and cmds.quit:
            return False
        hw.write(q_from + (q_to - q_from) * smoothstep(k / float(n)))
        hw.read()
        if on_tick is not None:
            on_tick(1.0 / hz)
        t_next += 1.0 / hz
        time.sleep(max(0.0, t_next - time.monotonic()))
    return True


def safety_check(s, hw, cfg):
    sf = cfg["safety"]
    tilt = np.degrees(np.arccos(np.clip(-s.gravity[2], -1.0, 1.0)))
    if tilt > sf["max_tilt_deg"]:
        raise HardwareFault("tipped over: tilt %.0f deg" % tilt)
    for sid, (v, temp, _a) in hw.health.items():
        if temp >= sf["max_servo_temp_c"]:
            raise HardwareFault("servo %d at %d C" % (sid, temp))
    if hw.battery is not None and 3.0 < hw.battery[0] < sf["min_battery_v"]:
        raise HardwareFault("battery low: %.1f V" % hw.battery[0])
    return tilt


def check_clocks():
    """Warn if the CPU may clock down between control steps (Jetson: run jetson_clocks)."""
    try:
        base = "/sys/devices/system/cpu/cpu0/cpufreq/"
        lo = int(open(base + "scaling_min_freq").read())
        hi = int(open(base + "cpuinfo_max_freq").read())
        if lo < hi:
            print("WARNING: CPU clocks are not pinned (min %d < max %d kHz): the loop will be slow and jittery."
                  " Run 'sudo jetson_clocks' (setup_jetson.sh installs it as a boot service)." % (lo, hi))
    except (IOError, OSError, ValueError):
        pass


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=os.path.join(HERE, "config.json"))
    ap.add_argument("--policy", default=None, help="override the policy file (.npz or .onnx)")
    ap.add_argument("--mock", action="store_true", help="fake hardware")
    ap.add_argument("--stand-only", action="store_true")
    ap.add_argument("--no-head", action="store_true", help="don't drive the neck/jaw servos")
    ap.add_argument("--on-stand", action="store_true",
                    help="mounted on the test stand: keep hip roll within 15 deg inward so no leg can reach the column")
    ap.add_argument("--seconds", type=float, default=0.0, help="stop after this long (0 = until x)")
    ap.add_argument("--vx", type=float, default=0.0, help="start with this forward command (m/s)")
    ap.add_argument("--log", default=None, help="CSV log path (default logs/run_<time>.csv)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if not args.mock:
        check_clocks()
    if not cfg["imu"].get("calibrated") and not args.mock:
        print("WARNING: IMU signs not calibrated yet - run calibrate.py imu first.")
    hz = cfg["loop_hz"]
    dt = 1.0 / hz
    policy = None if args.stand_only else Policy(os.path.join(HERE, args.policy or cfg["policy"]))
    ctrl = Controller(policy, cfg["action_scale"], cfg["action_filter_alpha"]) if policy else None

    cmds = Commands(cfg)
    cmds.target[0] = args.vx
    restore = cmds.start_keyboard()
    cmds.start_joystick()

    log_path = args.log or os.path.join(HERE, "logs", time.strftime("run_%Y%m%d_%H%M%S.csv"))
    if not os.path.isdir(os.path.dirname(log_path)):
        os.makedirs(os.path.dirname(log_path))
    logf = open(log_path, "w")
    log = csv.writer(logf)
    log.writerow(["t", "loop_ms", "vx_cmd", "vy_cmd", "wz_cmd", "wx", "wy", "wz", "gx", "gy", "gz", "tilt_deg"]
                 + ["q_" + j for j in JOINTS] + ["qd_" + j for j in JOINTS] + ["a_" + j for j in JOINTS]
                 + ["tgt_" + j for j in JOINTS] + ["batt_v", "batt_a", "servo_max_c", "neck_deg", "jaw_deg"])

    hw = MockHardware(cfg) if args.mock else Hardware(cfg)
    head = None
    if cfg.get("head", {}).get("enabled") and not args.no_head:
        if not cfg["head"].get("calibrated") and not args.mock:
            print("WARNING: head not calibrated yet - run calibrate.py head first.")
        head = Head(cfg, mock=args.mock)

    def head_tick(dt_, pitch_down=0.0, walking=False):
        if head is None:
            return
        head.neck_cmd = cmds.neck
        if cmds.chomp:
            head.chomp()
            cmds.chomp = False
        head.update(dt_, pitch_down, walking)
    q_now = None
    torque_on = False
    try:
        hw.check_servos()
        s = hw.read()
        tilt = np.degrees(np.arccos(np.clip(-s.gravity[2], -1.0, 1.0)))
        print("start pose read; body tilt %.0f deg" % tilt)
        if tilt > 35:
            raise HardwareFault("robot is not upright (%.0f deg) - set it on its feet first" % tilt)
        q_now = s.q.copy()
        hw.write(q_now)               # hold where it is, then switch torque on: no jump
        hw.torque(True)
        torque_on = True
        print("standing up...")
        head_tick(dt)                 # head pulses start at neutral
        if not ramp(hw, q_now, Q_STAND, 3.0, hz, cmds, on_tick=head_tick):
            raise KeyboardInterrupt
        q_now = Q_STAND.copy()
        if ctrl:
            ctrl.reset()
        print("running (%s). w/s a/d q/e, space = stop, x = sit & quit%s"
              % ("stand only" if not ctrl else policy.kind, " | head r/f/g, chomp c" if head else ""))

        t0 = time.monotonic()
        t_next = t0
        k = 0
        worst = 0.0
        while not cmds.quit:
            t_loop = time.monotonic()
            s = hw.read()
            tilt = safety_check(s, hw, cfg)
            cmd = cmds.update(dt)
            if ctrl:
                target = ctrl.step(s.ang_vel, s.gravity, cmd, s.q, s.qd)
                action = ctrl.last_action
                if args.on_stand:
                    target = np.clip(target, *stand_limits())
            else:
                target, action = Q_STAND, np.zeros(12)
            hw.write(target)
            q_now = target
            pitch_down = float(np.degrees(np.arcsin(np.clip(s.gravity[0], -1.0, 1.0))))
            head_tick(dt, pitch_down, walking=bool(np.abs(cmd).max() > 0.05))
            hw.poll_health()
            loop_ms = (time.monotonic() - t_loop) * 1000.0
            worst = max(worst, loop_ms)
            b = hw.battery or (0.0, 0.0, 0)
            tmax = max([h[1] for h in hw.health.values()] or [0])
            log.writerow(["%.4f" % (s.t - t0), "%.2f" % loop_ms] + ["%.3f" % v for v in cmd]
                         + ["%.4f" % v for v in np.concatenate([s.ang_vel, s.gravity])] + ["%.1f" % tilt]
                         + ["%.4f" % v for v in np.concatenate([s.q, s.qd, action, target])]
                         + ["%.2f" % b[0], "%.2f" % b[1], tmax,
                            "%.1f" % (head.neck.angle if head else 0.0), "%.1f" % (head.jaw.angle if head else 0.0)])
            if k % hz == 0:
                sys.stdout.write("\rcmd vx %+.2f vy %+.2f wz %+.2f | tilt %4.1f | batt %4.1f V %4.2f A | servo max %d C | loop %4.1f ms (worst %4.1f)   "
                                 % (cmd[0], cmd[1], cmd[2], tilt, b[0], b[1], tmax, loop_ms, worst))
                sys.stdout.flush()
            k += 1
            if args.seconds and time.monotonic() - t0 > args.seconds:
                break
            t_next += dt
            slack = t_next - time.monotonic()
            if slack > 0:
                time.sleep(slack)
            else:
                t_next = time.monotonic()          # overran: don't try to catch up
        print("\nsitting down...")
        if head is not None:
            cmds.neck = 0.0
        ramp(hw, np.asarray(q_now), Q_SIT, 2.0, hz, on_tick=head_tick)
        hw.torque(False)
        torque_on = False
    except HardwareFault as e:
        print("\nFAULT: %s -> torque off" % e)
        if head is not None:
            head.off()
    except KeyboardInterrupt:
        print("\ninterrupted -> sitting down")
        try:
            if torque_on and q_now is not None:
                cmds.neck = 0.0
                ramp(hw, np.asarray(q_now), Q_SIT, 2.0, hz, on_tick=head_tick)
        except Exception:
            pass
    finally:
        try:
            if torque_on:
                hw.torque(False)
        except Exception:
            pass
        cmds.quit = True
        if head is not None:
            head.off()
        hw.close()
        logf.close()
        if restore:
            restore()
        print("log:", log_path)


if __name__ == "__main__":
    main()
