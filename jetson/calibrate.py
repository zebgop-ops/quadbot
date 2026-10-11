#!/usr/bin/env python3
"""Guided calibration; results are written back into config.json.

  python3 calibrate.py scan     # every joint's servo answers?
  python3 calibrate.py zero     # joint zero offsets (legs hanging straight down)
  python3 calibrate.py dirs     # which way each servo counts
  python3 calibrate.py imu      # F3 axis signs (gravity + gyro)
  python3 calibrate.py show     # live joint angles / gravity / gyro, torque off (sanity check)
  python3 calibrate.py stand    # robot LIFTED: ease to the stand pose, report tracking error, sit, off
  python3 calibrate.py head     # neck/jaw: trim the neutral pulse, check which way each servo turns

Order on a new robot: scan -> zero -> dirs -> imu -> show -> stand -> head -> run_policy.py --stand-only.
"""
import json
import math
import os
import sys
import time

import numpy as np

from controller import JOINTS, POSITIVE_MOTION, Q_SIT, Q_STAND, gravity_from_attitude, gyro_to_body, ticks_to_rad

HERE = os.path.dirname(os.path.abspath(__file__))
CFG = os.path.join(HERE, "config.json")


def load():
    with open(CFG) as f:
        return json.load(f)


def save(cfg):
    with open(CFG + ".tmp", "w") as f:
        json.dump(cfg, f, indent=2)
    os.rename(CFG + ".tmp", CFG)
    print("saved", CFG)


def bus_of(cfg):
    from sts import STSBus
    return STSBus(cfg["servo_port"])


def f3_of(cfg):
    from msp import MSP
    return MSP(cfg["f3_port"])


def ids_of(cfg):
    return [cfg["joints"][j]["id"] for j in JOINTS]


def read_ticks(bus, ids, n=5):
    """Median of n sync reads -> {id: ticks}."""
    acc = {i: [] for i in ids}
    for _ in range(n):
        for i, r in bus.sync_read_pos_speed(ids).items():
            if r is not None:
                acc[i].append(r[0])
        time.sleep(0.01)
    return {i: (int(np.median(v)) if v else None) for i, v in acc.items()}


def wait(msg):
    try:
        input("\n" + msg + "\n  [Enter] ")
    except EOFError:
        pass


def cmd_scan(cfg):
    bus = bus_of(cfg)
    ticks = read_ticks(bus, ids_of(cfg), 3)
    for j in JOINTS:
        i = cfg["joints"][j]["id"]
        t = ticks[i]
        print("%-13s ID %2d  %s" % (j, i, "NO REPLY" if t is None else "ticks %4d" % t))
    bus.close()


def cmd_zero(cfg):
    bus = bus_of(cfg)
    bus.torque(ids_of(cfg), False)
    wait("Robot on a stand, torque is OFF. Hang every leg straight down:\n"
         "  - hip roll: leg vertical seen from the front\n"
         "  - hip pitch: leg vertical seen from the side\n"
         "  - knee: thigh and shin in one straight line\n"
         "Use a square against the body if you can. Hold them there.")
    ticks = read_ticks(bus, ids_of(cfg), 15)
    for j in JOINTS:
        t = ticks[cfg["joints"][j]["id"]]
        if t is None:
            print("%-13s no reply - skipped" % j)
            continue
        off = t - 2048
        flag = "" if abs(off) < 170 else "  <-- %.0f deg from 2048: horn a spline tooth out? (refit if > 1 tooth)" % (off * 360 / 4096.0)
        cfg["joints"][j]["offset"] = t
        print("%-13s offset %4d (%+5.1f deg)%s" % (j, t, off * 360 / 4096.0, flag))
    save(cfg)
    bus.close()


def cmd_dirs(cfg):
    bus = bus_of(cfg)
    bus.torque(ids_of(cfg), False)
    joints = [a for a in sys.argv[2:] if a in JOINTS] or JOINTS
    for j in joints:
        i = cfg["joints"][j]["id"]
        t0 = read_ticks(bus, [i])[i]
        wait("%s: move it so that the %s (about 20-30 deg)." % (j, POSITIVE_MOTION[j]))
        t1 = read_ticks(bus, [i])[i]
        if t0 is None or t1 is None or abs(t1 - t0) < 60:
            print("  no clear movement (%s -> %s) - skipped, run again: calibrate.py dirs %s" % (t0, t1, j))
            continue
        cfg["joints"][j]["dir"] = 1 if t1 > t0 else -1
        print("  %+d ticks -> dir %+d" % (t1 - t0, cfg["joints"][j]["dir"]))
    save(cfg)
    bus.close()


def _record(f3, seconds):
    t_end, rows = time.time() + seconds, []
    while time.time() < t_end:
        a, g = f3.attitude(), f3.gyro()
        if a and g:
            rows.append((time.time(),) + tuple(a) + tuple(g))
        time.sleep(0.01)
    return np.array(rows)


def _unwrap_deg(x):
    return np.degrees(np.unwrap(np.radians(x)))


def cmd_imu(cfg):
    f3 = f3_of(cfg)
    im = cfg["imu"]
    wait("Robot standing LEVEL on the floor, still.")
    lvl = _record(f3, 1.0)
    r0, p0 = lvl[:, 1].mean(), lvl[:, 2].mean()
    print("  level: roll %.1f pitch %.1f  gyro %s deg/s" % (r0, p0, np.round(lvl[:, 4:7].mean(0), 1)))
    wait("Tilt the NOSE DOWN about 30 deg and hold it.")
    a = _record(f3, 1.0)
    dp = a[:, 2].mean() - p0
    im["pitch_sign"] = -1 if dp > 0 else 1          # nose down must read as negative (nose-up positive)
    print("  pitch changed %+.1f deg -> pitch_sign %+d" % (dp, im["pitch_sign"]))
    wait("Level again, then tilt the RIGHT side DOWN about 30 deg and hold it.")
    a = _record(f3, 1.0)
    dr = a[:, 1].mean() - r0
    im["roll_sign"] = 1 if dr > 0 else -1           # right side down must read as positive
    print("  roll changed %+.1f deg -> roll_sign %+d" % (dr, im["roll_sign"]))
    if abs(dp) < 10 or abs(dr) < 10:
        print("  tilts too small (<10 deg) - run calibrate.py imu again")
        return

    # gyro: rock about each body axis; match the FC gyro axis that responds, and its sign
    axes, signs, used = [0, 0, 0], [1, 1, 1], set()
    moves = [(0, "ROCK it side to side (right side down / up) for 4 s", 1),
             (1, "ROCK it nose down / nose up for 4 s", 2),
             (2, "TURN it left and right on the floor (yaw) for 4 s", 3)]
    for body_axis, msg, att_col in moves:
        wait(msg + " - start moving, then press Enter.")
        d = _record(f3, 4.0)
        tt = d[:, 0]
        ang = _unwrap_deg(d[:, att_col])
        rate = np.gradient(ang, tt)
        if body_axis == 0:
            ref = im["roll_sign"] * rate            # + = rolling right side down = +wx
        elif body_axis == 1:
            ref = -im["pitch_sign"] * rate          # + = nose going down = +wy (y points left)
        else:
            ref = -rate                             # FC yaw is a heading (clockwise +); +wz = turning left
        best, best_c = None, 0.0
        for k in range(3):
            if k in used:
                continue
            c = np.corrcoef(d[:, 4 + k], ref)[0, 1]
            if best is None or abs(c) > abs(best_c):
                best, best_c = k, c
        used.add(best)
        axes[body_axis], signs[body_axis] = best, (1 if best_c > 0 else -1)
        print("  body %s <- FC gyro axis %d x %+d (correlation %.2f)%s" % ("xyz"[body_axis], best, signs[body_axis], best_c,
                                                                            "" if abs(best_c) > 0.7 else "  <-- weak, repeat"))
    im["gyro_axes"], im["gyro_signs"], im["calibrated"] = axes, signs, True
    save(cfg)
    f3.close()


def cmd_show(cfg):
    from hardware import Hardware
    hw = Hardware(cfg)
    hw.torque(False)
    print("torque OFF - move joints / tilt the robot; Ctrl-C to stop\n")
    try:
        while True:
            s = hw.read()
            line = " ".join("%s %+5.2f" % (j.replace("_hip", "").replace("_", ""), q) for j, q in zip(JOINTS, s.q))
            sys.stdout.write("\r%s | g %+.2f %+.2f %+.2f | w %+.2f %+.2f %+.2f " % ((line,) + tuple(s.gravity) + tuple(s.ang_vel)))
            sys.stdout.flush()
            time.sleep(0.1)
    except KeyboardInterrupt:
        print()
    hw.close()


def cmd_stand(cfg):
    from hardware import Hardware
    from run_policy import ramp
    hw = Hardware(cfg)
    wait("LIFT the robot (legs free). It will ease into the stand pose, hold 3 s, then sit and relax.")
    s = hw.read()
    hw.write(s.q)
    hw.torque(True)
    try:
        ramp(hw, s.q, Q_STAND, 3.0, 50)
        errs = []
        for _ in range(150):
            hw.write(Q_STAND)
            errs.append(hw.read().q - Q_STAND)
            time.sleep(0.02)
        e = np.degrees(np.mean(errs, axis=0))
        for j, v in zip(JOINTS, e):
            print("%-13s %+5.1f deg%s" % (j, v, "" if abs(v) < 3 else "  <-- check offset / dir / binding"))
        ramp(hw, Q_STAND, Q_SIT, 2.0, 50)
    finally:
        hw.torque(False)
        hw.close()


def _hold(joint, deg, seconds=1.0, hz=50):
    joint.set(deg)
    for _ in range(int(seconds * hz)):
        joint.update(1.0 / hz)
        time.sleep(1.0 / hz)


def cmd_head(cfg):
    from head import Head
    head = Head(cfg)
    jobs = [("neck", head.neck, "the head is LEVEL: face vertical, eyes looking straight ahead", 15.0,
             "Did the NOSE go DOWN?"),
            ("jaw", head.jaw, "the mouth is JUST closed: front teeth touching lightly, not pressing", 10.0,
             "Did the mouth OPEN?")]
    try:
        for name, j, level_msg, test_deg, question in jobs:
            c = cfg["head"][name]
            print("\n--- %s (PWM channel %d) ---" % (name, c["channel"]))
            _hold(j, 0.0)
            while True:
                try:
                    a = input("Trim until %s.\n  now %d us - type +N / -N (us), or Enter to accept: " % (level_msg, j.center)).strip()
                except EOFError:
                    a = ""
                if not a:
                    break
                try:
                    j.center = int(min(2200, max(800, j.center + int(a))))
                except ValueError:
                    continue
                _hold(j, 0.0, 0.3)
            c["center_us"] = j.center
            _hold(j, test_deg)
            try:
                ok = input("%s [y/n] " % question).strip().lower().startswith("y")
            except EOFError:
                ok = True
            _hold(j, 0.0)
            if not ok:
                j.dir = -j.dir
                c["dir"] = j.dir
                print("  direction flipped - checking again")
                _hold(j, test_deg)
                _hold(j, 0.0)
            print("  %s: center %d us, dir %+d" % (name, c["center_us"], c["dir"]))
        cfg["head"]["calibrated"] = True
        save(cfg)
    finally:
        head.off()


def main():
    cmds = {"scan": cmd_scan, "zero": cmd_zero, "dirs": cmd_dirs, "imu": cmd_imu, "show": cmd_show, "stand": cmd_stand,
            "head": cmd_head}
    if len(sys.argv) < 2 or sys.argv[1] not in cmds:
        print(__doc__)
        return
    cmds[sys.argv[1]](load())


if __name__ == "__main__":
    main()
