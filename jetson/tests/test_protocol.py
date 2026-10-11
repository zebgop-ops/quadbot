"""python3 -m pytest tests  (or python3 tests/test_protocol.py)"""
import math
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import msp  # noqa: E402
import sts  # noqa: E402
from controller import (JOINTS, Q_MAX, Q_MIN, Q_SIT, Q_STAND, Controller, Policy,  # noqa: E402
                        gravity_from_attitude, gyro_to_body, rad_to_ticks, ticks_to_rad)


def test_sts_packets():
    assert sts.make_packet(1, sts.INST_PING) == bytes([0xFF, 0xFF, 0x01, 0x02, 0x01, 0xFB])   # Feetech manual example
    # read 2 bytes from address 56 of ID 1
    assert sts.make_packet(1, sts.INST_READ, [56, 2]) == bytes([0xFF, 0xFF, 1, 4, 2, 56, 2, 0xBE])
    # status: ID 1, no error, position 2048
    body = bytes([1, 4, 0]) + struct.pack("<H", 2048)
    st = sts.parse_status(b"\x00\xff\xff" + body + bytes([sts.checksum(body)]) + b"\xff")
    assert st == [(1, 0, struct.pack("<H", 2048))]
    assert sts.sign_mag16(0x8000 | 100) == -100 and sts.sign_mag16(100) == 100


def test_sync_read_parse():
    pk = b""
    for sid, p, v in ((1, 2000, 50), (2, 2100, 0x8000 | 30)):
        body = bytes([sid, 6, 0]) + struct.pack("<HH", p, v)
        pk += b"\xff\xff" + body + bytes([sts.checksum(body)])
    got = {s[0]: struct.unpack("<HH", s[2]) for s in sts.parse_status(pk)}
    assert got == {1: (2000, 50), 2: (2100, 0x8000 | 30)}


def test_msp():
    assert msp.make_request(msp.MSP_ATTITUDE) == b"$M<\x00\x6c\x6c"
    payload = struct.pack("<hhh", -123, 45, 270)
    c = len(payload) ^ msp.MSP_ATTITUDE
    for b in payload:
        c ^= b
    frame = b"$M>" + bytes([len(payload), msp.MSP_ATTITUDE]) + payload + bytes([c])
    assert msp.parse_response(b"junk" + frame, msp.MSP_ATTITUDE) == payload


def test_conventions():
    assert np.allclose(gravity_from_attitude(0, 0), [0, 0, -1])
    g = gravity_from_attitude(0, -30)            # nose down -> gravity leans forward in the body
    assert g[0] > 0.49 and abs(g[1]) < 1e-9
    g = gravity_from_attitude(30, 0)             # right side down -> gravity toward -y (right)
    assert g[1] < -0.49
    assert np.allclose(gyro_to_body([10, 20, 30], [0, 1, 2], [1, -1, -1]), np.radians([10, -20, -30]))
    q = np.linspace(-1, 1, 12)
    assert np.allclose(ticks_to_rad(rad_to_ticks(q, 2000, -1), 2000, -1), q)
    assert abs(ticks_to_rad(2048 + 1024, 2048, 1) - math.pi / 2) < 1e-9
    assert np.all(Q_STAND >= Q_MIN) and np.all(Q_STAND <= Q_MAX) and np.all(Q_SIT >= Q_MIN) and np.all(Q_SIT <= Q_MAX)
    assert len(JOINTS) == 12


def test_policy_and_filter():
    here = os.path.join(os.path.dirname(__file__), "..", "models", "quadbot_cat_policy.npz")
    p = Policy(here)
    c = Controller(p)
    tgt = c.step(np.zeros(3), np.array([0, 0, -1.0]), np.zeros(3), Q_STAND, np.zeros(12))
    assert tgt.shape == (12,) and np.all(np.isfinite(tgt))
    a = c.last_action
    assert np.allclose(tgt, np.clip(Q_STAND + 0.4 * 0.25 * a, Q_MIN, Q_MAX))   # first step: filter from the stand pose
    assert np.abs(a).max() < 5.0


def test_head():
    import json
    from head import Head
    cfg = json.load(open(os.path.join(os.path.dirname(__file__), "..", "config.json")))
    h = Head(cfg, mock=True)
    h.update(0.02)
    assert h.neck.pwm.last == 1500000 and h.jaw.pwm.last == 1500000       # starts at neutral
    h.neck_cmd = 40.0
    h.update(0.1)                                                        # slew-limited: 90 deg/s
    assert abs(h.neck.angle - 9.0) < 1e-6
    for _ in range(100):
        h.update(0.02, walking=True)
    assert abs(h.neck.angle - cfg["head"]["walk_neck_limit_deg"]) < 1e-6  # capped while walking
    for _ in range(100):
        h.update(0.02, body_pitch_down_deg=10.0)                         # gaze stabilisation
    assert abs(h.neck.angle - 30.0) < 1e-6
    h.neck_cmd = -200
    for _ in range(200):
        h.update(0.02)
    assert h.neck.angle == cfg["head"]["neck"]["min_deg"]                # clamped to the CAD range
    assert h.neck.pulse(10) == 1500 + 10 * cfg["head"]["neck"]["us_per_deg"]
    h.chomp()
    opened = 0.0
    for _ in range(60):
        h.update(0.02)
        opened = max(opened, h.jaw.angle)
    assert opened > 40 and h.jaw.angle == 0.0                            # opened, then snapped shut


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
