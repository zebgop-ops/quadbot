"""Interference check: sweep joint angles and report collisions between rigid groups."""
import itertools
import sys

import numpy as np

import geom as g

TOL = 1.0  # mm^3 of overlap tolerated (coplanar horn contact etc.)


def rigid_groups():
    case = g.servo_case() + g.servo_plugs()
    body = g.union([g.part_body(), g.part_deck(), *g.head_groups(0, 0)]
                   + [g.xf(case, g.T_hip(l)) for l in g.LEGS])
    parts = {
        False: (g.part_hip_link(False)),
        True: (g.part_hip_link(True)),
    }
    return body, case, parts


BODY, CASE, LINK1 = rigid_groups()
FEMUR = g.part_femur()
TIBIA = {True: g.part_tibia(True) + g.part_foot(True),
         False: g.part_tibia(False) + g.part_foot(False)}


def leg_groups(leg, q1, q2, q3):
    F = g.leg_frames(leg, q1, q2, q3)
    mirror = g.LEGS[leg][3]
    G1 = g.xf(LINK1[mirror], F["L1"]) + g.xf(CASE, F["B"])
    G2 = g.xf(FEMUR, F["FEMUR"]) + g.xf(CASE, F["C"])
    G3 = g.xf(TIBIA[leg[0] == "L"], F["TIBIA"])
    return G1, G2, G3


def collisions(leg, q1, q2, q3):
    G1, G2, G3 = leg_groups(leg, q1, q2, q3)
    names = {"body": BODY, "link1": G1, "femur": G2, "tibia": G3}
    hits = []
    for a, b in itertools.combinations(names, 2):
        v = (names[a] ^ names[b]).volume()
        if v > TOL:
            hits.append(f"{a}/{b}:{v:.0f}")
    return hits


def free_range(leg, joint, lo, hi, step, base=(0.0, 0.0, 0.0)):
    ok = []
    for q in np.arange(lo, hi + 1e-9, step):
        qs = list(base)
        qs[joint] = q
        ok.append((q, collisions(leg, *qs)))
    return ok


def summarize(rows):
    good = [q for q, h in rows if not h]
    bad = [(q, h) for q, h in rows if h]
    return good, bad


if __name__ == "__main__":
    leg = sys.argv[1] if len(sys.argv) > 1 else "LF"
    print("zero pose:", collisions(leg, 0, 0, 0) or "clear")
    for j, (name, lo, hi, base) in enumerate([
        ("hip roll", -60, 60, (0, 0, 0)),
        ("hip pitch", -150, 150, (0, 0, 0)),
        ("knee", -170, 170, (0, 0, 0)),
    ]):
        rows = free_range(leg, j, lo, hi, 5, base)
        good, bad = summarize(rows)
        # contiguous range around 0
        zero_ok = [q for q in good]
        qs = sorted(zero_ok)
        lo_ok = hi_ok = 0
        s = set(round(q) for q in qs)
        while round(lo_ok - 5) in s:
            lo_ok -= 5
        while round(hi_ok + 5) in s:
            hi_ok += 5
        first_bad = [b for b in bad if b[0] in (lo_ok - 5, hi_ok + 5)]
        print(f"{name:10s} free {lo_ok:+.0f}..{hi_ok:+.0f} deg   first hits: {first_bad}")
    # random combos inside a working envelope
    rng = np.random.default_rng(0)
    n, fails = 400, []
    for _ in range(n):
        k = rng.uniform(-40, 140)
        q = (rng.uniform(-30, 30), rng.uniform(-90, 90), k if leg[0] == "L" else -k)
        h = collisions(leg, *q)
        if h:
            fails.append((tuple(round(x) for x in q), h))
    print(f"random combos in roll±30 pitch±90 knee -40..140 (fold side): {len(fails)}/{n} collide")
    for f in fails[:15]:
        print("  ", f)
