"""Collision sweep for the neck and jaw."""
import itertools

import numpy as np

import geom as g

BODY = g.union([g.part_body(), g.part_deck()]
               + [g.xf(g.servo_case(), g.T_hip(l)) for l in g.LEGS])


def hits(neck, jaw):
    st, hd, jw = g.head_groups(neck, jaw)
    grp = {"body": BODY + st, "head": hd, "jaw": jw}
    out = []
    for a, b in itertools.combinations(grp, 2):
        v = (grp[a] ^ grp[b]).volume()
        if v > 1.0:
            out.append(f"{a}/{b}:{v:.0f}")
    return out


def rng(fn, lo, hi, step=5):
    ok = {q for q in np.arange(lo, hi + 1, step) if not fn(q)}
    a = b = 0
    while a - step in ok:
        a -= step
    while b + step in ok:
        b += step
    return a, b, fn(a - step), fn(b + step)


print("rest:", hits(0, 0) or "clear")
print("neck (+ = nose down):", rng(lambda q: hits(q, 0), -90, 90))
print("jaw  (+ = open):     ", rng(lambda q: hits(0, q), -30, 90))
print("jaw open 40 across neck range:", rng(lambda q: hits(q, 40), -90, 90))
