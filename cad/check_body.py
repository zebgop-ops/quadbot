"""Section properties of the main body (with the deck bolted on) along its length."""
import numpy as np

import geom as g


def props(cs):
    A = Iy = Iz = Cy = Cz = 0.0
    for p in cs.to_polygons():
        p = np.asarray(p)
        u, v = p[:, 0], p[:, 1]
        un, vn = np.roll(u, -1), np.roll(v, -1)
        cr = u * vn - un * v
        A += cr.sum() / 2
        Cy += ((u + un) * cr).sum() / 6
        Cz += ((v + vn) * cr).sum() / 6
        Iy += ((u ** 2 + u * un + un ** 2) * cr).sum() / 12
        Iz += ((v ** 2 + v * vn + vn ** 2) * cr).sum() / 12
    cu, cv = Cy / A, Cz / A
    return A, Iy - A * cu ** 2, Iz - A * cv ** 2


def run(with_deck=True, step=3.0):
    m = g.part_body() + (g.part_deck() if with_deck else g.part_body() ^ g.box(0, 0.01, 0, 0.01, 0, 0.01))
    r = m.rotate([0, 90, 0])            # body x -> -z ; slice u = body z, v = body y
    out = []
    for x in np.arange(-118, 119, step):
        A, Ivert, Ilat = props(r.slice(-x))
        out.append((x, A, Ivert, Ilat))
    return out


if __name__ == "__main__":
    for x, A, Iv, Il in run():
        print(f"x {x:7.1f}  area {A:6.0f}  I_vertical {Iv / 1e3:7.1f}e3  I_lateral {Il / 1e3:7.1f}e3")
