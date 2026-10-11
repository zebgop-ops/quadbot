"""Build every printable part and write print-oriented STLs to ../stl/."""
import os
import sys
import time

import numpy as np
import trimesh

import geom as g

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "stl")

# name: (builder, how many to print, face that goes on the bed, material)
PARTS = {
    "body":              (g.part_body,                            1, "x+", "ABS"),
    "body_deck":         (g.part_deck,                            1, "z-", "ABS"),
    "hump":              (g.part_hump,                            1, "z+", "ABS"),
    "hip_link":          (lambda: g.part_hip_link(False),         2, "y-", "ABS"),
    "hip_link_mirror":   (lambda: g.part_hip_link(True),          2, "y+", "ABS"),
    "femur":             (g.part_femur,                           4, "x+", "ABS"),
    "tibia_left":        (lambda: g.part_tibia(True),             2, "rz50 y-", "ABS"),
    "tibia_right":       (lambda: g.part_tibia(False),            2, "rz-50 y+", "ABS"),
    "foot_left":         (lambda: g.part_foot(True),              2, "x-", "TPU 95A"),
    "foot_right":        (lambda: g.part_foot(False),             2, "x-", "TPU 95A"),
    "neck_base":         (g.part_neck_base,                       1, "z-", "ABS"),
    "head":              (g.part_head,                            1, "z+", "ABS"),
    "ear_cap":           (g.part_ear_cap,                         1, "z-", "ABS"),
    "jaw":               (g.part_jaw,                             1, "z-", "ABS"),
    "fit_test_cradle":   (g.part_fit_test_cradle,                 1, "x-", "ABS"),
    "fit_test_hip_roll": (g.part_fit_test_hip_roll,               1, "x-", "ABS"),
    "stand":             (g.part_stand,                           1, "rz45 z-", "ABS or PETG"),
    "fit_test_bracket":  (g.part_fit_test_bracket,                1, "y-", "ABS"),
}

# rotation (degrees about x, y) that puts the named face on the bed
ORIENT = {"z-": (0, 0), "z+": (180, 0), "y-": (90, 0), "y+": (-90, 0), "x-": (0, -90), "x+": (0, 90)}


def to_trimesh(m):
    mesh = m.to_mesh()
    return trimesh.Trimesh(np.asarray(mesh.vert_properties)[:, :3], np.asarray(mesh.tri_verts), process=False)


def print_oriented(m, face):
    if face.startswith("rz"):
        a, face = face.split()
        m = m.rotate([0, 0, float(a[2:])])
    rx, ry = ORIENT[face]
    m = m.rotate([rx, ry, 0])
    x0, y0, z0, x1, y1, z1 = m.bounding_box()
    return m.translate([-(x0 + x1) / 2, -(y0 + y1) / 2, -z0])


def main(only=None):
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for name, (fn, qty, face, mat) in PARTS.items():
        if only and name not in only:
            continue
        t = time.time()
        m = fn()
        bodies = sorted(m.decompose(), key=lambda p: -p.volume())
        if len(bodies) > 1 and sum(p.volume() for p in bodies[1:]) < 1.0:
            m = bodies[0]                      # drop zero-volume slivers from the rounding pass
        st = m.status()
        if str(st) != "Error.NoError":
            print(f"!! {name}: {st}")
        genus = m.genus()
        pm = print_oriented(m, face)
        tm = to_trimesh(pm)
        tm.export(os.path.join(OUT, f"{name}.stl"))
        x0, y0, z0, x1, y1, z1 = pm.bounding_box()
        vol = m.volume() / 1000.0
        rows.append((name, qty, mat, x1 - x0, y1 - y0, z1 - z0, vol, tm.is_watertight, len(m.decompose()), time.time() - t))
    print(f"{'part':18s} qty  material  {'size on bed (mm)':>22s}  solid cm3  watertight bodies")
    for r in rows:
        print(f"{r[0]:18s} {r[1]:3d}  {r[2]:8s} {r[3]:6.1f} x{r[4]:6.1f} x{r[5]:6.1f}  {r[6]:8.1f}   {str(r[7]):5s}  {r[8]}")
    return rows


if __name__ == "__main__":
    main(sys.argv[1:] or None)
