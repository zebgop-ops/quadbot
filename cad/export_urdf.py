"""Export a URDF (+ meshes) of the quadbot for simulation (Isaac Sim / Isaac Lab).

usage: python3 export_urdf.py OUTDIR      (default ../sim/quadbot_description)

Links: base_link (body, deck, hump, neck base, head + jaw fixed, electronics),
per leg {LEG}_hip -> {LEG}_thigh -> {LEG}_calf -> {LEG}_foot (fixed, sphere contact).
Joint frames come straight from geom.leg_frames, limits from the collision sweeps.
"""
import math
import os
import sys

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

import geom as g
from build import to_trimesh

MM = 0.001
RHO_PRINT = 0.60e-6      # kg/mm^3: ABS at ~4 walls + 25 % infill (effective)
RHO_TPU = 0.45e-6        # TPU paws at 15 % infill + 3 walls

# collision-free ranges (deg) from check.py; STS3215 at 12 V
LIMITS = {
    "LF": ((-60, 50), (-95, 95), (-45, 145)),
    "RF": ((-40, 60), (-95, 95), (-145, 45)),
    "LR": ((-50, 60), (-95, 95), (-45, 145)),
    "RR": ((-60, 50), (-95, 95), (-145, 45)),
}
EFFORT = 2.9          # N m stall (30 kg cm)
VELOCITY = 4.7        # rad/s (0.222 s / 60 deg at 12 V)
DAMPING, FRICTION = 0.05, 0.02


def mesh_mass(m, rho):
    """(mass kg, com m, inertia kg m^2 about com) of a manifold in mm at density rho (kg/mm^3)."""
    tm = to_trimesh(m)
    tm.density = rho
    return tm.mass, tm.center_mass * MM, tm.moment_inertia * MM * MM


def box_mass(mass, T, lo, hi):
    """Solid box (servo, battery...) given in a local frame T (mm)."""
    lo, hi = np.array(lo, float), np.array(hi, float)
    d = hi - lo
    c_local = (lo + hi) / 2
    c = (T @ np.append(c_local, 1))[:3]
    I_local = mass / 12 * np.diag([d[1] ** 2 + d[2] ** 2, d[0] ** 2 + d[2] ** 2, d[0] ** 2 + d[1] ** 2])
    R = T[:3, :3]
    return mass, c * MM, R @ I_local @ R.T * MM * MM


def combine(items):
    M = sum(i[0] for i in items)
    c = sum(i[0] * i[1] for i in items) / M
    I = np.zeros((3, 3))
    for m, ci, Ii in items:
        d = ci - c
        I += Ii + m * (np.dot(d, d) * np.eye(3) - np.outer(d, d))
    return M, c, I


def to_frame(item, T):
    """Express a (mass, com, inertia) given in the body/parent frame (m) in frame T (mm matrix)."""
    m, c, I = item
    Tm = np.array(T, float)
    Tm[:3, 3] *= MM
    Tinv = np.linalg.inv(Tm)
    R = Tinv[:3, :3]
    return m, (Tinv @ np.append(c, 1))[:3], R @ I @ R.T


def origin(T):
    t = np.array(T[:3, 3]) * MM
    rpy = Rotation.from_matrix(np.array(T)[:3, :3]).as_euler("xyz")
    return f'<origin xyz="{t[0]:.6f} {t[1]:.6f} {t[2]:.6f}" rpy="{rpy[0]:.6f} {rpy[1]:.6f} {rpy[2]:.6f}"/>'


def inertial(item):
    m, c, I = item
    return (f'<inertial><origin xyz="{c[0]:.6f} {c[1]:.6f} {c[2]:.6f}" rpy="0 0 0"/><mass value="{m:.5f}"/>'
            f'<inertia ixx="{I[0,0]:.3e}" ixy="{I[0,1]:.3e}" ixz="{I[0,2]:.3e}" iyy="{I[1,1]:.3e}" '
            f'iyz="{I[1,2]:.3e}" izz="{I[2,2]:.3e}"/></inertial>')


def main(out):
    os.makedirs(os.path.join(out, "meshes"), exist_ok=True)
    os.makedirs(os.path.join(out, "urdf"), exist_ok=True)

    def save(name, m):
        tm = to_trimesh(m.simplify(0.15).scale([MM, MM, MM]))     # visual only: 0.15 mm is plenty
        tm.export(os.path.join(out, "meshes", name + ".stl"))
        return f"package://quadbot_description/meshes/{name}.stl"

    links, joints = [], []

    # ------------------------------------------------ base
    TH, TJ = g.head_frames(-12.0, 0.0)
    head_all = g.xf(g.part_head() + g.part_ear_cap(), TH) + g.xf(g.part_jaw(), TJ)
    base_vis = g.union([g.part_body(), g.part_deck(), g.part_hump(), g.part_neck_base(), head_all])
    items = [mesh_mass(g.part_body(), RHO_PRINT), mesh_mass(g.part_deck(), RHO_PRINT),
             mesh_mass(g.part_hump(), RHO_PRINT), mesh_mass(g.part_neck_base(), RHO_PRINT),
             mesh_mass(head_all, RHO_PRINT)]
    I4 = np.eye(4)
    sx, sy, sz = g.SERVO_X_OUT, g.SERVO_HALF_W, g.SERVO_HALF_H
    for leg in g.LEGS:                                       # hip-roll servos live in the body
        items.append(box_mass(0.055, g.T_hip(leg), (-g.SERVO_X_MOTOR, -sy, -sz), (sx, sy, sz)))
    items.append(box_mass(0.060, g.T_NECK, (-30, -10, -40.4), (10, 10, 0)))                  # neck DS3218
    items.append(box_mass(0.060, TH @ g.T_JAW, (-30, -10, -40.4), (10, 10, 0)))              # jaw DS3218
    items.append(box_mass(0.430, I4, (g.BATTERY_X - 36, -32.5, -25), (g.BATTERY_X + 36, 32.5, 18)))   # 3S2P
    items.append(box_mass(0.150, I4, (-50, -39, 34), (50, 41, 64)))                          # Jetson + heatsink (board offset on its holes)
    items.append(box_mass(0.060, I4, (-45, -30, -24), (45, 30, -12)))                        # bus adapter, buck, F3, wiring
    base = combine(items)
    base_col = [((0, 0, 0), (240, 77, 56)), ((0, 0, 51), (136, 92, 46))]                     # body, hump
    hc = (TH @ np.array([45, g.HEAD_YC, 2, 1]))[:3]
    base_col.append((tuple(hc), (110, 95, 90)))                                              # head (axis-aligned approx)
    col = "".join(f'<collision><origin xyz="{c[0]*MM:.4f} {c[1]*MM:.4f} {c[2]*MM:.4f}" rpy="0 0 0"/>'
                  f'<geometry><box size="{s[0]*MM:.4f} {s[1]*MM:.4f} {s[2]*MM:.4f}"/></geometry></collision>'
                  for c, s in base_col)
    links.append(f'<link name="base_link">{inertial(base)}<visual><geometry><mesh filename="{save("base", base_vis)}"/>'
                 f'</geometry><material name="honey"/></visual>{col}</link>')

    # ------------------------------------------------ legs
    case = g.servo_case()
    for leg in g.LEGS:
        left = leg[0] == "L"
        mir = g.LEGS[leg][3]
        F0 = g.leg_frames(leg, 0, 0, 0)
        M1B = g.M_1B_MIRROR if mir else g.M_1B
        # hip (link1) in its own joint frame
        hip_m = g.part_hip_link(mir)
        hip = combine([mesh_mass(hip_m, RHO_PRINT), box_mass(0.055, M1B, (-g.SERVO_X_MOTOR, -sy, -sz), (sx, sy, sz))])
        thigh_m = g.part_femur()
        thigh = combine([mesh_mass(thigh_m, RHO_PRINT),
                         box_mass(0.055, g.trans(g.L1, 0, 0), (-g.SERVO_X_MOTOR, -sy, -sz), (sx, sy, sz))])
        calf_m = g.part_tibia(left)
        foot_m = g.part_foot(left)
        calf = combine([mesh_mass(calf_m, RHO_PRINT), mesh_mass(foot_m, RHO_TPU)])
        pr = f"package://quadbot_description/meshes/"
        links.append(f'<link name="{leg}_hip">{inertial(hip)}<visual><geometry><mesh filename="{save(leg + "_hip", hip_m + g.xf(case, M1B))}"/></geometry>'
                     f'<material name="honey"/></visual><collision><origin xyz="{0.030:.4f} {(-0.012 if mir else 0.012):.4f} 0" rpy="0 0 0"/>'
                     f'<geometry><box size="0.05 0.05 0.045"/></geometry></collision></link>')
        links.append(f'<link name="{leg}_thigh">{inertial(thigh)}<visual><geometry><mesh filename="{save(leg + "_thigh", thigh_m + g.xf(case, g.trans(g.L1, 0, 0)))}"/></geometry>'
                     f'<material name="honey"/></visual><collision><origin xyz="{g.L1 * MM / 2:.4f} 0 -0.004" rpy="0 0 0"/>'
                     f'<geometry><box size="{(g.L1 + 30) * MM:.4f} 0.034 0.05"/></geometry></collision></link>')
        axis_mid = (g.L2 - g.PAW_R) / 2
        links.append(f'<link name="{leg}_calf">{inertial(calf)}<visual><geometry><mesh filename="{save(leg + "_calf", calf_m + foot_m)}"/></geometry>'
                     f'<material name="honey"/></visual><collision><origin xyz="{axis_mid * MM:.4f} 0 0" rpy="0 1.570796 0"/>'
                     f'<geometry><cylinder radius="0.011" length="{(g.L2 - g.PAW_R - 20) * MM:.4f}"/></geometry></collision></link>')
        links.append(f'<link name="{leg}_foot"><inertial><origin xyz="0 0 0"/><mass value="0.001"/>'
                     f'<inertia ixx="1e-8" ixy="0" ixz="0" iyy="1e-8" iyz="0" izz="1e-8"/></inertial>'
                     f'<collision><geometry><sphere radius="{g.PAW_R * MM:.4f}"/></geometry></collision></link>')
        lim = LIMITS[leg]
        chain = [("hip_roll", "base_link", f"{leg}_hip", g.T_hip(leg), lim[0]),
                 ("hip_pitch", f"{leg}_hip", f"{leg}_thigh", M1B, lim[1]),
                 ("knee", f"{leg}_thigh", f"{leg}_calf", g.trans(g.L1, 0, 0), lim[2])]
        for jn, parent, child, T, (lo, hi) in chain:
            joints.append(f'<joint name="{leg}_{jn}" type="revolute"><parent link="{parent}"/><child link="{child}"/>'
                          f'{origin(T)}<axis xyz="0 0 1"/><limit lower="{math.radians(lo):.4f}" upper="{math.radians(hi):.4f}" '
                          f'effort="{EFFORT}" velocity="{VELOCITY}"/><dynamics damping="{DAMPING}" friction="{FRICTION}"/></joint>')
        joints.append(f'<joint name="{leg}_foot_fixed" type="fixed"><parent link="{leg}_calf"/><child link="{leg}_foot"/>'
                      f'{origin(g.trans(g.L2 - g.PAW_R, 0, 0))}</joint>')

    urdf = ('<?xml version="1.0"?>\n<robot name="quadbot">\n<material name="honey"><color rgba="0.80 0.50 0.22 1"/></material>\n'
            + "\n".join(links) + "\n" + "\n".join(joints) + "\n</robot>\n")
    open(os.path.join(out, "urdf", "quadbot.urdf"), "w").write(urdf)
    open(os.path.join(out, "package.xml"), "w").write(
        '<?xml version="1.0"?><package format="3"><name>quadbot_description</name><version>0.1.0</version>'
        '<description>quadbot</description><maintainer email="none@none">none</maintainer><license>MIT</license></package>\n')

    # summary + a stand pose for the training config
    total = base[0] + sum(0 for _ in [])
    print(f"base mass {base[0]:.3f} kg, com {np.round(base[1]*1000,1)} mm")
    print("stand pose (rad):", [round(math.radians(v), 4) for v in g.STAND_POSE])


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sim", "quadbot_description"))
