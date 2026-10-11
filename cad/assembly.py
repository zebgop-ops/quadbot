"""Export a posed assembly (world coordinates) for rendering / sanity checks.

usage: python3 assembly.py OUTDIR [q1 q2 q3]   (left-leg angles; right legs mirrored)
"""
import json
import os
import sys

import geom as g
from build import to_trimesh


def pose_components(q=None, head=(-12.0, 0.0)):
    q = q or g.STAND_POSE
    q1, q2, q3 = q
    comps = []  # (name, manifold, color)
    comps += [("body", g.part_body(), "frame"), ("deck", g.part_deck(), "frame"), ("hump", g.part_hump(), "frame")]
    case, horns = g.servo_case(), g.servo_horns()
    link = {False: g.part_hip_link(False), True: g.part_hip_link(True)}
    femur = g.part_femur()
    tib = {True: g.part_tibia(True), False: g.part_tibia(False)}
    foot = g.part_foot()
    for leg in g.LEGS:
        s = 1 if leg[0] == "L" else -1
        F = g.leg_frames(leg, s * q1, s * q2, s * q3)
        comps += [
            (f"{leg}_servo_hip_roll", g.xf(case, F["A"]), "servo"),
            (f"{leg}_servo_hip_pitch", g.xf(case, F["B"]), "servo"),
            (f"{leg}_servo_knee", g.xf(case, F["C"]), "servo"),
            (f"{leg}_horns", g.union([g.xf(horns, F["L1"]), g.xf(horns, F["FEMUR"]), g.xf(horns, F["TIBIA"])]), "metal"),
            (f"{leg}_hip_link", g.xf(link[g.LEGS[leg][3]], F["L1"]), "link"),
            (f"{leg}_femur", g.xf(femur, F["FEMUR"]), "link"),
            (f"{leg}_tibia", g.xf(tib[leg[0] == "L"], F["TIBIA"]), "link"),
            (f"{leg}_foot", g.xf(g.part_foot(s > 0), F["TIBIA"]), "rubber"),
        ]
    # neck + head (head frame -> body via head_frames)
    TH, TJ = g.head_frames(*head)
    H0 = g.np.linalg.inv(g.trans(g.NECK_X, 0, g.NECK_Z))
    comps += [("neck_base", g.part_neck_base(), "frame"),
              ("neck_servo", g.xf(g.ds_case(), g.T_NECK), "servo"),
              ("HEAD_skull", g.xf(g.part_head(), TH), "accent"),
              ("HEAD_jaw_servo", g.xf(g.xf(g.ds_case(), g.T_JAW), TH), "servo"),
              ("HEAD_neck_lever", g.xf(g.xf(g.ds_horn(0.0), H0 @ g.T_NECK), TH), "metal"),
              ("HEAD_camera", g.xf(g.box(g.FRONT - 4.6, g.FRONT - 3, g.CAM_YC - 12.5, g.CAM_YC + 12.5,
                                         g.CAM_ZH - 13, g.CAM_ZH + 11)
                                   + g.cylx(4.2, g.FRONT - 3, g.FRONT + 1, g.CAM_YC, g.CAM_ZH - 1.2), TH), "pcb"),
              ("HEAD_ears", g.xf(g.part_ear_cap(), TH), "accent"),
              ("HEAD_eyes", g.xf(g.union([g.cylx(g.EYE_R - 0.4, g.FRONT - 1.1, g.FRONT - 0.5, g.HEAD_YC + s_ * g.EYE_DY, g.EYE_Z)
                                          for s_ in (1, -1)]), TH), "eye"),
              ("HEAD_nose", g.xf(g.nose().translate([0.3, 0, 0]) - g.muzzle(), TH), "eye"),
              ("JAW_jaw", g.xf(g.part_jaw(), TJ), "accent"),
              ("JAW_lever", g.xf(g.xf(g.ds_horn(g.jaw_lever_angle_ds()), g.T_JAW), TJ), "metal")]
    # proxies for bought parts
    z1 = g.BODY_Z_TOP + g.DECK_T + g.JETSON_STANDOFF_H
    bx, by, bz = g.BATTERY
    fz = g.BODY_Z_BOT + 3.0 + 8
    gx, gy, gz = g.GNSS
    zt = g.BODY_Z_TOP + g.DECK_T
    comps += [
        # board: 5 mm past the holes on the right, 17 mm on the left (port edge); ports stick out ~2 mm
        ("jetson_carrier", g.box(-50, 50, g.JETSON_HOLES_Y0 - 34, g.JETSON_HOLES_Y0 + 29 + g.JETSON_PORT_MARGIN, z1, z1 + 1.6)
         + g.box(-45, 45, g.JETSON_HOLES_Y0 + 29 + g.JETSON_PORT_MARGIN - 14, g.JETSON_HOLES_Y0 + 29 + g.JETSON_PORT_MARGIN + 2,
                 z1 + 1.6, z1 + 15.6), "pcb"),
        ("jetson_module", g.box(-30, 30, -35, 35, z1 + 6, z1 + 9) + g.box(-28, 28, -28, 28, z1 + 9, z1 + 30), "metal"),
        ("battery", g.union([g.cylx(11.0, g.BATTERY_X - 36, g.BATTERY_X + 36, y, g.BODY_Z_BOT + 3.0 + 11 + 21.5 * k)
                             for y in (-21.5, 0.0, 21.5) for k in (0, 1)]), "battery"),   # 3S2P 21700 + wrap
        ("fc_f3", g.box(g.FC_X - 18, g.FC_X + 18, -18, 18, fz, fz + 1.6), "pcb"),
        ("gnss_be222p", g.box(-g.GNSS_X - gx / 2, -g.GNSS_X + gx / 2, -gy / 2, gy / 2, zt - gz, zt), "pcb"),
    ]
    return comps


if __name__ == "__main__":
    out = sys.argv[1]
    q = tuple(float(v) for v in sys.argv[2:5]) if len(sys.argv) >= 5 else g.STAND_POSE
    os.makedirs(out, exist_ok=True)
    manifest = []
    zmin = 1e9
    for name, m, color in pose_components(q):
        to_trimesh(m).export(os.path.join(out, name + ".stl"))
        manifest.append({"name": name, "color": color})
        if name.endswith("_foot"):
            zmin = min(zmin, m.bounding_box()[2])
    json.dump({"parts": manifest, "ground_z": zmin}, open(os.path.join(out, "manifest.json"), "w"), indent=1)
    print("ground z", zmin, "parts", len(manifest))
