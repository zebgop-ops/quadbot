"""Export parts in their joint-local frames + a hierarchy manifest, for blend_build.py.

usage: python3 export_blend.py OUTDIR
"""
import json
import os
import sys

import numpy as np

import geom as g
from assembly import pose_components
from build import to_trimesh

MM = 0.001

# collision-free ranges from check.py (deg): roll, pitch, knee
LIMITS = {
    "LF": ((-60, 50), (-95, 95), (-45, 145)),
    "RF": ((-40, 60), (-95, 95), (-145, 45)),
    "LR": ((-50, 60), (-95, 95), (-45, 145)),
    "RR": ((-60, 50), (-95, 95), (-145, 45)),
}
STAND = g.STAND_POSE  # left legs; right legs negated


def m_to_list(T):
    T = np.array(T, float).copy()
    T[:3, 3] *= MM
    return T.tolist()


def main(out):
    os.makedirs(out, exist_ok=True)
    meshes, objs, joints = {}, [], []

    def mesh(key, m):
        if key not in meshes:
            to_trimesh(m.scale([MM, MM, MM])).export(os.path.join(out, key + ".stl"))
            meshes[key] = key + ".stl"
        return key

    # static body: take everything that is not a leg part from the posed assembly
    comps = pose_components(STAND)
    for name, m, color in comps:
        if name[:2] in g.LEGS and not name.endswith("servo_hip_roll"):
            continue
        if name.startswith(("HEAD_", "JAW_")):
            continue
        coll = "bought parts" if color in ("pcb", "battery") or name.startswith("jetson") else "body"
        objs.append(dict(name=name, mesh=mesh(name, m), color=color, parent=None, collection=coll))

    case, horns = g.servo_case(), g.servo_horns()
    link = {False: g.part_hip_link(False), True: g.part_hip_link(True)}
    femur = g.part_femur()
    tib = {True: g.part_tibia(True), False: g.part_tibia(False)}
    foot = g.part_foot()
    for leg in g.LEGS:
        left = leg[0] == "L"
        mir = g.LEGS[leg][3]
        s = 1 if left else -1
        names = [f"{leg}_hip_roll", f"{leg}_hip_pitch", f"{leg}_knee"]
        mounts = [g.T_hip(leg), g.M_1B_MIRROR if mir else g.M_1B, g.trans(g.L1, 0, 0)]
        parents = [None, names[0], names[1]]
        for j in range(3):
            joints.append(dict(name=names[j], mount=m_to_list(mounts[j]), parent=parents[j], collection=leg,
                               limits=LIMITS[leg][j], angle=s * STAND[j]))
        B = g.M_1B_MIRROR if mir else g.M_1B
        objs += [
            dict(name=f"{leg}_hip_link", mesh=mesh(f"hip_link{'_mirror' if mir else ''}", link[mir]), color="link",
                 parent=names[0], collection=leg),
            dict(name=f"{leg}_servo_hip_pitch", mesh=mesh(f"servo_in_link{'_mirror' if mir else ''}", g.xf(case, B)),
                 color="servo", parent=names[0], collection=leg),
            dict(name=f"{leg}_horns_roll", mesh=mesh("horns", horns), color="metal", parent=names[0], collection=leg),
            dict(name=f"{leg}_femur", mesh=mesh("femur", femur), color="link", parent=names[1], collection=leg),
            dict(name=f"{leg}_servo_knee", mesh=mesh("servo_in_femur", g.xf(case, g.trans(g.L1, 0, 0))),
                 color="servo", parent=names[1], collection=leg),
            dict(name=f"{leg}_horns_pitch", mesh=mesh("horns", horns), color="metal", parent=names[1], collection=leg),
            dict(name=f"{leg}_tibia", mesh=mesh(f"tibia_{'left' if left else 'right'}", tib[left]), color="link",
                 parent=names[2], collection=leg),
            dict(name=f"{leg}_foot", mesh=mesh(f"foot_{'left' if left else 'right'}",
                                                 g.part_foot(left)),
                 color="rubber", parent=names[2], collection=leg),
            dict(name=f"{leg}_horns_knee", mesh=mesh("horns", horns), color="metal", parent=names[2], collection=leg),
        ]
    # neck + jaw: joint empties rotate about local Z = body +Y (+ = nose down / mouth open)
    R = g.frame((1, 0, 0), (0, 0, -1), (0, 1, 0), (0, 0, 0))
    joints.append(dict(name="neck", mount=m_to_list(g.trans(g.NECK_X, 0, g.NECK_Z) @ R), parent=None,
                       collection="head", limits=(-55, 55), angle=-12.0))
    joints.append(dict(name="jaw", mount=m_to_list(R.T @ g.trans(g.JX, 0, g.JZ) @ R), parent="neck",
                       collection="head", limits=(0, 60), angle=0.0))
    TH0, _ = g.head_frames(-12.0, 0)
    to_neck = R.T @ np.linalg.inv(TH0)                     # body (rest pose) -> neck-joint local
    to_jaw = R.T @ g.trans(-g.JX, 0, -g.JZ) @ np.linalg.inv(TH0)
    for name, m, color in comps:
        if name.startswith("HEAD_"):
            objs.append(dict(name=name[5:], mesh=mesh(name, g.xf(m, to_neck)), color=color, parent="neck",
                             collection="head"))
        elif name.startswith("JAW_"):
            objs.append(dict(name="jaw_" + name[4:], mesh=mesh(name, g.xf(m, to_jaw)), color=color, parent="jaw",
                             collection="head"))
    # test / display stand, mounted on the belly rail (body frame); the floor goes to the stand's table
    objs.append(dict(name="stand", mesh=mesh("stand", g.part_stand()), color="servo", parent=None, collection="stand"))
    ground = g.STAND_TABLE_Z * MM
    json.dump(dict(objects=objs, joints=joints, ground_z=ground), open(os.path.join(out, "scene.json"), "w"), indent=1)
    print(len(meshes), "meshes,", len(objs), "objects,", len(joints), "joints")


if __name__ == "__main__":
    main(sys.argv[1])
