"""Build a posable quadbot.blend from export_blend.py output.

blender -b -P blend_build.py -- EXPORT_DIR OUT.blend

Each joint is an empty named e.g. LF_knee: rotate it about its local Z
(only that channel is unlocked, limits = collision-free range).
"""
import json
import math
import os
import sys

import bpy
from mathutils import Matrix, Quaternion

argv = sys.argv[sys.argv.index("--") + 1:]
src, out = argv[0], argv[1]
scn_data = json.load(open(os.path.join(src, "scene.json")))

bpy.ops.wm.read_homefile(use_empty=False)
for ob in list(bpy.data.objects):
    bpy.data.objects.remove(ob, do_unlink=True)
scene = bpy.context.scene
us = scene.unit_settings
us.system, us.length_unit, us.scale_length = "METRIC", "MILLIMETERS", 1.0

COLORS = {
    "frame": (0.80, 0.50, 0.22, 1), "link": (0.80, 0.50, 0.22, 1), "accent": (0.80, 0.50, 0.22, 1),
    "servo": (0.03, 0.03, 0.03, 1), "metal": (0.55, 0.50, 0.45, 1), "rubber": (0.20, 0.11, 0.06, 1),
    "pcb": (0.05, 0.35, 0.12, 1), "battery": (0.15, 0.25, 0.75, 1), "eye": (0.02, 0.02, 0.02, 1),
}
mats = {}
for k, c in COLORS.items():
    m = bpy.data.materials.new(k)
    m.use_nodes = True
    bsdf = next(n for n in m.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    bsdf.inputs["Base Color"].default_value = c
    bsdf.inputs["Roughness"].default_value = 0.35 if k == "metal" else 0.55
    if k == "metal":
        bsdf.inputs["Metallic"].default_value = 0.9
    m.diffuse_color = c
    mats[k] = m

colls = {}


def coll(name):
    if name not in colls:
        c = bpy.data.collections.new(name)
        scene.collection.children.link(c)
        colls[name] = c
    return colls[name]


root = bpy.data.objects.new("quadbot", None)
root.empty_display_type = "ARROWS"
root.empty_display_size = 0.05
coll("body").objects.link(root)

# joints: fixed "mount" empty + rotating joint empty under it
joint_obj = {}
for j in scn_data["joints"]:
    mount = bpy.data.objects.new(j["name"] + "_mount", None)
    mount.empty_display_size = 0.005
    mount.hide_viewport = False
    coll(j["collection"]).objects.link(mount)
    mount.parent = joint_obj[j["parent"]] if j["parent"] else root
    mount.matrix_parent_inverse = Matrix.Identity(4)
    mount.matrix_basis = Matrix(j["mount"])
    mount.hide_select = True

    jo = bpy.data.objects.new(j["name"], None)
    jo.empty_display_type = "CIRCLE"   # circle lies in the local XY plane = the joint's plane of rotation
    jo.empty_display_size = 0.028
    jo.show_in_front = True
    coll(j["collection"]).objects.link(jo)
    jo.parent = mount
    jo.matrix_parent_inverse = Matrix.Identity(4)
    jo.rotation_mode = "XYZ"
    jo.lock_location = (True, True, True)
    jo.lock_rotation = (True, True, False)
    jo.lock_scale = (True, True, True)
    lo, hi = j["limits"]
    con = jo.constraints.new("LIMIT_ROTATION")
    con.use_limit_z = True
    con.min_z, con.max_z = math.radians(lo), math.radians(hi)
    con.owner_space = "LOCAL"
    jo.rotation_euler = (0, 0, math.radians(j["angle"]))
    jo["range_deg"] = f"{lo}..{hi}"
    joint_obj[j["name"]] = jo

mesh_cache = {}
for o in scn_data["objects"]:
    key = o["mesh"]
    if key not in mesh_cache:
        bpy.ops.wm.stl_import(filepath=os.path.join(src, key + ".stl"))
        tmp = bpy.context.selected_objects[0]
        me = tmp.data
        me.name = key
        me.materials.append(mats[o["color"]])
        bpy.data.objects.remove(tmp, do_unlink=True)
        mesh_cache[key] = me
    ob = bpy.data.objects.new(o["name"], mesh_cache[key])
    coll(o["collection"]).objects.link(ob)
    ob.parent = joint_obj[o["parent"]] if o["parent"] else root
    ob.matrix_parent_inverse = Matrix.Identity(4)
    if o["collection"] != "body" and not o["name"].startswith(("LF", "RF", "LR", "RR")):
        pass
for me in mesh_cache.values():
    for p in me.polygons:
        p.use_smooth = False

# ground plane at the stand-pose foot height
bpy.ops.mesh.primitive_plane_add(size=1.2, location=(0, 0, scn_data["ground_z"]))
gnd = bpy.context.active_object
gnd.name = "ground"
gm = bpy.data.materials.new("ground")
gm.diffuse_color = (0.75, 0.75, 0.72, 1)
gnd.data.materials.append(gm)
gnd.hide_select = True
for c in gnd.users_collection:
    c.objects.unlink(gnd)
coll("helpers").objects.link(gnd)

# a camera + sun so F12 works
sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN"))
sun.data.energy = 3.5
sun.rotation_euler = (math.radians(40), math.radians(10), math.radians(-35))
coll("helpers").objects.link(sun)
cam = bpy.data.objects.new("camera", bpy.data.cameras.new("camera"))
cam.data.lens = 55
cam.location = (0.55, 0.62, 0.2)
cam.rotation_euler = (math.radians(72), 0, math.radians(138))
coll("helpers").objects.link(cam)
scene.camera = cam
cam.data.clip_start = 0.005

# viewports: material colours, sensible clipping, framed on the robot
for screen in bpy.data.screens:
    for area in screen.areas:
        if area.type != "VIEW_3D":
            continue
        for sp in area.spaces:
            if sp.type == "VIEW_3D":
                sp.shading.type = "SOLID"
                sp.shading.color_type = "MATERIAL"
                sp.clip_start, sp.clip_end = 0.001, 100.0
                r3d = sp.region_3d
                r3d.view_location = (0.0, 0.0, -0.06)
                r3d.view_distance = 0.75
                r3d.view_rotation = Quaternion((0.78, 0.43, 0.22, 0.40)).normalized()
                r3d.view_perspective = "PERSP"

for ob in bpy.context.view_layer.objects:
    ob.select_set(False)
bpy.context.view_layer.objects.active = joint_obj["LF_knee"]
joint_obj["LF_knee"].select_set(True)

bpy.ops.wm.save_as_mainfile(filepath=out)
print("saved", out)
