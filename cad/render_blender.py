"""Render an exported assembly with Blender (headless).

blender -b -P render_blender.py -- ASSEMBLY_DIR OUT_PREFIX [--quick]

Default: Cycles (GPU if available) with bounce light, soft key + shadowless fill + rim, AgX, so
the shadow sides keep their detail. --quick: the old fast EEVEE look.
"""
import json
import math
import os
import sys

import bpy
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1:]
src, out = argv[0], argv[1]
QUICK = "--quick" in argv
man = json.load(open(os.path.join(src, "manifest.json")))

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene

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

for p in man["parts"]:
    path = os.path.join(src, p["name"] + ".stl")
    bpy.ops.wm.stl_import(filepath=path)
    ob = bpy.context.selected_objects[0]
    ob.name = p["name"]
    ob.scale = (0.001, 0.001, 0.001)
    ob.data.materials.append(mats[p["color"]])
    for poly in ob.data.polygons:
        poly.use_smooth = False

gz = man["ground_z"] * 0.001
bpy.ops.mesh.primitive_plane_add(size=4, location=(0, 0, gz))
floor = bpy.context.active_object
fm = bpy.data.materials.new("floor")
fm.use_nodes = True
fb = next(n for n in fm.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
fb.inputs["Base Color"].default_value = (0.8, 0.8, 0.78, 1)
fb.inputs["Roughness"].default_value = 0.9
floor.data.materials.append(fm)

world = bpy.data.worlds.new("w")
scene.world = world
world.use_nodes = True
bg = next(n for n in world.node_tree.nodes if n.type == "BACKGROUND")
bg.inputs["Color"].default_value = (0.9, 0.92, 0.95, 1)
bg.inputs["Strength"].default_value = 0.6 if QUICK else 0.75


def light(name, energy, rot, angle_deg, shadow=True):
    o = bpy.data.objects.new(name, bpy.data.lights.new(name, "SUN"))
    o.data.energy = energy
    o.data.angle = math.radians(angle_deg)
    o.data.use_shadow = shadow
    o.rotation_euler = tuple(math.radians(v) for v in rot)
    scene.collection.objects.link(o)


if QUICK:
    light("sun", 3.5, (40, 10, -35), 8)
else:
    light("key", 3.6, (40, 10, -35), 18)                    # soft key: wide sun = soft shadow edges
    light("fill", 0.8, (65, 0, 150), 30, shadow=False)     # opens up the shadow side
    light("rim", 1.2, (-60, 0, 10), 10, shadow=False)      # separates the outline from the floor

cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
cam.data.lens = 55
scene.collection.objects.link(cam)
scene.camera = cam

if QUICK:
    try:
        scene.render.engine = "BLENDER_EEVEE_NEXT"
    except TypeError:
        scene.render.engine = "BLENDER_EEVEE"
    scene.view_settings.view_transform = "Standard"
else:
    scene.render.engine = "CYCLES"
    try:                                                     # GPU if Blender can see one
        prefs = bpy.context.preferences.addons["cycles"].preferences
        for dev_type in ("OPTIX", "CUDA", "HIP", "ONEAPI"):
            try:
                prefs.compute_device_type = dev_type
                prefs.get_devices()
                if any(d.type == dev_type for d in prefs.devices):
                    for d in prefs.devices:
                        d.use = d.type == dev_type
                    scene.cycles.device = "GPU"
                    print("cycles device:", dev_type)
                    break
            except TypeError:
                continue
    except Exception as e:
        print("cycles GPU setup failed, using CPU:", e)
    scene.cycles.samples = 160
    scene.cycles.use_denoising = True
    scene.cycles.max_bounces = 6
    try:
        scene.view_settings.view_transform = "AgX"
        scene.view_settings.look = "AgX - Punchy"
    except TypeError:
        scene.view_settings.view_transform = "Filmic"
scene.render.resolution_x, scene.render.resolution_y = 1600, 1100
scene.render.film_transparent = False

target = Vector((0.0, 0.0, gz * 0.45))
views = {
    "iso": (Vector((0.55, 0.62, 0.28)), 1.0),
    "side": (Vector((0.0, 0.85, 0.02)), 1.0),
    "front": (Vector((0.85, 0.0, 0.05)), 1.0),
    "top": (Vector((0.001, 0.0, 0.9)), 1.0),
}
for name, (pos, _) in views.items():
    cam.location = target + pos
    d = target - cam.location
    cam.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
    scene.render.filepath = f"{out}_{name}.png"
    bpy.ops.render.render(write_still=True)
print("done")
