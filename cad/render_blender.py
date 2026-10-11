"""Render an exported assembly with Blender (headless).

blender -b -P render_blender.py -- ASSEMBLY_DIR OUT_PREFIX
"""
import json
import math
import os
import sys

import bpy
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1:]
src, out = argv[0], argv[1]
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
bg.inputs["Strength"].default_value = 0.6

sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN"))
sun.data.energy = 3.5
sun.data.angle = math.radians(8)
sun.rotation_euler = (math.radians(40), math.radians(10), math.radians(-35))
scene.collection.objects.link(sun)

cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
cam.data.lens = 55
scene.collection.objects.link(cam)
scene.camera = cam

try:
    scene.render.engine = "BLENDER_EEVEE_NEXT"
except TypeError:
    try:
        scene.render.engine = "BLENDER_EEVEE"
    except TypeError:
        scene.render.engine = "CYCLES"
scene.render.resolution_x, scene.render.resolution_y = 1600, 1100
scene.render.film_transparent = False
scene.view_settings.view_transform = "Standard"

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
