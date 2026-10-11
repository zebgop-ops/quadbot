"""Split each print-oriented STL into ok / steep-overhang / long-bridge faces and render it from
below with Blender, so problem faces show up red (overhang) and orange (bridge).
  python3 render_audit.py OUTDIR part [part ...]"""
import os
import subprocess
import sys

import numpy as np
import trimesh

import audit

HERE = os.path.dirname(os.path.abspath(__file__))
out, names = sys.argv[1], sys.argv[2:]
os.makedirs(out, exist_ok=True)
BLEND = r'''
import bpy, sys, math
from mathutils import Vector
a = sys.argv[sys.argv.index("--") + 1:]
d, name = a[0], a[1]
bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene
cols = {"ok": (0.78, 0.74, 0.62, 1), "over": (0.9, 0.08, 0.05, 1), "bridge": (1.0, 0.55, 0.0, 1)}
lo = Vector((1e9, 1e9, 1e9)); hi = -lo
import os
for k, c in cols.items():
    p = os.path.join(d, name + "_" + k + ".stl")
    if not os.path.exists(p):
        continue
    bpy.ops.wm.stl_import(filepath=p)
    ob = bpy.context.selected_objects[0]
    m = bpy.data.materials.new(k); m.use_nodes = True
    b = next(n for n in m.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    b.inputs["Base Color"].default_value = c; b.inputs["Roughness"].default_value = 0.6
    ob.data.materials.append(m)
    for v in ob.bound_box:
        w = ob.matrix_world @ Vector(v)
        lo = Vector(map(min, lo, w)); hi = Vector(map(max, hi, w))
ctr = (lo + hi) / 2; size = max(hi - lo)
w = bpy.data.worlds.new("w"); sc.world = w; w.use_nodes = True
next(n for n in w.node_tree.nodes if n.type == "BACKGROUND").inputs["Strength"].default_value = 1.0
for rot in ((40, 10, -35), (-130, 10, 140), (180, 0, 0)):
    s = bpy.data.objects.new("s", bpy.data.lights.new("s", "SUN")); s.data.energy = 3
    s.rotation_euler = [math.radians(v) for v in rot]; sc.collection.objects.link(s)
cam = bpy.data.objects.new("c", bpy.data.cameras.new("c")); sc.collection.objects.link(cam); sc.camera = cam
cam.data.lens = 50; cam.data.clip_end = 10000
try: sc.render.engine = "BLENDER_EEVEE_NEXT"
except TypeError: sc.render.engine = "BLENDER_EEVEE"
sc.render.resolution_x, sc.render.resolution_y = 900, 700
sc.view_settings.view_transform = "Standard"
for i, dvec in enumerate([(1.2, -1.4, -1.1), (-1.3, 1.2, -1.0)]):   # from below: overhangs face down
    cam.location = ctr + Vector(dvec) * size * 1.25
    cam.rotation_euler = (ctr - cam.location).to_track_quat("-Z", "Y").to_euler()
    sc.render.filepath = os.path.join(d, name + "_%d.png" % i); bpy.ops.render.render(write_still=True)
'''
open(os.path.join(out, "_render.py"), "w").write(BLEND)
for name in names:
    mesh = trimesh.load(os.path.join(HERE, "..", "stl", name + ".stl"), process=True)
    nz = mesh.face_normals[:, 2]
    zmin = mesh.vertices[mesh.faces].reshape(-1, 3, 3)[:, :, 2].min(1)
    over = (nz < audit.OVERHANG_NZ) & (zmin > 0.3)
    cls = np.zeros(len(mesh.faces), int)             # 0 ok, 1 overhang, 2 long bridge
    for area, (lo, hi), f in audit.patches(mesh, over):
        ang = np.degrees(np.arcsin(np.clip(-nz[f], 0, 1)))
        mean_ang = np.average(ang, weights=mesh.area_faces[f])
        span = min(hi[0] - lo[0], hi[1] - lo[1])
        if mean_ang > 80 and span > audit.MAX_BRIDGE:
            cls[f] = 2
        elif mean_ang <= 80 and mean_ang > audit.MAX_OVERHANG and area > 8:
            cls[f] = 1
    for k, v in (("ok", 0), ("over", 1), ("bridge", 2)):
        p = os.path.join(out, "%s_%s.stl" % (name, k))
        if (cls == v).any():
            mesh.submesh([np.nonzero(cls == v)[0]], append=True).export(p)
        elif os.path.exists(p):
            os.remove(p)
    subprocess.run(["snap", "run", "blender", "-b", "-P", os.path.join(out, "_render.py"), "--", out, name],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(name, "rendered")
