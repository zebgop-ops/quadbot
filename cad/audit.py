"""Printability audit of the print-oriented STLs in ../stl (run build.py first).

  python3 audit.py [part ...]

Per part: bed fit, bed contact area, unsupported overhang patches (steeper than 45 deg, not on the
bed; flat ones are reported as bridges with their span), and thin walls found slice by slice.
"""
import os
import sys

import numpy as np
import trimesh

HERE = os.path.dirname(os.path.abspath(__file__))
STL = os.path.join(HERE, "..", "stl")
BED = (256.0, 256.0, 256.0)
OVERHANG_NZ = -0.72          # face normal z below this = steeper than ~44 deg from vertical
BRIDGE_NZ = -0.985           # ~flat ceiling: printable as a bridge if the span is short
MAX_BRIDGE = 12.0            # mm, comfortable ABS bridge
MIN_WALL = 1.2               # mm, 3 perimeters at 0.4
MAX_OVERHANG = 55.0          # deg from vertical that ABS prints cleanly without support


def patches(mesh, mask):
    """Connected groups of flagged faces -> list of (area, bbox, face ids)."""
    idx = np.nonzero(mask)[0]
    if len(idx) == 0:
        return []
    sub = set(idx.tolist())
    adj = mesh.face_adjacency
    keep = np.array([a in sub and b in sub for a, b in adj]) if len(adj) else np.zeros(0, bool)
    import networkx as nx
    G = nx.Graph()
    G.add_nodes_from(idx.tolist())
    G.add_edges_from(adj[keep].tolist())
    out = []
    for comp in nx.connected_components(G):
        f = np.fromiter(comp, int)
        pts = mesh.vertices[mesh.faces[f].reshape(-1)]
        out.append((float(mesh.area_faces[f].sum()), (pts.min(0), pts.max(0)), f))
    return sorted(out, key=lambda p: -p[0])


def thin_walls(mesh, step=0.5, half=MIN_WALL / 2):
    """Slice every `step` mm like a slicer; in each layer, material that vanishes under an opening
    of radius `half` (shrink then regrow) is a wall thinner than MIN_WALL. -> [(z, area, bounds)]"""
    from manifold3d import Manifold, Mesh
    m = Manifold(Mesh(vert_properties=np.asarray(mesh.vertices, np.float32), tri_verts=np.asarray(mesh.faces, np.uint32)))
    out = []
    for z in np.arange(step / 2, mesh.bounds[1][2], step):
        cs = m.slice(float(z))
        if cs.area() <= 0:
            continue
        opened = cs.offset(-half).offset(half)
        thin = cs - opened
        for piece in thin.decompose():          # corner slivers are tiny; a real thin wall is a strip
            a = piece.area()
            if a > 1.0:
                out.append((float(z), a, piece.bounds()))
    return out


def audit(name):
    mesh = trimesh.load(os.path.join(STL, name + ".stl"), process=True)
    ext = mesh.bounds[1] - mesh.bounds[0]
    fits = all(e <= b for e, b in zip(sorted(ext[:2]) + [ext[2]], sorted(BED[:2]) + [BED[2]]))
    nz = mesh.face_normals[:, 2]
    zmin = mesh.vertices[mesh.faces].reshape(-1, 3, 3)[:, :, 2].min(1)
    on_bed = (nz < -0.99) & (zmin < 0.05)
    bed_area = mesh.area_faces[on_bed].sum()
    over = (nz < OVERHANG_NZ) & (zmin > 0.3)
    rows = []
    for area, (lo, hi), f in patches(mesh, over):
        if area < 4.0:
            continue
        w = mesh.area_faces[f]
        ang = np.degrees(np.arcsin(np.clip(-nz[f], 0, 1)))      # 0 = vertical wall, 90 = flat ceiling
        mean_ang = float(np.average(ang, weights=w))
        flat = mean_ang > 80.0
        span = min(hi[0] - lo[0], hi[1] - lo[1])
        kind = ("bridge %.1f mm" % span) if flat else ("overhang %2.0f deg" % mean_ang)
        bad = (span > MAX_BRIDGE) if flat else (mean_ang > MAX_OVERHANG and area > 8.0)
        rows.append((area, lo, hi, kind, bad))
    print("=" * 96)
    print("%s  %.1f x %.1f x %.1f mm  %s | %.1f cm3 | bed contact %.0f mm2"
          % (name, ext[0], ext[1], ext[2], "fits" if fits else "DOES NOT FIT", mesh.volume / 1000, bed_area))
    nbad = sum(r[4] for r in rows)
    print("  unsupported patches >= 4 mm2: %d (%d need attention)" % (len(rows), nbad))
    for area, lo, hi, kind, bad in rows[:12]:
        print("   %s %6.1f mm2  %-16s at x %.0f..%.0f y %.0f..%.0f z %.0f..%.0f"
              % ("!!" if bad else "  ", area, kind, lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]))
    thin = thin_walls(mesh)
    thin_area = sum(t[1] for t in thin)
    print("  thin-wall strips (< %.1f mm, > 1 mm2 per layer): %d, %.0f mm2 layer-area in total"
          % (MIN_WALL, len(thin), thin_area))
    for z, a, b in sorted(thin, key=lambda t: -t[1])[:5]:
        print("     z %5.1f: %5.1f mm2 at x %.0f..%.0f y %.0f..%.0f" % (z, a, b[0], b[2], b[1], b[3]))
    return dict(name=name, fits=fits, volume=mesh.volume / 1000, bed=bed_area, bad=nbad, thin=thin_area)


if __name__ == "__main__":
    names = sys.argv[1:] or sorted(f[:-4] for f in os.listdir(STL) if f.endswith(".stl"))
    res = [audit(n) for n in names]
    print("\n%-18s %6s %8s %9s %9s %12s" % ("part", "fits", "cm3", "bed mm2", "overhang", "thin mm2"))
    for r in res:
        print("%-18s %6s %8.1f %9.0f %9d %12.0f" % (r["name"], "yes" if r["fits"] else "NO", r["volume"], r["bed"], r["bad"], r["thin"]))
