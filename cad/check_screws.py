"""Horn screws (supplied 5.8 mm): under every head there must be HORN_GRIP of plastic,
and nothing may sit in the head's path out to the surface (screwdriver access)."""
import math
import geom as g

def probe(part, rot, label, mirror=False):
    ok = True
    for k in range(4):
        a = math.radians(45 + 90 * k + rot) * (-1 if mirror else 1)   # mirror in y: angle -> -angle
        x, y = g.HORN_PCD / 2 * math.cos(a), g.HORN_PCD / 2 * math.sin(a)
        sides = [("out", g.Z_OUT0, g.Z_OUT0 + g.HORN_GRIP, +1)]
        if k % 2 == 0:
            sides.append(("idl", g.Z_BOSS1 - g.HORN_GRIP, g.Z_BOSS1, -1))
        for side, z0, z1, d in sides:
            ann = g.cylz(3.0, z0 + 0.05, z1 - 0.05, x, y, 32) - g.cylz(1.9, z0 - 1, z1 + 1, x, y, 32)
            fill = (part ^ ann).volume() / ann.volume()
            seat = z1 if d > 0 else z0
            path = g.cylz(2.9, seat + 0.05, seat + 30, x, y, 32) if d > 0 else g.cylz(2.9, seat - 30, seat - 0.05, x, y, 32)
            blocked = (part ^ path).volume()
            good = fill > 0.97 and blocked < 0.05
            ok &= good
            print(f"  {label:10s} screw {k} {side}: under-head fill {fill*100:5.1f}%  "
                  f"grip {z1-z0:.1f} mm  thread {g.HORN_SCREW_L-(z1-z0):.1f} mm  head path blocked {blocked:.2f} mm3  {'OK' if good else 'BAD'}")
    return ok

res = [probe(g.part_femur(), 0.0, "femur"), probe(g.part_hip_link(False), 0.0, "hip_link"),
       probe(g.part_hip_link(True), 0.0, "hip_link_m", True), probe(g.part_tibia(True), g.KNEE_BIAS, "tibia_L"),
       probe(g.part_tibia(False), g.KNEE_BIAS, "tibia_R", True)]
print("all screws OK" if all(res) else "SCREW PROBLEMS")


# ---- servo case screws (PA2.0): every one needs a straight screwdriver path and a solid seat
def case_screws(name, part, T, faces, reach=60):
    ok = True
    for face in faces:
        holes = g.CASE_HOLES_TOP if face == "top" else g.CASE_HOLES_BOT
        seat, sgn = (g.CAP_TOP - 1.0, 1) if face == "top" else (g.CAP_BOT + 2.0, -1)
        case_face = g.RAISED_TOP + g.CLR if face == "top" else g.RAISED_BOT - g.CLR
        for x, y in holes:
            drv = g.cylz(1.8, min(seat + sgn * 0.05, seat + sgn * reach), max(seat + sgn * 0.05, seat + sgn * reach), x, y, 24)
            grip = (g.cylz(g.M2_HEAD_R, min(seat, case_face), max(seat, case_face), x, y, 24)
                    - g.cylz(g.M2_CLR + 0.1, -50, 50, x, y, 24))
            blocked = (part ^ g.xf(drv, T)).volume()
            fill = (part ^ g.xf(grip, T)).volume() / grip.volume()
            good = blocked < 0.3 and fill > 0.97
            ok &= good
            print("  %-12s %s y %+5.1f: driver path (%d mm) blocked %.1f mm3, under-head fill %3.0f%%  %s"
                  % (name, face, y, reach, blocked, 100 * fill, "OK" if good else "BAD"))
    return ok


body = g.part_body()
res2 = [case_screws("body " + l, body, g.T_hip(l), ("top", "bot")) for l in g.LEGS]
res2 += [case_screws("hip_link", g.part_hip_link(False), g.M_1B, ("top", "bot")),
         case_screws("hip_link_m", g.part_hip_link(True), g.M_1B_MIRROR, ("top", "bot")),
         case_screws("femur", g.part_femur(), g.trans(g.L1, 0, 0), ("top", "bot"))]
print("all case screws OK" if all(res2) else "CASE SCREW PROBLEMS")
