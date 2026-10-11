"""Static fit: every servo (case + horns + plugged connectors) vs the part that holds it,
and every bracket vs the servo whose horns it clamps. Catches skins/lips over the servo
that the joint sweep can't see (same rigid group)."""
import geom as g
g.RIBS = False           # check the plain pockets; the crush ribs overlap the servo on purpose (below)

servo = g.servo_case() + g.servo_horns() + g.servo_plugs()
case_only = g.servo_case() + g.servo_plugs()
rows = [
    ("body / 4 hip-roll servos", g.part_body(), g.union([g.xf(servo, g.T_hip(l)) for l in g.LEGS])),
    ("hip_link / hip-pitch servo", g.part_hip_link(False), g.xf(servo, g.M_1B)),
    ("femur / knee servo", g.part_femur(), g.xf(servo, g.trans(g.L1, 0, 0))),
    ("fit_test_cradle / servo", g.part_fit_test_cradle(), servo),
    ("fit_test_hip_roll / servo", g.part_fit_test_hip_roll(), servo),
    ("hip_link bracket / hip-roll servo case", g.part_hip_link(False), case_only),
    ("femur bracket / hip-pitch servo case", g.part_femur(), case_only),
    ("tibia bracket / knee servo case", g.part_tibia(True), case_only),
    ("neck_base / neck DS3218", g.part_neck_base(), g.xf(g.ds_case(), g.T_NECK)),
    ("head / jaw DS3218", g.part_head(), g.xf(g.ds_case(), g.T_JAW)),
]
bad = 0
for name, part, other in rows:
    v = (part ^ other).volume()
    flag = "OK " if v < 1.0 else "HIT"
    bad += v >= 1.0
    print(f"{flag} {name:42s} overlap {v:8.2f} mm3")
print("all clear" if not bad else f"{bad} problems")

# pivots: the bearing side of every servo-driven joint must have material on BOTH parts
# around the bolt axis (a missing boss is invisible to the collision sweeps)
TH, TJ = g.head_frames(0, 0)
head_b, jaw_b, nb = g.xf(g.part_head(), TH), g.xf(g.part_jaw(), TJ), g.part_neck_base()


def ring(ax_x, ax_z, y0, y1, T=None):
    r = g.cyly(9, y0, y1, ax_x, ax_z) - g.cyly(2.5, y0 - 1, y1 + 1, ax_x, ax_z)
    return g.xf(r, T) if T is not None else r


pivots = [
    ("neck pivot: head bearing plate", head_b, ring(0, 0, g.HN_OUT, g.HN_IN, TH)),
    ("neck pivot: neck-base boss", nb, ring(g.NECK_X, g.NECK_Z, g.HN_IN + 0.5, g.N_BOT - 0.5)),
    ("jaw pivot: jaw bearing arm", jaw_b, ring(g.JX, g.JZ, g.HN_OUT - 5.0, g.HN_OUT - 0.5, TH)),
    ("jaw pivot: head plate + boss", head_b, ring(g.JX, g.JZ, g.HN_OUT, g.HN_IN + 2.5, TH)),
]
for name, part, zone in pivots:
    fill = (part ^ zone).volume() / zone.volume()
    ok = fill > 0.6
    bad += not ok
    print(f"{'OK ' if ok else 'MISSING'} {name:42s} {fill * 100:5.1f} % of the zone is material")
print("all clear" if not bad else f"{bad} problems")

# crush ribs: each should bite RIB_INTERF into the servo case and nothing else
import importlib
g.RIBS = True
for name, ribs, T in (("hip roll (body)", g.cradle_ribs(g.HR_SADDLE_X1, g.HR_SADDLE_X1, bottom_holes=True), g.np.eye(4)),
                      ("hip pitch (hip link)", g.cradle_ribs(bottom_holes=True), g.np.eye(4)),
                      ("knee (femur)", g.cradle_ribs(), g.np.eye(4))):
    case = g.servo_case()
    bite = (ribs ^ case)
    depth = bite.bounding_box()
    print("ribs %-22s %2d ribs, overlap with the case %5.1f mm3 (squashed on insertion)"
          % (name, len(ribs.decompose()), bite.volume()))
