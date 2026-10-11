"""Geometry for the STS3215 quadruped. Pure manifold3d CSG, no CAD kernel needed."""
import math
from functools import lru_cache

import numpy as np
from manifold3d import Manifold, CrossSection, OpType

from params import *  # noqa: F401,F403

# ------------------------------------------------------------------ helpers


def box(x0, x1, y0, y1, z0, z1):
    return Manifold.cube([x1 - x0, y1 - y0, z1 - z0]).translate([x0, y0, z0])


def cylz(r, z0, z1, x=0.0, y=0.0, seg=SEG):
    return Manifold.cylinder(z1 - z0, r, r, seg).translate([x, y, z0])


def cylx(r, x0, x1, y=0.0, z=0.0, seg=SEG):
    return Manifold.cylinder(x1 - x0, r, r, seg).rotate([0, 90, 0]).translate([x0, y, z])


def cyly(r, y0, y1, x=0.0, z=0.0, seg=SEG):
    return Manifold.cylinder(y1 - y0, r, r, seg).rotate([-90, 0, 0]).translate([x, y0, z])


def rbox(x0, x1, y0, y1, z0, z1, r):
    """Box with its vertical (Z) edges rounded -- ABS corners lift less."""
    return hull(*[cylz(r, z0, z1, x, y) for x in (x0 + r, x1 - r) for y in (y0 + r, y1 - r)])


def rbox_x(x0, x1, y0, y1, z0, z1, r):
    """Box with its X-parallel edges rounded."""
    return hull(*[cylx(r, x0, x1, y, z) for y in (y0 + r, y1 - r) for z in (z0 + r, z1 - r)])


def diamond(axis, a, b, h, t0, t1):
    """Prism with a 45-degree-rotated square section (half-diagonal h) running along
    `axis` from t0 to t1, centred at (a, b) = (y, z) for x, (x, z) for y, (x, y) for z.
    45 degree edges print without support in any orientation within its plane."""
    p = CrossSection.square([h * math.sqrt(2)] * 2, True).rotate(45).extrude(t1 - t0).translate([0, 0, t0])
    M = {"z": [[1, 0, 0, a], [0, 1, 0, b], [0, 0, 1, 0]],
         "y": [[0, 1, 0, a], [0, 0, 1, 0], [1, 0, 0, b]],
         "x": [[0, 0, 1, 0], [1, 0, 0, a], [0, 1, 0, b]]}[axis]
    return p.transform(np.array(M, float))


def teardrop(axis, a, b, r, t0, t1, up, drop=0.0):
    """Round window with a 45-degree point toward `up` (2-vector in the (a, b) plane,
    see diamond() for the axis convention) - prints without support and has no sharp
    stress-raising corners except the self-supporting tip. `drop` stretches it away
    from the tip into an organic drop shape."""
    ua, ub_ = np.array(up, float) / np.linalg.norm(up)
    du, dv = (ub_, ua) if axis == "y" else (ua, ub_)
    tip = r * math.sqrt(2) - 0.3
    cs = [CrossSection.circle(r, 48), CrossSection.circle(0.3, 8).translate([du * tip, dv * tip])]
    if drop:
        cs.append(CrossSection.circle(r * 0.8, 48).translate([-du * drop, -dv * drop]))
    p = CrossSection.batch_hull(cs).extrude(t1 - t0).translate([0, 0, t0])
    M = {"z": [[1, 0, 0, a], [0, 1, 0, b], [0, 0, 1, 0]],
         "y": [[0, 1, 0, a], [0, 0, 1, 0], [1, 0, 0, b]],
         "x": [[0, 0, 1, 0], [1, 0, 0, a], [0, 1, 0, b]]}[axis]
    return p.transform(np.array(M, float))


@lru_cache(None)
def _ball(r):
    return Manifold.sphere(r, 12)


def _dilate(m, r):
    return m.minkowski_sum(_ball(r)).simplify(0.04)


def _erode(m, r):
    # Minkowski difference is slow on non-convex parts; erode = complement of the
    # dilated complement, which only needs the (fast) Minkowski sum.
    x0, y0, z0, x1, y1, z1 = m.bounding_box()
    p = r + 3
    B = box(x0 - p, x1 + p, y0 - p, y1 + p, z0 - p, z1 + p)
    return (B - _dilate(B - m, r)).simplify(0.04)


def pod(m, r=3.0):
    """Intersect with m's bounding box rounded to radius r on every edge and corner:
    softens the silhouette without thinning walls the way a large opening would."""
    x0, y0, z0, x1, y1, z1 = m.bounding_box()
    b = hull(*[Manifold.sphere(r, 24).translate([x, y, z])
               for x in (x0 + r, x1 - r) for y in (y0 + r, y1 - r) for z in (z0 + r, z1 - r)])
    return m ^ b


def organic(m, fillet=2.0, rnd=1.0):
    """Fillet every inside corner (morphological closing) and round every outside
    edge (opening). Applied to structural bodies *before* precise features are cut,
    so bores, pockets, bed faces and clearances stay exact."""
    m = m.simplify(0.02)
    if fillet:
        m = _erode(_dilate(m, fillet), fillet)
    if rnd:
        m = _dilate(_erode(m, rnd), rnd)
    return m


def union(parts):
    parts = [p for p in parts if p is not None]
    return Manifold.batch_boolean(parts, OpType.Add) if len(parts) > 1 else parts[0]


def hull(*parts):
    return Manifold.batch_hull(list(parts))


def xf(m, T):
    return m.transform(np.asarray(T, dtype=float)[:3, :4])


def frame(xa, ya, za, o):
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = xa, ya, za, o
    return T


def rz(deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    T = np.eye(4)
    T[:2, :2] = [[c, -s], [s, c]]
    return T


def trans(x, y, z):
    T = np.eye(4)
    T[:3, 3] = [x, y, z]
    return T


# ------------------------------------------------------------------ servo


def servo_case():
    """Static envelope of an STS3215 (servo frame), incl. plugged connectors."""
    main = box(-SERVO_X_MOTOR, SERVO_X_OUT, -SERVO_HALF_W, SERVO_HALF_W, -SERVO_HALF_H, SERVO_HALF_H)
    raised = box(-SERVO_X_MOTOR, RAISED_X_END, -SERVO_HALF_W, SERVO_HALF_W, RAISED_BOT, RAISED_TOP)
    raised = raised - cylz(RAISED_ARC_R, -30, 30)
    foot = box(FOOT_X[0], FOOT_X[1], -FOOT_HALF_W, FOOT_HALF_W, FOOT_Z, RAISED_BOT)
    spline = cylz(SPLINE_R, SERVO_HALF_H, SPLINE_TOP)
    hub = cylz(IDLER_HUB_R, IDLER_HUB_END, -SERVO_HALF_H)
    return union([main, raised, foot, spline, hub])


def servo_plugs():
    return box(PLUG_X[0], PLUG_X[1], -PLUG_HALF_W, PLUG_HALF_W, RAISED_BOT - PLUG_H, RAISED_BOT)


def servo_horns():
    out = cylz(HORN_R, HORN_FACE_OUT - 2.5, HORN_FACE_OUT) + cylz(4.5, SERVO_HALF_H + 0.3, HORN_FACE_OUT)
    idl = cylz(HORN_R, HORN_FACE_IDL, HORN_FACE_IDL + 2.1)
    return out + idl


def servo_pocket(ext=80.0):
    """Cavity the servo slides into along -X (swept toward +X so it can)."""
    c = CLR
    main = box(-SERVO_X_MOTOR - c, ext, -SERVO_HALF_W - c, SERVO_HALF_W + c, -SERVO_HALF_H - c, SERVO_HALF_H + c)
    raised = box(-SERVO_X_MOTOR - c, ext, -SERVO_HALF_W - c, SERVO_HALF_W + c, RAISED_BOT - c, RAISED_TOP + c)
    foot = box(FOOT_X[0] - c, ext, -FOOT_HALF_W - c, FOOT_HALF_W + c, FOOT_Z - c, RAISED_BOT + 0.5)
    return union([main, raised, foot])


CAP_X1 = -SERVO_X_MOTOR - CLR - 2.65  # outer face of end cap (-38.0)
# sleeve side walls: up to just under the output arm plane, down to the cap's lower lip
# (the idler arm sits at -26 and the plugs are inboard of the walls)
SLEEVE_Z = (-21.5, 16.3)
SLEEVE_HW = SERVO_HALF_W + CLR + WALL  # 15.1


def cradle_solid():
    """Sleeve + end cap that holds a servo (servo frame). Pocket NOT removed."""
    sleeve = box(CAP_X1, SLEEVE_FRONT, -SLEEVE_HW, SLEEVE_HW, SLEEVE_Z[0], SLEEVE_Z[1])
    cap = box(CAP_X1, CAP_X0, -SLEEVE_HW, SLEEVE_HW, CAP_BOT, CAP_TOP)
    return sleeve + cap


@lru_cache(None)
def cradle_cuts(top=True, bot=True, up=(0.0, 1.0)):
    """Pocket + PA2.0 screw holes into the case (servo frame). `up` = print-up
    direction in the servo's (x, z) plane, for the wall windows.

    Only the face you can reach with a screwdriver gets holes: 2 screws per
    servo plus the snug pocket is plenty."""
    cuts = [servo_pocket(),
            # open faces: nothing may cover the servo's top/bottom outside the cap
            box(CAP_X0, SLEEVE_FRONT + 1, -SERVO_HALF_W - CLR, SERVO_HALF_W + CLR,
                SLEEVE_Z[0] - 1, SLEEVE_Z[1] + 1)]
    for x, y in CASE_HOLES_TOP if top else []:
        cuts.append(cylz(M2_CLR, 10, 40, x, y, 24))
        cuts.append(cylz(M2_HEAD_R, CAP_TOP - 1.0, 40, x, y, 24))
    for x, y in CASE_HOLES_BOT if bot else []:
        cuts.append(cylz(M2_CLR, -40, -10, x, y, 24))
        cuts.append(cylz(M2_HEAD_R, -40, CAP_BOT + 2.0, x, y, 24))
    # window in the end wall: less bridging, cable/air access
    cuts.append(box(CAP_X1 - 1, CAP_X1 + 3, -8, 8, -10, 10))
    # (sleeve walls stay solid: no pointed cut-outs on the outside)
    return union(cuts)


# ---- crush ribs + hip-roll saddle (servos rocked on their 2 screws; fixed 2026-09-24)
RIBS = True              # check_fit.py turns them off to check the plain pocket
RIB_R = 0.8
RIB_INTERF = 0.2         # how far each rib reaches into the servo: squashed on insertion
HR_SADDLE_X1 = -13.0     # hip-roll saddle + rails reach this far toward the output end


def _rib(x0, x1, y, z, ny, nz):
    """Half-round ridge along x (the insertion direction) on the pocket wall through (y, z);
    (ny, nz) = the wall's inward normal. Its crest sits CLR + RIB_INTERF inside the wall."""
    reach = CLR + RIB_INTERF
    return cylx(RIB_R, x0, x1, y + ny * (reach - RIB_R), z + nz * (reach - RIB_R), 16)


@lru_cache(None)
def cradle_ribs(x_top=None, x_bot=None, bottom_holes=False):
    """Crush ribs inside a cradle pocket (servo frame): 3 per side wall near the output end,
    2 under the top lip, 2 on the bottom lip (beside the foot channel). They take up the
    pocket clearance so the servo can't rock, without needing a precise print."""
    wy = SERVO_HALF_W + CLR
    zt, zb = RAISED_TOP + CLR, RAISED_BOT - CLR
    xs = -SERVO_X_MOTOR + 1.5
    ribs = []
    for sy in (1, -1):
        for z in (-8.0, 0.0, 8.0):
            ribs.append(_rib(-20.0, SLEEVE_FRONT - 0.5, sy * wy, z, -sy, 0))
        ribs.append(_rib(xs, CAP_X0 if x_top is None else x_top, sy * 6.0, zt, 0, -1))
        ribs.append(_rib(-31.0 if bottom_holes else xs, CAP_X0 if x_bot is None else x_bot, sy * 11.8, zb, 0, 1))
    return union(ribs)


@lru_cache(None)
def hip_roll_saddle():
    """Hip roll only turns -60..+50 deg, so the bracket never sweeps the motor half of the
    servo: extend the end cap's grip toward the output end - a plate over the horn face and
    two rails on the idler face (outside the foot channel and clear of the plugs)."""
    x0, x1 = CAP_X0 - 0.5, HR_SADDLE_X1
    top = box(x0, x1, -SLEEVE_HW, SLEEVE_HW, RAISED_TOP + CLR, CAP_TOP)
    rails = [box(x0, x1, *sorted((sy * 11.0, sy * SLEEVE_HW)), CAP_BOT, RAISED_BOT - CLR) for sy in (1, -1)]
    return union([top] + rails)


# U bracket arm planes
Z_OUT0, Z_OUT1 = HORN_FACE_OUT, HORN_FACE_OUT + ARM_T
Z_BOSS0, Z_BOSS1 = HORN_FACE_IDL - IDLER_STANDOFF, HORN_FACE_IDL
Z_IDL0, Z_IDL1 = Z_BOSS0 - ARM_T, Z_BOSS0
U_ZC = (Z_OUT1 + Z_IDL0) / 2
HORN_GRIP = HORN_SCREW_L - HORN_THREAD   # plastic under each horn-screw head (2.5)
SPACER_R = 10.3                           # plugs sweep from r 11.3


SPACER_FLAT = 8.5      # underside trimmed flat this far from the axis: an 11.6 mm bridge in print


def idler_spacer(down_deg=None):
    """Solid round standoff between the idler horn and the idler arm. The supplied
    5.8 mm horn screws sit deep inside it (heads HORN_GRIP above the horn), so it is a
    disc that stays inside the servo plugs' sweep (r 11.3), not a thin dog-bone.
    Added after the organic pass (no fillet) to keep the servo plugs clear."""
    z0 = Z_BOSS0 - 0.8                   # sink into the arm so it fuses
    disc = cylz(SPACER_R, z0, Z_BOSS1)
    if down_deg is None:
        return disc
    # lying on its side in the print its underside would be a 7 mm unsupported shelf; a teardrop
    # would reach the plugs' sweep (r 11.3), so trim it flat instead (bridge) - screw heads unaffected
    return disc ^ box(-50, SPACER_FLAT, -50, 50, z0 - 1, Z_BOSS1 + 1).rotate([0, 0, down_deg])


def u_solid(half_w=U_HALF_W, base_out=U_R_OUT):
    """Link side of a joint: two arms on the horns + base (no spacer, no holes)."""
    def arm(z0, z1):
        return hull(cylz(PAD_R, z0, z1), box(U_R_IN, base_out, -half_w, half_w, z0, z1))
    return union([arm(Z_OUT0, Z_OUT1), arm(Z_IDL0, Z_IDL1),
                  box(U_R_IN, base_out, -half_w, half_w, Z_IDL0, Z_OUT1)])


@lru_cache(None)
def u_cuts():
    """Horn screw holes for the supplied 5.8 mm screws: HORN_GRIP of plastic under
    every head on both horns, deep counterbores (open-sided in the idler spacer)."""
    cuts = [cylz(3.4, Z_IDL0 - 1, Z_OUT1 + 1)]  # centre screw access
    seat_idl = Z_BOSS1 - HORN_GRIP
    for k in range(4):
        a = math.radians(45 + 90 * k)
        c, s_ = math.cos(a), math.sin(a)
        x, y = HORN_PCD / 2 * c, HORN_PCD / 2 * s_
        cuts.append(cylz(M3_CLR, Z_OUT0 - 0.5, Z_OUT1 + 1, x, y, 24))
        cuts.append(cylz(M3_HEAD_R, Z_OUT0 + HORN_GRIP, Z_OUT1 + 1, x, y, 24))   # 4x output side
        if k % 2 == 0:  # idler horn only carries radial load: 2 screws
            cuts.append(cylz(M3_CLR, seat_idl - 0.5, Z_BOSS1 + 0.5, x, y, 24))
            cuts.append(cylz(M3_HEAD_R, Z_IDL0 - 1, seat_idl, x, y, 24))
            # the counterbore breaks out of the spacer side: open it as a clean slot
            ro = SPACER_R + 2.0 - HORN_PCD / 2
            cuts.append(hull(cylz(M3_HEAD_R, Z_BOSS0, seat_idl, x, y, 24),
                             cylz(M3_HEAD_R, Z_BOSS0, seat_idl, x + ro * c, y + ro * s_, 24)))
    return union(cuts)


def joint_button(rot):
    """45-deg cone 'knee cap' on the idler arm's outer face (prints any way up)."""
    return Manifold.cylinder(6.0, 11.0, 17.0, 64).translate([0, 0, Z_IDL0 - 6.0 + 0.01]).rotate([0, 0, rot])


def joint_button_holes(rot):
    p = HORN_PCD / 2 / math.sqrt(2)
    return union([cylz(3.6, Z_IDL0 - 8, Z_IDL0 + 0.5), cylz(3.3, Z_IDL0 - 8, Z_IDL0 + 0.5, p, p),
                  cylz(3.3, Z_IDL0 - 8, Z_IDL0 + 0.5, -p, -p)]).rotate([0, 0, rot])


def u_bracket(half_w=U_HALF_W, base_out=U_R_OUT):
    return (organic(u_solid(half_w, base_out)) + idler_spacer(-90.0)) - u_cuts()


# ------------------------------------------------------------------ leg frames
# body frame: X forward, Y left, Z up
LEGS = {
    #       A-servo x axis   A-servo z axis  hip position        link1 mirrored
    "LF": ((0, 1, 0), (1, 0, 0), (HIP_X, HIP_Y, 0.0), False),
    "RF": ((0, -1, 0), (1, 0, 0), (HIP_X, -HIP_Y, 0.0), True),
    "LR": ((0, 1, 0), (-1, 0, 0), (-HIP_X, HIP_Y, 0.0), True),
    "RR": ((0, -1, 0), (-1, 0, 0), (-HIP_X, -HIP_Y, 0.0), False),
}


def T_hip(leg):
    xa, za, o, _ = LEGS[leg]
    xa, za = np.array(xa, float), np.array(za, float)
    return frame(xa, np.cross(za, xa), za, o)


# hip-pitch servo in the link1 frame
M_1B = frame((0, -1, 0), (0, 0, 1), (-1, 0, 0), (B_OFFSET, 0, 0))
M_1B_MIRROR = frame((0, 1, 0), (0, 0, -1), (-1, 0, 0), (B_OFFSET, 0, 0))


def leg_frames(leg, q1=0.0, q2=0.0, q3=0.0):
    """Joint frames. q* in degrees, 0 = leg straight down."""
    TA = T_hip(leg)
    T1 = TA @ rz(q1)
    TB = T1 @ (M_1B_MIRROR if LEGS[leg][3] else M_1B)
    T2 = TB @ rz(q2)
    TC = T2 @ trans(L1, 0, 0)
    T3 = TC @ rz(q3)
    return dict(A=TA, L1=T1, B=TB, FEMUR=T2, C=TC, TIBIA=T3)


# ------------------------------------------------------------------ parts (local frames)


@lru_cache(None)
def part_hip_link(mirror=False):
    """link1: clamps the hip-roll servo, carries the hip-pitch servo."""
    ext_x1 = U_R_IN + 5.7
    top = -CAP_X1                      # flush with the pitch servo's end cap
    B = M_1B
    Bo = B_OFFSET
    shoulder = pod(union([
        hull(box(U_R_IN, ext_x1, -U_HALF_W, top, -SLEEVE_HW, SLEEVE_HW),          # riser + bridge as
             box(U_R_IN, Bo - CAP_TOP + 3.0, 25.5, top, -SLEEVE_HW, SLEEVE_HW)),   # one swept shoulder
        xf(cradle_solid(), B)]), 4.5)
    parts = [u_solid(), shoulder]
    m = organic(union(parts)) + idler_spacer(-90.0)
    m = m - u_cuts() - xf(cradle_cuts(up=(-1.0, 0.0)), B)         # 2 + 2 case screws
    m = m - teardrop("x", 17.0, 0.0, 5.0, U_R_IN - 1, ext_x1 + 2, (1.0, 0.0))    # riser window
    # keep the femur's inner arm sweep clear
    fa0 = Bo - HORN_FACE_OUT - ARM_T - 1.0
    fa1 = Bo - HORN_FACE_OUT + 0.5
    m = m - cylx(24.5, fa0, fa1)
    m = m - cylx(HORN_R + 1.0, fa1 - 0.1, Bo - SERVO_HALF_H - 1.5)   # the pitch servo's own output horn
    if RIBS:
        m = m + xf(cradle_ribs(bottom_holes=True), B)
    return m.mirror([0, 1, 0]) if mirror else m


def _smooth(t):
    t = max(0.0, min(1.0, t))
    return t * t * (3 - 2 * t)


FEMUR_WAIST_X = 47.0


def femur_profile(x):
    """(z_lo, z_hi) of the femur side walls: full bracket height at the hip, a waist
    mid-shaft, flaring again into the knee cradle. The hip-side flare stays below
    45 deg because the femur prints standing on its knee end."""
    x_u, x_k = U_R_OUT, L1 + CAP_X0 - 10.0
    hi_u, hi_w, hi_k = Z_OUT1, 11.0, CAP_TOP
    # inner side (+z, toward the body) keeps the waist; the outer side is a chunky thigh
    if x <= FEMUR_WAIST_X:
        hi = hi_u + (hi_w - hi_u) * _smooth((x - x_u) / (FEMUR_WAIST_X - x_u))
    else:
        hi = hi_w + (hi_k - hi_w) * _smooth((x - FEMUR_WAIST_X) / (x_k - FEMUR_WAIST_X))
    bulge = 1.2 * math.sin(math.pi * max(0.0, min(1.0, (x - x_u) / (FEMUR_WAIST_X - x_u))) / 2) if x <= FEMUR_WAIST_X else 1.2
    lo = Z_IDL0 - bulge if x <= FEMUR_WAIST_X else \
        (Z_IDL0 - 1.2) + ((CAP_BOT) - (Z_IDL0 - 1.2)) * _smooth((x - FEMUR_WAIST_X) / (x_k - FEMUR_WAIST_X))
    return lo, hi


def femur_wall(y0, y1, x0, x1, band=None):
    """Extrude the femur profile between y0..y1; `band` = thickness of an outer skin only."""
    xs = np.linspace(x0, x1, 60)
    lo = [(x, femur_profile(x)[0]) for x in xs]
    hi = [(x, femur_profile(x)[0] + band if band else femur_profile(x)[1]) for x in xs[::-1]]
    poly = lo + hi
    cs = CrossSection([poly])
    if cs.area() < 0:
        cs = CrossSection([poly[::-1]])
    # extrude along +z (local) then map local y -> world z, local z -> world y
    return cs.extrude(y1 - y0).transform(np.array([[1, 0, 0, 0], [0, 0, 1, y0], [0, 1, 0, 0]], float))


@lru_cache(None)
def part_femur():
    hw = SLEEVE_HW
    x0, x1 = U_R_IN, L1 + CAP_X0 - 1.0
    zc = (CAP_BOT + CAP_TOP) / 2
    parts = [
        u_solid(half_w=hw),
        femur_wall(hw - WALL, hw, x0, x1),                   # side walls
        femur_wall(-hw, -hw + WALL, x0, x1),
        femur_wall(-hw, hw, x0, x1, band=2.5),               # outer skin: a closed, chunky thigh
        box(x0, x1, -hw, hw, -1.5, 1.5),                     # web
        xf(pod(cradle_solid(), 3.0), trans(L1, 0, 0)),
    ]
    m = organic(union(parts)) + idler_spacer(0.0)
    m = m - u_cuts() - xf(cradle_cuts(up=(-1.0, 0.0)), trans(L1, 0, 0))   # 2 + 2 case screws
    for xc in (40.0, 56.0):                                   # web windows (also cable / zip-tie points)
        for yc in (6.0, -6.0):
            m = m - teardrop("z", xc, yc, 4.0, -4, 4, (-1.0, 0.0), drop=5.0)
    m = m + joint_button(0.0) - joint_button_holes(0.0)                     # hip-pitch button
    # tibia arm sweep near the knee must stay clear
    m = m - (cylz(24.5, Z_OUT0 - 0.8, Z_OUT1 + 1).translate([L1, 0, 0]))
    m = m - (cylz(24.5, Z_IDL0 - 1, Z_IDL1 + 0.8).translate([L1, 0, 0]))
    m = m - (cylz(11.5, Z_BOSS0, Z_BOSS1 + 0.8).translate([L1, 0, 0]))       # spacer sweep
    if RIBS:
        m = m + xf(cradle_ribs(bottom_holes=True), trans(L1, 0, 0))
    # it prints standing on the knee end: cut off the edge rounding so the whole end sits on the bed
    x_end = m.bounding_box()[3]
    return m - box(x_end - 1.0, x_end + 5, -60, 60, -60, 60)


def octagon_x(across_flats, x0, x1, y, z):
    r = across_flats / 2 / math.cos(math.radians(22.5))
    cs = CrossSection.circle(r, 8).rotate(22.5)
    return cs.extrude(x1 - x0).rotate([0, 90, 0]).translate([x0, y, z])


PEG_AF = 14.0
PEG_X = (L2 - 27.0, L2 - 8.0)
TIBIA_WINDOWS = ((38.0, -9.5, 4.5), (52.0, -7.0, 4.0), (66.0, -4.0, 4.0))  # (along, across, half-diag)
KNEE_BIAS = -50.0   # U bracket rotated so the knee folds 150 deg one way (left legs)


def foot_y(left=True):
    return 0.0


@lru_cache(None)
def part_tibia(left=True):
    """Left-leg tibia; the right-leg one is its mirror image.

    Prints lying on the knee bracket (bracket frame y = -11): the beam then climbs
    at ~65 deg and the peg at 50 deg, so no supports. Windows are laid out in that
    bracket frame so their 45-degree edges stay 45 degrees on the printer."""
    zc = U_ZC
    ub = rz(KNEE_BIAS)                       # bracket frame -> tibia frame
    u = u_solid().rotate([0, 0, KNEE_BIAS])

    def ell(c, r):
        return Manifold.sphere(1.0, 40).scale(list(r)).translate(list(c))

    # knee end (in the bracket frame, rotated in), calf, ankle
    knee = ell((U_R_IN + 4.0, 0.0, zc), (5.0, U_HALF_W, 10.0)).rotate([0, 0, KNEE_BIAS])
    kx, ky = (rz(KNEE_BIAS) @ np.array([U_R_IN + 4.0, 0, 0, 1]))[:2]
    tip = np.array([L2 - 28.0, 0.0])
    calf_c = np.array([kx, ky]) + (tip - np.array([kx, ky])) * 0.42
    calf = ell((calf_c[0], calf_c[1], zc * 0.6), (11.0, 12.0, 10.5))
    ankle = ell((L2 - 28.0, 0.0, 0.0), (6.0, PEG_AF / 2 + 2.0, PEG_AF / 2 + 2.0))
    beam = hull(knee, calf, ankle)
    peg = octagon_x(PEG_AF, PEG_X[0], PEG_X[1], 0.0, 0.0)
    m = organic(union([u, beam])) + idler_spacer(-90.0).rotate([0, 0, KNEE_BIAS]) + peg
    m = m - u_cuts().rotate([0, 0, KNEE_BIAS])
    flat = box(-40, 40, -U_HALF_W, 40, -60, 60).rotate([0, 0, KNEE_BIAS])   # keep the bracket's bed face flat
    m = m + (joint_button(KNEE_BIAS) ^ flat) - joint_button_holes(KNEE_BIAS)   # knee button
    return m if left else m.mirror([0, 1, 0])


PAW_R = 21.0
STAND_POSE = (0.0, -49.3, 92.8)   # left legs: lower, playful crouch (~145 mm hip height)


def paw_forward(left=True):
    """Body +x (forward) expressed in the tibia frame at the default stand pose."""
    leg = "LF" if left else "RF"
    q = STAND_POSE if left else tuple(-v for v in STAND_POSE)
    R = leg_frames(leg, *q)["TIBIA"][:3, :3]
    return R.T @ np.array([1.0, 0.0, 0.0])


@lru_cache(None)
def part_foot(left=True):
    """TPU bean paw: round sole (still rolls at any shin angle), cuff round the ankle,
    three toe bumps facing forward in the stand pose. Print cuff-down, 15 % infill."""
    c = np.array([L2 - PAW_R, 0.0, 0.0])
    sole = Manifold.sphere(PAW_R, 64).translate(list(c))
    cuff = cylx(15.0, L2 - 30.0, L2 - 29.0)
    d = paw_forward(left)
    d = d - np.array([1, 0, 0]) * d[0]; d = d / np.linalg.norm(d)            # perpendicular to the shin
    toes = [Manifold.sphere(5.5, 32).translate(list(c + d * (PAW_R - 1.5) + np.array([1.5, 0, dz])))
            for dz in (-10.0, 0.0, 10.0)]
    paw = hull(sole, cuff) + union(toes)
    sock = octagon_x(PEG_AF + 0.2, PEG_X[0] - 5, PEG_X[1] + 0.3, 0.0, 0.0)
    # 45-deg roof over the peg (4 mm high, 6 mm flat): TPU can't bridge the 14 mm octagon
    sock = sock + hull(octagon_x(PEG_AF + 0.2, PEG_X[1] + 0.29, PEG_X[1] + 0.3, 0.0, 0.0),
                       octagon_x(PEG_AF + 0.2 - 8.0, PEG_X[1] + 4.29, PEG_X[1] + 4.3, 0.0, 0.0))
    return paw - sock - box(L2 - 60, L2 - 30, -40, 40, -40, 40)


# ------------------------------------------------------------------ body


def hip_servo_frames_front():
    return [T_hip("LF"), T_hip("RF")]


INSERT_POS = [(12.0, 22.0), (12.0, -22.0), (-12.0, 22.0), (-12.0, -22.0)]   # (y, z) on the flange
HEAD_BOLTS = [(100.0, 16.0), (100.0, -16.0), (116.0, 16.0), (116.0, -16.0)]
GNSS = (22.0, 20.0, 6.0)           # Beitian BE-222P (VERIFY)
GNSS_X = 106.0
CABLE_CH = (5.0, 14.0, -4.5, 4.5)  # y0, y1, z0, z1 (mirrored for the right side)


@lru_cache(None)
def part_hip_module():
    """Front hip module (the rear one is the same part turned 180 deg).

    A light box: end flange to the tray, top + bottom skins, a centre web and two
    side webs, all tying the two hip-roll cradles together."""
    x0 = MODULE_X[0]
    xe = HIP_X + CAP_TOP                    # front face = the caps' outer face (bed face)
    hy = MODULE_HALF_Y
    zt, zb = BODY_Z_TOP + DECK_T, BODY_Z_BOT
    t = 3.0
    gx, gy, gz = GNSS
    frames = hip_servo_frames_front()
    parts = [
        rbox_x(x0, x0 + 3.4, -hy, hy, zb, zt, 4.0),              # flange to the tray
        rbox(x0, xe, -hy, hy, zt - t, zt, 4.0),                  # top skin
        rbox(x0, xe, -hy, hy, zb, zb + t, 4.0),                  # bottom skin
        box(x0, xe, -2.0, 2.0, zb, zt),                          # centre web
        box(x0, xe, 10.0, 12.5, zb, zt), box(x0, xe, -12.5, -10.0, zb, zt),   # side webs
        box(GNSS_X - gx / 2 - 1.3, xe, -gy / 2 - 1.3, gy / 2 + 1.3, zt - gz - 1.8, zt),  # GNSS tub
    ]
    parts += [hull(cylx(4.2, x0, x0 + 6.0, y, z), cylx(0.5, x0, x0 + 9.7, y, z))
              for y, z in INSERT_POS]                                          # insert bosses (cone)
    parts += [hull(cylz(4.5, zt - 8, zt, x, y), cylz(0.5, zt - 8, zt, x + 6.4, y))
              for x, y in HEAD_BOLTS]                                          # neck-base bosses (teardrop)
    m = organic(union(parts + [xf(cradle_solid(), T) for T in frames]))
    keep = union([xf(cradle_solid(), T) for T in frames])
    cuts = [cylx(25.0, x0 + 3.4, xe + 10, s_ * HIP_Y, 0.0) - keep for s_ in (1, -1)]   # hip-bracket sweep
    cuts += [xf(cradle_cuts(bot=False, up=(0.0, -1.0)), T) for T in frames]
    cuts.append(box(xe, xe + 10, -40, 40, -40, 40))          # bed face stays flat
    for s_ in (1, -1):
        y0, y1, z0, z1 = CABLE_CH
        ya, yb = sorted((s_ * y0, s_ * y1))
        cuts.append(box(x0 - 1, HIP_X - 17.0, ya, yb, z0, z1))                 # servo cable
    for y, z in INSERT_POS:
        cuts.append(cylx(M3_INSERT_R, x0 - 1, x0 + 5.5, y, z, 24))
    for x, y in HEAD_BOLTS:
        cuts.append(cylz(M3_INSERT_R, zt - 6, zt + 1, x, y, 24))
    cuts.append(box(GNSS_X - gx / 2 - 0.3, GNSS_X + gx / 2 + 0.3, -gy / 2 - 0.3, gy / 2 + 0.3, zt - gz - 0.5, zt + 1))
    cuts.append(box(GNSS_X - gx / 2 - 3, GNSS_X, -2.5, 2.5, zt - 3, zt + 1))  # GNSS lead
    # skins: teardrop windows pointing up the print (-x)
    for zz in (zt, zb):
        for yc in (-21.0, 21.0):
            for xc in (80.0, 101.0):
                cuts.append(teardrop("z", xc, yc, 5.5, zz - 4, zz + 4, (-1.0, 0.0), drop=2.5))
        for yc in (-6.0, 6.0):
            cuts.append(teardrop("z", 80.0, yc, 2.6, zz - 4, zz + 4, (-1.0, 0.0)))
    for yc in (-6.0, 6.0):
        cuts.append(teardrop("z", 101.0, yc, 2.6, zb - 4, zb + 4, (-1.0, 0.0)))
    # flange windows between the bolts
    cuts.append(rbox_x(x0 - 1, x0 + 5, 15.0, 27.0, -15.0, 15.0, 3.0))
    cuts.append(rbox_x(x0 - 1, x0 + 5, -27.0, -15.0, -15.0, 15.0, 3.0))
    return m - union(cuts)


HUMP_BOSS_X = (-40.0, 40.0)     # as printed (body is final); only the right side's are used now
HUMP_TAB_X = (-55.0, 55.0)      # left side: hump tabs clamp on the deck's left corner screws instead
DECK_SIDE_X = (-30.0, 0.0, 30.0)
DECK_SIDE_Y = TRAY_HALF_Y - 5.5
HUMP_BOSS_Z = 18.0
BATTERY_X = -9.0           # battery centre (x); the F3 sits flat on the floor ahead of it
FC_X = 50.0
TRANS_X0, TRANS_X1 = 60.0, 69.4   # bay -> hip-section taper (ends 0.5 mm before the hip arm plane)


def _mx(s_, a, b):
    """x-interval [a, b] (front coordinates) mirrored for the rear end (s_ = -1)."""
    return min(s_ * a, s_ * b), max(s_ * a, s_ * b)


def _hip_end(s_):
    """Structural solids of one hip section (s_ = +1 front, -1 rear) in body coordinates."""
    zt, zb = BODY_Z_TOP + DECK_T, BODY_Z_BOT
    xe = HIP_X + CAP_TOP
    hy = MODULE_HALF_Y
    legs = ("LF", "RF") if s_ > 0 else ("LR", "RR")
    frames = [T_hip(l) for l in legs]
    xa, xb = _mx(s_, TRANS_X0 + 2, xe)
    t = 5.0                                   # skins: the only full-width material past the arm sweep
    # their inner ends grow out of the taper at 45 deg: the rear section prints on top, where a flat
    # skin end would be a 60 mm ceiling over the open taper
    xi, xf_ = _mx(s_, TRANS_X0 + 2, TRANS_X0 + 2 + 0.01)
    xs0, xs1 = _mx(s_, TRANS_X0 + 2 + t, xe)
    parts = [hull(rbox(xs0, xs1, -hy, hy, zt - t, zt, 4.0), rbox(xi, xf_, -hy, hy, zt - 0.4, zt, 4.0)),
             hull(rbox(xs0, xs1, -hy, hy, zb, zb + t, 4.0), rbox(xi, xf_, -hy, hy, zb, zb + 0.4, 4.0))]
    for y0, y1 in ((-2.0, 2.0), (8.5, 12.5), (-12.5, -8.5)):                 # centre + 4 mm side webs,
        parts.append(box(*_mx(s_, 70.0, xe), y0, y1, zb, zt))                # run into the taper
        parts.append(box(*_mx(s_, TRANS_X0 + 2, 70.5), y0, y1, (-10.0 if s_ > 0 else zb), zt))  # (clear of the F3)
    parts += [xf(cradle_solid(), T) for T in frames]
    if s_ > 0:   # neck-base bosses, teardrop toward the bed (+x)
        parts += [hull(cylz(4.5, zt - 8, zt, x, y), cylz(0.5, zt - 8, zt, x + 6.4, y)) for x, y in HEAD_BOLTS]
    else:        # GNSS tub, 45-deg underside toward the bed
        gx, gy, gz = GNSS
        g0 = GNSS_X - gx / 2 - 1.3
        parts.append(hull(box(-xe, -g0, -gy / 2 - 1.3, gy / 2 + 1.3, zt - gz - 1.8, zt),
                          box(-xe, -g0 + gz + 1.8, -gy / 2 - 1.3, gy / 2 + 1.3, zt - 0.1, zt)))
    return union(parts), frames


def _hip_end_cuts(s_, frames):
    zt, zb = BODY_Z_TOP + DECK_T, BODY_Z_BOT
    xe = HIP_X + CAP_TOP
    up = (0.0, -1.0) if s_ > 0 else (0.0, 1.0)          # print up (-x) in the hip servos' (x, z) plane
    keep = union([xf(cradle_solid(), T) for T in frames])
    cuts = [cylx(25.0, *_mx(s_, TRANS_X1, xe + 10), y_ * HIP_Y, 0.0) - keep for y_ in (1, -1)]
    cuts += [xf(cradle_cuts(bot=True, up=up), T) for T in frames]   # 2 + 2 screws; the inner pair from the bay
    for y_ in (1, -1):                                   # servo cable notch through the side webs
        ya, yb = sorted((y_ * 8.5, y_ * 14.0))
        cuts.append(box(*_mx(s_, 72.0, HIP_X - 17.0), ya, yb, -4.5, 4.5))
    for zz in (zt, zb):                                  # skin windows (point up the print = -x)
        for yc in (-21.0, 21.0):
            for xc in (80.0, 101.0):
                cuts.append(cylz(4.0, zz - 4, zz + 4, s_ * xc, yc, 32))
    if s_ > 0:
        for x, y in HEAD_BOLTS:
            cuts.append(cylz(M3_INSERT_R, zt - 6, zt + 1, x, y, 24))
    else:
        gx, gy, gz = GNSS
        cuts.append(box(-GNSS_X - gx / 2 - 0.3, -GNSS_X + gx / 2 + 0.3, -gy / 2 - 0.3, gy / 2 + 0.3, zt - gz - 0.5, zt + 1))
        # GNSS lead: room for the connector + the wires' bend (15 x 8 mm, through the 5 mm top skin so
        # the cable can drop into the hip section), then a channel to the bay under the deck edge
        ex = -GNSS_X + gx / 2 + 0.3
        cuts.append(box(ex - 1.0, ex + 8.0, -7.5, 7.5, zt - 5.5, zt + 1))
        cuts.append(box(ex + 7.0, -TRAY_HALF_X + 2.0, -4.0, 4.0, zt - 3.5, zt + 1))
    return cuts


@lru_cache(None)
def part_body():
    """The whole main body in one print (fits 256 x 256; prints standing on its
    front face, 239 mm tall): battery bay, tapers, both hip sections, F3 + GNSS
    mounts. The deck is the lid."""
    hx, hy = TRAY_HALF_X, TRAY_HALF_Y
    zt, zb = BODY_Z_TOP, BODY_Z_BOT
    w = 3.0
    xe = HIP_X + CAP_TOP
    # loaf cross-section: round belly (r 13, fits the round cells of the pack), soft top
    loaf = CrossSection.batch_hull([CrossSection.circle(13, 64).translate([sy * (hy - 13), zb + 13]) for sy in (1, -1)]
                                   + [CrossSection.circle(5, 32).translate([sy * (hy - 5), zt - 5]) for sy in (1, -1)])
    loaf_in = loaf.offset(-w)
    def along_x(cs, x0, x1):
        return cs.extrude(x1 - x0).transform(np.array([[0, 0, 1, x0], [1, 0, 0, 0], [0, 1, 0, 0]], float))
    inner_open = hull(along_x(loaf_in, -hx - 1, hx + 1), along_x(loaf_in, -hx - 1, hx + 1).translate([0, 0, 20]))
    bay = along_x(loaf, -hx, hx) - inner_open
    parts = [bay]
    for s_ in (1, -1):   # tapers: loaf -> hip-section cross-section, open inside
        o = hull(along_x(loaf, *_mx(s_, TRANS_X0 - 0.1, TRANS_X0)),
                 box(*_mx(s_, TRANS_X1 - 0.1, TRANS_X1), -MODULE_HALF_Y, MODULE_HALF_Y, zb, zt + DECK_T))
        lt = loaf.offset(-3.5)
        i = hull(along_x(lt, *_mx(s_, TRANS_X0 - 1.1, TRANS_X0 - 1.0)).translate([0, 0, 0]),
                 along_x(lt, *_mx(s_, TRANS_X0 - 1.1, TRANS_X0 - 1.0)).translate([0, 0, 20]),
                 box(*_mx(s_, TRANS_X1 + 1.0, TRANS_X1 + 1.1), -MODULE_HALF_Y + 3.5, MODULE_HALF_Y - 3.5, zb + 3.5, zt + 10))
        parts.append(o - i)
    # stub tail on the rear face (top of the print), clear of the hip-servo screw holes
    parts.append(hull(cylx(8.0, -xe - 0.1, -xe + 2, 0.0, -4.0), Manifold.sphere(6.0, 32).translate([-xe - 4, 0, -3])))
    # bosses for the hump cover's 4 side screws (teardrop toward the bed, +x)
    parts += [hull(cyly(4.5, *sorted((sy * (hy - 3), sy * (hy + 5))), x, HUMP_BOSS_Z),
                   cyly(0.5, *sorted((sy * (hy - 3), sy * (hy + 5))), x + 6.3, HUMP_BOSS_Z))
              for x in HUMP_BOSS_X for sy in (1, -1)]
    ends = [_hip_end(1), _hip_end(-1)]
    parts += [e[0] for e in ends]
    # deck bosses (teardrop toward the bed, +x) and F3 standoffs on the floor
    bos = [(sx * (hx - 5), sy * (hy - 5.5)) for sx in (1, -1) for sy in (1, -1)]
    parts += [hull(cylz(4.2, zb, zt, x, y), cylz(0.5, zb, zt, x + 5.9, y)) for x, y in bos]
    # 6 more deck screws along the long edges: bolted down every ~30 mm the deck closes the
    # open-topped bay into a torsion box (short bosses hang from the wall top, above the pack)
    side = [(x, sy * DECK_SIDE_Y) for x in DECK_SIDE_X for sy in (1, -1)]
    parts += [hull(cylz(3.6, zt - 5.5, zt, x, y), cylz(0.5, zt - 5.5, zt, x + 5.0, y)) for x, y in side]
    h = FC_HOLES / 2
    fc = [(FC_X + sx * h, sy * h) for sx in (1, -1) for sy in (1, -1)]
    parts += [hull(cylz(3.5, zb, zb + w + 8, x, y), cylz(0.5, zb, zb + w + 8, x + 4.9, y)) for x, y in fc]
    m = organic(union(parts))
    cuts = [box(xe, xe + 20, -60, 60, -60, 60),                              # bed face (front) stays flat
            box(-hx - 1, hx + 1, -60, 60, zt, zt + 10)]                      # deck seat stays flat
    for s_, (_, frames) in zip((1, -1), ends):
        cuts += _hip_end_cuts(s_, frames)
    cuts += [cylz(M3_INSERT_R, zt - 6, zt + 1, x, y, 24) for x, y in bos]
    cuts += [cylz(M3_INSERT_R, zt - 5, zt + 1, x, y, 24) for x, y in side]
    cuts += [cylz(M3_INSERT_R, zb + w + 2, zb + w + 9, x, y, 24) for x, y in fc]
    cuts += [cyly(M3_INSERT_R, *sorted((sy * (hy - 1), sy * (hy + 6))), x, HUMP_BOSS_Z, 24)
             for x in HUMP_BOSS_X for sy in (1, -1)]
    # left wall: power switch (12 mm) + charge jack (8 mm)
    cuts.append(cyly(6.1, hy - 5, hy + 1, 45.0, 0.0))
    cuts.append(cyly(4.0, hy - 5, hy + 1, 28.0, 0.0))
    # a few small round vents in the side walls (no pointed cut-outs on the outside)
    for y_ in (1, -1):
        for xc in ((-35.0, -15.0) if y_ > 0 else (-35.0, -15.0, 5.0, 25.0)):
            cuts.append(cyly(3.0, -hy - 6, hy + 6, xc, -6.0) ^ box(-200, 200, *sorted((y_ * (hy - 6), y_ * (hy + 6))), -60, 60))
    m = m - union(cuts)             # (the floor is the box's bottom flange: no windows)
    grip = hip_roll_saddle() + cradle_ribs(HR_SADDLE_X1, HR_SADDLE_X1, bottom_holes=True) if RIBS else hip_roll_saddle()
    for _, frames in ends:
        for T in frames:
            m = m + xf(grip, T)
    return m + stand_rail()


@lru_cache(None)
def part_tray():
    """Battery bay and the body's main beam: the side walls are trusses,
    the floor a frame the pack sits on."""
    hx, hy = TRAY_HALF_X, TRAY_HALF_Y
    zt, zb = BODY_Z_TOP, BODY_Z_BOT
    w = 2.5
    m = rbox(-hx, hx, -hy, hy, zb, zt, CORNER_R) - rbox(-hx + w, hx - w, -hy + w, hy - w, zb + w, zt + 1, CORNER_R - w)
    bosses = [cylz(4.5, zb, zt, sx * (hx - 7), sy * (hy - 7)) for sx in (1, -1) for sy in (1, -1)]
    h = FC_HOLES / 2
    fc = [cylx(3.5, hx - w - 8, hx - w + 0.5, sy * h, sz * h) for sy in (1, -1) for sz in (1, -1)]
    m = organic(union([m] + bosses + fc))
    cuts = [box(-hx - 5, hx + 5, -hy - 5, hy + 5, zt, zt + 10)]           # deck face stays flat
    cuts += [cylz(M3_INSERT_R, zt - 6, zt + 1, sx * (hx - 7), sy * (hy - 7), 24) for sx in (1, -1) for sy in (1, -1)]
    cuts += [cylx(M3_INSERT_R, hx - w - 9, hx - w - 2, sy * h, sz * h, 24) for sy in (1, -1) for sz in (1, -1)]
    for sx in (1, -1):
        for y, z in INSERT_POS:
            cuts.append(cylx(1.7, sx * hx - 5, sx * hx + 5, y, z, 24))
        for s_ in (1, -1):
            y0, y1, z0, z1 = CABLE_CH
            ya, yb = sorted((s_ * y0, s_ * y1))
            cuts.append(box(sx * hx - 5, sx * hx + 5, ya, yb, z0, z1))
    # battery strap slots (20 mm hook-and-loop)
    for xc in (-22.0, 22.0):
        for sy in (1, -1):
            cuts.append(box(xc - 11, xc + 11, sy * 38.5 - 1.5, sy * 38.5 + 1.5, zb - 1, zb + w + 1))
    # left wall: power switch (12 mm) + charge jack (8 mm)
    cuts.append(cyly(6.1, hy - 5, hy + 1, 45.0, 0.0))
    cuts.append(cyly(4.0, hy - 5, hy + 1, 28.0, 0.0))
    # side walls: organic drop windows (point up)
    for sy in (1, -1):
        for xc in ((-42.0, -14.0, 12.0) if sy > 0 else (-42.0, -14.0, 14.0, 42.0)):
            y0, y1 = (hy - 5, hy + 1) if sy > 0 else (-hy - 1, -hy + 5)
            cuts.append(teardrop("y", xc, 1.0, 9.0, y0, y1, (0.0, 1.0), drop=9.0))
    # floor: a frame + cross, the pack rests on it
    for sx in (1, -1):
        for sy in (1, -1):
            cuts.append(rbox(min(sx * 6, sx * 34), max(sx * 6, sx * 34), min(sy * 7, sy * 31), max(sy * 7, sy * 31),
                             zb - 1, zb + w + 1, 4.0))
        cuts.append(rbox(min(sx * 44, sx * 56), max(sx * 44, sx * 56), -25, 25, zb - 1, zb + w + 1, 3.0))
    return m - union(cuts)


HUMP = dict(x=68.0, y=46.0, z0=29.0, zwall=58.0, ztop=74.0, skirt_z=12.0)


@lru_cache(None)
def part_hump():
    """Back hump over the Jetson: flat plateau (bed face), graded shoulders, vertical
    walls; side skirts drop past the deck and screw into the body's side bosses. The gap
    under the rim is the air intake, round vents on the plateau the exhaust."""
    hx, hy, z0, zw, zt = HUMP["x"], HUMP["y"], HUMP["z0"], HUMP["zwall"], HUMP["ztop"]

    def ring(off, za, zb_):
        return rbox(-hx + 16 - off, hx - 16 + off, -hy + 16 - off, hy - 16 + off, za, zb_, 8 + off)
    outer = hull(ring(0, zt - 0.1, zt), ring(10, zt - 10.1, zt - 10), ring(13, zt - 14.1, zt - 14),
                 ring(16, z0, zw))
    shell = outer - _erode(outer, 2.5) - box(-100, 100, -100, 100, 0, z0 + 2.5)
    skirt_in = hy - 2.5
    # right skirt screws into the body's side bosses; the left wall is the Jetson port bay, so the
    # left side is held by two tabs on the deck's left corner screws (reached through the shoulders)
    ty = TRAY_HALF_Y - 5.5
    skirts = [box(-50, 50, -hy, -skirt_in, HUMP["skirt_z"], z0 + 3)]
    skirts += [hull(cylz(4.5, BODY_Z_TOP + DECK_T, z0 + 4, x, ty),
                    box(x - 4.5, x + 4.5, skirt_in - 1.0, skirt_in + 0.5, BODY_Z_TOP + DECK_T, z0 + 9)) for x in HUMP_TAB_X]
    m = organic(union([shell] + skirts), 2.0, 0.8)
    cuts = [cyly(M3_CLR, -hy - 5, -hy + 5, x, HUMP_BOSS_Z, 24) for x in HUMP_BOSS_X]
    cuts += [cyly(M3_HEAD_R, -hy - 5, -hy + 1.0, x, HUMP_BOSS_Z, 24) for x in HUMP_BOSS_X]
    for x in HUMP_TAB_X:                                    # tab screw + driver access through the shoulder
        cuts.append(cylz(M3_CLR, BODY_Z_TOP, z0 + 5, x, ty, 24))
        cuts.append(cylz(3.6, z0 + 4, zt + 5, x, ty, 32))
    # Jetson ports (USB-C, HDMI, USB, Ethernet) face the left wall: open it from the rim up past the
    # tallest port. Open at the rim, so printed plateau-down it needs no bridge.
    pz1 = BODY_Z_TOP + DECK_T + JETSON_STANDOFF_H + 1.6 + 16.0
    cuts.append(hull(box(-50.0, 50.0, hy - 6.0, hy + 2.0, HUMP["skirt_z"] - 1, pz1 - 5.0),
                     box(-45.0, 45.0, hy - 6.0, hy + 2.0, pz1 - 0.1, pz1)))     # 45-deg top corners
    for i in range(-3, 4):                                  # round vents on the plateau
        for j in range(-1, 2):
            cuts.append(cylz(2.8, zt - 5, zt + 1, i * 12.0 + (6.0 if j % 2 else 0.0), j * 11.0, 24))
    return m - union(cuts)


@lru_cache(None)
def part_deck():
    """Top flange of the body's torsion box + Jetson standoffs. Screwed down at 10 points
    and triangulated (perimeter band, centre cross, diagonals) so it carries shear."""
    hx, hy = TRAY_HALF_X, TRAY_HALF_Y
    z0, z1 = BODY_Z_TOP, BODY_Z_TOP + DECK_T
    jx, jy = JETSON_HOLES
    jpts = [(sx * jx / 2, JETSON_HOLES_Y0 + sy * jy / 2) for sx in (1, -1) for sy in (1, -1)]
    m = rbox(-hx, hx, -hy, hy, z0, z1, CORNER_R)
    so = [cylz(4.0, z0, z1 + JETSON_STANDOFF_H, x, y) for x, y in jpts]
    m = organic(union([m] + so), 2.0, 0.8)
    cuts = [box(-hx - 5, hx + 5, -hy - 5, hy + 5, z0 - 5, z0)]           # bed face stays flat
    cuts += [cylz(M3_INSERT_R, z1 + JETSON_STANDOFF_H - 6, z1 + JETSON_STANDOFF_H + 1, x, y, 24)
             for x, y in jpts]  # M3 heat-set inserts
    screws = [(sx * (hx - 5), sy * (hy - 5.5)) for sx in (1, -1) for sy in (1, -1)]
    screws += [(x, sy * DECK_SIDE_Y) for x in DECK_SIDE_X for sy in (1, -1)]
    for x, y in screws:
        cuts.append(cylz(M3_CLR, z0 - 1, z1 + 1, x, y, 24))
        cuts.append(cylz(3.1, z1 - 1.2, z1 + 1, x, y, 24))
    # truss windows: inner field minus centre cross and diagonals, corners rounded
    ix, iy, band = hx - 9.0, hy - 10.5, 8.0
    field = CrossSection.square([2 * ix, 2 * iy], True)
    diag = math.degrees(math.atan2(iy, ix))
    L = 2 * math.hypot(ix, iy) + 10
    bars = CrossSection.batch_boolean([CrossSection.square([band, 2 * iy + 2], True),
                                       CrossSection.square([2 * ix + 2, band], True),
                                       CrossSection.square([L, band], True).rotate(diag),
                                       CrossSection.square([L, band], True).rotate(-diag)], OpType.Add)
    feet = CrossSection.batch_boolean([CrossSection.circle(6.5, 32).translate([x, y]) for x, y in jpts], OpType.Add)
    win = (field - bars - feet).offset(-2.0).offset(2.0)      # solid plate under every standoff
    cuts.append(win.extrude(z1 - z0 + 2).translate([0, 0, z0 - 1]))
    return m - union(cuts)


# ------------------------------------------------------------------ neck + head (2x DS3218)
# DS frame: origin on the shaft axis at the case top, +Z = shaft out, +X = toward the output end


def ds_case():
    case = box(-DS_L_BACK, DS_L_OUT, -DS_HALF_W, DS_HALF_W, -DS_H, 0)
    mid = (DS_L_OUT - DS_L_BACK) / 2
    ears = box(mid - DS_EAR_HALF_SPAN, mid + DS_EAR_HALF_SPAN, -DS_HALF_W + 0.5, DS_HALF_W - 0.5,
               DS_EAR_UNDER, DS_EAR_UNDER + DS_EAR_T)
    return case + ears


@lru_cache(None)
def ds_envelope(c=0.4):
    """Case + ears grown by c: cut from holders so fillets never touch the servo."""
    case = box(-DS_L_BACK - c, DS_L_OUT + c, -DS_HALF_W - c, DS_HALF_W + c, -DS_H - c, c)
    mid = (DS_L_OUT - DS_L_BACK) / 2
    ears = box(mid - DS_EAR_HALF_SPAN - c, mid + DS_EAR_HALF_SPAN + c, -DS_HALF_W, DS_HALF_W,
               DS_EAR_UNDER, DS_EAR_UNDER + DS_EAR_T + 3)
    return case + ears


def ds_horn(angle=0.0):
    """Hub + flat lever of the aluminium arm; lever points along `angle` (deg, about +Z)."""
    hub = cylz(DS_HUB_R, 0, DS_HUB_H)
    lever = hull(cylz(DS_HUB_R, DS_HUB_H, DS_HUB_H + DS_LEVER_T),
                 cylz(4.0, DS_HUB_H, DS_HUB_H + DS_LEVER_T, DS_LEVER_L - 4, 0))
    return hub + lever.rotate([0, 0, angle])


@lru_cache(None)
def ds_bulkhead_cut():
    """Through-cut for the case below the ears; M3 clearance + captive hex-nut traps
    on the back of the 4 mm bulkhead for the 4 ears (M3x8 + nut)."""
    # only through the 4 mm bulkhead itself -- it used to run 10 mm past the case bottom and
    # silently cut away the neck pivot boss on the far side (the case room comes from ds_envelope)
    c = [box(-DS_L_BACK - 0.3, DS_L_OUT + 0.3, -DS_HALF_W - 0.3, DS_HALF_W + 0.3, DS_EAR_UNDER - 6.0, DS_EAR_UNDER + 0.01)]
    back = DS_EAR_UNDER - 4.0
    for x in DS_EAR_HOLE_X:
        for y in (DS_EAR_HOLE_Y, -DS_EAR_HOLE_Y):
            c.append(cylz(M3_CLR, DS_EAR_UNDER - 9, DS_EAR_UNDER + 1, x, y, 24))
            c.append(cylz(M3_NUT_R, back - 5, back + 2.5, x, y, 6))
    return union(c)


def lever_mount_cuts(angle, y0, y1):
    """Centre-screw hole + radial slot for the lever screws (hole spacing is unknown,
    a slot takes any). In an x-z plate spanning y0..y1 (head frame axes)."""
    d = (math.cos(math.radians(angle)), math.sin(math.radians(angle)))
    return union([cyly(3.5, y0 - 1, y1 + 1),
                  hull(cyly(M3_CLR, y0 - 1, y1 + 1, 9 * d[0], 9 * d[1], 24),
                       cyly(M3_CLR, y0 - 1, y1 + 1, 25 * d[0], 25 * d[1], 24))])


Y0 = HEAD_Y0
N_TOP = Y0 + 17.4                      # neck servo case top (its +y face)
N_BOT = N_TOP - DS_H
HP_IN = N_TOP + DS_HUB_H + DS_LEVER_T  # head +y plate sits on the neck lever
HP_OUT = HP_IN + 4.0
HN_OUT, HN_IN = N_BOT - 12.0, N_BOT - 7.5    # head -y plate (bearing side)
J_TOP = HP_IN - 0.5                    # jaw servo case top; its lever clears the (3 mm) +y plate
J_BOT = J_TOP - DS_H
JP_IN = J_TOP + DS_HUB_H + DS_LEVER_T  # jaw +y arm sits on the jaw lever
JX, JZ = JAW_AXIS
JAW_LEVER_ANGLE = -43.0                # lever/slot points from hinge toward the chin
FRONT = 80.0                           # face plane (head frame x)
HEAD_YC = (HN_OUT + HP_OUT) / 2        # head centre line (y)
EYE_Z, EYE_DY = 6.0, 22.0              # eyes on the flat face; the camera is the left eye
EYE_R = 11.0
CAM_ZH = EYE_Z + 1.2
CAM_YC = HEAD_YC + EYE_DY

# neck servo in the body frame, jaw servo in the head frame
T_NECK = frame((1, 0, 0), (0, 0, -1), (0, 1, 0), (NECK_X, N_TOP, NECK_Z))
T_JAW = frame((0, 0, -1), (-1, 0, 0), (0, 1, 0), (JX, J_TOP, JZ))


def ry(deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    T = np.eye(4)
    T[0, 0], T[0, 2], T[2, 0], T[2, 2] = c, s, -s, c
    return T


def head_frames(neck=0.0, jaw=0.0):
    """+neck = nose down, +jaw = mouth opens."""
    TH = trans(NECK_X, 0, NECK_Z) @ ry(neck)
    TJ = TH @ trans(JX, 0, JZ) @ ry(jaw) @ trans(-JX, 0, -JZ)
    return TH, TJ


@lru_cache(None)
def part_neck_base():
    """Bolts onto the front hip module; holds the neck servo, pivot for the head."""
    z0 = BODY_Z_TOP + DECK_T
    ear_y = N_TOP + DS_EAR_UNDER
    py0 = HN_IN + 0.5
    floor = rbox(94, 154, py0, Y0 + 20, z0, z0 + 3, 4)
    bulk = box(94, 154, ear_y - 4, ear_y, z0, NECK_Z + 15)
    pivot = box(118, 154, N_BOT - 4.0, N_BOT - 0.5, z0, NECK_Z + 13)        # 3.5 plate...
    boss = cyly(6.5, py0, N_BOT - 0.5, NECK_X, NECK_Z)                      # ...+ pivot boss
    web = box(94, 98, py0, ear_y, z0, NECK_Z + 5)
    # bridge over the neck servo: pivot plate <-> bulkhead, so the pivot isn't a lone cantilever
    bridge = box(120, 152, N_BOT - 4.0, ear_y, NECK_Z + DS_HALF_W + 1.0, NECK_Z + 15)
    m = organic(union([floor, bulk, pivot, boss, web, bridge]), 2.0, 0.8)
    cuts = [box(80, 170, -60, 60, z0 - 10, z0),                          # bed face stays flat
            xf(ds_bulkhead_cut(), T_NECK), xf(ds_envelope(), T_NECK),
            cyly(M3_INSERT_R, py0 - 1, py0 + 5.5, NECK_X, NECK_Z, 24)]
    for x, y in HEAD_BOLTS:
        cuts.append(cylz(M3_CLR, z0 - 1, z0 + 4, x, y, 24))
    cuts.append(hull(box(150, 160, -40, 40, z0 - 1, z0 + 3.01), box(154, 160, -40, 40, z0 - 1, z0 + 7)))
    # floor windows under the servo
    cuts.append(rbox(122, 146, py0 + 4, ear_y - 6, z0 - 1, z0 + 4, 3))
    cuts.append(rbox(104, 112, -8, 10, z0 - 1, z0 + 4, 3))
    cuts.append(teardrop("y", 128.0, 40.0, 4.5, N_BOT - 5, N_BOT + 1, (0.0, 1.0)))
    return m - union(cuts)


HEAD_TOP = 40.0                        # flat crown (bed face, head prints upside-down)
HEAD_EQ = (-10.0, 18.0)                # vertical band of the face
HEAD_HW = 47.5                         # half width at the cheeks: wider than the body
MOUTH_Z = -32.0                        # upper lip
MUZZLE_TIP = 96.0
EAR_DY = 17.0
EAR_BOLTS = [(16.0, HEAD_YC), (56.0, HEAD_YC)]


@lru_cache(None)
def head_outer():
    """Teddy head: flat crown, 45-deg shoulders (exactly: same corner centres, radius +16),
    vertical face band, round cheeks/jowls."""
    def ring(off, z0, z1):   # crown outline grown by `off`: same corner centres, radius 8 + off
        return rbox(8 - off, 64 + off, HEAD_YC - 31.5 - off, HEAD_YC + 31.5 + off, z0, z1, 8 + off)
    # shoulder curve: 45 deg off the crown, then steeper (53, 63, 76 deg) into the face band --
    # every step is >= 45 deg so it prints crown-down, but it reads round, not chamfered
    rings = [ring(0, HEAD_TOP - 0.1, HEAD_TOP), ring(10, 29.9, 30), ring(13, 25.9, 26), ring(15, 21.9, 22),
             ring(16, HEAD_EQ[0], HEAD_EQ[1])]
    cheeks = [Manifold.sphere(18, 48).translate([x, HEAD_YC + s_ * (HEAD_HW - 18), HEAD_EQ[0] - 6])
              for x in (12, 60) for s_ in (1, -1)]
    return hull(*rings, *cheeks)


def muzzle():
    """Short round muzzle; its top falls at 45 deg to the nose (printable upside-down)."""
    root = rbox_x(70, FRONT, HEAD_YC - 24, HEAD_YC + 24, MOUTH_Z, EYE_Z - EYE_R - 1, 6)
    front = [Manifold.sphere(5, 32).translate([MUZZLE_TIP - 5, HEAD_YC + s_ * 13, MOUTH_Z + 5]) for s_ in (1, -1)]
    return hull(root, *front)


def nose():
    return hull(Manifold.sphere(3.2, 24).translate([MUZZLE_TIP - 3.5, HEAD_YC - 5.5, MOUTH_Z + 10.0]),
                Manifold.sphere(3.2, 24).translate([MUZZLE_TIP - 3.5, HEAD_YC + 5.5, MOUTH_Z + 10.0]),
                Manifold.sphere(2.6, 24).translate([MUZZLE_TIP - 1.2, HEAD_YC, MOUTH_Z + 5.0]))


UPPER_CANINES = [(88.0, HEAD_YC + 9.0), (88.0, HEAD_YC - 9.0)]
LOWER_CANINES = [(83.0, HEAD_YC + 13.0), (83.0, HEAD_YC - 13.0)]


def canine(x, y, z_base, up):
    """4 mm canine with a rounded tip; hidden in a pocket in the other jaw when closed."""
    zt = z_base + up * 4.0
    return hull(cylz(2.0, min(z_base, z_base - up * 0.5), max(z_base, z_base - up * 0.5), x, y, 20),
                Manifold.sphere(0.9, 12).translate([x, y, zt - up * 0.9]))


@lru_cache(None)
def part_head():
    """Cub head: shell (crown, shoulders, face, cheeks) around the neck yoke, jaw servo and
    camera. Head frame, prints crown-down."""
    outer = head_outer()
    shell = outer - _erode(outer, 2.5)
    shell = shell - hull(cyly(32, -80, 80), box(-90, -80, -80, 80, -90, -10))   # rounded neck opening
    shell = shell - hull(box(-60, 18, -80, 80, -80, -14), box(-60, 10, -80, 80, -80, -6))  # no thin fins behind the jaw hinge
    shell = shell - box(-60, 200, HEAD_YC - 40.5, HEAD_YC + 40.5, -80, HEAD_EQ[0] - 2)  # jaw opening
    shell = shell - cyly(4.0, -80, 80, JX, JZ)                                   # cheek dimples = jaw screw access

    def plate(y0, y1):
        return union([hull(cyly(14, y0, y1), box(0, 70, y0, y1, -16, HEAD_TOP)), cyly(12, y0, y1, JX, JZ)])

    ear_y = J_TOP + DS_EAR_UNDER
    inner = union([
        plate(HP_IN, HP_OUT - 1.0),                                          # +y yoke plate (3 mm)
        plate(HN_OUT, HN_IN),                                                # -y yoke plate (bearing)
        box(24, 64, ear_y - 4, ear_y, -27, HEAD_TOP),                        # jaw servo bulkhead
        cyly(5.5, HN_IN - 0.1, HN_IN + 2.5, JX, JZ),                         # jaw pivot boss
    ] + [cylz(4.5, HEAD_TOP - 8, HEAD_TOP, x, y) for x, y in EAR_BOLTS])
    inner = organic(inner, 2.0, 0.8) ^ outer
    m = union([shell, inner, muzzle(), nose()]
              + [canine(x, y, MOUTH_Z, -1.0) for x, y in UPPER_CANINES])
    # camera on 4 printed pins behind the left eye (heat-stake or CA)
    cx, cz = CAM_HOLES
    for sy in (1, -1):
        for sz in (1, -1):
            m = m + cylx(0.95, FRONT - 5.1, FRONT - 2.4, CAM_YC + sy * cx / 2, CAM_ZH + sz * cz / 2, 16)
    m = m - box(-60, 200, -80, 80, HEAD_TOP, 100)                            # crown stays flat
    cuts = [
        lever_mount_cuts(0.0, HP_IN, HP_OUT),                                # neck lever
        cyly(DS_HUB_R + 1, HP_IN - 1, HP_OUT + 1, JX, JZ),                   # jaw hub...
        box(JX - DS_HUB_R - 1, JX + DS_HUB_R + 1, HP_IN - 1, HP_OUT + 1, -80, JZ),   # ...slides in from below
        cyly(5.05, HN_OUT - 1, HN_OUT + 4.0),                                # 623ZZ bearing, neck pivot
        cyly(3.25, HN_OUT - 1, HN_IN + 1),
        cyly(M3_INSERT_R, HN_OUT - 1, HN_IN + 1.5, JX, JZ, 24),              # jaw pivot insert
        xf(ds_bulkhead_cut(), T_JAW), xf(ds_envelope(), T_JAW),
        # the jaw servo goes in from below through the open mouth: slot the bulkhead downward
        box(JX - DS_HALF_W - 0.4, JX + DS_HALF_W + 0.4, ear_y - 5, ear_y + 0.01, -80, JZ + DS_L_BACK),
    ]
    for x, y in EAR_BOLTS:
        cuts.append(cylz(M3_INSERT_R, HEAD_TOP - 6, HEAD_TOP + 1, x, y, 24))
    for s_ in (1, -1):                                                       # eyes: shallow sockets
        cuts.append(cylx(EYE_R, FRONT - 1.2, FRONT + 5, HEAD_YC + s_ * EYE_DY, EYE_Z))
    cuts.append(cylx(4.6, FRONT - 6, FRONT + 5, CAM_YC, EYE_Z))                 # left eye = camera lens
    cuts.append(cylx(4.6, FRONT - 2.4, FRONT + 5, HEAD_YC - EYE_DY, EYE_Z))     # right eye pupil
    for x, y in LOWER_CANINES:                                               # pockets for the lower canines
        cuts.append(cylz(2.7, MOUTH_Z - 0.1, MOUTH_Z + 4.8, x, y, 20))
    return m - union(cuts)


@lru_cache(None)
def part_ear_cap():
    """Two round cub ears on a small crown plate (2x M3 into the head). Prints plate-down."""
    z0 = HEAD_TOP
    plate = rbox(8, 64, HEAD_YC - 31.5, HEAD_YC + 31.5, z0, z0 + 2.0, 8)
    ears = []
    for s_ in (1, -1):
        e = cylx(13.0, 12.0, 19.0, 0.0, 13.0, 48) - cylx(8.5, 17.0, 19.5, 0.0, 13.0, 48)   # cupped front
        e = e.rotate([s_ * 15.0, 0, 0])                                                    # splay outward
        ears.append(e.translate([0, HEAD_YC + s_ * EAR_DY, z0 + 1.0]))
    m = organic(union([plate] + ears), 1.5, 0.6)
    m = m - box(-20, 60, -80, 80, z0 - 10, z0)                              # flat underside
    for x, y in EAR_BOLTS:
        m = m - cylz(M3_CLR, z0 - 1, z0 + 10, x, y, 24) - cylz(M3_HEAD_R, z0 + 1.2, z0 + 30, x, y, 24)
    return m


JAW_YB0, JAW_YA1 = HN_OUT - 5.0, JP_IN + 4.0     # outer faces of the jaw arms
CHIN_R = 22.0


@lru_cache(None)
def part_jaw():
    """Lower jaw: a round chin pod under the muzzle, hinged on the jaw servo (+y) and a
    bearing (-y). Head frame, mouth closed. Prints on its floor."""
    ya0, ya1 = JP_IN, JAW_YA1              # +y arm on the lever
    yb0, yb1 = JAW_YB0, HN_OUT - 0.5       # -y arm with bearing
    lip = MOUTH_Z - 0.5
    zf = lip - 14.0                        # underside of the chin

    def arm(y0, y1):
        return hull(cyly(10, y0, y1, JX, JZ), box(46, 58, y0, y1, zf, zf + 7))

    pod = hull(box(46, 60, HEAD_YC - CHIN_R, HEAD_YC + CHIN_R, zf, lip),
               cylz(CHIN_R, zf, lip, MUZZLE_TIP - 2 - CHIN_R, HEAD_YC, 64))
    bar = box(46, 58, yb0, ya1, zf, zf + 7)                                   # ties both arms to the pod
    m = organic(union([arm(ya0, ya1), arm(yb0, yb1), pod, bar]), 2.0, 0.8)
    m = m - box(0, 150, -80, 80, zf - 10, zf)                                 # bed face stays flat
    m = m + union([canine(x, y, lip, 1.0) for x, y in LOWER_CANINES])
    ang = JAW_LEVER_ANGLE
    d = (math.cos(math.radians(ang)), math.sin(math.radians(ang)))
    cuts = [cyly(3.5, ya0 - 1, ya1 + 1, JX, JZ),
            hull(cyly(M3_CLR, ya0 - 1, ya1 + 1, JX + 9 * d[0], JZ + 9 * d[1], 24),
                 cyly(M3_CLR, ya0 - 1, ya1 + 1, JX + 25 * d[0], JZ + 25 * d[1], 24)),
            cyly(5.05, yb0 - 1, yb0 + 4.0, JX, JZ),
            cyly(3.25, yb0 - 1, yb1 + 1, JX, JZ)]
    cuts += [cylz(2.7, lip - 4.8, lip + 0.1, x, y, 20) for x, y in UPPER_CANINES]   # upper canine pockets
    return m - union(cuts)


@lru_cache(None)
def part_fit_test_cradle():
    """Stand-alone cradle for checking the servo pocket, screw holes and crush ribs."""
    m = organic(cradle_solid()) - cradle_cuts(up=(1.0, 0.0)) - box(CAP_X1 - 10, CAP_X1, -40, 40, -40, 40)
    return m + cradle_ribs() if RIBS else m


@lru_cache(None)
def part_fit_test_hip_roll():
    """The body's hip-roll grip on its own (cradle + saddle + rails + ribs): print this before
    reprinting the body to check the servo slides in firmly and doesn't rock."""
    m = organic(cradle_solid()) - cradle_cuts(up=(1.0, 0.0)) - box(CAP_X1 - 10, CAP_X1, -40, 40, -40, 40)
    m = m + hip_roll_saddle()
    return m + cradle_ribs(HR_SADDLE_X1, HR_SADDLE_X1, bottom_holes=True) if RIBS else m


@lru_cache(None)
def part_fit_test_bracket():
    """Bare U bracket for checking horn spacing / hole pattern."""
    return u_bracket()


def jaw_lever_angle_ds():
    a = math.radians(JAW_LEVER_ANGLE)
    return math.degrees(math.atan2(-math.cos(a), -math.sin(a)))


def head_groups(neck=0.0, jaw=0.0):
    """(static neck parts in body frame, moving head, moving jaw) incl. servos and levers."""
    TH, TJ = head_frames(neck, jaw)
    H0 = np.linalg.inv(trans(NECK_X, 0, NECK_Z))
    static = part_neck_base() + xf(ds_case(), T_NECK)
    head = xf(part_head() + xf(ds_case(), T_JAW) + xf(ds_horn(0.0), H0 @ T_NECK), TH)
    jaw_g = xf(part_jaw() + xf(ds_horn(jaw_lever_angle_ds()), T_JAW), TJ)
    return static, head, jaw_g


# ------------------------------------------------------------------ test / display stand
# A dovetail rail under the belly slides into the stand's socket from the rear until the 45-deg
# ramps meet; one M3 thumb screw from the side locks it. The body prints standing on its front
# face, so the rail is a vertical extrusion there and its front ramp overhangs only 45 deg.
RAIL_X = (-25.0, 45.0)     # flat part of the rail, around the robot's centre of mass (x ~ +15)
RAIL_D = 5.0               # depth below the belly
RAIL_ROOT_HW = 7.0         # half width at the belly; 45-deg flanks -> 12 at the tip
STAND_CLR = 0.3            # socket clearance per side
STAND_COL_X = 15.0         # column under the centre of mass
STAND_TABLE_Z = -233.0     # table, in body coordinates: belly 205 mm up, paws (lowest -210) clear
STAND_ARM = 160.0          # base arm length from the column (base 320 mm across, printed diagonally)


def _ccw(pts):
    cs = CrossSection([pts])
    return cs if cs.area() > 0 else CrossSection([pts[::-1]])


def _rail_section(clr=0.0):
    """Dovetail cross-section in (y, z): narrow at the belly, wide at the tip."""
    zb = BODY_Z_BOT
    r, t = RAIL_ROOT_HW + clr, RAIL_ROOT_HW + RAIL_D + clr
    return _ccw([(-r, zb + 0.5), (r, zb + 0.5), (t, zb - RAIL_D - clr), (-t, zb - RAIL_D - clr)])


def _along_x(cs, x0, x1):
    return cs.extrude(x1 - x0).transform(np.array([[0, 0, 1, x0], [1, 0, 0, 0], [0, 1, 0, 0]], float))


@lru_cache(None)
def stand_rail(clr=0.0, x_open=None):
    """The rail (clr=0) or the socket cavity (clr>0, open past x_open at the rear)."""
    x0, x1 = RAIL_X
    x0 = x0 - clr if x_open is None else x_open
    main = _along_x(_rail_section(clr), x0, x1 + clr)
    root = _along_x(_ccw([(-RAIL_ROOT_HW - clr, BODY_Z_BOT + 0.5), (RAIL_ROOT_HW + clr, BODY_Z_BOT + 0.5),
                          (RAIL_ROOT_HW + clr, BODY_Z_BOT), (-RAIL_ROOT_HW - clr, BODY_Z_BOT)]),
                    x1 + RAIL_D + clr, x1 + RAIL_D + clr + 0.01)
    ramp = hull(_along_x(_rail_section(clr), x1 + clr - 0.01, x1 + clr), root)
    return main + ramp


@lru_cache(None)
def part_stand():
    """Test / display stand, in body coordinates (robot mounted). One print, base on the bed."""
    zb, zt = BODY_Z_BOT, STAND_TABLE_Z
    x0, x1 = RAIL_X
    hx0, hx1 = x0, x1 + RAIL_D + 7.0                 # socket head along the rail
    hw = RAIL_ROOT_HW + RAIL_D + 7.0                 # 19: inside the belly's flat (25.5)
    hz0 = zb - RAIL_D - 11.0                         # head bottom
    head = box(hx0, hx1, -hw, hw, hz0, zb)
    col_r, flare = 17.0, 26.0
    cz_top = hz0 - flare                             # 45-deg flare from the column into the head
    under = hull(box(hx0, hx1, -hw, hw, hz0, hz0 + 0.1), cylz(col_r, cz_top - 0.1, cz_top, STAND_COL_X, 0.0))
    column = cylz(col_r, zt, cz_top + 0.1, STAND_COL_X, 0.0)
    arms = []
    for ang in (0.0, 90.0, 180.0, 270.0):            # between the legs (paws hang at +-100, +-84)
        a = math.radians(ang)
        tip = (STAND_COL_X + STAND_ARM * math.cos(a), STAND_ARM * math.sin(a))
        arms.append(hull(cylz(7.0, zt, zt + 9.0, STAND_COL_X, 0.0), cylz(7.0, zt, zt + 4.5, *tip)))
    hub = cylz(24.0, zt, zt + 9.0, STAND_COL_X, 0.0)
    m = organic(union([head, under, column, hub] + arms), 3.0, 1.0)
    # hollow column: open to the bed, 45-deg cone roof (no bridge), 3.5 mm wall
    ri = col_r - 3.5
    m = m - cylz(ri, zt - 1, cz_top - ri, STAND_COL_X, 0.0) - hull(cylz(ri, cz_top - ri - 0.01, cz_top - ri, STAND_COL_X, 0.0),
                                                              cylz(0.3, cz_top - 0.3, cz_top, STAND_COL_X, 0.0))
    # the socket: open at the rear, ramp stop at the front; top face stays flat against the belly
    m = m - stand_rail(STAND_CLR, x_open=hx0 - 5.0) - box(hx0 - 10, hx1 + 10, -40, 40, zb, zb + 20)
    # side lock: M3 thumb screw through the +y wall onto the rail flank, nut captured in the wall
    sx, sz = x0 + 14.0, zb - RAIL_D / 2
    m = m - cyly(M3_CLR, 0.0, hw + 1, sx, sz, 24)
    nut = CrossSection.circle(M3_NUT_R, 6).rotate(30).extrude(3.0).rotate([-90, 0, 0]).translate([sx, hw - 5.0, sz])
    m = m - nut - box(sx - 2.9, sx + 2.9, hw - 5.0, hw - 2.0, sz, zb + 1)   # nut drops in from the top face
    return m
