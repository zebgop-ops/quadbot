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
    # windows in the sleeve walls (the cap + screws do the holding)
    for y0, y1 in ((SERVO_HALF_W, SLEEVE_HW + 1), (-SLEEVE_HW - 1, -SERVO_HALF_W)):
        cuts.append(teardrop("y", -9.0, -1.5, 7.0, y0, y1, up))
    return union(cuts)


# U bracket arm planes
Z_OUT0, Z_OUT1 = HORN_FACE_OUT, HORN_FACE_OUT + ARM_T
Z_BOSS0, Z_BOSS1 = HORN_FACE_IDL - IDLER_STANDOFF, HORN_FACE_IDL
Z_IDL0, Z_IDL1 = Z_BOSS0 - ARM_T, Z_BOSS0
U_ZC = (Z_OUT1 + Z_IDL0) / 2


def idler_spacer():
    """Only the 2 idler screws + centre need backing: a dog-bone, not a disc.
    Added after the organic pass (no fillet) to keep the servo plugs clear."""
    p = HORN_PCD / 2 / math.sqrt(2)
    z0 = Z_BOSS0 - 0.8                   # sink into the arm so it fuses
    return (hull(cylz(3.6, z0, Z_BOSS1, p, p), cylz(3.6, z0, Z_BOSS1, -p, -p))
            + cylz(5.6, z0, Z_BOSS1))


def u_solid(half_w=U_HALF_W, base_out=U_R_OUT):
    """Link side of a joint: two arms on the horns + base (no spacer, no holes)."""
    def arm(z0, z1):
        return hull(cylz(PAD_R, z0, z1), box(U_R_IN, base_out, -half_w, half_w, z0, z1))
    return union([arm(Z_OUT0, Z_OUT1), arm(Z_IDL0, Z_IDL1),
                  box(U_R_IN, base_out, -half_w, half_w, Z_IDL0, Z_OUT1)])


@lru_cache(None)
def u_cuts():
    cuts = [cylz(3.4, Z_IDL0 - 1, Z_OUT1 + 1)]  # centre screw access
    for k in range(4):
        a = math.radians(45 + 90 * k)
        x, y = HORN_PCD / 2 * math.cos(a), HORN_PCD / 2 * math.sin(a)
        cuts.append(cylz(M3_CLR, Z_OUT0 - 0.5, Z_OUT1 + 1, x, y, 24))
        cuts.append(cylz(M3_HEAD_R, Z_OUT1 - 1.5, Z_OUT1 + 1, x, y, 24))   # 4x M3x6 output side
        if k % 2 == 0:  # idler horn only carries radial load: 2x M3x10
            cuts.append(cylz(M3_CLR, Z_IDL0 - 1, Z_BOSS1 + 0.5, x, y, 24))
            cuts.append(cylz(M3_HEAD_R, Z_IDL0 - 1, Z_IDL0 + 3.0, x, y, 24))
    return union(cuts)


def u_bracket(half_w=U_HALF_W, base_out=U_R_OUT):
    return (organic(u_solid(half_w, base_out)) + idler_spacer()) - u_cuts()


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
        xf(cradle_solid(), B)]), 3.0)
    parts = [u_solid(), shoulder]
    m = organic(union(parts)) + idler_spacer()
    m = m - u_cuts() - xf(cradle_cuts(top=False, up=(-1.0, 0.0)), B)
    m = m - teardrop("x", 17.0, 0.0, 5.0, U_R_IN - 1, ext_x1 + 2, (1.0, 0.0))    # riser window
    # keep the femur's inner arm sweep clear
    fa0 = Bo - HORN_FACE_OUT - ARM_T - 1.0
    fa1 = Bo - HORN_FACE_OUT + 0.5
    m = m - cylx(24.5, fa0, fa1)
    m = m - cylx(HORN_R + 1.0, fa1 - 0.1, Bo - SERVO_HALF_H - 1.5)   # the pitch servo's own output horn
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
    lo_u, hi_u, lo_w, hi_w, lo_k, hi_k = Z_IDL0, Z_OUT1, -14.0, 11.0, CAP_BOT, CAP_TOP
    if x <= FEMUR_WAIST_X:
        s_ = _smooth((x - x_u) / (FEMUR_WAIST_X - x_u))
        return lo_u + (lo_w - lo_u) * s_, hi_u + (hi_w - hi_u) * s_
    s_ = _smooth((x - FEMUR_WAIST_X) / (x_k - FEMUR_WAIST_X))
    return lo_w + (lo_k - lo_w) * s_, hi_w + (hi_k - hi_w) * s_


def femur_wall(y0, y1, x0, x1):
    xs = np.linspace(x0, x1, 60)
    lo = [(x, femur_profile(x)[0]) for x in xs]
    hi = [(x, femur_profile(x)[1]) for x in xs[::-1]]
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
        femur_wall(hw - WALL, hw, x0, x1),                   # bone-profile side walls
        femur_wall(-hw, -hw + WALL, x0, x1),
        box(x0, x1, -hw, hw, -1.5, 1.5),                     # web
        xf(pod(cradle_solid(), 3.0), trans(L1, 0, 0)),
    ]
    m = organic(union(parts)) + idler_spacer()
    m = m - u_cuts() - xf(cradle_cuts(bot=False, up=(-1.0, 0.0)), trans(L1, 0, 0))
    for xc in (40.0, 56.0):                                   # web windows (also cable / zip-tie points)
        for yc in (6.0, -6.0):
            m = m - teardrop("z", xc, yc, 4.0, -4, 4, (-1.0, 0.0), drop=5.0)
    for xc, r_ in ((36.0, 6.5), (58.0, 6.0)):                 # side-wall windows follow the waist
        lo, hi = femur_profile(xc)
        for y0, y1 in ((hw - WALL - 1, hw + 1), (-hw - 1, -hw + WALL + 1)):
            m = m - teardrop("y", xc, (lo + hi) / 2, r_, y0, y1, (-1.0, 0.0), drop=4.0)
    # tibia arm sweep near the knee must stay clear
    m = m - (cylz(24.5, Z_OUT0 - 0.8, Z_OUT1 + 1).translate([L1, 0, 0]))
    m = m - (cylz(24.5, Z_IDL0 - 1, Z_IDL1 + 0.8).translate([L1, 0, 0]))
    m = m - (cylz(11.5, Z_BOSS0, Z_BOSS1 + 0.8).translate([L1, 0, 0]))       # spacer sweep
    return m


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
    knee = ell((U_R_IN + 4.0, 0.0, zc), (4.0, U_HALF_W, 7.5)).rotate([0, 0, KNEE_BIAS])
    kx, ky = (rz(KNEE_BIAS) @ np.array([U_R_IN + 4.0, 0, 0, 1]))[:2]
    tip = np.array([L2 - 28.0, 0.0])
    calf_c = np.array([kx, ky]) + (tip - np.array([kx, ky])) * 0.42
    calf = ell((calf_c[0], calf_c[1], zc * 0.6), (9.0, 9.0, 7.0))
    ankle = ell((L2 - 28.0, 0.0, 0.0), (5.0, PEG_AF / 2 + 0.5, PEG_AF / 2 + 0.5))
    beam = hull(knee, calf, ankle)
    peg = octagon_x(PEG_AF, PEG_X[0], PEG_X[1], 0.0, 0.0)
    m = organic(union([u, beam])) + idler_spacer().rotate([0, 0, KNEE_BIAS]) + peg
    m = m - u_cuts().rotate([0, 0, KNEE_BIAS])
    for t, off, h in TIBIA_WINDOWS:                         # windows (>= 3 mm walls at mid-plane)
        p = np.linalg.inv(ub) @ np.array([t, off, 0, 1])
        m = m - xf(teardrop("z", p[0], p[1], h * 0.8, -40, 40, (0.0, 1.0)), ub)
    return m if left else m.mirror([0, 1, 0])


@lru_cache(None)
def part_foot():
    """TPU (95A) boot that presses onto the octagonal tibia peg (same part for all legs)."""
    fy = foot_y(True)
    r = 12.5
    boot = hull(Manifold.sphere(r, SEG).translate([L2 - r, fy, 0]),
                cylx(11.0, L2 - 25.0, L2 - 24.0, fy, 0))
    sock = octagon_x(PEG_AF + 0.2, PEG_X[0] - 5, PEG_X[1] + 0.3, fy, 0.0)
    return boot - sock


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
    parts = [rbox(xa, xb, -hy, hy, zt - 3, zt, 4.0), rbox(xa, xb, -hy, hy, zb, zb + 3, 4.0),     # skins
             box(*_mx(s_, 70.0, xe), -2.0, 2.0, zb, zt),                                          # centre web
             box(*_mx(s_, 70.0, xe), 10.0, 12.5, zb, zt), box(*_mx(s_, 70.0, xe), -12.5, -10.0, zb, zt)]   # side webs
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
    cuts += [xf(cradle_cuts(bot=False, up=up), T) for T in frames]
    for y_ in (1, -1):                                   # servo cable notch through the side webs
        ya, yb = sorted((y_ * 8.5, y_ * 14.0))
        cuts.append(box(*_mx(s_, 72.0, HIP_X - 17.0), ya, yb, -4.5, 4.5))
    for zz in (zt, zb):                                  # skin windows (point up the print = -x)
        for yc in (-21.0, 21.0):
            for xc in (80.0, 101.0):
                cuts.append(teardrop("z", s_ * xc, yc, 5.5, zz - 4, zz + 4, (-1.0, 0.0), drop=2.5))
    if s_ > 0:
        for x, y in HEAD_BOLTS:
            cuts.append(cylz(M3_INSERT_R, zt - 6, zt + 1, x, y, 24))
    else:
        gx, gy, gz = GNSS
        cuts.append(box(-GNSS_X - gx / 2 - 0.3, -GNSS_X + gx / 2 + 0.3, -gy / 2 - 0.3, gy / 2 + 0.3, zt - gz - 0.5, zt + 1))
        cuts.append(box(-GNSS_X + gx / 2, -GNSS_X + gx / 2 + 6, -2.5, 2.5, zt - 3, zt + 1))   # GNSS lead
    return cuts


@lru_cache(None)
def part_body():
    """The whole main body in one print (fits 256 x 256; prints standing on its
    front face, 239 mm tall): battery bay, tapers, both hip sections, F3 + GNSS
    mounts. The deck is the lid."""
    hx, hy = TRAY_HALF_X, TRAY_HALF_Y
    zt, zb = BODY_Z_TOP, BODY_Z_BOT
    w = 2.5
    xe = HIP_X + CAP_TOP
    bay = rbox(-hx, hx, -hy, hy, zb, zt, CORNER_R) - box(-hx - 1, hx + 1, -hy + w, hy - w, zb + w, zt + 10)
    parts = [bay]
    for s_ in (1, -1):   # tapers: bay cross-section -> hip-section cross-section, open inside
        o = hull(box(*_mx(s_, TRANS_X0 - 0.1, TRANS_X0), -hy, hy, zb, zt),
                 box(*_mx(s_, TRANS_X1 - 0.1, TRANS_X1), -MODULE_HALF_Y, MODULE_HALF_Y, zb, zt + DECK_T))
        i = hull(box(*_mx(s_, TRANS_X0 - 1.1, TRANS_X0 - 1.0), -hy + w, hy - w, zb + w, zt + 10),
                 box(*_mx(s_, TRANS_X1 + 1.0, TRANS_X1 + 1.1), -MODULE_HALF_Y + w, MODULE_HALF_Y - w, zb + w, zt + 10))
        parts.append(o - i)
    ends = [_hip_end(1), _hip_end(-1)]
    parts += [e[0] for e in ends]
    # deck bosses (teardrop toward the bed, +x) and F3 standoffs on the floor
    bos = [(sx * (hx - 5), sy * (hy - 5.5)) for sx in (1, -1) for sy in (1, -1)]
    parts += [hull(cylz(4.2, zb, zt, x, y), cylz(0.5, zb, zt, x + 5.9, y)) for x, y in bos]
    h = FC_HOLES / 2
    fc = [(FC_X + sx * h, sy * h) for sx in (1, -1) for sy in (1, -1)]
    parts += [hull(cylz(3.5, zb, zb + w + 8, x, y), cylz(0.5, zb, zb + w + 8, x + 4.9, y)) for x, y in fc]
    m = organic(union(parts))
    cuts = [box(xe, xe + 20, -60, 60, -60, 60),                              # bed face (front) stays flat
            box(-hx - 1, hx + 1, -60, 60, zt, zt + 10)]                      # deck seat stays flat
    for s_, (_, frames) in zip((1, -1), ends):
        cuts += _hip_end_cuts(s_, frames)
    cuts += [cylz(M3_INSERT_R, zt - 6, zt + 1, x, y, 24) for x, y in bos]
    cuts += [cylz(M3_INSERT_R, zb + w + 2, zb + w + 9, x, y, 24) for x, y in fc]
    # left wall: power switch (12 mm) + charge jack (8 mm)
    cuts.append(cyly(6.1, hy - 5, hy + 1, 45.0, 0.0))
    cuts.append(cyly(4.0, hy - 5, hy + 1, 28.0, 0.0))
    # side-wall windows and floor windows, all pointing up the print (-x)
    for y_ in (1, -1):
        for xc in ((-40.0, -12.0) if y_ > 0 else (-40.0, -12.0, 16.0, 44.0)):
            y0, y1 = (hy - 5, hy + 1) if y_ > 0 else (-hy - 1, -hy + 5)
            cuts.append(teardrop("y", xc, 1.0, 8.5, y0, y1, (-1.0, 0.0), drop=7.0))
    for xc in (-36.0, -8.0, 20.0):
        for yc in (-17.0, 17.0):
            cuts.append(teardrop("z", xc, yc, 6.5, zb - 1, zb + w + 1, (-1.0, 0.0), drop=5.0))
    return m - union(cuts)


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


@lru_cache(None)
def part_deck():
    """Top skin of the body beam + Jetson standoffs: a frame, not a plate."""
    hx, hy = TRAY_HALF_X, TRAY_HALF_Y
    z0, z1 = BODY_Z_TOP, BODY_Z_TOP + DECK_T
    jx, jy = JETSON_HOLES
    m = rbox(-hx, hx, -hy, hy, z0, z1, CORNER_R)
    so = [cylz(4.0, z0, z1 + JETSON_STANDOFF_H, sx * jx / 2, sy * jy / 2) for sx in (1, -1) for sy in (1, -1)]
    m = organic(union([m] + so), 2.0, 0.8)
    cuts = [box(-hx - 5, hx + 5, -hy - 5, hy + 5, z0 - 5, z0)]           # bed face stays flat
    cuts += [cylz(M3_INSERT_R, z1 + JETSON_STANDOFF_H - 6, z1 + JETSON_STANDOFF_H + 1, sx * jx / 2, sy * jy / 2, 24)
             for sx in (1, -1) for sy in (1, -1)]  # M3 heat-set inserts
    for sx in (1, -1):
        for sy in (1, -1):
            cuts.append(cylz(M3_CLR, z0 - 1, z1 + 1, sx * (hx - 5), sy * (hy - 5.5), 24))
            cuts.append(cylz(3.1, z1 - 1.2, z1 + 1, sx * (hx - 5), sy * (hy - 5.5), 24))
            cuts.append(rbox(min(sx * 6, sx * 34), max(sx * 6, sx * 34), min(sy * 6, sy * 26), max(sy * 6, sy * 26),
                             z0 - 1, z1 + 1, 4.0))
        cuts.append(rbox(min(sx * 47, sx * 51), max(sx * 47, sx * 51), -18, 18, z0 - 1, z1 + 1, 2.0))  # cables
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
    c = [box(-DS_L_BACK - 0.3, DS_L_OUT + 0.3, -DS_HALF_W - 0.3, DS_HALF_W + 0.3, -DS_H - 10, DS_EAR_UNDER + 0.01)]
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
J_TOP = HP_IN + 0.5                    # jaw servo case top: its lever clears the head plate
J_BOT = J_TOP - DS_H
JP_IN = J_TOP + DS_HUB_H + DS_LEVER_T  # jaw +y arm sits on the jaw lever
JX, JZ = JAW_AXIS
JAW_LEVER_ANGLE = -43.0                # lever/slot points from hinge toward the chin
FRONT = 73.0                           # head front face (head frame x)
CAM_ZH = 15.5
CAM_YC = (HN_OUT + HP_OUT) / 2

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
    m = organic(union([floor, bulk, pivot, boss, web]), 2.0, 0.8)
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


@lru_cache(None)
def part_head():
    """Skull: yoke on the neck servo + jaw servo mount + camera. Head frame H."""
    def plate(y0, y1):
        return union([hull(cyly(14, y0, y1), box(0, FRONT, y0, y1, -16, 33)),
                      cyly(12, y0, y1, JX, JZ)])

    ear_y = J_TOP + DS_EAR_UNDER
    parts = [
        plate(HP_IN, HP_OUT),
        plate(HN_OUT, HN_IN),
        box(22, FRONT, HN_OUT, HP_OUT, 30, 33),                     # top
        box(FRONT - 3, FRONT, HN_OUT, HP_OUT, -10, 33),             # face
        box(24, 46, ear_y - 4, ear_y, -27, 31),                     # jaw servo bulkhead...
        box(46, FRONT - 9, ear_y - 4, ear_y, -9, 31),               # ...kept out of the mouth and off the camera
        cyly(5.5, HN_IN - 0.1, HN_IN + 2.5, JX, JZ),                # jaw pivot boss
    ]
    parts.append(upper_snout())
    body = union(parts)
    # sloped forehead: trim everything beyond the plane x + z = c, close it with a 3 mm plate
    c = FRONT + 24.0

    def beyond(cc):
        tri = CrossSection([[(cc - 60, 60), (120, cc - 120), (120, 60)]])
        return tri.extrude(200).transform(np.array([[1, 0, 0, 0], [0, 0, -1, 100], [0, 1, 0, 0]], float))

    body = body - beyond(c)
    slab = (box(40, FRONT, HN_OUT, HP_OUT, 5, 33) ^ beyond(c - 3 * math.sqrt(2))) - beyond(c)
    body = body + slab
    m = organic(body, 2.0, 0.8)
    m = m - box(-40, 120, -60, 60, 33, 60)                                   # bed face (top) stays flat
    # upper teeth: cones hanging from the gum line, staggered between the lower teeth
    teeth = [tooth(c, n, lip_top(c[0]) + GUM_GAP + 0.4, up=-1.0)
             for c, n in tooth_sites(phase=0.5, x_min=FRONT + 1)]
    xf_ = JAW_TIP + 1.0
    g_tip = lip_top(xf_) + GUM_GAP
    teeth.append(fang(xf_, JAW_YC, g_tip + 0.8, up=-1.0))
    xk = 84.0                                            # keel: fang root -> thick part of the snout
    teeth.append(hull(cylz(2.7, g_tip - 0.8, g_tip + 1.5, xf_, JAW_YC, 20),
                      Manifold.sphere(3.2, 24).translate([xk, JAW_YC, lip_top(xk) + GUM_GAP + 1.0])))
    # nothing may rise above the snout's 45-deg top (it's the underside when printing)
    top45 = prism_xz([(40, -60), (140, -60), (140, SNOUT_TOP - (140 - FRONT)), (40, SNOUT_TOP + (FRONT - 40))])
    m = m + (union(teeth) ^ top45)
    # camera: 4 printed pins through the board's holes (heat-stake or a drop of CA)
    cx, cz = CAM_HOLES
    for sy in (1, -1):
        for sz in (1, -1):
            m = m + cylx(0.95, FRONT - 5.5, FRONT - 2.9, CAM_YC + sy * cx / 2, CAM_ZH - 1.2 + 1.3 + sz * cz / 2, 16)
    cuts = [
        lever_mount_cuts(0.0, HP_IN, HP_OUT),                        # neck lever
        cyly(DS_HUB_R + 1, HP_IN - 1, HP_OUT + 1, JX, JZ),           # jaw hub passes through
        xf(box(-DS_L_BACK - 0.4, DS_L_OUT + 0.4, -DS_HALF_W - 0.4, DS_HALF_W + 0.4, -1.0, 0.6), T_JAW),
        cyly(5.05, HN_OUT - 1, HN_OUT + 4.0),                        # 623ZZ bearing, neck pivot
        cyly(3.25, HN_OUT - 1, HN_IN + 1),
        cyly(M3_INSERT_R, HN_OUT - 1, HN_IN + 1.5, JX, JZ, 24),      # insert for the jaw pivot bolt
        xf(ds_bulkhead_cut(), T_JAW), xf(ds_envelope(), T_JAW),
    ]
    # windows: cheeks (both plates), eyes either side of the camera, vents on top
    for y0, y1 in ((HP_IN - 1, HP_OUT + 1), (HN_OUT - 1, HN_IN + 1)):
        cuts.append(teardrop("y", 57.0, 9.0, 8.0, y0, y1, (0.0, -1.0), drop=6.0))
        # top-back corner only existed for a flat print edge: 45-deg slope instead
        cuts.append(hull(box(-20, -19, y0, y1, 14.5 - 20, 60), box(-20, 18.5, y0, y1, 59, 60),
                         box(18.4, 18.5, y0, y1, 33.0, 60)))
    for yc in (HN_OUT + 9.5, HP_OUT - 9.5):                                  # eyes
        cuts.append(teardrop("x", yc, CAM_ZH + 2, 5.0, FRONT - 4, FRONT + 1, (0.0, -1.0), drop=2.0))
    for dy in (-4.5, 4.5):                                                    # nostrils on the snout
        cuts.append(cylz(1.7, -30, 30, JAW_TIP - 12.0, JAW_YC + dy, 24))
    for yc in (-24.0, -15.0, 20.5):                                            # leaf vents on top
        cuts.append(hull(cylz(2.0, 29, 34, 30, yc), cylz(3.2, 29, 34, 48, yc), cylz(1.0, 29, 34, 64, yc)))
    # CSI camera lens opening (board sits on the pins behind the face)
    yc = CAM_YC
    cuts.append(hull(cylx(5.5, FRONT - 4, FRONT + 12, yc, CAM_ZH - 2.2), cylx(5.5, FRONT - 4, FRONT + 12, yc, CAM_ZH - 0.2)))
    return m - union(cuts)


JAW_TIP = 98.0      # head-frame x of the jaw point
JAW_YB0, JAW_YA1 = HN_OUT - 5.0, JP_IN + 4.0     # outer faces of the jaw arms
JAW_YC = (JAW_YB0 + JAW_YA1) / 2
JAW_LIP = -21.0     # lower lip height at the corner of the mouth
GUM_GAP = 8.5       # upper gum line sits this far above the lower lip
SNOUT_TOP = 8.5     # snout top at the face; falls at 45 deg to the tip (prints upside-down)


def jaw_vee(inset, z0, z1, r=4.0):
    """V footprint shared by the lower jaw and the upper snout."""
    return hull(box(46, 48, JAW_YB0 + inset, JAW_YA1 - inset, z0, z1),
                cylz(max(r - inset, 0.6), z0, z1, JAW_TIP - inset, JAW_YC))


def lip_top(x):
    """Lower jaw line: drops 6 mm from the corner of the mouth to the point."""
    return JAW_LIP - 6.0 * max(0.0, min(1.0, (x - 57.0) / (JAW_TIP - 57.0)))


def prism_xz(poly):
    """Counter-clockwise (x, z) polygon extruded through y = -100..100."""
    return CrossSection([poly]).extrude(200).transform(np.array([[1, 0, 0, 0], [0, 0, -1, 100], [0, 1, 0, 0]], float))


TOOTH_L = 5.5       # tooth length; upper and lower rows interleave by ~2.5 mm
FANG_L = 6.5


def _edges():
    """The two straight sides of the V (outer faces), each as (p0, p1, outward normal)."""
    out = []
    for yb, sgn in ((JAW_YB0, -1.0), (JAW_YA1, 1.0)):
        p0 = np.array([48.0, yb])
        p1 = np.array([JAW_TIP - 1.5, JAW_YC + sgn * 3.8])
        d = (p1 - p0) / np.linalg.norm(p1 - p0)
        n = np.array([-d[1], d[0]])
        if n[1] * sgn < 0:
            n = -n
        out.append((p0, p1, n))
    return out


def tooth_sites(phase=0.0, x_min=57.0):
    """Tooth centres on the outer edge of the V (0.8 mm in), lower row phase 0, upper 0.5."""
    sites = []
    for p0, p1, n in _edges():
        for f in np.arange(0.0, 1.0, 0.13) + 0.13 * phase:
            if f >= 0.97:
                continue
            c = p0 + (p1 - p0) * f - n * 0.8
            if c[0] > x_min:
                sites.append((c, n))
    return sites


def tooth(c, n, z_base, up=1.0, length=TOOTH_L, lean=1.2):
    """Cone sticking out past the jaw outline, with a root blended down the outer
    wall so the overhanging half prints without support. `up` = +1 lower jaw, -1 upper;
    `lean` pushes the tip outward (mm)."""
    x, y = c
    tip = c + n * (0.8 + lean)
    cone = hull(cylz(2.0, min(z_base, z_base - up * 0.5), max(z_base, z_base - up * 0.5), x, y, 16),
                cylz(0.45, min(z_base + up * length, z_base + up * (length - 0.3)),
                     max(z_base + up * length, z_base + up * (length - 0.3)), tip[0], tip[1], 8))
    r = c + n * 0.4
    root = hull(cylz(2.0, min(z_base, z_base - up * 0.5), max(z_base, z_base - up * 0.5), x, y, 16),
                cylz(1.2, min(z_base - up * 4.0, z_base - up * 3.5), max(z_base - up * 4.0, z_base - up * 3.5),
                     r[0], r[1], 12))
    return cone + root


def fang(x, y, z_base, up=1.0, length=FANG_L, reach=5.0, r=2.7):
    """Front fang leaning forward past the tip (the forward lean stays under 45 deg
    from vertical except the last few mm, so it prints either way up)."""
    return hull(cylz(r, min(z_base, z_base - up * 0.8), max(z_base, z_base - up * 0.8), x, y, 20),
                cylz(0.55, min(z_base + up * length, z_base + up * (length - 0.3)),
                     max(z_base + up * length, z_base + up * (length - 0.3)), x + reach, y, 8))


def upper_snout():
    """Upper jaw: same V as the lower jaw, underside following the jaw line."""
    x0, x1 = FRONT - 3.0, 130.0
    gum = lambda x: lip_top(x) + GUM_GAP
    below_top = prism_xz([(x0 - 5, -60), (x1, -60), (x1, SNOUT_TOP - (x1 - FRONT)), (x0 - 5, SNOUT_TOP + (FRONT - x0 + 5))])
    above_gum = prism_xz([(x0 - 5, gum(x0 - 5)), (57.0, gum(57.0)), (JAW_TIP, gum(JAW_TIP)), (x1, gum(x1)),
                          (x1, 60), (x0 - 5, 60)])
    return jaw_vee(0, -40, 40) ^ box(x0, x1, -60, 60, -60, 60) ^ below_top ^ above_gum




@lru_cache(None)
def part_jaw():
    """Lower jaw: a V-shaped snout hinged on the jaw servo (+y) and a bearing (-y).
    Head frame, mouth closed. Prints on its floor. Same V as the upper snout."""
    ya0, ya1 = JP_IN, JAW_YA1              # +y arm on the lever
    yb0, yb1 = JAW_YB0, HN_OUT - 0.5       # -y arm with bearing
    zf0, zf1, zl = -36.0, -32.0, JAW_LIP   # floor, lip top

    def arm(y0, y1):
        return hull(cyly(10, y0, y1, JX, JZ), box(46, 56, y0, y1, zf0, zf0 + 6))

    lip = (jaw_vee(0, zf1, zl) - jaw_vee(4, zf1 - 1, zl + 1)) - box(0, 54, -60, 60, zf1 - 1, zl + 1)
    lip = lip - hull(box(57, 57.1, -60, 60, zl, zl + 20), box(JAW_TIP + 5, JAW_TIP + 5.1, -60, 60, zl - 6.5, zl + 20))
    parts = [arm(ya0, ya1), arm(yb0, yb1), jaw_vee(0, zf0, zf1), lip]
    m = organic(union(parts), 2.0, 0.8)
    m = m - box(0, 150, -60, 60, zf0 - 10, zf0)                             # bed face stays flat
    teeth = [tooth(c, n, lip_top(c[0]) - 0.4, up=1.0) for c, n in tooth_sites(phase=0.0, x_min=66.0)]
    xf_ = JAW_TIP + 1.0
    zb_ = lip_top(xf_) - 0.6
    for dy in (-2.6, 2.6):
        teeth.append(fang(xf_, JAW_YC + dy, zb_, up=1.0))
    teeth.append(hull(cylz(2.7, zb_ - 2.0, zb_, xf_, JAW_YC - 2.6, 20),     # web between the pair
                      cylz(2.7, zb_ - 2.0, zb_, xf_, JAW_YC + 2.6, 20),
                      Manifold.sphere(4.0, 24).translate([JAW_TIP - 5.0, JAW_YC, zf1 + 1.0])))   # chin root
    m = m + union(teeth)
    ang = JAW_LEVER_ANGLE
    d = (math.cos(math.radians(ang)), math.sin(math.radians(ang)))
    cuts = [cyly(3.5, ya0 - 1, ya1 + 1, JX, JZ),
            hull(cyly(M3_CLR, ya0 - 1, ya1 + 1, JX + 9 * d[0], JZ + 9 * d[1], 24),
                 cyly(M3_CLR, ya0 - 1, ya1 + 1, JX + 25 * d[0], JZ + 25 * d[1], 24)),
            cyly(5.05, yb0 - 1, yb0 + 4.0, JX, JZ),
            cyly(3.25, yb0 - 1, yb1 + 1, JX, JZ),
            jaw_vee(9, zf0 - 1, zf1 + 1) - box(0, 55, -60, 60, zf0 - 2, zf1 + 2)]   # the floor is a V frame
    return m - union(cuts)


# ------------------------------------------------------------------ fit tests


@lru_cache(None)
def part_fit_test_cradle():
    """Stand-alone cradle for checking the servo pocket and screw holes."""
    return organic(cradle_solid()) - cradle_cuts(up=(1.0, 0.0)) - box(CAP_X1 - 10, CAP_X1, -40, 40, -40, 40)


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
