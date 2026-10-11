"""All dimensions in mm. Edit here, re-run build.py.

Servo numbers come from the Feetech STS3215 drawing (spec A/0, 2020-04-10,
page 6/8 + horn drawing page 7/8). Values marked VERIFY were scaled off the
drawing rather than dimensioned on it -- measure one servo with calipers and
correct them before printing everything (print fit_test_* first).
"""

# ---------------------------------------------------------------- STS3215
# Servo frame: origin on the output shaft axis at the mid-plane of the case,
# +Z = output spline side, +X = toward the output end, motor end at -X.
SERVO_X_OUT = 10.1       # shaft axis -> output end face            (VERIFY)
SERVO_X_MOTOR = 35.1     # shaft axis -> motor end face  (45.23 total)
SERVO_HALF_W = 24.73 / 2
SERVO_HALF_H = 14.5      # case is 29 thick at the output end
RAISED_TOP = 16.3        # motor-end half of case is 33 thick (fit test 2026-09-24: +1 mm, split top/bottom)
RAISED_BOT = -16.5
RAISED_X_END = -6.0      # raised region runs from motor end to here    (VERIFY)
RAISED_ARC_R = 11.6      # raised region is cut back around the horn    (VERIFY)
FOOT_X = (-29.9, -18.1)  # little foot on the idler side of motor end   (VERIFY)
FOOT_Z = -19.8           # 36.1 mm overall height incl. foot (fit test: +1 mm)
FOOT_HALF_W = 10.0
SPLINE_R, SPLINE_TOP = 2.95, 17.9        # 25T, protrudes 3.4
IDLER_HUB_R, IDLER_HUB_END = 3.0, -18.6  # rear hub protrudes 4.1
HORN_FACE_OUT = 18.8     # outer face of metal output horn              (VERIFY)
HORN_FACE_IDL = -19.1    # outer face of idler horn                     (VERIFY)
HORN_R = 19.95 / 2
HORN_PCD = 14.0          # 4x M3 on 14 PCD, both horns
HORN_SCREW_L = 5.8       # the M3 screws supplied with the STS3215 (both horns)
HORN_THREAD = 3.3        # thread engaged in the horn -> head sits HORN_SCREW_L - HORN_THREAD above it
# PA2.0 self-tap holes in the case faces (only the motor-end pair is used)
CASE_HOLES_TOP = [(-29.0, 10.5), (-29.0, -10.5)]      # (x, y)  (VERIFY)
CASE_HOLES_BOT = [(-32.9, 10.5), (-32.9, -10.5)]      # (VERIFY)
# the two 5264 sockets on the idler face + a plugged-in housing
PLUG_X = (-16.1, -11.3)
PLUG_HALF_W = 10.0
PLUG_H = 8.6

# ---------------------------------------------------------------- printing
# Printed in ABS. Geometry here is the TRUE size: turn on the slicer's
# filament shrinkage compensation (e.g. OrcaSlicer "Shrinkage (XY)" 99.4 %)
# instead of padding these clearances.
CLR = 0.25               # pocket clearance per side
CORNER_R = 6.0           # rounded vertical corners on tray/deck (less warp)
M3_CLR = 1.65            # radius
M3_HEAD_R = 3.1
M3_INSERT_R = 2.0        # heat-set insert hole (M3 x 5.7, 4.0 mm hole)
M3_NUT_R = 3.3          # hex nut trap (5.5 AF + clearance), circumradius
M2_CLR = 1.2           # PA2.0 self-tappers into the STS3215 case (the only non-M3 screw)
M2_HEAD_R = 2.1
SEG = 64

# ---------------------------------------------------------------- joints
ARM_T = 4.0              # U-bracket arm thickness
IDLER_STANDOFF = 7.0     # spacer on the idler side so the arm clears the plugs
PAD_R = 11.0
BOSS_R = 10.0
U_R_IN = 17.0            # U base inner face (clears case corner r=16)
U_R_OUT = 21.0
U_HALF_W = 11.0
WALL = 2.5
SLEEVE_FRONT = 6.0       # sleeve walls stop here so the U base can swing past
CAP_X0 = -24.5           # cradle end cap starts here (outside U sweep r~23.7)
CAP_TOP = 19.5
CAP_BOT = -22.3

# ---------------------------------------------------------------- layout
HIP_X = 100.0            # hip-roll axis offset fore/aft of centre
HIP_Y = 37.5             # hip-roll servo shaft lateral position
B_OFFSET = 47.0          # hip-roll axis -> hip-pitch servo centre (lateral)
L1 = 100.0               # hip pitch axis -> knee axis
L2 = 110.0               # knee axis -> foot contact
FOOT_Y = -4.0            # foot offset in the tibia frame (flat print face)

# body
BODY_Z_TOP = 25.0        # tray top / deck underside
DECK_T = 3.0
BODY_Z_BOT = -28.0
TRAY_HALF_X = 60.0
TRAY_HALF_Y = 38.5
MODULE_X = (66.0, 122.0)
MODULE_HALF_Y = 30.0

# battery: 3S2P 21700 (6 cells, ~63 x 42 x 70 bare) + wrap + BMS
BATTERY = (78.0, 68.0, 46.0)   # x, y, z envelope

# Jetson Nano 2GB carrier 100 x 80, holes 86 x 58               (VERIFY)
JETSON_BOARD = (100.0, 80.0)
JETSON_HOLES = (86.0, 58.0)
# measured 2026-09-29: the holes are 17 mm from the port edge (ports face the robot's left, +y) and
# 80 - 58 - 17 = 5 mm from the other edge, so the board is NOT centred on its holes
JETSON_PORT_MARGIN = 17.0
JETSON_HOLES_Y0 = -5.0   # hole-pattern centre: board spans y -39..+41 on the 77 mm deck
JETSON_STANDOFF_H = 6.0

# ZOSKAY DS3218 (neck pitch + jaw): 40 x 20 x 40.4 case, ears 54.5 span,
# holes 49.5 x 10, ear underside 27.7 above the case bottom (Dsservo drawing)
DS_L_OUT, DS_L_BACK = 10.0, 30.0   # shaft -> output end / -> back end    (VERIFY)
DS_HALF_W = 10.0
DS_H = 40.4
DS_EAR_HALF_SPAN = 27.25
DS_EAR_HOLE_X = (14.75, -34.75)    # along the case, from the shaft
DS_EAR_HOLE_Y = 5.0
DS_EAR_UNDER = 27.7 - DS_H         # ear underside relative to the case top (-12.7)
DS_EAR_T = 2.5                                                          # (VERIFY)
# 25T clamp-style aluminium arm that ships with it                         (VERIFY)
DS_HUB_R = 7.0           # arm hub radius
DS_HUB_H = 4.0           # case top -> underside of the flat lever
DS_LEVER_T = 3.0
DS_LEVER_L = 26.0

# neck / head layout (body frame); head frame H = origin on the neck axis
NECK_X, NECK_Z = 136.0, 45.0
HEAD_Y0 = 2.0            # shifts the whole head so it sits centred
JAW_AXIS = (34.0, -8.0)  # jaw hinge in the head frame (x, z)

# Betaflight F3 AIO: 36 x 36 board, 30.5 x 30.5 M3 pattern
FC_BOARD = 36.0
FC_HOLES = 30.5
# Raspberry Pi camera v2 style CSI camera (IMX219): 25 x 24, holes 21 x 12.5
CAM_HOLES = (21.0, 12.5)
