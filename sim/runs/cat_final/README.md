# Quadbot "cat" locomotion policy (final)

Trained in Isaac Lab 2.3.2 / Isaac Sim 5.1 on the comfy server (`~/quadbot_sim`), task
`Isaac-Velocity-Cat11-Quadbot-v0` (source: `~/quadbot/sim/isaaclab_quadbot/cat_env_cfg.py`, versions 1..11
are cumulative), checkpoint iteration 6600.

Files: `quadbot_cat_policy.onnx` / `.pt` (TorchScript) - obs[1,45] -> actions[1,12], observation
normalisation baked in. `quadbot_cat_checkpoint.pt` = raw rsl_rl checkpoint (resume/fine-tune).

## Running it on the robot (50 Hz loop)

Joint order (Isaac Lab articulation order - use it for obs AND actions):
LF_hip_roll, LR_hip_roll, RF_hip_roll, RR_hip_roll, LF_hip_pitch, LR_hip_pitch, RF_hip_pitch, RR_hip_pitch,
LF_knee, LR_knee, RF_knee, RR_knee.  Default (stand) pose: hip_roll 0, hip_pitch -0.8604 (L) / +0.8604 (R),
knee 1.6197 (L) / -1.6197 (R) rad (joint axes/signs as in the URDF).

Observation (45, no scaling, in this order):
 1. base angular velocity, body frame (3) rad/s  - F3 gyro
 2. projected gravity, body frame (3) unit vector - F3 accel/attitude (gravity direction, not accel)
 3. command vx, vy (m/s), yaw rate (rad/s) (3) - for heading hold use yaw_rate = clip(0.5*heading_err, +-2.5)
 4. joint positions minus default pose (12) rad
 5. joint velocities (12) rad/s
 6. last policy output (12) - the raw action, before scaling/filtering

Action -> servo target:  target = default + 0.25*action, then low-pass
`filtered = 0.4*target + 0.6*filtered_prev` (the policy was trained with this filter; skip it and the
gait gets rougher). Send `filtered` to the STS3215s as position goals every 20 ms.

## Results (simulation)

Gait scorecard (flat ground, scripted course: trot 0.4 m/s, sprint 0.8, turn 2 rad/s, sidestep,
stop, reverse; 64 robots + 64 shoved at 0.6 m/s). Smoothness is scored over the cruise phases;
the 0.8 m/s sprint drives the servos to their 4.7 rad/s limit ~25 % of the time and is judged on
speed/response instead.

| metric | target | first flat policy | final |
|---|---|---|---|
| falls (scripted course) | <= 0.0 | 0.000 ✓ | 0.000 ✓ |
| falls after a 0.6 m/s shove | <= 0.05 | 0.000 ✓ | 0.000 ✓ |
| speed tracking (mean) worst cruise | <= 0.05 | 0.025 ✓ | 0.041 ✓ |
| within-stride surge, worst cruise | <= 0.08 | 0.025 ✓ | 0.070 ✓ |
| turn-rate error (rad/s rms) | <= 0.2 | 0.127 ✓ | 0.197 ✓ |
| sprint speed at 0.8 cmd (m/s) | >= 0.75 | 0.635 ✗ | 0.788 ✓ |
| 0.4->0.8 m/s response (s) | <= 0.6 | – ✗ | 0.340 ✓ |
| 2 rad/s turn response (s) | <= 0.4 | 0.100 ✓ | 0.340 ✓ |
| wobble, worst cruise phase | <= 0.25 | 0.429 ✗ | 0.213 ✓ |
| bounce, worst cruise phase | <= 0.04 | 0.068 ✗ | 0.042 ✗ |
| jitter, worst cruise phase | <= 0.08 | 0.257 ✗ | 0.085 ✗ |
| joint accel (rad/s2 rms) | <= 60.0 | 85.046 ✗ | 54.426 ✓ |
| paw speed at touchdown (m/s) | <= 0.25 | 0.315 ✗ | 0.229 ✓ |
| peak foot load / body weight | <= 1.0 | 0.849 ✓ | 0.806 ✓ |
| foot slip in stance (m/s) | <= 0.05 | 0.186 ✗ | 0.020 ✓ |
| foot clearance (m) | >= 0.03 | 0.018 ✗ | 0.031 ✓ |
| steps per foot per s | 1.5–3.0 | 4.066 ✗ | 1.812 ✓ |


Hills (`hills_speed_efficiency.png`, slope simulated by tilting gravity, heading held):
top speed 0.84 m/s flat, 0.94 downhill, 0.50 m/s up 25 deg, 0.53 m/s across a 25 deg side slope;
best cost of transport 1.35 (18.7 W incl. 5.5 W electronics at 0.70 m/s) vs 2.59 for the first policy.

Known limits: 2 rad/s turns bounce ~5 mm per stride (0.042 m/s rms vs 0.04 target) and servo-target
jitter 0.085 (target 0.08); fast (>=0.6 m/s) crossings of 20-25 deg side slopes drift downhill
0.1-0.3 m/s; reversing on the steepest bumpy tiles slides sideways. All sim - verify the servo
model (stiffness 12, damping 0.25, armature 0.005) against a real STS3215 before trusting the numbers.
