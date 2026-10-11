"""Drive the Isaac Sim quadbot through the *Jetson* code path (~/quadbot_sim/jetson):
sim state -> fake raw sensors (servo ticks with random offsets/directions, Betaflight-style attitude
in 0.1 deg and gyro in integer deg/s) -> hardware.py conversions -> controller.py (obs, numpy policy,
filter) -> raw action into the env. Checks obs/targets against the env's own and that it walks."""
import argparse
from isaaclab.app import AppLauncher
ap = argparse.ArgumentParser(); ap.add_argument("--task", default="Isaac-Velocity-Cat11-Quadbot-Play-v0")
ap.add_argument("--vx", type=float, default=0.4); ap.add_argument("--seconds", type=float, default=8.0)
args = ap.parse_args()
app = AppLauncher(headless=True).app

import math, os, sys
import numpy as np, torch, gymnasium as gym
import isaaclab_tasks  # noqa
from isaaclab_tasks.utils import parse_env_cfg
sys.path.insert(0, os.path.expanduser("~/quadbot_sim/jetson"))
from controller import JOINTS, Controller, Policy, RAD_PER_TICK, gravity_from_attitude, gyro_to_body, ticks_to_rad

cfg = parse_env_cfg(args.task, num_envs=1)
cfg.scene.terrain.terrain_type = "plane"; cfg.scene.terrain.terrain_generator = None
for k in ("terrain_levels", "speed_levels"):
    if getattr(cfg.curriculum, k, None) is not None: setattr(cfg.curriculum, k, None)
cfg.events.add_base_mass = None; cfg.events.push_robot = None; cfg.events.base_external_force_torque = None
cfg.events.reset_base.params = {"pose_range": {"x": (0, 0), "y": (0, 0), "yaw": (0, 0)},
                                "velocity_range": {k: (0.0, 0.0) for k in ("x", "y", "z", "roll", "pitch", "yaw")}}
c = cfg.commands.base_velocity; c.heading_command = False; c.rel_standing_envs = 0.0; c.resampling_time_range = (1e6, 1e6)
env = gym.make(args.task, cfg=cfg); u = env.unwrapped
rob = u.scene["robot"]
assert list(rob.joint_names) == JOINTS, rob.joint_names

rng = np.random.default_rng(0)
off_true = 2048 + rng.integers(-150, 150, 12)            # horn fitted a bit off
dir_true = rng.choice([-1, 1], 12)                        # some servos count the other way
ctrl = Controller(Policy(os.path.expanduser("~/quadbot_sim/jetson/quadbot_cat_policy.npz")))
cmd = np.array([args.vx, 0.0, 0.0])

obs, _ = env.reset()
ctrl.reset()
p0 = rob.data.root_pos_w[0, :2].clone()
d_obs, d_tgt = [], []
for k in range(int(args.seconds / u.step_dt)):
    u.command_manager.get_term("base_velocity").vel_command_b[:] = torch.tensor(cmd, device=u.device, dtype=torch.float32)
    obs = u.observation_manager.compute()
    # ---- raw sensors as the real hardware would report them
    q = rob.data.joint_pos[0].cpu().numpy(); qd = rob.data.joint_vel[0].cpu().numpy()
    ticks = np.round(off_true + dir_true * q / RAD_PER_TICK)
    speed = np.round(dir_true * qd / RAD_PER_TICK)
    g = rob.data.projected_gravity_b[0].cpu().numpy()
    roll = round(math.degrees(math.atan2(-g[1], -g[2])), 1); pitch = round(math.degrees(math.asin(-g[0])), 1)
    w = rob.data.root_ang_vel_b[0].cpu().numpy()
    gyro_bf = np.round(np.degrees([w[0], -w[1], -w[2]]))
    # ---- Jetson side (same calibration the robot would have after calibrate.py)
    q_j = ticks_to_rad(ticks, off_true, dir_true)
    qd_j = dir_true * speed * RAD_PER_TICK
    tgt = ctrl.step(gyro_to_body(gyro_bf, [0, 1, 2], [1, -1, -1]), gravity_from_attitude(roll, pitch), cmd, q_j, qd_j)
    d_obs.append(np.abs(ctrl.obs - obs["policy"][0].cpu().numpy()))
    env.step(torch.tensor(ctrl.last_action, device=u.device, dtype=torch.float32).unsqueeze(0))
    env_tgt = u.action_manager.get_term("joint_pos").processed_actions[0].cpu().numpy()
    d_tgt.append(np.abs(ctrl.target_f - env_tgt))
d_obs, d_tgt = np.array(d_obs), np.array(d_tgt)
names = ["angvel"] * 3 + ["grav"] * 3 + ["cmd"] * 3 + ["q"] * 12 + ["qd"] * 12 + ["act"] * 12
worst = {}
for i, n in enumerate(names):
    worst[n] = max(worst.get(n, 0.0), float(d_obs[:, i].max()))
disp = (rob.data.root_pos_w[0, :2] - p0).cpu().numpy()
print("obs max |jetson - env| per group:", {k: round(v, 4) for k, v in worst.items()}, flush=True)
print("servo target max |jetson - env| (rad): %.2e" % d_tgt.max(), flush=True)
print("walked %.2f m forward, %.2f m sideways in %.0f s at cmd %.2f m/s (expected %.2f m)"
      % (disp[0], disp[1], args.seconds, args.vx, args.vx * args.seconds), flush=True)
env.close(); app.close()
