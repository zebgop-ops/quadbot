"""Speed x slope sweep for an exported quadbot policy.
A slope is simulated exactly by tilting gravity over a flat plane (the IMU sees the tilt too).
python eval_sweep.py POLICY.pt --slope DEG --dir up|down|side --out result.json"""
import argparse
from isaaclab.app import AppLauncher
ap = argparse.ArgumentParser()
ap.add_argument("policy"); ap.add_argument("--slope", type=float, default=0.0)
ap.add_argument("--dir", default="up", choices=["up", "down", "side"])
ap.add_argument("--speeds", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2")
ap.add_argument("--reps", type=int, default=8); ap.add_argument("--out", required=True)
ap.add_argument("--task", default="Isaac-Velocity-Rough-Quadbot-Play-v0")
ap.add_argument("--no_heading_hold", action="store_true", help="command zero yaw rate instead of holding heading")
args = ap.parse_args()
app = AppLauncher(headless=True).app

import json, math, torch, gymnasium as gym
import isaaclab_tasks  # noqa
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_tasks.manager_based.locomotion.velocity.config.quadbot import mdp_extra

ELECTRONICS_W = 5.5          # Jetson Nano 2GB (~5 W walking-policy load) + F3 + servo logic
speeds = [float(v) for v in args.speeds.split(",")]
n = len(speeds) * args.reps
cfg = parse_env_cfg(args.task, num_envs=n)
cfg.scene.terrain.terrain_type = "plane"; cfg.scene.terrain.terrain_generator = None
if getattr(cfg.curriculum, "terrain_levels", None) is not None: cfg.curriculum.terrain_levels = None
if getattr(cfg.curriculum, "speed_levels", None) is not None: cfg.curriculum.speed_levels = None
cfg.events.add_base_mass = None
cfg.events.base_external_force_torque = None
cfg.events.push_robot = None
cfg.events.physics_material.params["static_friction_range"] = (0.8, 0.8)
cfg.events.physics_material.params["dynamic_friction_range"] = (0.6, 0.6)
cfg.events.reset_base.params = {"pose_range": {"x": (0, 0), "y": (0, 0), "yaw": (0, 0)},
                                "velocity_range": {k: (0.0, 0.0) for k in ("x", "y", "z", "roll", "pitch", "yaw")}}
cfg.scene.env_spacing = 1.0
c = cfg.commands.base_velocity
c.heading_command = not args.no_heading_hold; c.rel_heading_envs = 1.0; c.rel_standing_envs = 0.0
c.resampling_time_range = (1e6, 1e6); c.ranges.heading = (0.0, 0.0)
th = math.radians(args.slope); g = 9.81
# walking direction is world +x; "up" = ground rises along +x
gx = {"up": -g * math.sin(th), "down": g * math.sin(th), "side": 0.0}[args.dir]
gy = -g * math.sin(th) if args.dir == "side" else 0.0
cfg.sim.gravity = (gx, gy, -g * math.cos(th))

env = gym.make(args.task, cfg=cfg); u = env.unwrapped
pol = torch.jit.load(args.policy, map_location=u.device).eval()
obs, _ = env.reset()
cmd_term = u.command_manager.get_term("base_velocity")
vx = torch.tensor(speeds, device=u.device).repeat_interleave(args.reps)


def set_cmd():
    """vx per robot, no sidestep; yaw rate from the heading-hold loop (as in training / on the robot)
    unless --no_heading_hold."""
    cmd_term.vel_command_b[:, 0] = vx
    cmd_term.vel_command_b[:, 1] = 0.0
    if args.no_heading_hold:
        cmd_term.vel_command_b[:, 2] = 0.0
    else:
        cmd_term.heading_target[:] = 0.0
        cmd_term.is_heading_env[:] = True
        cmd_term.is_standing_env[:] = False


rob = u.scene["robot"]
mass = float(rob.root_physx_view.get_masses()[0].sum())
dt = u.step_dt
T_SETTLE, T_MEAS = 2.0, 6.0
fell = torch.zeros(n, dtype=torch.bool, device=u.device)
p_sum = torch.zeros(n, device=u.device); p0 = None
with torch.inference_mode():
    set_cmd(); obs = u.observation_manager.compute()
    for i in range(int((T_SETTLE + T_MEAS) / dt)):
        set_cmd()
        obs, _, term, trunc, _ = env.step(pol(obs["policy"]))
        fell |= term
        if i == int(T_SETTLE / dt):
            p0 = rob.data.root_pos_w[:, :2].clone(); p_sum.zero_()
        if p0 is not None:
            p_sum += mdp_extra.servo_electrical_power(u, mdp_extra.SceneEntityCfg("robot"))
d = (rob.data.root_pos_w[:, :2] - p0) / T_MEAS
steps = int(T_MEAS / dt)
rows = []
for k, s in enumerate(speeds):
    idx = slice(k * args.reps, (k + 1) * args.reps)
    ok = ~fell[idx]
    v_fwd = d[idx, 0][ok].mean().item() if ok.any() else float("nan")
    v_lat = d[idx, 1][ok].mean().item() if ok.any() else float("nan")
    p_servo = (p_sum[idx][ok] / (steps + 1)).mean().item() if ok.any() else float("nan")
    p_tot = p_servo + ELECTRONICS_W
    cot = p_tot / (mass * g * v_fwd) if v_fwd > 0.02 else float("nan")
    rows.append(dict(cmd=s, fall_rate=float(fell[idx].float().mean()), v=v_fwd, v_lat=v_lat,
                     p_servo=p_servo, p_total=p_tot, cot=cot))
    print(f"slope {args.slope:4.1f} {args.dir:4s} cmd {s:.2f}  v {v_fwd:6.3f}  drift {v_lat:6.3f}  fall {rows[-1]['fall_rate']:.2f}"
          f"  servo {p_servo:5.1f} W  total {p_tot:5.1f} W  CoT {cot:5.2f}", flush=True)
json.dump(dict(slope=args.slope, dir=args.dir, mass=mass, electronics_w=ELECTRONICS_W, rows=rows), open(args.out, "w"), indent=1)
env.close(); app.close()
