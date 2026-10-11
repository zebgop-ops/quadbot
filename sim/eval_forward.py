"""Fixed-command evaluation of an exported quadbot policy: does it actually go where it's told?
python eval_forward.py POLICY.pt [vx] [--video OUT.mp4]"""
import argparse
from isaaclab.app import AppLauncher
ap = argparse.ArgumentParser()
ap.add_argument("policy"); ap.add_argument("--vx", type=float, default=0.3)
ap.add_argument("--video", default=None); ap.add_argument("--seconds", type=float, default=6.0)
args = ap.parse_args()
app = AppLauncher(headless=True, enable_cameras=bool(args.video)).app

import math, torch, gymnasium as gym
import isaaclab_tasks  # noqa
from isaaclab_tasks.utils import parse_env_cfg

cfg = parse_env_cfg("Isaac-Velocity-Flat-Quadbot-Play-v0", num_envs=1 if args.video else 64)
cfg.events.base_external_force_torque = None
c = cfg.commands.base_velocity
c.heading_command = False
c.rel_standing_envs = 0.0
c.resampling_time_range = (1e6, 1e6)
c.ranges.lin_vel_x = (args.vx, args.vx); c.ranges.lin_vel_y = (0.0, 0.0); c.ranges.ang_vel_z = (0.0, 0.0)
cfg.events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
if args.video:
    # fixed side camera on env 0: the robot walks across the frame
    cfg.viewer.origin_type = "env"; cfg.viewer.env_index = 0
    cfg.viewer.eye = (0.9, 1.7, 0.45); cfg.viewer.lookat = (0.9, 0.0, 0.08)
env = gym.make("Isaac-Velocity-Flat-Quadbot-Play-v0", cfg=cfg, render_mode="rgb_array" if args.video else None)
u = env.unwrapped
pol = torch.jit.load(args.policy, map_location=u.device).eval()
obs, _ = env.reset()
rob = u.scene["robot"]
p0 = rob.data.root_pos_w.clone()
frames = []
steps = int(args.seconds / u.step_dt)
vb = []
with torch.inference_mode():
    for i in range(steps):
        obs, _, _, _, _ = env.step(pol(obs["policy"]))
        if i > steps // 3:
            vb.append(rob.data.root_lin_vel_b[:, :2].clone())
        if args.video and i % 1 == 0:
            frames.append(env.render())
d = rob.data.root_pos_w - p0
v = torch.stack(vb).mean(0)
print(f"cmd vx={args.vx}  body-frame vx mean {v[:,0].mean():.3f} (min {v[:,0].min():.3f}, max {v[:,0].max():.3f})  vy {v[:,1].mean():.3f}", flush=True)
print(f"world displacement over {args.seconds}s: x {d[:,0].mean():.3f} m  y {d[:,1].mean():.3f} m  (expected {args.vx*args.seconds:.2f})", flush=True)
print(f"robot0 displacement {d[0,:2].tolist()}", flush=True)
if args.video:
    import imageio.v3 as iio
    iio.imwrite(args.video, frames, fps=int(round(1 / u.step_dt)), codec="libx264", quality=8)
    print("wrote", args.video, flush=True)
env.close(); app.close()
