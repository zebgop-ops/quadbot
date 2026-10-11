"""Zero-action rollout: does the quadbot stand on the STS3215 actuator model?"""
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True).app

import torch
import gymnasium as gym
import isaaclab_tasks  # noqa
from isaaclab_tasks.utils import parse_env_cfg

cfg = parse_env_cfg("Isaac-Velocity-Flat-Quadbot-Play-v0", num_envs=4)
cfg.events.base_external_force_torque = None
cfg.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
env = gym.make("Isaac-Velocity-Flat-Quadbot-Play-v0", cfg=cfg).unwrapped
env.reset()
rob = env.scene["robot"]
cs = env.scene["contact_forces"]
print("bodies:", rob.body_names)
print("joints:", rob.joint_names)
print("total mass kg:", float(rob.root_physx_view.get_masses()[0].sum()))
act = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
for i in range(150):  # 3 s
    env.step(act)
    if i % 25 == 24:
        z = rob.data.root_pos_w[:, 2]
        g = rob.data.projected_gravity_b[0]
        feet = [j for j, n in enumerate(cs.body_names) if n.endswith("_foot")]
        fz = cs.data.net_forces_w[0, feet, 2]
        print(f"t={0.02*(i+1):.1f}s base_z={z.mean():.4f} grav={g.tolist()} foot_Fz={[round(float(f),2) for f in fz]}"
              f" tau={[round(float(t),2) for t in rob.data.applied_torque[0]]}")
print("joint_pos err (rad):", (rob.data.joint_pos[0] - rob.data.default_joint_pos[0]).abs().max().item())
env.close()
app.close()
