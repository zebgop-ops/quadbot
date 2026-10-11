"""Gait quality ("smooth and nimble as a cat") for an exported quadbot policy, on flat ground.
Scripted command profile: trot, sprint, turn, sidestep, stop, reverse; half the robots get a shove.
python eval_gait.py POLICY.pt --out gait.json [--task ...]"""
import argparse
from isaaclab.app import AppLauncher
ap = argparse.ArgumentParser()
ap.add_argument("policy"); ap.add_argument("--out", required=True)
ap.add_argument("--task", default="Isaac-Velocity-Rough-Quadbot-Play-v0")
ap.add_argument("--video", default=None)
ap.add_argument("--sim_dt", type=float, default=None)
ap.add_argument("--terrain", action="store_true", help="keep the hills/slopes terrain (play config)")
args = ap.parse_args()
app = AppLauncher(headless=True, enable_cameras=bool(args.video)).app

import json, math, torch, gymnasium as gym
import isaaclab_tasks  # noqa
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_tasks.manager_based.locomotion.velocity.config.quadbot import mdp_extra

# (t_start, vx, vy, wz, name)
PROFILE = [(0.0, 0.4, 0.0, 0.0, "trot"), (3.0, 0.8, 0.0, 0.0, "sprint"), (5.0, 0.3, 0.0, 2.0, "turn"),
           (7.0, 0.0, 0.3, 0.0, "sidestep"), (9.0, 0.0, 0.0, 0.0, "stop"), (10.5, -0.3, 0.0, 0.0, "reverse")]
T_END, PUSH_T, PUSH_V = 12.5, 1.5, 0.6
N = 1 if args.video else 128

cfg = parse_env_cfg(args.task, num_envs=N)
if not args.terrain:
    cfg.scene.terrain.terrain_type = "plane"; cfg.scene.terrain.terrain_generator = None
else:
    cfg.scene.terrain.terrain_generator.difficulty_range = (0.6, 1.0)
for k in ("terrain_levels", "speed_levels"):
    if getattr(cfg.curriculum, k, None) is not None: setattr(cfg.curriculum, k, None)
cfg.events.add_base_mass = None; cfg.events.base_external_force_torque = None; cfg.events.push_robot = None
cfg.events.physics_material.params["static_friction_range"] = (0.8, 0.8)
cfg.events.physics_material.params["dynamic_friction_range"] = (0.6, 0.6)
cfg.events.reset_base.params = {"pose_range": {"x": (0, 0), "y": (0, 0), "yaw": (0, 0)},
                                "velocity_range": {k: (0.0, 0.0) for k in ("x", "y", "z", "roll", "pitch", "yaw")}}
cfg.scene.env_spacing = 1.0
if args.sim_dt:
    cfg.decimation = int(round(cfg.sim.dt * cfg.decimation / args.sim_dt)); cfg.sim.dt = args.sim_dt
    cfg.sim.render_interval = cfg.decimation; cfg.scene.contact_forces.update_period = cfg.sim.dt
c = cfg.commands.base_velocity
c.heading_command = False; c.rel_standing_envs = 0.0; c.resampling_time_range = (1e6, 1e6)
if args.video:
    cfg.viewer.origin_type = "asset_root"; cfg.viewer.asset_name = "robot"; cfg.viewer.env_index = 0
    cfg.viewer.eye = (0.05, 0.85, 0.22); cfg.viewer.lookat = (0.05, 0.0, 0.05); cfg.viewer.resolution = (1280, 720)
env = gym.make(args.task, cfg=cfg, render_mode="rgb_array" if args.video else None); u = env.unwrapped
pol = torch.jit.load(args.policy, map_location=u.device).eval()
obs, _ = env.reset()
rob, cs = u.scene["robot"], u.scene["contact_forces"]
feet_r = rob.find_bodies(".*_foot")[0]
feet_s = cs.find_bodies(".*_foot")[0]
cmd_term = u.command_manager.get_term("base_velocity")
dt = u.step_dt
mass = float(rob.root_physx_view.get_masses()[0].sum()); W = mass * 9.81
push = torch.zeros(N, dtype=torch.bool, device=u.device); push[N // 2:] = True if N > 1 else False


def cmd_at(t):
    cur = PROFILE[0]
    for p in PROFILE:
        if t >= p[0]: cur = p
    return cur


keys = ["t", "phase", "v", "w", "cmd", "act", "tgt", "fzmax", "fvz", "qd", "tau", "qacc", "fz", "fh", "fv", "rp", "vz", "p"]
log = {k: [] for k in keys}
fell = torch.zeros(N, dtype=torch.bool, device=u.device)
last_act = None; last_qd = rob.data.joint_vel.clone()
frames = []
with torch.inference_mode():
    for i in range(int(T_END / dt)):
        t = i * dt
        ph = cmd_at(t)
        cmd_term.vel_command_b[:] = torch.tensor(ph[1:4], device=u.device)
        obs = u.observation_manager.compute()
        a = pol(obs["policy"])
        obs, _, term, _, _ = env.step(a)
        if abs(t - PUSH_T) < dt / 2 and push.any():
            v = rob.data.root_vel_w.clone(); v[push, 1] += PUSH_V
            rob.write_root_velocity_to_sim(v[push], env_ids=push.nonzero().squeeze(-1))
        fell |= term
        qd = rob.data.joint_vel
        log["t"].append(t); log["phase"].append(PROFILE.index(ph))
        log["v"].append(rob.data.root_lin_vel_b[:, :2].clone()); log["w"].append(rob.data.root_ang_vel_b[:, 2].clone())
        log["rp"].append(rob.data.root_ang_vel_b[:, :2].clone()); log["vz"].append(rob.data.root_lin_vel_w[:, 2].clone())
        log["cmd"].append(cmd_term.vel_command_b.clone()); log["act"].append(a.clone())
        log["tgt"].append(u.action_manager.get_term("joint_pos").processed_actions.clone() / 0.25)
        log["qacc"].append((qd - last_qd) / dt); last_qd = qd.clone()
        log["qd"].append(qd.clone()); log["tau"].append(rob.data.applied_torque.clone())
        log["fz"].append(cs.data.net_forces_w[:, feet_s, 2].clone())
        log["fzmax"].append(cs.data.net_forces_w_history[:, :, feet_s, 2].max(dim=1).values.clone())  # spikes between policy steps
        log["fvz"].append(rob.data.body_lin_vel_w[:, feet_r, 2].clone())
        log["fh"].append(rob.data.body_pos_w[:, feet_r, 2].clone() - 0.021)
        vfc = rob.data.body_lin_vel_w[:, feet_r] + torch.cross(rob.data.body_ang_vel_w[:, feet_r],
              torch.tensor([0.0, 0.0, -0.021], device=u.device).expand_as(rob.data.body_lin_vel_w[:, feet_r]), dim=-1)
        log["fv"].append(torch.norm(vfc[..., :2], dim=-1))
        log["p"].append(mdp_extra.servo_electrical_power(u, mdp_extra.SceneEntityCfg("robot")))
        if args.video: frames.append(env.render())

S = {k: (torch.stack(v).cpu() if k not in ("t", "phase") else torch.tensor(v)) for k, v in log.items()}
fell, push = fell.cpu(), push.cpu()
ok = ~fell; calm = ok & ~push
m = {}
m["fall_rate"] = float(fell[~push].float().mean()) if (~push).any() else float(fell.float().mean())
m["push_fall_rate"] = float(fell[push].float().mean()) if push.any() else float("nan")
T = S["t"]
settle = torch.zeros_like(T, dtype=torch.bool)
for k, p in enumerate(PROFILE):
    t1 = PROFILE[k + 1][0] if k + 1 < len(PROFILE) else T_END
    settle |= (T >= p[0] + 0.7) & (T < t1)
nopush_win = (T < PUSH_T - 0.05) | (T > PUSH_T + 1.0)
sel = settle & nopush_win
ev = (S["v"] - S["cmd"][..., :2])[sel][:, calm]
ew = (S["w"] - S["cmd"][..., 2])[sel][:, calm]
m["track_v_rms"] = float(ev.norm(dim=-1).pow(2).mean().sqrt())
m["track_yaw_rms"] = float(ew.pow(2).mean().sqrt())
for k, p in enumerate(PROFILE):
    t1 = PROFILE[k + 1][0] if k + 1 < len(PROFILE) else T_END
    w_ = (T >= p[0] + 0.7) & (T < t1)
    m[f"v_{p[4]}"] = [round(float(x), 3) for x in S["v"][w_][:, calm].mean((0, 1))] + [round(float(S["w"][w_][:, calm].mean()), 3)]


def resp(t0, comp, target, frac=0.9):
    """time after t0 until the calm-robot mean of component reaches frac of the step"""
    idx = (T >= t0).nonzero().squeeze(-1)
    base = {"vx": S["v"][..., 0], "wz": S["w"], "vy": S["v"][..., 1]}[comp][:, calm].mean(1)
    start = float(base[idx[0] - 1])
    for i in idx:
        if (float(base[i]) - start) / (target - start) >= frac:
            return float(T[i] - t0)
    return float("nan")


for k, p in enumerate(PROFILE):                      # speed-tracking error per phase
    t1 = PROFILE[k + 1][0] if k + 1 < len(PROFILE) else T_END
    w_ = (T >= p[0] + 0.7) & (T < t1) & nopush_win
    e_ = (S["v"] - S["cmd"][..., :2])[w_][:, calm]
    m[f"track_{p[4]}"] = round(float(e_.norm(dim=-1).pow(2).mean().sqrt()), 4)
    vm = S["v"][w_][:, calm]                                          # (time, robot, 2)
    m[f"meanerr_{p[4]}"] = round(float((vm.mean(0) - S["cmd"][w_][:, calm][..., :2].mean(0)).norm(dim=-1).mean()), 4)
    m[f"surge_{p[4]}"] = round(float((vm - vm.mean(0, keepdim=True)).norm(dim=-1).pow(2).mean().sqrt()), 4)
m["resp_sprint_s"] = resp(3.0, "vx", 0.8)
m["resp_turn_s"] = resp(5.0, "wz", 2.0)
m["resp_side_s"] = resp(7.0, "vy", 0.3)
m["resp_stop_s"] = resp(9.0, "vy", 0.0)
moving = sel & (S["phase"] != 4)
m["rollpitch_rate_rms"] = float(S["rp"][moving][:, calm].pow(2).mean().sqrt())
m["bounce_vz_rms"] = float(S["vz"][moving][:, calm].pow(2).mean().sqrt())
da = (S["tgt"][1:] - S["tgt"][:-1])[moving[1:]][:, calm]
m["action_rate_rms"] = float(da.pow(2).mean().sqrt())
# jitter = high-frequency part of the servo targets: second difference (a smooth 2 Hz gait ~0.05)
d2 = (S["tgt"][2:] - 2 * S["tgt"][1:-1] + S["tgt"][:-2])[moving[2:]][:, calm]
m["target_jitter_rms"] = float(d2.pow(2).mean().sqrt())
m["joint_acc_rms"] = float(S["qacc"][moving][:, calm].pow(2).mean().sqrt())
fz, fh, fv = S["fz"][:, calm], S["fh"][:, calm], S["fv"][:, calm]
contact = fz > 1.0
st = contact & moving[:, None, None]
m["duty_factor"] = float(st.float().sum() / moving[:, None, None].expand_as(contact).float().sum())
td = contact[1:] & ~contact[:-1]                               # touchdowns
peaks = []
for i in td.nonzero():
    s, e, f = i[0] + 1, min(i[0] + 6, fz.shape[0]), (i[1], i[2])
    peaks.append(float(fz[s:e, f[0], f[1]].max()))
m["impact_peak_over_weight"] = (sum(peaks) / len(peaks) / W) if peaks else float("nan")
# touchdown spike vs that stance's mean load (soft paw ~1, slap >1.5); history max catches the 5 ms spikes
fzm = S["fzmax"][:, calm]
ratios = []
for i in td.nonzero():
    s0, e0, f0 = int(i[0]) + 1, int(i[0]) + 1, (int(i[1]), int(i[2]))
    while e0 < contact.shape[0] and contact[e0, f0[0], f0[1]]:
        e0 += 1
    if e0 - s0 >= 3:
        mean_st = float(fz[s0:e0, f0[0], f0[1]].mean())
        if mean_st > 1.0:
            ratios.append(float(fzm[s0:s0 + 2, f0[0], f0[1]].max()) / mean_st)
fvz = S["fvz"][:, calm]
tdv = [max(0.0, -float(fvz[int(i[0]), int(i[1]), int(i[2])])) for i in td.nonzero()]   # step before contact
m["touchdown_speed_mps"] = (sum(tdv) / len(tdv)) if tdv else float("nan")
m["touchdown_spike_ratio"] = (sum(ratios) / len(ratios)) if ratios else float("nan")
m["touchdowns_per_foot_s"] = len(peaks) / (4 * int(calm.sum()) * T_END) if peaks else 0.0
m["foot_slip_mps"] = float(fv[contact & moving[:, None, None]].mean())
sw = (~contact) & moving[:, None, None]
m["swing_clearance_m"] = float(fh[sw].quantile(0.9)) if sw.any() else 0.0
for k, p in enumerate(PROFILE):                      # servo saturation per phase
    t1 = PROFILE[k + 1][0] if k + 1 < len(PROFILE) else T_END
    w_ = (T >= p[0] + 0.7) & (T < t1)
    m[f"sat_{p[4]}"] = [round(float((S["qd"][w_][:, calm].abs() > 0.9 * 4.7).float().mean()), 3),
                        round(float((S["tau"][w_][:, calm].abs() > 0.9 * 2.5).float().mean()), 3)]
for k, p in enumerate(PROFILE):                      # smoothness per phase: wobble, bounce, jitter
    t1 = PROFILE[k + 1][0] if k + 1 < len(PROFILE) else T_END
    w_ = (T >= p[0] + 0.7) & (T < t1) & nopush_win
    j_ = w_[2:]
    d2p = (S["tgt"][2:] - 2 * S["tgt"][1:-1] + S["tgt"][:-2])[j_][:, calm]
    m[f"smooth_{p[4]}"] = [round(float(S["rp"][w_][:, calm].pow(2).mean().sqrt()), 3),
                           round(float(S["vz"][w_][:, calm].pow(2).mean().sqrt()), 3),
                           round(float(d2p.pow(2).mean().sqrt()), 3)]
m["servo_power_w"] = float(S["p"][moving][:, calm].mean())
m["phase_names"] = [p[4] for p in PROFILE]
json.dump(m, open(args.out, "w"), indent=1)
for k, v in m.items(): print(f"{k:24s} {v}", flush=True)
if args.video:
    import imageio.v3 as iio
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 30)
    except OSError:
        font = ImageFont.load_default()
    out = []
    for i, fr in enumerate(frames[1:], start=1):
        im = Image.fromarray(fr); d = ImageDraw.Draw(im)
        c = S["cmd"][i, 0]; v = S["v"][i, 0]; w = S["w"][i, 0]
        txt = (f"{PROFILE[int(S['phase'][i])][4]:9s}  t={float(T[i]):4.1f}s\n"
               f"command  vx {float(c[0]):+.2f}  vy {float(c[1]):+.2f} m/s  yaw {float(c[2]):+.1f} rad/s\n"
               f"actual   vx {float(v[0]):+.2f}  vy {float(v[1]):+.2f} m/s  yaw {float(w):+.1f} rad/s")
        d.rectangle([12, 12, 760, 128], fill=(0, 0, 0))
        d.multiline_text((24, 18), txt, fill=(255, 255, 255), font=font, spacing=6)
        out.append(np.asarray(im))
    iio.imwrite(args.video, out, fps=int(round(1 / dt)), codec="libx264", quality=8); print("wrote", args.video)
env.close(); app.close()
