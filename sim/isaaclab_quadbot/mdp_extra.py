import torch

from isaaclab.managers import SceneEntityCfg  # noqa: F401 (re-exported for eval scripts)


def joint_power_l1(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Mechanical power sum |tau * qdot| (W) - proxy for the F3 current sensor reading."""
    asset = env.scene[asset_cfg.name]
    tau = asset.data.applied_torque[:, asset_cfg.joint_ids]
    qd = asset.data.joint_vel[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(tau * qd), dim=1)


# STS3215-12V: 2.94 N.m / 2.7 A stall at 12 V  ->  R = 4.4 ohm, kt(output) = 1.09 N.m/A
STS_R = 12.0 / 2.7
STS_KT = 2.94 / 2.7
K_COPPER = STS_R / STS_KT**2          # W per (N.m)^2 of output torque


def servo_electrical_power(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Estimated battery power drawn by the leg servos (W): positive mechanical power
    (braking is not recovered) + copper loss. This is what the F3 current sensor sees,
    minus the constant Jetson/electronics load."""
    asset = env.scene[asset_cfg.name]
    tau = asset.data.applied_torque[:, asset_cfg.joint_ids]
    qd = asset.data.joint_vel[:, asset_cfg.joint_ids]
    return torch.sum(torch.clamp(tau * qd, min=0.0) + K_COPPER * tau**2, dim=1)


def command_speed_levels(env, env_ids, reward_term_name: str = "track_lin_vel_xy_exp",
                         max_vx: tuple = (-0.8, 1.2), max_vy: float = 0.5, step: float = 0.1,
                         threshold: float = 0.8):
    """Widen the commanded speed range whenever the robots track the current one well.
    Score = running mean over every finished episode of (tracking reward / its max), falls count
    as short episodes; checked once per episode length."""
    rng = env.command_manager.get_term("base_velocity").cfg.ranges
    if len(env_ids) > 0:
        w = env.reward_manager.get_term_cfg(reward_term_name).weight
        s = env.reward_manager._episode_sums[reward_term_name][env_ids] / env.max_episode_length_s / w
        prev = getattr(env, "_speed_score", None)
        env._speed_score = float(s.mean()) if prev is None else 0.98 * prev + 0.02 * float(s.mean())
    if env.common_step_counter % env.max_episode_length == 0 and getattr(env, "_speed_score", 0.0) > threshold:
        rng.lin_vel_x = (max(rng.lin_vel_x[0] - step, max_vx[0]), min(rng.lin_vel_x[1] + step, max_vx[1]))
        rng.lin_vel_y = (max(rng.lin_vel_y[0] - step / 2, -max_vy), min(rng.lin_vel_y[1] + step / 2, max_vy))
        env._speed_score = threshold - 0.1            # re-earn the next step at the new range
    return rng.lin_vel_x[1]


from isaaclab.managers import ManagerTermBase, RewardTermCfg  # noqa: E402


class joint_range_of_motion(ManagerTermBase):
    """Reward using the joints' range: per joint, (max - min) of its angle over a sliding window
    (~one stride), capped at a target swing so flailing earns nothing extra. Mean over the joints,
    in [0, 1]; only paid while a walking command is active."""

    def __init__(self, cfg: RewardTermCfg, env):
        super().__init__(cfg, env)
        self.asset = env.scene[cfg.params.get("asset_cfg", SceneEntityCfg("robot")).name]
        self.window = int(round(cfg.params.get("window_s", 0.6) / env.step_dt))
        names = self.asset.joint_names
        targets = cfg.params["targets"]                   # {regex: target swing (rad)}
        import re
        ids, tgt = [], []
        for j, n in enumerate(names):
            for pat, t in targets.items():
                if re.fullmatch(pat, n):
                    ids.append(j); tgt.append(t); break
        self.ids = torch.tensor(ids, device=env.device)
        self.tgt = torch.tensor(tgt, device=env.device)
        self.hist = self.asset.data.joint_pos[:, self.ids].unsqueeze(1).repeat(1, self.window, 1).clone()
        self.ptr = 0

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        self.hist[env_ids] = self.asset.data.joint_pos[env_ids][:, self.ids].unsqueeze(1)

    def __call__(self, env, targets, window_s=0.6, command_name="base_velocity", asset_cfg=None):
        self.hist[:, self.ptr] = self.asset.data.joint_pos[:, self.ids]
        self.ptr = (self.ptr + 1) % self.window
        rom = self.hist.max(dim=1).values - self.hist.min(dim=1).values
        r = torch.clamp(rom / self.tgt, max=1.0).mean(dim=1)
        return r * (torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1)


def foot_clearance_rel(env, asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg, target: float,
                       std: float, tanh_mult: float) -> torch.Tensor:
    """Swing feet should rise `target` above the feet that are on the ground (works on slopes and
    hills where world z is meaningless); weighted by how fast the foot moves (swinging)."""
    asset = env.scene[asset_cfg.name]
    cs = env.scene.sensors[sensor_cfg.name]
    z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    contact = cs.data.current_contact_time[:, sensor_cfg.body_ids] > 0.0
    n_c = contact.sum(dim=1, keepdim=True)
    ground = torch.where(n_c > 0, (z * contact).sum(dim=1, keepdim=True) / n_c.clamp(min=1), z.min(dim=1, keepdim=True).values)
    err = torch.square((z - ground) - target) * (~contact)
    vel = torch.tanh(tanh_mult * torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2))
    return torch.exp(-torch.sum(err * vel, dim=1) / std)


class action_jerk(ManagerTermBase):
    """Second difference of the actions |a_t - 2 a_t-1 + a_t-2|^2 (smooth servo targets)."""

    def __init__(self, cfg: RewardTermCfg, env):
        super().__init__(cfg, env)
        self.a1 = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
        self.a2 = torch.zeros_like(self.a1)

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        self.a1[env_ids] = 0.0
        self.a2[env_ids] = 0.0

    def __call__(self, env):
        a = env.action_manager.action
        j = torch.sum(torch.square(a - 2 * self.a1 + self.a2), dim=1)
        self.a2[:] = self.a1
        self.a1[:] = a
        return j


def fast_tracking_bonus(env, std: float, above: float, command_name: str = "base_velocity",
                        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Extra tracking reward that grows with commanded speed above `above` m/s, so smoothness
    penalties can't make 'slow' the easy answer."""
    asset = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)[:, :2]
    err = torch.sum(torch.square(cmd - asset.data.root_lin_vel_b[:, :2]), dim=1)
    return torch.exp(-err / std**2) * torch.clamp(torch.norm(cmd, dim=1) - above, min=0.0) / (1.0 - above)


def foot_contact_slip(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg, radius: float = 0.021) -> torch.Tensor:
    """Horizontal speed of the contact point of each ball foot in stance (rolling is not slip)."""
    asset = env.scene[asset_cfg.name]
    cs = env.scene.sensors[sensor_cfg.name]
    contact = cs.data.net_forces_w_history[:, :, sensor_cfg.body_ids, 2].max(dim=1)[0] > 1.0
    v = asset.data.body_lin_vel_w[:, asset_cfg.body_ids]
    w = asset.data.body_ang_vel_w[:, asset_cfg.body_ids]
    r = torch.zeros_like(v); r[..., 2] = -radius
    vc = v + torch.cross(w, r, dim=-1)
    return torch.sum(torch.norm(vc[..., :2], dim=-1) * contact, dim=1)


def air_time_cadence(env, asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg, t_slow: float, t_fast: float,
                     v_fast: float, velocity_threshold: float = 0.15) -> torch.Tensor:
    """Spot-style air/stance time reward, but the target step time shrinks with commanded speed
    (cats raise cadence to go faster instead of over-striding): t_slow at 0, t_fast at v_fast."""
    cs = env.scene.sensors[sensor_cfg.name]
    asset = env.scene[asset_cfg.name]
    air = cs.data.current_air_time[:, sensor_cfg.body_ids]
    con = cs.data.current_contact_time[:, sensor_cfg.body_ids]
    cmd = torch.norm(env.command_manager.get_command("base_velocity")[:, :2], dim=1)
    mode_time = (t_slow + (t_fast - t_slow) * torch.clamp(cmd / v_fast, 0.0, 1.0)).unsqueeze(1)
    t_max = torch.max(air, con)
    t_min = torch.minimum(t_max, mode_time)
    stance = torch.clamp(con - air, -mode_time, mode_time)
    moving = (torch.norm(env.command_manager.get_command("base_velocity"), dim=1) > 0.0) | \
             (torch.norm(asset.data.root_lin_vel_b[:, :2], dim=1) > velocity_threshold)
    r = torch.where(moving.unsqueeze(1), torch.where(t_max < mode_time, t_min, torch.zeros_like(t_min)), stance)
    return torch.sum(r, dim=1) / mode_time.squeeze(1) * 0.25     # normalised to the v1 scale


class touchdown_speed(ManagerTermBase):
    """Soft paws: squared downward speed of each foot in the step before it touches down."""

    def __init__(self, cfg: RewardTermCfg, env):
        super().__init__(cfg, env)
        self.asset = env.scene[cfg.params["asset_cfg"].name]
        self.cs = env.scene.sensors[cfg.params["sensor_cfg"].name]
        self.feet_a = self.asset.find_bodies(".*_foot")[0]
        self.feet_s = self.cs.find_bodies(".*_foot")[0]
        self.vz_prev = torch.zeros(env.num_envs, len(self.feet_a), device=env.device)

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        self.vz_prev[env_ids] = 0.0

    def __call__(self, env, asset_cfg, sensor_cfg):
        first = self.cs.compute_first_contact(env.step_dt)[:, self.feet_s]
        pen = torch.sum(first * torch.square(torch.clamp(-self.vz_prev, min=0.0)), dim=1)
        self.vz_prev[:] = self.asset.data.body_lin_vel_w[:, self.feet_a, 2]
        return pen


def command_progress(env, command_name: str = "base_velocity", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
                     min_cmd: float = 0.1) -> torch.Tensor:
    """Linear progress toward the commanded velocity: fraction achieved along the command (capped
    at 1) minus sideways drift. Unlike the exp tracking terms it keeps a gradient when the command
    is out of reach (steep hills), so 'freeze and hold' never beats 'climb as fast as you can'."""
    asset = env.scene[asset_cfg.name]
    c = env.command_manager.get_command(command_name)[:, :2]
    n = torch.norm(c, dim=1)
    u = c / n.clamp(min=1e-6).unsqueeze(1)
    v = asset.data.root_lin_vel_b[:, :2]
    along = torch.sum(v * u, dim=1)
    perp = torch.norm(v - along.unsqueeze(1) * u, dim=1)
    r = torch.clamp(along / n.clamp(min=min_cmd), -1.0, 1.0) - torch.clamp(perp / n.clamp(min=0.2), 0.0, 1.0)
    return r * (n > min_cmd)
