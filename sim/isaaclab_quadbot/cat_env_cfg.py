"""'Smooth and nimble as a cat': the terrain task + gait shaping (trot timing, longer steps, foot
clearance, soft landings, no slip, smooth servo targets, steady body) and quicker command changes."""
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.config.spot import mdp as spot_mdp

from . import mdp_extra
from .rough_env_cfg import QuadbotRoughEnvCfg, QuadbotRoughEnvCfg_PLAY

FEET = SceneEntityCfg("robot", body_names=".*_foot")
FEET_S = SceneEntityCfg("contact_forces", body_names=".*_foot")


def apply_cat(cfg):
    r = cfg.rewards
    # gait timing: diagonal pairs together, ~0.25 s swing/stance, all feet alike
    r.feet_air_time = None
    r.gait = RewTerm(func=spot_mdp.GaitReward, weight=1.5,
                     params={"std": 0.1, "max_err": 0.2, "velocity_threshold": 0.15,
                             "synced_feet_pair_names": (("LF_foot", "RR_foot"), ("RF_foot", "LR_foot")),
                             "asset_cfg": SceneEntityCfg("robot"), "sensor_cfg": SceneEntityCfg("contact_forces")})
    r.air_time = RewTerm(func=spot_mdp.air_time_reward, weight=0.8,
                         params={"mode_time": 0.25, "velocity_threshold": 0.15,
                                 "asset_cfg": SceneEntityCfg("robot"), "sensor_cfg": FEET_S})
    r.air_time_variance = RewTerm(func=spot_mdp.air_time_variance_penalty, weight=-0.5, params={"sensor_cfg": FEET_S})
    r.foot_clearance = RewTerm(func=mdp_extra.foot_clearance_rel, weight=0.5,
                               params={"asset_cfg": FEET, "sensor_cfg": FEET_S, "target": 0.04, "std": 0.004,
                                       "tanh_mult": 4.0})
    # feet: no skating, soft touchdowns (~0.75 x body weight per foot max)
    r.feet_slide.weight = -0.4
    r.soft_landing = RewTerm(func=mdp.contact_forces, weight=-0.02, params={"sensor_cfg": FEET_S, "threshold": 16.0})
    # smooth body and servo targets
    r.base_motion = RewTerm(func=spot_mdp.base_motion_penalty, weight=-0.5, params={"asset_cfg": SceneEntityCfg("robot")})
    r.action_rate_l2.weight = -0.03
    r.action_jerk = RewTerm(func=mdp_extra.action_jerk, weight=-0.02)
    r.dof_acc_l2.weight = -1.0e-6
    # posture: calm when told to stand, free legs when walking
    r.pose_deviation = RewTerm(func=spot_mdp.joint_position_penalty, weight=-0.05,
                               params={"asset_cfg": SceneEntityCfg("robot"), "stand_still_scale": 5.0,
                                       "velocity_threshold": 0.15})
    r.hip_roll = RewTerm(func=mdp.joint_deviation_l1, weight=-0.1,
                         params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_roll"])})
    # nimble: commands change often, sharp turns, stops
    c = cfg.commands.base_velocity
    c.resampling_time_range = (3.0, 6.0)
    c.rel_heading_envs = 0.5
    c.rel_standing_envs = 0.1
    c.ranges.ang_vel_z = (-2.5, 2.5)


@configclass
class QuadbotCatEnvCfg(QuadbotRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        apply_cat(self)


@configclass
class QuadbotCatEnvCfg_PLAY(QuadbotRoughEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        apply_cat(self)


def apply_cat2(cfg):
    """Cat v2: v1 + much stronger smoothness terms + a gentle range-of-motion reward
    (on its own the ROM reward made the legs flail; here smoothness reins it in)."""
    r = cfg.rewards
    r.action_rate_l2.weight = -0.08
    r.action_jerk.weight = -0.06
    r.base_motion.weight = -1.5
    r.ang_vel_xy_l2.weight = -0.15
    r.lin_vel_z_l2.weight = -3.0
    r.dof_acc_l2.weight = -3.0e-6
    r.soft_landing.weight = -0.05
    r.soft_landing.params["threshold"] = 13.0
    r.feet_slide.weight = -0.8
    r.foot_clearance.weight = 1.0
    r.foot_clearance.params["target"] = 0.045
    r.range_of_motion = RewTerm(func=mdp_extra.joint_range_of_motion, weight=0.15,
                                params={"targets": {".*_hip_pitch": 0.5, ".*_knee": 0.7}, "window_s": 0.6})


@configclass
class QuadbotCat2EnvCfg(QuadbotRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        apply_cat(self)
        apply_cat2(self)


@configclass
class QuadbotCat2EnvCfg_PLAY(QuadbotRoughEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        apply_cat(self)
        apply_cat2(self)


def apply_cat3(cfg):
    """Cat v3: v2 + low-pass filtered servo targets (~8 Hz corner), a fast-tracking bonus,
    and slip measured at the paw's contact point (a rolling ball foot is not slipping)."""
    from .filtered_action import FilteredJointPositionActionCfg
    cfg.actions.joint_pos = FilteredJointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.25,
                                                           use_default_offset=True, alpha=0.5)
    r = cfg.rewards
    r.fast_tracking = RewTerm(func=mdp_extra.fast_tracking_bonus, weight=1.0, params={"std": 0.2, "above": 0.4})
    r.feet_slide = RewTerm(func=mdp_extra.foot_contact_slip, weight=-0.8, params={"sensor_cfg": FEET_S, "asset_cfg": FEET})
    # curricula are not stored in checkpoints: resume from where cat v2 had got to
    cfg.commands.base_velocity.ranges.lin_vel_x = (-0.6, 0.9)
    cfg.commands.base_velocity.ranges.lin_vel_y = (-0.35, 0.35)
    cfg.scene.terrain.max_init_terrain_level = 5


@configclass
class QuadbotCat3EnvCfg(QuadbotRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        apply_cat(self)
        apply_cat2(self)
        apply_cat3(self)


@configclass
class QuadbotCat3EnvCfg_PLAY(QuadbotRoughEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        apply_cat(self)
        apply_cat2(self)
        apply_cat3(self)


def apply_cat4(cfg):
    """Cat v4: v3 + cadence that rises with speed."""
    cfg.rewards.air_time = RewTerm(func=mdp_extra.air_time_cadence, weight=0.8,
                                   params={"asset_cfg": SceneEntityCfg("robot"), "sensor_cfg": FEET_S,
                                           "t_slow": 0.28, "t_fast": 0.16, "v_fast": 0.9})
    cfg.scene.terrain.max_init_terrain_level = 4


@configclass
class QuadbotCat4EnvCfg(QuadbotRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        apply_cat(self); apply_cat2(self); apply_cat3(self); apply_cat4(self)


@configclass
class QuadbotCat4EnvCfg_PLAY(QuadbotRoughEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        apply_cat(self); apply_cat2(self); apply_cat3(self); apply_cat4(self)


def apply_cat5(cfg):
    """Cat v5: v4 + soft paw placement (slow each foot before it touches down)."""
    cfg.rewards.touchdown = RewTerm(func=mdp_extra.touchdown_speed, weight=-8.0,
                                    params={"asset_cfg": SceneEntityCfg("robot"), "sensor_cfg": SceneEntityCfg("contact_forces")})


@configclass
class QuadbotCat5EnvCfg(QuadbotRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        apply_cat(self); apply_cat2(self); apply_cat3(self); apply_cat4(self); apply_cat5(self)


@configclass
class QuadbotCat5EnvCfg_PLAY(QuadbotRoughEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        apply_cat(self); apply_cat2(self); apply_cat3(self); apply_cat4(self); apply_cat5(self)


def apply_cat6(cfg):
    """Cat v6: v5 + sharper speed tracking so smoothness can't be bought by running slow."""
    r = cfg.rewards
    r.fast_tracking.weight = 2.5
    r.fast_tracking.params["std"] = 0.12
    r.track_lin_vel_xy_exp.params["std"] = 0.2


@configclass
class QuadbotCat6EnvCfg(QuadbotRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        apply_cat(self); apply_cat2(self); apply_cat3(self); apply_cat4(self); apply_cat5(self); apply_cat6(self)


@configclass
class QuadbotCat6EnvCfg_PLAY(QuadbotRoughEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        apply_cat(self); apply_cat2(self); apply_cat3(self); apply_cat4(self); apply_cat5(self); apply_cat6(self)


def apply_cat7(cfg):
    """Cat v7: v6 + stronger target filter (alpha 0.4, ~5 Hz corner), more jerk penalty, softer paws."""
    cfg.actions.joint_pos.alpha = 0.4
    cfg.rewards.action_jerk.weight = -0.1
    cfg.rewards.touchdown.weight = -16.0


@configclass
class QuadbotCat7EnvCfg(QuadbotRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        for f in (apply_cat, apply_cat2, apply_cat3, apply_cat4, apply_cat5, apply_cat6, apply_cat7):
            f(self)


@configclass
class QuadbotCat7EnvCfg_PLAY(QuadbotRoughEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        for f in (apply_cat, apply_cat2, apply_cat3, apply_cat4, apply_cat5, apply_cat6, apply_cat7):
            f(self)


def apply_cat8(cfg):
    """Cat v8: v7 + practise sharp turns while moving (80 % direct yaw-rate commands),
    a steadier body and softer paws."""
    cfg.commands.base_velocity.rel_heading_envs = 0.2
    cfg.rewards.base_motion.weight = -2.0
    cfg.rewards.touchdown.weight = -20.0


@configclass
class QuadbotCat8EnvCfg(QuadbotRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        for f in (apply_cat, apply_cat2, apply_cat3, apply_cat4, apply_cat5, apply_cat6, apply_cat7, apply_cat8):
            f(self)


@configclass
class QuadbotCat8EnvCfg_PLAY(QuadbotRoughEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        for f in (apply_cat, apply_cat2, apply_cat3, apply_cat4, apply_cat5, apply_cat6, apply_cat7, apply_cat8):
            f(self)


def apply_cat9(cfg):
    """Cat v9: v8 + stronger vertical-bounce penalty (the last miss: bounce in sharp turns)."""
    cfg.rewards.lin_vel_z_l2.weight = -6.0


@configclass
class QuadbotCat9EnvCfg(QuadbotRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        for f in (apply_cat, apply_cat2, apply_cat3, apply_cat4, apply_cat5, apply_cat6, apply_cat7, apply_cat8, apply_cat9):
            f(self)


@configclass
class QuadbotCat9EnvCfg_PLAY(QuadbotRoughEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        for f in (apply_cat, apply_cat2, apply_cat3, apply_cat4, apply_cat5, apply_cat6, apply_cat7, apply_cat8, apply_cat9):
            f(self)


def apply_cat10(cfg):
    """Cat v10: v8's rewards, re-trained on the steep tiles (fine-tunes restart the terrain
    curriculum, and v8 never climbed back to the steep slopes)."""
    cfg.scene.terrain.max_init_terrain_level = 7


@configclass
class QuadbotCat10EnvCfg(QuadbotRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        for f in (apply_cat, apply_cat2, apply_cat3, apply_cat4, apply_cat5, apply_cat6, apply_cat7, apply_cat8, apply_cat10):
            f(self)


@configclass
class QuadbotCat10EnvCfg_PLAY(QuadbotRoughEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        for f in (apply_cat, apply_cat2, apply_cat3, apply_cat4, apply_cat5, apply_cat6, apply_cat7, apply_cat8, apply_cat10):
            f(self)


def apply_cat11(cfg):
    """Cat v11: v10 + linear command progress (no freezing on out-of-reach hills, no downhill drift)."""
    cfg.rewards.progress = RewTerm(func=mdp_extra.command_progress, weight=0.6)


@configclass
class QuadbotCat11EnvCfg(QuadbotRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        for f in (apply_cat, apply_cat2, apply_cat3, apply_cat4, apply_cat5, apply_cat6, apply_cat7, apply_cat8,
                  apply_cat10, apply_cat11):
            f(self)


@configclass
class QuadbotCat11EnvCfg_PLAY(QuadbotRoughEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        for f in (apply_cat, apply_cat2, apply_cat3, apply_cat4, apply_cat5, apply_cat6, apply_cat7, apply_cat8,
                  apply_cat10, apply_cat11):
            f(self)
