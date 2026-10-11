import math

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
    ObservationsCfg,
)

from . import mdp_extra
from .quadbot_cfg import QUADBOT_CFG


@configclass
class QuadbotObservationsCfg(ObservationsCfg):
    """Policy sees only what the robot has (F3 IMU gyro + gravity, servo positions/speeds).
    The critic additionally gets the true base linear velocity."""

    @configclass
    class CriticCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    critic: CriticCfg = CriticCfg()


@configclass
class QuadbotFlatEnvCfg(LocomotionVelocityRoughEnvCfg):
    observations: QuadbotObservationsCfg = QuadbotObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = QUADBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # flat ground, no height scan
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.observations.policy.base_lin_vel = None  # not measurable on the real robot
        self.curriculum.terrain_levels = None

        # small robot: small actions, small commands
        self.actions.joint_pos.scale = 0.25
        cmd = self.commands.base_velocity.ranges
        cmd.lin_vel_x = (-0.3, 0.4)
        cmd.lin_vel_y = (-0.2, 0.2)
        cmd.ang_vel_z = (-1.0, 1.0)

        # events
        self.events.push_robot = None
        self.events.physics_material.params["static_friction_range"] = (0.5, 1.1)
        self.events.physics_material.params["dynamic_friction_range"] = (0.4, 0.9)
        self.events.add_base_mass.params["mass_distribution_params"] = (-0.15, 0.35)
        self.events.add_base_mass.params["asset_cfg"].body_names = "base_link"
        self.events.base_external_force_torque.params["asset_cfg"].body_names = "base_link"
        if self.events.base_com is not None:
            self.events.base_com.params["asset_cfg"].body_names = "base_link"
            self.events.base_com.params["com_range"] = {"x": (-0.015, 0.015), "y": (-0.01, 0.01), "z": (-0.01, 0.01)}
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {k: (0.0, 0.0) for k in ("x", "y", "z", "roll", "pitch", "yaw")},
        }

        # rewards (tracking std scaled to the slow speeds)
        self.rewards.track_lin_vel_xy_exp.weight = 1.5
        self.rewards.track_lin_vel_xy_exp.params["std"] = 0.2
        self.rewards.track_ang_vel_z_exp.weight = 0.75
        self.rewards.track_ang_vel_z_exp.params["std"] = math.sqrt(0.15)
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = ".*_foot"
        self.rewards.feet_air_time.params["threshold"] = 0.25
        self.rewards.feet_air_time.weight = 0.25
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = ".*_thigh"
        self.rewards.flat_orientation_l2.weight = -2.5
        self.rewards.dof_torques_l2.weight = -2.0e-4
        self.rewards.dof_acc_l2.weight = -2.5e-7
        self.rewards.dof_pos_limits.weight = -0.5
        self.rewards.joint_power = RewTerm(func=mdp_extra.joint_power_l1, weight=-0.01)

        # terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = "base_link"


@configclass
class QuadbotFlatEnvCfg_PLAY(QuadbotFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 0.8
        # camera follows robot 0 (the default view is sized for a Go1-scale robot)
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"
        self.viewer.env_index = 0
        self.viewer.eye = (0.55, 0.75, 0.35)
        self.viewer.lookat = (0.0, 0.0, 0.05)
        self.viewer.resolution = (1280, 720)
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
