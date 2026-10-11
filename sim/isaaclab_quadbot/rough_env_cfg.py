"""Hills, up/down slopes, side slopes and bumps, with a terrain curriculum and a speed curriculum.
The policy stays blind (IMU + joints, like the real robot); the critic also gets a height scan."""
import math

import isaaclab.terrains as terrain_gen
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

from . import mdp_extra
from .flat_env_cfg import QuadbotObservationsCfg
from .quadbot_cfg import QUADBOT_CFG

# tiles sized for a 15 cm tall robot; difficulty rows 0..9
QUADBOT_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(4.0, 4.0),
    border_width=10.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.05,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=True,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.1),
        "hills": terrain_gen.HfWaveTerrainCfg(proportion=0.25, amplitude_range=(0.01, 0.12), num_waves=2,
                                             border_width=0.25),
        "slope_up": terrain_gen.HfPyramidSlopedTerrainCfg(proportion=0.2, slope_range=(0.0, 0.45),
                                                        platform_width=0.8, border_width=0.25),
        "slope_down": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(proportion=0.2, slope_range=(0.0, 0.45),
                                                                  platform_width=0.8, border_width=0.25),
        "bumps": terrain_gen.HfRandomUniformTerrainCfg(proportion=0.25, noise_range=(0.005, 0.03),
                                                      noise_step=0.005, downsampled_scale=0.1, border_width=0.25),
    },
)


@configclass
class QuadbotRoughObservationsCfg(QuadbotObservationsCfg):
    @configclass
    class CriticCfg(QuadbotObservationsCfg.CriticCfg):
        height_scan = ObsTerm(func=mdp.height_scan, params={"sensor_cfg": SceneEntityCfg("height_scanner")},
                              clip=(-1.0, 1.0))

    critic: CriticCfg = CriticCfg()


@configclass
class QuadbotRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    observations: QuadbotRoughObservationsCfg = QuadbotRoughObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = QUADBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.terrain.terrain_generator = QUADBOT_TERRAINS_CFG
        self.scene.terrain.max_init_terrain_level = 3
        self.scene.height_scanner = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_link",
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
            ray_alignment="yaw",
            pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=[0.6, 0.4]),
            debug_vis=False,
            mesh_prim_paths=["/World/ground"],
        )
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        self.observations.policy.height_scan = None      # blind policy
        self.observations.policy.base_lin_vel = None

        # commands: start where the flat policy was, the curriculum widens them
        self.actions.joint_pos.scale = 0.25
        cmd = self.commands.base_velocity.ranges
        cmd.lin_vel_x = (-0.3, 0.4)
        cmd.lin_vel_y = (-0.2, 0.2)
        cmd.ang_vel_z = (-1.5, 1.5)
        self.curriculum.speed_levels = CurrTerm(func=mdp_extra.command_speed_levels)

        # events
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
            "pose_range": {"x": (-0.4, 0.4), "y": (-0.4, 0.4), "yaw": (-3.14, 3.14)},
            "velocity_range": {k: (0.0, 0.0) for k in ("x", "y", "z", "roll", "pitch", "yaw")},
        }
        self.events.push_robot.params["velocity_range"] = {"x": (-0.3, 0.3), "y": (-0.3, 0.3)}

        # rewards
        self.rewards.track_lin_vel_xy_exp.weight = 1.5
        self.rewards.track_lin_vel_xy_exp.params["std"] = 0.25
        self.rewards.track_ang_vel_z_exp.weight = 0.75
        self.rewards.track_ang_vel_z_exp.params["std"] = math.sqrt(0.2)
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = ".*_foot"
        self.rewards.feet_air_time.params["threshold"] = 0.2
        self.rewards.feet_air_time.weight = 0.5
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = ".*_thigh"
        self.rewards.flat_orientation_l2.weight = 0.0     # the body must tilt on slopes
        self.rewards.dof_torques_l2.weight = -2.0e-4
        self.rewards.dof_acc_l2.weight = -2.5e-7
        self.rewards.dof_pos_limits.weight = -0.5
        self.rewards.electrical_power = RewTerm(func=mdp_extra.servo_electrical_power, weight=-0.01)
        self.rewards.pose_deviation = RewTerm(func=mdp.joint_deviation_l1, weight=-0.03)  # keep legs under the body
        self.rewards.feet_slide = RewTerm(
            func=mdp.feet_slide, weight=-0.1,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
                    "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot")})

        # terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = "base_link"


@configclass
class QuadbotRoughEnvCfg_PLAY(QuadbotRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.terrain.max_init_terrain_level = None
        self.scene.terrain.terrain_generator.num_rows = 5
        self.scene.terrain.terrain_generator.num_cols = 5
        self.scene.terrain.terrain_generator.curriculum = False
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.curriculum.speed_levels = None


@configclass
class QuadbotRoughROMEnvCfg(QuadbotRoughEnvCfg):
    """Same as QuadbotRoughEnvCfg + a range-of-motion reward; the stand-pose penalty is kept on
    hip roll only (anti-sprawl) so hip pitch and knee are free to swing."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.pose_deviation = RewTerm(
            func=mdp.joint_deviation_l1, weight=-0.1,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_roll"])})
        self.rewards.range_of_motion = RewTerm(
            func=mdp_extra.joint_range_of_motion, weight=0.4,
            params={"targets": {".*_hip_pitch": 0.6, ".*_knee": 0.8}, "window_s": 0.6})


@configclass
class QuadbotRoughROMEnvCfg_PLAY(QuadbotRoughROMEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.terrain.max_init_terrain_level = None
        self.scene.terrain.terrain_generator.num_rows = 5
        self.scene.terrain.terrain_generator.num_cols = 5
        self.scene.terrain.terrain_generator.curriculum = False
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.curriculum.speed_levels = None
