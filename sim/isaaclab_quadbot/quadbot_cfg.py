"""Quadbot articulation: 12x Feetech STS3215-12V, URDF from ~/quadbot/cad/export_urdf.py."""
import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg
from isaaclab.assets.articulation import ArticulationCfg

URDF_PATH = os.environ.get("QUADBOT_URDF", os.path.expanduser("~/quadbot_sim/quadbot_description/urdf/quadbot_isaac.urdf"))

# stand pose from the CAD (left legs; right legs are mirrored)
HIP_ROLL, HIP_PITCH, KNEE = 0.0, -0.8604, 1.6197

# STS3215-12V: 30 kg.cm stall (2.9 N.m), 0.222 s/60deg no-load (4.7 rad/s), ~345:1 gearbox
STS3215_CFG = DCMotorCfg(
    joint_names_expr=[".*_hip_roll", ".*_hip_pitch", ".*_knee"],
    saturation_effort=2.9,
    effort_limit=2.5,
    velocity_limit=4.7,
    stiffness=12.0,
    damping=0.25,
    armature=0.005,
    friction=0.02,
)

QUADBOT_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=URDF_PATH,
        fix_base=False,
        merge_fixed_joints=False,  # keep the *_foot links for contact sensing
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0)
        ),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.17),
        joint_pos={
            "L._hip_roll": HIP_ROLL,
            "R._hip_roll": -HIP_ROLL,
            "L._hip_pitch": HIP_PITCH,
            "R._hip_pitch": -HIP_PITCH,
            "L._knee": KNEE,
            "R._knee": -KNEE,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={"legs": STS3215_CFG},
)
