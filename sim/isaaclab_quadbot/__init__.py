import gymnasium as gym

from . import agents

gym.register(
    id="Isaac-Velocity-Flat-Quadbot-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:QuadbotFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:QuadbotFlatPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-Quadbot-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:QuadbotFlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:QuadbotFlatPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Quadbot-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:QuadbotRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:QuadbotRoughPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Quadbot-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:QuadbotRoughEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:QuadbotRoughPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Quadbot-ROM-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:QuadbotRoughROMEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:QuadbotRoughROMPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Quadbot-ROM-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:QuadbotRoughROMEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:QuadbotRoughROMPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Cat-Quadbot-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cat_env_cfg:QuadbotCatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:QuadbotCatPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Cat-Quadbot-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cat_env_cfg:QuadbotCatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:QuadbotCatPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Cat2-Quadbot-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cat_env_cfg:QuadbotCat2EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:QuadbotCat2PPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Cat2-Quadbot-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cat_env_cfg:QuadbotCat2EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:QuadbotCat2PPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Cat3-Quadbot-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cat_env_cfg:QuadbotCat3EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:QuadbotCat3PPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Cat3-Quadbot-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cat_env_cfg:QuadbotCat3EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:QuadbotCat3PPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Cat4-Quadbot-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cat_env_cfg:QuadbotCat4EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:QuadbotCat4PPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Cat4-Quadbot-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cat_env_cfg:QuadbotCat4EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:QuadbotCat4PPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Cat5-Quadbot-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cat_env_cfg:QuadbotCat5EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:QuadbotCat5PPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Cat5-Quadbot-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cat_env_cfg:QuadbotCat5EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:QuadbotCat5PPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Cat6-Quadbot-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cat_env_cfg:QuadbotCat6EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:QuadbotCat6PPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Cat6-Quadbot-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cat_env_cfg:QuadbotCat6EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:QuadbotCat6PPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Cat7-Quadbot-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cat_env_cfg:QuadbotCat7EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:QuadbotCat7PPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Cat7-Quadbot-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cat_env_cfg:QuadbotCat7EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:QuadbotCat7PPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Cat8-Quadbot-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cat_env_cfg:QuadbotCat8EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:QuadbotCat8PPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Cat8-Quadbot-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cat_env_cfg:QuadbotCat8EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:QuadbotCat8PPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Cat9-Quadbot-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cat_env_cfg:QuadbotCat9EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:QuadbotCat9PPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Cat9-Quadbot-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cat_env_cfg:QuadbotCat9EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:QuadbotCat9PPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Cat10-Quadbot-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cat_env_cfg:QuadbotCat10EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:QuadbotCat10PPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Cat10-Quadbot-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cat_env_cfg:QuadbotCat10EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:QuadbotCat10PPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Cat11-Quadbot-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cat_env_cfg:QuadbotCat11EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:QuadbotCat11PPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Cat11-Quadbot-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cat_env_cfg:QuadbotCat11EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:QuadbotCat11PPORunnerCfg",
    },
)
