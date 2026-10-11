from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class QuadbotFlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 100
    experiment_name = "quadbot_flat"
    obs_groups = {"policy": ["policy"], "critic": ["critic"]}
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[256, 128, 128],
        critic_hidden_dims=[256, 128, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class QuadbotRoughPPORunnerCfg(QuadbotFlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 3000
        self.save_interval = 200
        self.experiment_name = "quadbot_rough"
        self.policy.actor_hidden_dims = [512, 256, 128]
        self.policy.critic_hidden_dims = [512, 256, 128]


@configclass
class QuadbotRoughROMPPORunnerCfg(QuadbotRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "quadbot_rough_rom"


@configclass
class QuadbotCatPPORunnerCfg(QuadbotRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "quadbot_cat"


@configclass
class QuadbotCat2PPORunnerCfg(QuadbotRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "quadbot_cat2"
        self.max_iterations = 1500


@configclass
class QuadbotCat3PPORunnerCfg(QuadbotRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "quadbot_cat3"
        self.max_iterations = 1500


@configclass
class QuadbotCat4PPORunnerCfg(QuadbotRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "quadbot_cat4"
        self.max_iterations = 1500


@configclass
class QuadbotCat5PPORunnerCfg(QuadbotRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "quadbot_cat5"
        self.max_iterations = 1500


@configclass
class QuadbotCat6PPORunnerCfg(QuadbotRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "quadbot_cat6"
        self.max_iterations = 1500


@configclass
class QuadbotCat7PPORunnerCfg(QuadbotRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "quadbot_cat7"
        self.max_iterations = 1500


@configclass
class QuadbotCat8PPORunnerCfg(QuadbotRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "quadbot_cat8"
        self.max_iterations = 1200


@configclass
class QuadbotCat9PPORunnerCfg(QuadbotRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "quadbot_cat9"
        self.max_iterations = 800


@configclass
class QuadbotCat10PPORunnerCfg(QuadbotRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "quadbot_cat10"
        self.max_iterations = 1500


@configclass
class QuadbotCat11PPORunnerCfg(QuadbotRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "quadbot_cat11"
        self.max_iterations = 1500
