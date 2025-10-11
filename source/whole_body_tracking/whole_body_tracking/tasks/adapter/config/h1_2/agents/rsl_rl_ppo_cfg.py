from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)
from dataclasses import MISSING
from typing import Literal


@configclass
class AdapterRslRlPpoActorCriticCfg(RslRlPpoActorCriticCfg):
    """Configuration for the Adapter PPO actor-critic networks."""

    class_name: str = "ActorCriticAdapter"
    """The policy class name. Default is ActorCritic."""

    history_encoder_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the history encoder network."""

    world_model_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the world model network."""

    estimator_future_frame: int = MISSING
    frozen_actor_path: str = ""


@configclass
class H1_2FlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 30000
    save_interval = 500
    experiment_name = "h1_2_flat_adapter"
    empirical_normalization = True
    policy = AdapterRslRlPpoActorCriticCfg(
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        history_encoder_hidden_dims=[512, 256, 128],
        # history_encoder_hidden_dims=[16384, 4096, 1024],
        world_model_hidden_dims=[128, 256, 512],
        # world_model_hidden_dims=[1024, 4096, 8192],
        estimator_future_frame=20,
        activation="elu",
        init_noise_std=0.8,
        frozen_actor_path="/home/hpx/HPX_LOCO_2/whole_body_tracking"
        + "/logs/rsl_rl/temp/exported/policy.pt",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="PPO_Adapter",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
