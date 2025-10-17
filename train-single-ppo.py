import os
from ray import tune
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from ray.air.integrations.wandb import WandbLoggerCallback

from car_racing_env import CarRacingEnv
from wandbvideocallback import WandbVideoCallback

# Configure the algorithm.
config = (
    PPOConfig()
    .environment(
        CarRacingEnv,
        env_config={"lap_complete_percent": 0.95},
        render_env=False,
    )
    .rl_module(
        model_config=DefaultModelConfig(
            conv_bias_initializer_kwargs={"dtype": "uint8"},
        ),
    )
    .env_runners(
        num_env_runners=6, num_envs_per_env_runner=16, sample_timeout_s=1500
    )  # makes sense to have as many runners and therefore as much data as possible
    .learners(num_learners=1, num_gpus_per_learner=1)
    # only 1 runner and low interval for evaluation as we have new data every iteration anyways
    .training(
        gamma=0.95,
        use_critic=True,
        use_gae=True,
        train_batch_size=256,
        shuffle_batch_per_epoch=True,
        lr=0.0001,
        grad_clip=0.1,
        kl_coeff=0.1,
        grad_clip_by="norm",
        num_epochs=5,
    )
    .evaluation(
        evaluation_interval=5,
        evaluation_num_env_runners=1,
        evaluation_sample_timeout_s=3000,
        evaluation_duration=5,
        evaluation_duration_unit="episodes",
        evaluation_config={
            "env_config": {
                "lap_complete_percent": 0.95,
                "max_timesteps": 5000,
            }
        },
    )
    .callbacks(WandbVideoCallback)
)

ray.init()

# Train through Ray Tune.
results = tune.Tuner(
    "PPO",
    tune_config=tune.TuneConfig(
        reuse_actors=True,
    ),
    param_space=config,
    run_config=tune.RunConfig(
        stop={"training_iteration": 100},
        verbose=1,
        storage_path=os.path.join(os.getcwd(), "results/single-agent"),
        callbacks=[
            WandbLoggerCallback(
                group="car-racing",
                project="car-racing-single-agent",
                log_config=True,
                upload_checkpoints=True,
            )
        ],
    ),
).fit()
