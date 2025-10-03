import os
from ray import tune
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from ray.air.integrations.wandb import WandbLoggerCallback, setup_wandb

from car_racing_env import CarRacingEnv

NUM_OF_STEPS = 3000

# Configure the algorithm.
config = (
    PPOConfig()
    .environment(
        CarRacingEnv,
        env_config={"total_episode_steps": NUM_OF_STEPS, "lap_complete_percent": 0.95},
        render_env=False,
    )
    .rl_module(
        model_config=DefaultModelConfig(
            conv_bias_initializer_kwargs={"dtype": "uint8"},
        ),
    )
    .env_runners(
        num_env_runners=6, sample_timeout_s=1000
    )  # makes sense to have as many runners and therefore as possible
    .learners(num_learners=1, num_gpus_per_learner=1)
    # only 1 runner and low interval for evaluation as we have new data every iteration anyways
    .training(
        use_critic=True,
        use_gae=True,
        train_batch_size=500,
        lr=0.0003,
        num_epochs=5,
    )
    .evaluation(
        evaluation_interval=5,
        evaluation_num_env_runners=1,
        evaluation_sample_timeout_s=1000,
        evaluation_duration=5,
        evaluation_duration_unit="episodes",
    )
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
        stop={"training_iteration": 20},
        verbose=1,
        storage_path=os.path.join(os.getcwd(), "results/single-agent"),
        callbacks=[
            WandbLoggerCallback(
                project="car-racing-single-agent",
                log_config=True,
                upload_checkpoints=True
            )
        ]
    ),
).fit()
