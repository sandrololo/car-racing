import os
from ray import tune
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig

from car_racing_env import CarRacingEnv

NUM_OF_STEPS = 3000

# Configure the algorithm.
config = (
    PPOConfig()
    .environment(
        CarRacingEnv,
        env_config={
            "total_episode_steps": NUM_OF_STEPS,
            "lap_complete_percent": 0.95
        },
        render_env=False,
    )
    .rl_module(
        model_config=DefaultModelConfig(
            conv_bias_initializer_kwargs={"dtype": "uint8"},
        ),
    )
    .env_runners(num_env_runners=3, sample_timeout_s=1000)
    .learners(num_learners=1, num_gpus_per_learner=1)
    .evaluation(evaluation_interval=1, evaluation_num_env_runners=3, evaluation_sample_timeout_s=1000, evaluation_duration=5, evaluation_duration_unit="episodes")
)

ray.init()

# Train through Ray Tune.
results = tune.Tuner(
    "PPO",
    tune_config=tune.TuneConfig(
        reuse_actors=True,
    ),
    param_space=config,
    run_config=tune.RunConfig(stop={"training_iteration": 50}, verbose=1, storage_path=os.path.join(os.getcwd(), "results/single-agent")),
).fit()
