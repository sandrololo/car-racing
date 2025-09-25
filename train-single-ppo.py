from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig

from car_racing_env import CarRacingEnv


# Configure the algorithm.
config = (
    PPOConfig()
    .environment(
        CarRacingEnv,
        render_env=False,
    )
    .rl_module(
        model_config=DefaultModelConfig(
            conv_bias_initializer_kwargs={"dtype": "uint8"},
        ),
    )
    .env_runners(num_env_runners=1, sample_timeout_s=180)
    .evaluation(evaluation_interval=1, evaluation_num_env_runners=1)
)

# Train through Ray Tune.
results = tune.Tuner(
    "PPO",
    tune_config=tune.TuneConfig(
        reuse_actors=True,
    ),
    param_space=config,
    run_config=tune.RunConfig(stop={"training_iteration": 5}, verbose=1),
).fit()
