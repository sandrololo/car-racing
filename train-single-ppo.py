from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
import logging

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
    .env_runners(num_env_runners=1)
    .evaluation(evaluation_interval=1, evaluation_num_env_runners=1)
)


# Build the algorithm.
algo = config.build_algo()

# Train it for 5 iterations ...
for _ in range(5):
    logging.info(algo.train())

# ... and evaluate it.
logging.info(algo.evaluate())

# Release the algo's resources (remote actors, like EnvRunners and Learners).
algo.stop()
