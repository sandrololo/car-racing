from ray import tune
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from ray.air.integrations.wandb import WandbLoggerCallback

from car_racing_env import CarRacingEnv
from wandbvideocallback import WandbVideoCallback

TRAIN_BATCH_SIZE = 512
NUM_ITERATIONS = 20_000

# Configure the algorithm.
config = (
    PPOConfig()
    .environment(
        CarRacingEnv,
        env_config={
            "lap_complete_percent": 0.95,
            "gray_scale": True,
            "frame_stack": 4,
            "frame_skip": 4,
        },
        render_env=False,
    )
    .rl_module(
        model_config=DefaultModelConfig(
            conv_bias_initializer_kwargs={"dtype": "uint8"},
        ),
    )
    # don't use more than one num_envs_per_env_runner so that training happens more often
    .env_runners(
        num_env_runners=6, sample_timeout_s=1500
    )  # makes sense to have as many runners and therefore as much data as possible
    .learners(num_learners=1, num_gpus_per_learner=1)
    # only 1 runner and low interval for evaluation as we have new data every iteration anyways
    .training(
        gamma=0.99,
        use_critic=True,
        use_gae=True,
        train_batch_size=TRAIN_BATCH_SIZE,
        minibatch_size=64,
        shuffle_batch_per_epoch=True,
        lr=[[0, 0.0001], [TRAIN_BATCH_SIZE * NUM_ITERATIONS, 0.000001]],
        grad_clip=0.1,
        kl_coeff=0.2,
        grad_clip_by="norm",
        num_epochs=3,
    )
    .evaluation(
        evaluation_interval=1000,
        evaluation_num_env_runners=1,
        evaluation_sample_timeout_s=3000,
        evaluation_duration=5,
        evaluation_duration_unit="episodes",
        evaluation_config={
            "env_config": {
                "lap_complete_percent": 0.95,
                "max_timesteps": 4000,
                "gray_scale": True,
                "frame_stack": 4,
                "frame_skip": 4,
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
        stop={"training_iteration": NUM_ITERATIONS},
        verbose=1,
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
