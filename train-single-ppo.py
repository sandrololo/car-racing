import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from ray.air.integrations.wandb import WandbLoggerCallback

from environments import SingleAgentCarRacingEnv
from wandbvideocallback import SingleAgentWandbVideoCallback
import config


# Configure the algorithm.
ppo_config = (
    PPOConfig()
    .environment(
        SingleAgentCarRacingEnv,
        env_config={
            "lap_complete_percent": 0.95,
            "gray_scale": config.OBS_GRAY_SCALE,
            "frame_stack": config.OBS_FRAME_STACK,
            "frame_skip": config.OBS_FRAME_SKIP,
            "max_timesteps": config.TRAIN_MAX_TIMESTEPS,
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
        gamma=config.TRAIN_GAMMA,
        use_critic=True,
        use_gae=True,
        lambda_=0.95,
        train_batch_size=config.TRAIN_BATCH_SIZE,
        minibatch_size=config.MINI_BATCH_SIZE,
        shuffle_batch_per_epoch=True,
        lr=[
            [0, config.LR_SCHEDULE_START],
            [
                config.TRAIN_BATCH_SIZE * config.TRAIN_NUM_ITERATIONS,
                config.LR_SCHEDULE_END,
            ],
        ],
        num_epochs=config.TRAIN_NUM_EPOCHS,
        clip_param=config.TRAIN_CLIP_PARAM,
    )
    .evaluation(
        evaluation_interval=config.EVAL_INTERVAL,
        evaluation_num_env_runners=1,
        evaluation_sample_timeout_s=3000,
        evaluation_duration=config.EVAL_DURATION,
        evaluation_duration_unit="episodes",
        evaluation_config={
            "env_config": {
                "lap_complete_percent": 0.95,
                "max_timesteps": config.EVAL_MAX_TIMESTEPS,
                "gray_scale": config.OBS_GRAY_SCALE,
                "frame_stack": config.OBS_FRAME_STACK,
                "frame_skip": config.OBS_FRAME_SKIP,
                "render_mode": "rgb_array",
                "record_video": True,
            }
        },
    )
    .callbacks(SingleAgentWandbVideoCallback)
)

ray.init()

# Train through Ray Tune.
results = tune.Tuner(
    "PPO",
    tune_config=tune.TuneConfig(
        reuse_actors=True,
    ),
    param_space=ppo_config,
    run_config=tune.RunConfig(
        stop={"training_iteration": config.TRAIN_NUM_ITERATIONS},
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
