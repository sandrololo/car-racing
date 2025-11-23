import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from ray.air.integrations.wandb import WandbLoggerCallback

from environments import SingleAgentCarRacingEnv
from wandbvideocallback import SingleAgentWandbVideoCallback
import single_agent_config


# Configure the algorithm.
ppo_config = (
    PPOConfig()
    .environment(
        SingleAgentCarRacingEnv,
        env_config={
            "lap_complete_percent": 0.95,
            "gray_scale": single_agent_config.OBS_GRAY_SCALE,
            "frame_stack": single_agent_config.OBS_FRAME_STACK,
            "frame_skip": single_agent_config.OBS_FRAME_SKIP,
            "normalize_rewards": single_agent_config.NORMALIZE_REWARDS,
            "max_timesteps": single_agent_config.TRAIN_MAX_TIMESTEPS,
            "max_timesteps_per_episode_increase": single_agent_config.TRAIN_MAX_TIMESTEPS_PER_EPISODE_INCREASE,
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
        num_env_runners=single_agent_config.NUM_ENV_RUNNERS,
        sample_timeout_s=1500,
        rollout_fragment_length=single_agent_config.ROLLOUT_FRAGMENT_LENGTH,
    )  # makes sense to have as many runners and therefore as much data as possible
    .learners(num_learners=1, num_gpus_per_learner=1)
    # only 1 runner and low interval for evaluation as we have new data every iteration anyways
    .training(
        gamma=single_agent_config.TRAIN_GAMMA,
        use_critic=True,
        use_gae=True,
        lambda_=0.95,
        train_batch_size=single_agent_config.NUM_ENV_RUNNERS
        * single_agent_config.ROLLOUT_FRAGMENT_LENGTH,
        minibatch_size=single_agent_config.MINI_BATCH_SIZE,
        shuffle_batch_per_epoch=True,
        lr=[
            [0, single_agent_config.LR_SCHEDULE_START],
            [
                single_agent_config.NUM_ENV_RUNNERS
                * single_agent_config.ROLLOUT_FRAGMENT_LENGTH
                * single_agent_config.TRAIN_NUM_ITERATIONS,
                single_agent_config.LR_SCHEDULE_END,
            ],
        ],
        num_epochs=single_agent_config.TRAIN_NUM_EPOCHS,
        clip_param=single_agent_config.TRAIN_CLIP_PARAM,
        kl_coeff=0.1,
    )
    .evaluation(
        evaluation_interval=single_agent_config.EVAL_INTERVAL,
        evaluation_num_env_runners=1,
        evaluation_sample_timeout_s=3000,
        evaluation_duration=single_agent_config.EVAL_DURATION,
        evaluation_duration_unit="episodes",
        evaluation_config={
            "env_config": {
                "lap_complete_percent": 0.95,
                "max_timesteps": single_agent_config.EVAL_MAX_TIMESTEPS,
                "gray_scale": single_agent_config.OBS_GRAY_SCALE,
                "frame_stack": single_agent_config.OBS_FRAME_STACK,
                "frame_skip": single_agent_config.OBS_FRAME_SKIP,
                "normalize_rewards": False,
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
        stop={"training_iteration": single_agent_config.TRAIN_NUM_ITERATIONS},
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
