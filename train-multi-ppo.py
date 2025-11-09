import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from ray.air.integrations.wandb import WandbLoggerCallback
import gymnasium

from environments import MultiAgentCarRacingEnv
from environments.multiagentwrappers import (
    RecordVideo,
    TimeLimit,
    GrayscaleObservation,
    FrameStackObservation,
)
from wandbvideocallback import MultiAgentWandbVideoCallback
import config as training_config


class WrappedEnv(gymnasium.Wrapper):
    def __init__(
        self,
        config: dict = None,
        *args,
        **kwargs,
    ):
        max_timesteps = config.get("max_timesteps", None)
        gray_scale = config.get("gray_scale", False)
        frame_stack = config.get("frame_stack", 1)
        record_video = config.get("record_video", False)
        self.env = MultiAgentCarRacingEnv(config, *args, **kwargs)
        if record_video:
            self.env = RecordVideo(
                self.env,
                video_folder="/tmp/multi-agent-videos",
                video_length=0,
                episode_trigger=lambda episode_id: episode_id
                % training_config.EVAL_DURATION
                == 0,
                name_prefix="car-racing-env",
            )
        if max_timesteps is not None:
            self.env = TimeLimit(self.env, max_timesteps)
        if gray_scale:
            self.env = GrayscaleObservation(self.env)
        if frame_stack > 1:
            self.env = FrameStackObservation(self.env, frame_stack)
        super().__init__(self.env)


ppo_config = (
    PPOConfig()
    .environment(
        WrappedEnv,
        env_config={
            "lap_complete_percent": 0.95,
            "num_cars": training_config.NUM_CARS,
            "max_timesteps": training_config.TRAIN_MAX_TIMESTEPS,
            "frame_skip": training_config.OBS_FRAME_SKIP,
            "record_video": False,
        },
        render_env=False,
    )
    .multi_agent(
        policies={"p0"},
        # All agents map to the exact same policy.
        policy_mapping_fn=(lambda aid, *args, **kwargs: "p0"),
    )
    .rl_module(
        model_config=DefaultModelConfig(
            conv_bias_initializer_kwargs={"dtype": "uint8"},
        ),
    )
    # don't use more than one num_envs_per_env_runner so that training happens more often
    .env_runners(
        num_env_runners=6, sample_timeout_s=1500, rollout_fragment_length=100
    )  # makes sense to have as many runners and therefore as much data as possible
    .learners(num_learners=1, num_gpus_per_learner=1)
    # only 1 runner and low interval for evaluation as we have new data every iteration anyways
    .training(
        gamma=training_config.TRAIN_GAMMA,
        use_critic=True,
        use_gae=True,
        train_batch_size=600,
        minibatch_size=training_config.MINI_BATCH_SIZE,
        shuffle_batch_per_epoch=True,
        lr=[
            [0, training_config.LR_SCHEDULE_START],
            [
                training_config.TRAIN_BATCH_SIZE * training_config.TRAIN_NUM_ITERATIONS,
                training_config.LR_SCHEDULE_END,
            ],
        ],
        num_epochs=2,
        clip_param=0.1,
    )
    .evaluation(
        evaluation_interval=10,
        evaluation_num_env_runners=1,
        evaluation_sample_timeout_s=3000,
        evaluation_duration=training_config.EVAL_DURATION,
        evaluation_duration_unit="episodes",
        evaluation_config={
            "env_config": {
                "lap_complete_percent": 0.95,
                "num_cars": training_config.NUM_CARS,
                "max_timesteps": training_config.TRAIN_MAX_TIMESTEPS,
                "frame_skip": training_config.OBS_FRAME_SKIP,
                "record_video": True,
                "render_mode": "rgb_array",
            },
        },
    )
    .callbacks(MultiAgentWandbVideoCallback)
)

ray.init()

results = tune.Tuner(
    "PPO",
    tune_config=tune.TuneConfig(
        reuse_actors=True,
    ),
    param_space=ppo_config,
    run_config=tune.RunConfig(
        stop={"training_iteration": 50},
        verbose=1,
        callbacks=[
            WandbLoggerCallback(
                group="car-racing",
                project="car-racing-multi-agent",
                log_config=True,
                upload_checkpoints=True,
            )
        ],
    ),
).fit()
