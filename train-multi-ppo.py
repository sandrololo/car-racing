import numpy as np
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.air.integrations.wandb import WandbLoggerCallback
from gymnasium.spaces import Box

from environments import MultiAgentCarRacingEnv
from environments.multiagent.multiagentwrappers import (
    MultiAgentEnvWrapper,
    RecordVideo,
    GrayscaleObservation,
    TimeLimit,
    NormalizeObservation,
    FrameStackObservation,
    NormalizeReward,
)
from wandbvideocallback import MultiAgentWandbVideoCallback
import multi_agent_config as training_config


class WrappedEnv(MultiAgentEnvWrapper):
    def __init__(
        self,
        config: dict = None,
        *args,
        **kwargs,
    ):
        max_timesteps = config.get("max_timesteps", None)
        record_video = config.get("record_video", False)
        normalize_rewards = config.get("normalize_rewards", False)
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
        self.env = GrayscaleObservation(self.env)
        self.env = FrameStackObservation(self.env, training_config.OBS_FRAME_STACK)
        if normalize_rewards is True:
            self.env = NormalizeReward(self.env)
        self.env = NormalizeObservation(self.env)
        super().__init__(self.env)


ppo_config = (
    PPOConfig()
    .environment(
        WrappedEnv,
        env_config={
            "lap_complete_percent": 0.95,
            "num_cars": training_config.NUM_CARS,
            "cars_configs": training_config.CAR_CONFIGS,
            "max_timesteps": training_config.TRAIN_MAX_TIMESTEPS,
            "normalize_rewards": training_config.NORMALIZE_REWARDS,
            "record_video": False,
        },
        render_env=False,
    )
    .multi_agent(
        policies={"p0"},
        # All agents map to the exact same policy.
        policy_mapping_fn=(lambda aid, *args, **kwargs: "p0"),
        count_steps_by="env_steps",
    )
    .rl_module(
        rl_module_spec=MultiRLModuleSpec(
            rl_module_specs={
                "p0": RLModuleSpec(
                    observation_space=Box(
                        low=0.0,
                        high=1.0,
                        shape=(96, 96, training_config.OBS_FRAME_STACK),
                        dtype=np.float32,
                    )
                )
            }
        ),
    )
    # don't use more than one num_envs_per_env_runner so that training happens more often
    .env_runners(
        num_env_runners=training_config.NUM_ENV_RUNNERS,
        sample_timeout_s=1500,
        rollout_fragment_length=training_config.ROLLOUT_FRAGMENT_LENGTH,
    )  # makes sense to have as many runners and therefore as much data as possible
    .learners(num_learners=1, num_gpus_per_learner=1)
    # only 1 runner and low interval for evaluation as we have new data every iteration anyways
    .training(
        gamma=training_config.TRAIN_GAMMA,
        use_critic=True,
        use_gae=True,
        train_batch_size=training_config.NUM_ENV_RUNNERS
        * training_config.ROLLOUT_FRAGMENT_LENGTH,
        minibatch_size=training_config.MINI_BATCH_SIZE,
        shuffle_batch_per_epoch=True,
        lr=[
            [0, training_config.LR_SCHEDULE_START],
            [
                training_config.NUM_ENV_RUNNERS
                * training_config.ROLLOUT_FRAGMENT_LENGTH
                * training_config.TRAIN_NUM_ITERATIONS,
                training_config.LR_SCHEDULE_END,
            ],
        ],
        num_epochs=training_config.TRAIN_NUM_EPOCHS,
        clip_param=0.1,
    )
    .evaluation(
        evaluation_interval=training_config.EVAL_INTERVAL,
        evaluation_num_env_runners=1,
        evaluation_sample_timeout_s=3000,
        evaluation_duration=training_config.EVAL_DURATION,
        evaluation_duration_unit="episodes",
        evaluation_config={
            "env_config": {
                "lap_complete_percent": 0.95,
                "num_cars": training_config.NUM_CARS,
                "cars_configs": training_config.CAR_CONFIGS,
                "max_timesteps": training_config.EVAL_MAX_TIMESTEPS,
                "normalize_rewards": False,
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
        stop={"training_iteration": training_config.TRAIN_NUM_ITERATIONS},
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
