import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig

from environments import MultiAgentCarRacingEnv
import config

ppo_config = (
    PPOConfig()
    .environment(
        MultiAgentCarRacingEnv,
        env_config={"lap_complete_percent": 0.95, "num_agents": config.NUM_CARS},
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
        num_env_runners=6, sample_timeout_s=1500
    )  # makes sense to have as many runners and therefore as much data as possible
    .learners(num_learners=1, num_gpus_per_learner=1)
    # only 1 runner and low interval for evaluation as we have new data every iteration anyways
    .training(
        gamma=config.TRAIN_GAMMA,
        use_critic=True,
        use_gae=True,
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
        num_epochs=2,
        clip_param=0.1,
    )
    .evaluation(
        evaluation_interval=1000,
        evaluation_num_env_runners=1,
        evaluation_sample_timeout_s=3000,
        evaluation_duration=config.EVAL_DURATION,
        evaluation_duration_unit="episodes",
        evaluation_config={
            "env_config": {"lap_complete_percent": 0.95, "num_agents": config.NUM_CARS},
        },
    )
)
print(ppo_config.is_multi_agent)

ray.init()

results = tune.Tuner(
    "PPO",
    tune_config=tune.TuneConfig(
        reuse_actors=True,
    ),
    param_space=ppo_config,
    run_config=tune.RunConfig(
        stop={"training_iteration": config.TRAIN_NUM_ITERATIONS},
        verbose=1,
    ),
).fit()
