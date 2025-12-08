from typing import Optional
import gymnasium
from ray.rllib.callbacks.callbacks import RLlibCallback
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.env.env_runner import EnvRunner
from ray.rllib.utils.metrics.metrics_logger import MetricsLogger
from ray.rllib.utils.metrics import EPISODE_RETURN_MEAN, ENV_RUNNER_RESULTS
from .cars import CarConfig


def _get_wrapped_base_envs(env):
    envs = []
    current = env
    if env is not None:
        if hasattr(env, "envs") and type(env.envs) == list:
            for e in env.envs:
                base_envs = _get_wrapped_base_envs(e)
                envs.extend(base_envs)
        else:
            while hasattr(current, "env"):
                current = current.env
            envs.append(current)
    return envs


def _set_num_cars(env_runner: EnvRunner, num_cars: int):
    if env_runner.env is not None:
        base_envs = _get_wrapped_base_envs(env_runner.env)
        if len(base_envs):
            for env in base_envs:
                if hasattr(env, "set_num_cars"):
                    env.set_num_cars(num_cars)
                    print(f"Setting num_cars to {num_cars} in env {env}")
                else:
                    gymnasium.logger.warn("Base env has no method 'set_num_cars'!")
        else:
            gymnasium.logger.warn("No base envs found!")


def update(algorithm: Algorithm, num_cars: int):
    algorithm.config.environment(
        env_config={
            "num_cars": num_cars,
        }
    )
    algorithm.env_runner_group.foreach_env_runner(
        lambda env_runner: _set_num_cars(env_runner, num_cars)
    )
    algorithm.eval_env_runner_group.foreach_env_runner(
        lambda env_runner: _set_num_cars(env_runner, num_cars)
    )


class Curriculum(RLlibCallback):
    def on_algorithm_init(
        self,
        *,
        algorithm: "Algorithm",
        metrics_logger: Optional[MetricsLogger] = None,
        **kwargs,
    ) -> None:
        assert hasattr(
            algorithm.config, "env_config"
        ), "Curriculum callback requires 'env_config' in algorithm config!"
        assert "curriculum_config" in algorithm.config.get(
            "env_config", {}
        ), "Curriculum callback requires 'curriculum_config' in env_config!"
        config: CurriculumConfig = algorithm.config.get("env_config", {})[
            "curriculum_config"
        ]
        algorithm._counters["num_cars"] = config.num_cars_start
        if metrics_logger is not None:
            metrics_logger.log_value("curriculum/num_cars", config.num_cars_start)

    def on_evaluate_end(
        self,
        *,
        algorithm: Algorithm,
        metrics_logger: Optional[MetricsLogger] = None,
        evaluation_metrics: dict,
        **kwargs,
    ) -> None:
        config: CurriculumConfig = algorithm.config.get("env_config", {})[
            "curriculum_config"
        ]
        if ENV_RUNNER_RESULTS not in evaluation_metrics:
            gymnasium.logger.warn(
                "Curriculum callback: No env runner results found in evaluation metrics!"
            )
            return
        mean_return = evaluation_metrics[ENV_RUNNER_RESULTS][EPISODE_RETURN_MEAN]
        print(f"Curriculum callback: eval reward mean = {mean_return}")
        current = algorithm._counters.get("num_cars", config.num_cars_start)
        for entry in config.entries:
            if mean_return >= entry.min_reward and current < entry.num_cars:
                update(algorithm, entry.num_cars)
                algorithm._counters["num_cars"] = entry.num_cars
                if metrics_logger is not None:
                    metrics_logger.log_value("curriculum/num_cars", entry.num_cars)


class CurriculumStep:
    def __init__(self, num_cars: int, min_reward: float):
        self.num_cars = num_cars
        self.min_reward = min_reward


class CurriculumConfig:
    def __init__(
        self,
        car_configs: list[CarConfig],
        num_cars_start: int,
        entries: list[CurriculumStep],
    ):
        assert all(entry.num_cars < len(car_configs) for entry in entries)
        assert all(
            entries[i].min_reward < entries[i + 1].min_reward
            for i in range(len(entries) - 1)
        )
        assert all(
            entries[i].num_cars < entries[i + 1].num_cars
            for i in range(len(entries) - 1)
        )
        self.num_cars_start = num_cars_start
        self.entries = sorted(entries, key=lambda entry: entry.num_cars, reverse=True)
