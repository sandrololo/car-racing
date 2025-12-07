import gymnasium
from ray.rllib.callbacks.callbacks import RLlibCallback
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.env.env_runner import EnvRunner
from ray.rllib.utils.metrics import EPISODE_RETURN_MEAN, ENV_RUNNER_RESULTS


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
        **kwargs,
    ) -> None:
        algorithm._counters["num_cars"] = 1

    def on_evaluate_end(
        self,
        *,
        algorithm: Algorithm,
        metrics_logger=None,
        evaluation_metrics: dict,
        **kwargs,
    ) -> None:
        if ENV_RUNNER_RESULTS not in evaluation_metrics:
            gymnasium.logger.warn(
                "Curriculum callback: No env runner results found in evaluation metrics!"
            )
            return
        mean_return = evaluation_metrics[ENV_RUNNER_RESULTS][EPISODE_RETURN_MEAN]
        gymnasium.logger.warn(f"Curriculum callback: eval reward mean = {mean_return}")
        current = algorithm._counters.get("num_cars", 1)
        if mean_return >= 150 and current <= 4:
            update(algorithm, 4)
            algorithm._counters["num_cars"] = 4
        if mean_return >= 100 and current <= 3:
            update(algorithm, 3)
            algorithm._counters["num_cars"] = 3
        if mean_return >= 50 and current <= 2:
            update(algorithm, 2)
            algorithm._counters["num_cars"] = 2
