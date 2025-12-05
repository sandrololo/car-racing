from typing import Any, SupportsFloat
from copy import deepcopy
import gymnasium as gym
from gymnasium.core import ActType, ObsType, RenderFrame, WrapperObsType
from gymnasium.envs.registration import EnvSpec


class IncreasingTimeLimit(
    gym.Wrapper[ObsType, ActType, ObsType, ActType], gym.utils.RecordConstructorArgs
):
    """Limits the number of steps for an environment through truncating the environment if a maximum number of timesteps is exceeded.
    The maximum number of timesteps increases by a fixed amount after each episode.

    If a truncation is not defined inside the environment itself, this is the only place that the truncation signal is issued.
    Critically, this is different from the `terminated` signal that originates from the underlying environment as part of the MDP.
    No vector wrapper exists.


    """

    def __init__(
        self,
        env: gym.Env,
        max_episode_steps_start: int,
        max_episode_steps_increase: float,
    ):
        """Initializes the :class:`TimeLimit` wrapper with an environment and the number of maximum episode steps, as well as the increase in maximum episode steps after each episode.

        Args:
            env: The environment to apply the wrapper
            max_episode_steps_start: the initial number of steps after which the episode is truncated (``elapsed >= max_episode_steps``)
            max_episode_steps_increase: the amount by which the maximum episode steps increase after each episode
        """
        assert (
            isinstance(max_episode_steps_start, int) and max_episode_steps_start > 0
        ), f"Expect the `max_episode_steps_start` to be positive, actually: {max_episode_steps_start}"
        assert (
            isinstance(max_episode_steps_increase, (int, float))
            and max_episode_steps_increase > 0
        ), f"Expect the `max_episode_steps_increase` to be positive, actually: {max_episode_steps_increase}"
        gym.utils.RecordConstructorArgs.__init__(
            self,
            max_episode_steps_start=max_episode_steps_start,
            max_episode_steps_increase=max_episode_steps_increase,
        )
        gym.Wrapper.__init__(self, env)

        self._max_episode_steps_start = max_episode_steps_start
        self._max_episode_steps_increase = max_episode_steps_increase
        self._elapsed_steps = None
        self._elapsed_episodes = 0

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = self.env.step(action)
        self._elapsed_steps += 1

        if (
            self._elapsed_steps
            >= self._max_episode_steps_start
            + self._max_episode_steps_increase * self._elapsed_episodes
        ):
            truncated = True

        return observation, reward, terminated, truncated, info

    def reset(
        self, *, seed: int = None, options: dict[str, Any] = None
    ) -> tuple[ObsType, dict[str, Any]]:
        self._elapsed_steps = 0
        self._elapsed_episodes += 1
        return super().reset(seed=seed, options=options)

    @property
    def spec(self) -> EnvSpec:
        """Modifies the environment spec to include the `max_episode_steps_start=self._max_episode_steps_start` and `max_episode_steps_increase=self._max_episode_steps_increase`."""
        if self._cached_spec is not None:
            return self._cached_spec

        env_spec = self.env.spec
        if env_spec is not None:
            try:
                env_spec = deepcopy(env_spec)
                env_spec.max_episode_steps_start = self._max_episode_steps_start
                env_spec.max_episode_steps_increase = self._max_episode_steps_increase
            except Exception as e:
                gym.logger.warn(
                    f"An exception occurred ({e}) while copying the environment spec={env_spec}"
                )
                return None

        self._cached_spec = env_spec
        return env_spec
