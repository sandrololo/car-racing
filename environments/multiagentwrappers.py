import os
from typing import Any, Callable, List
from copy import deepcopy
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.wrappers import TransformObservation, RecordVideo as GymRecordVideo
from gymnasium.core import ActType, ObsType, RenderFrame, WrapperObsType
from gymnasium.envs.registration import EnvSpec


class TimeLimit(
    gym.Wrapper[ObsType, ActType, ObsType, ActType], gym.utils.RecordConstructorArgs
):
    """Limits the number of steps for an environment through truncating the environment if a maximum number of timesteps is exceeded."""

    def __init__(
        self,
        env: gym.Env,
        max_episode_steps: int,
    ):
        assert (
            isinstance(max_episode_steps, int) and max_episode_steps > 0
        ), f"Expect the `max_episode_steps` to be positive, actually: {max_episode_steps}"
        gym.utils.RecordConstructorArgs.__init__(
            self, max_episode_steps=max_episode_steps
        )
        gym.Wrapper.__init__(self, env)

        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = None

    def step(self, action: ActType) -> tuple[ObsType, dict, dict, dict, dict]:
        observation, reward, terminated, truncated, info = self.env.step(action)
        self._elapsed_steps += 1

        if self._elapsed_steps >= self._max_episode_steps:
            for key, value in terminated.items():
                truncated[key] = True
            truncated["__all__"] = True

        return observation, reward, terminated, truncated, info

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[ObsType, dict]:
        self._elapsed_steps = 0
        return super().reset(seed=seed, options=options)

    @property
    def spec(self) -> EnvSpec | None:
        """Modifies the environment spec to include the `max_episode_steps=self._max_episode_steps`."""
        if self._cached_spec is not None:
            return self._cached_spec

        env_spec = self.env.spec
        if env_spec is not None:
            try:
                env_spec = deepcopy(env_spec)
                env_spec.max_episode_steps = self._max_episode_steps
            except Exception as e:
                gym.logger.warn(
                    f"An exception occurred ({e}) while copying the environment spec={env_spec}"
                )
                return None

        self._cached_spec = env_spec
        return env_spec


def create_grayscale_observation(obs: dict) -> dict:
    gray_obs = {}
    for key, value in obs.items():
        gray_obs[key] = np.sum(
            np.multiply(value, np.array([0.2125, 0.7154, 0.0721])), axis=-1
        ).astype(np.uint8)
    return gray_obs


class GrayscaleObservation(
    TransformObservation[WrapperObsType, ActType, ObsType],
    gym.utils.RecordConstructorArgs,
):
    """Converts an image observation computed by ``reset`` and ``step`` from RGB to Grayscale."""

    def __init__(self, env: gym.Env[ObsType, ActType]):
        for key, value in env.observation_space.spaces.items():
            assert isinstance(value, spaces.Box)
            assert len(value.shape) == 3 and value.shape[-1] == 3
            assert (
                np.all(value.low == 0)
                and np.all(value.high == 255)
                and value.dtype == np.uint8
            )

        gym.utils.RecordConstructorArgs.__init__(self)

        self.observation_spaces = {
            key: spaces.Box(low=0, high=255, shape=value.shape[:2], dtype=np.uint8)
            for key, value in env.observation_space.spaces.items()
        }

        new_observation_space = spaces.Box(
            low=0, high=255, shape=env.observation_space.shape[:2], dtype=np.uint8
        )
        TransformObservation.__init__(
            self,
            env=env,
            func=create_grayscale_observation,
            observation_space=new_observation_space,
        )
