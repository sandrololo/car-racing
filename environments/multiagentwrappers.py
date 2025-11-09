from collections import deque
from typing import Any, Final, SupportsFloat
from copy import deepcopy
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.wrappers import TransformObservation, RecordVideo as GymRecordVideo
from gymnasium.core import ActType, ObsType, WrapperObsType, WrapperActType
from gymnasium.envs.registration import EnvSpec
from gymnasium.vector.utils import batch_space, concatenate, create_empty_array
from gymnasium.wrappers.utils import create_zero_array


class RecordVideo(GymRecordVideo):
    """Records videos of environment episodes using the environment's render function."""

    def _capture_frame(self):
        assert self.recording, "Cannot capture a frame, recording wasn't started."
        frame = self.env._render("video")
        assert isinstance(frame, np.ndarray), (
            "Expected the type of frame returned by render to be a numpy array, "
            f"got instead {type(frame)}."
        )
        self.recorded_frames.append(frame)


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
        self, *, seed: int = None, options: dict[str, Any] = None
    ) -> tuple[ObsType, dict]:
        self._elapsed_steps = 0
        return super().reset(seed=seed, options=options)

    @property
    def spec(self) -> EnvSpec:
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


class FrameStackObservation(
    gym.Wrapper[WrapperObsType, ActType, ObsType, ActType],
    gym.utils.RecordConstructorArgs,
):
    """Stacks the observations from the last ``N`` time steps in a rolling manner.

    For example, if the number of stacks is 4, then the returned observation contains
    the most recent 4 observations. For environment 'Pendulum-v1', the original observation
    is an array with shape [3], so if we stack 4 observations, the processed observation
    has shape [4, 3].
    """

    def __init__(self, env: gym.Env[ObsType, ActType], stack_size: int):
        """Observation wrapper that stacks the observations in a rolling manner.

        Args:
            env: The environment to apply the wrapper
            stack_size: The number of frames to stack.
        """
        gym.utils.RecordConstructorArgs.__init__(self, stack_size=stack_size)
        gym.Wrapper.__init__(self, env)

        if not np.issubdtype(type(stack_size), np.integer):
            raise TypeError(
                f"The stack_size is expected to be an integer, actual type: {type(stack_size)}"
            )
        if not 0 < stack_size:
            raise ValueError(
                f"The stack_size needs to be greater than zero, actual value: {stack_size}"
            )
        self.padding_value: ObsType = {
            key: create_zero_array(value)
            for key, value in env.observation_space.spaces.items()
        }

        self.observation_spaces = {
            key: spaces.Box(
                low=0, high=255, shape=(stack_size, *value.shape), dtype=np.uint8
            )
            for key, value in env.observation_space.spaces.items()
        }
        self.stack_size: Final[int] = stack_size

        self.obs_queue = {
            key: deque(
                [self.padding_value[key] for _ in range(self.stack_size)],
                maxlen=self.stack_size,
            )
            for key in env.observation_space.spaces.keys()
        }
        self.stacked_obs = {
            key: create_empty_array(
                batch_space(
                    env.observation_space.spaces[key],
                    (self.stack_size,),
                )
            )
            for key in env.observation_space.spaces.keys()
        }

    def step(
        self, action: WrapperActType
    ) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Steps through the environment, appending the observation to the frame buffer.

        Args:
            action: The action to step through the environment with

        Returns:
            Stacked observations, reward, terminated, truncated, and info from the environment
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        for key in obs.keys():
            self.obs_queue[key].append(obs[key])

        updated_obs = {
            key: deepcopy(
                concatenate(
                    self.env.observation_space.spaces[key],
                    self.obs_queue[key],
                    self.stacked_obs[key],
                )
            )
            for key in self.env.observation_space.spaces.keys()
        }
        return updated_obs, reward, terminated, truncated, info

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[WrapperObsType, dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)

        self.padding_value = obs
        for _ in range(self.stack_size - 1):
            for key in obs.keys():
                self.obs_queue[key].append(self.padding_value[key])
        for key in obs.keys():
            self.obs_queue[key].append(obs[key])

        updated_obs = {
            key: deepcopy(
                concatenate(
                    self.env.observation_space.spaces[key],
                    self.obs_queue[key],
                    self.stacked_obs[key],
                )
            )
            for key in self.env.observation_space.spaces.keys()
        }
        return updated_obs, info
