from collections import deque
from typing import Any, Final, SupportsFloat, Callable
from copy import deepcopy
import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils.typing import MultiAgentDict
import gymnasium as gym
from gymnasium import spaces
from gymnasium.core import ActType, ObsType, WrapperObsType, WrapperActType
from gymnasium.envs.registration import EnvSpec
from gymnasium.wrappers import RecordVideo as GymRecordVideo
from gymnasium.wrappers.utils import RunningMeanStd


class MultiAgentEnvWrapper(gym.Wrapper):
    def __init__(self, env: MultiAgentEnv):
        super().__init__(env)
        self.observation_spaces = env.observation_spaces
        self.action_spaces = env.action_spaces


class TransformObservation(MultiAgentEnvWrapper):
    """Applies a transformation function to the observations returned by ``reset`` and ``step``."""

    def __init__(
        self,
        env: MultiAgentEnv,
        transform_function: Callable,
        new_observation_space: dict,
    ):
        super().__init__(env)
        self.transform_function = transform_function
        self.observation_spaces = new_observation_space

    def step(
        self, action: WrapperActType
    ) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = super().step(action)
        transformed_obs = self.transform_function(obs)
        return transformed_obs, reward, terminated, truncated, info

    def reset(
        self, *, seed: int = None, options: dict[str, Any] = None
    ) -> tuple[WrapperObsType, dict[str, Any]]:
        obs, info = super().reset(seed=seed, options=options)
        transformed_obs = self.transform_function(obs)
        return transformed_obs, info


class RecordVideo(MultiAgentEnvWrapper):
    """Records videos of environment episodes using the environment's render function."""

    def __init__(
        self,
        env: MultiAgentEnv,
        video_folder: str,
        episode_trigger=None,
        step_trigger=None,
        video_length: int = 0,
        name_prefix: str = "rl-video",
        fps: int = None,
        disable_logger: bool = True,
    ):
        super().__init__(env)
        self.record_video = GymRecordVideo(
            env,
            video_folder,
            episode_trigger=episode_trigger,
            step_trigger=step_trigger,
            video_length=video_length,
            name_prefix=name_prefix,
            fps=fps,
            disable_logger=disable_logger,
        )

    def _capture_frame(self):
        assert (
            self.record_video.recording
        ), "Cannot capture a frame, recording wasn't started."
        frame = self.env._render("video")
        assert isinstance(frame, np.ndarray), (
            "Expected the type of frame returned by render to be a numpy array, "
            f"got instead {type(frame)}."
        )
        self.record_video.recorded_frames.append(frame)

    def reset(
        self, *, seed: int = None, options: dict[str, Any] = None
    ) -> tuple[ObsType, dict[str, Any]]:
        """Reset the environment and eventually starts a new recording."""
        obs, info = super().reset(seed=seed, options=options)
        self.record_video.episode_id += 1

        if self.record_video.recording and self.record_video.video_length == float(
            "inf"
        ):
            self.stop_recording()

        if self.record_video.episode_trigger and self.record_video.episode_trigger(
            self.record_video.episode_id
        ):
            self.start_recording(
                f"{self.record_video.name_prefix}-episode-{self.record_video.episode_id}"
            )
        if self.record_video.recording:
            self._capture_frame()
            if len(self.record_video.recorded_frames) > self.record_video.video_length:
                self.stop_recording()

        return obs, info

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Steps through the environment using action, recording observations if :attr:`self.recording`."""
        obs, rew, terminated, truncated, info = self.env.step(action)
        self.record_video.step_id += 1

        if self.record_video.step_trigger and self.record_video.step_trigger(
            self.record_video.step_id
        ):
            self.start_recording(
                f"{self.record_video.name_prefix}-step-{self.record_video.step_id}"
            )
        if self.record_video.recording:
            self._capture_frame()

            if len(self.record_video.recorded_frames) > self.record_video.video_length:
                self.stop_recording()

        return obs, rew, terminated, truncated, info

    def render(self):
        return self.env.render()

    def close(self):
        self.record_video.close()

    def start_recording(self, video_name: str):
        self.record_video.start_recording(video_name)

    def stop_recording(self):
        self.record_video.stop_recording()

    def __del__(self):
        self.record_video.__del__()


class IncreasingTimeLimit(MultiAgentEnvWrapper, gym.utils.RecordConstructorArgs):
    """Limits the number of steps for an environment through truncating the environment if a maximum number of timesteps is exceeded."""

    def __init__(
        self,
        env: MultiAgentEnv,
        max_episode_steps_start: int,
        max_episode_steps_increase: float,
    ):
        assert (
            isinstance(max_episode_steps_start, int) and max_episode_steps_start > 0
        ), f"Expect the `max_episode_steps_start` to be positive, actually: {max_episode_steps_start}"
        assert (
            isinstance(max_episode_steps_increase, (int, float))
            and max_episode_steps_increase > 0
        ), f"Expect the `max_episode_steps_increase` to be positive, actually: {max_episode_steps_increase}"
        gym.utils.RecordConstructorArgs.__init__(
            self,
            max_episode_steps=max_episode_steps_start,
            max_episode_steps_increase=max_episode_steps_increase,
        )
        super().__init__(env)

        self._max_episode_steps_start = max_episode_steps_start
        self._max_episode_steps_increase = max_episode_steps_increase
        self._elapsed_steps = None
        self._elapsed_episodes = 0

    def step(self, action: ActType) -> tuple[ObsType, dict, dict, dict, dict]:
        observation, reward, terminated, truncated, info = self.env.step(action)
        self._elapsed_steps += 1

        if (
            self._elapsed_steps
            >= self._max_episode_steps_start
            + self._max_episode_steps_increase * self._elapsed_episodes
        ):
            for key in observation.keys():
                truncated[key] = True
            truncated["__all__"] = True

        return observation, reward, terminated, truncated, info

    def reset(
        self, *, seed: int = None, options: dict[str, Any] = None
    ) -> tuple[ObsType, dict[str, Any]]:
        self._elapsed_steps = 0
        self._elapsed_episodes += 1
        return super().reset(seed=seed, options=options)

    @property
    def spec(self) -> EnvSpec:
        """Modifies the environment spec to include the `max_episode_steps_start` and `max_episode_steps_increase`."""
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


def _preprocess_obs(obs: MultiAgentDict) -> MultiAgentDict:
    """Convert uint8 observations to normalized float32."""
    normalized_obs = {}
    for key, value in obs.items():
        normalized_obs[key] = value.astype(np.float32) / 255.0
    return normalized_obs


class NormalizeObservation(TransformObservation, gym.utils.RecordConstructorArgs):
    """Normalizes observations to the range [0.0, 1.0]."""

    def __init__(self, env: MultiAgentEnv):
        for value in env.observation_spaces.values():
            assert isinstance(value, spaces.Box)
            assert np.issubdtype(value.dtype, np.integer)
            assert np.all(value.low == 0)
            assert np.all(value.high == 255)

        gym.utils.RecordConstructorArgs.__init__(self)
        new_observation_spaces = {
            key: spaces.Box(
                low=0.0,
                high=1.0,
                shape=value.shape,
                dtype=np.float32,
            )
            for key, value in env.observation_spaces.items()
        }
        super().__init__(env, _preprocess_obs, new_observation_spaces)


def _create_grayscale_observation(obs: MultiAgentDict) -> MultiAgentDict:
    gray_obs = {}
    for key, value in obs.items():
        gray_obs[key] = np.sum(
            np.multiply(value, np.array([0.2125, 0.7154, 0.0721])), axis=-1
        ).astype(np.uint8)
        gray_obs[key] = np.expand_dims(gray_obs[key], axis=-1)
    return gray_obs


class GrayscaleObservation(TransformObservation, gym.utils.RecordConstructorArgs):
    """Converts an image observation computed by ``reset`` and ``step`` from RGB to Grayscale."""

    def __init__(self, env: MultiAgentEnv):
        for value in env.observation_spaces.values():
            assert isinstance(value, spaces.Box)
            assert len(value.shape) == 3 and value.shape[-1] == 3
            assert (
                np.all(value.low == 0)
                and np.all(value.high == 255)
                and value.dtype == np.uint8
            )

        gym.utils.RecordConstructorArgs.__init__(self)
        new_observation_spaces = {
            key: spaces.Box(
                low=0, high=255, shape=value.shape[:2] + (1,), dtype=np.uint8
            )
            for key, value in env.observation_spaces.items()
        }
        super().__init__(env, _create_grayscale_observation, new_observation_spaces)


class FrameStackObservation(MultiAgentEnvWrapper, gym.utils.RecordConstructorArgs):
    """Stacks the observations from the last ``N`` time steps in a rolling manner.

    For example, if the number of stacks is 4, then the returned observation contains
    the most recent 4 observations. For environment 'Pendulum-v1', the original observation
    is an array with shape [3], so if we stack 4 observations, the processed observation
    has shape [4, 3].
    """

    def __init__(self, env: MultiAgentEnv, stack_size: int):
        """Observation wrapper that stacks the observations in a rolling manner.

        Args:
            env: The environment to apply the wrapper
            stack_size: The number of frames to stack.
        """
        gym.utils.RecordConstructorArgs.__init__(self, stack_size=stack_size)
        super().__init__(env)

        if not np.issubdtype(type(stack_size), np.integer):
            raise TypeError(
                f"The stack_size is expected to be an integer, actual type: {type(stack_size)}"
            )
        if not 0 < stack_size:
            raise ValueError(
                f"The stack_size needs to be greater than zero, actual value: {stack_size}"
            )
        self.observation_spaces = {
            key: spaces.Box(
                low=value.low.min(),
                high=value.high.max(),
                shape=(
                    (*value.shape[:-1], stack_size)
                    if len(value.shape) >= 3
                    else (*value.shape, stack_size)
                ),
                dtype=value.dtype,
            )
            for key, value in env.observation_spaces.items()
        }
        self.stack_size: Final[int] = stack_size
        self.padding_value = {
            key: np.zeros(value.shape, value.dtype).squeeze()
            for key, value in env.observation_spaces.items()
        }
        self.obs_queue = {
            key: deque(
                [self.padding_value[key] for _ in range(self.stack_size)],
                maxlen=self.stack_size,
            )
            for key in env.observation_spaces.keys()
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
        for key, value in obs.items():
            self.obs_queue[key].append(value.squeeze())

        updated_obs = {
            key: deepcopy(np.stack(value, axis=-1))
            for key, value in self.obs_queue.items()
        }
        return updated_obs, reward, terminated, truncated, info

    def reset(
        self, *, seed: int = None, options: dict[str, Any] = None
    ) -> tuple[WrapperObsType, dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)

        for _ in range(self.stack_size - 1):
            for key, value in self.padding_value.items():
                self.obs_queue[key].append(value.squeeze())
        for key, value in obs.items():
            self.obs_queue[key].append(value.squeeze())

        updated_obs = {
            key: deepcopy(np.stack(value, axis=-1))
            for key, value in self.obs_queue.items()
        }
        return updated_obs, info


class NormalizeReward(MultiAgentEnvWrapper, gym.utils.RecordConstructorArgs):
    r"""Normalizes immediate rewards such that their exponential moving average has an approximately fixed variance.
    The reward is avaraged over all agents, assuming they share their policy.

    The property `_update_running_mean` allows to freeze/continue the running mean calculation of the reward
    statistics. If `True` (default), the `RunningMeanStd` will get updated every time `self.normalize()` is called.
    If False, the calculated statistics are used but not updated anymore; this may be used during evaluation.

    Note:
        The scaling depends on past trajectories and rewards will not be scaled correctly if the wrapper was newly
        instantiated or the policy was changed recently.
    """

    def __init__(
        self,
        env: MultiAgentEnv,
        gamma: float = 0.99,
        epsilon: float = 1e-8,
    ):
        """This wrapper will normalize immediate rewards s.t. their exponential moving average has an approximately fixed variance.

        Args:
            env (env): The environment to apply the wrapper
            epsilon (float): A stability parameter
            gamma (float): The discount factor that is used in the exponential moving average.
        """
        gym.utils.RecordConstructorArgs.__init__(self, gamma=gamma, epsilon=epsilon)
        super().__init__(env)

        self.return_rms = RunningMeanStd(shape=())
        self.discounted_reward: np.array = np.array([0.0])
        self.gamma = gamma
        self.epsilon = epsilon
        self._update_running_mean = True

    @property
    def update_running_mean(self) -> bool:
        """Property to freeze/continue the running mean calculation of the reward statistics."""
        return self._update_running_mean

    @update_running_mean.setter
    def update_running_mean(self, setting: bool):
        """Sets the property to freeze/continue the running mean calculation of the reward statistics."""
        self._update_running_mean = setting

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Steps through the environment, normalizing the reward returned."""
        obs, reward_dict, terminated, truncated, info = super().step(action)
        keys = reward_dict.keys()

        # Using the `discounted_reward` rather than `reward` makes no sense but for backward compatibility, it is being kept
        for key in keys:
            self.discounted_reward = self.discounted_reward * self.gamma * (
                1 - terminated[key]
            ) + float(reward_dict[key])
            if self._update_running_mean:
                self.return_rms.update(self.discounted_reward)

        # We don't (reward - self.return_rms.mean) see https://github.com/openai/baselines/issues/538
        normalized_rewards = {
            key: reward_dict[key] / np.sqrt(self.return_rms.var + self.epsilon)
            for key in keys
        }
        return obs, normalized_rewards, terminated, truncated, info
