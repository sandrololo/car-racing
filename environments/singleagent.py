import numpy as np
from gymnasium.envs.box2d.car_racing import CarRacing, FPS
from gymnasium.spaces import Box
from gymnasium import wrappers
import gymnasium


import config as training_config


class SingleAgentCarRacingEnv(gymnasium.Wrapper):
    def __init__(
        self,
        config: dict = None,
        *args,
        **kwargs,
    ):
        if config:
            lap_complete_percent = config.get("lap_complete_percent", 0.95)
            render_mode = config.get("render_mode", None)
            max_timesteps = config.get("max_timesteps", None)
            gray_scale = config.get("gray_scale", False)
            frame_stack = config.get("frame_stack", 1)
            frame_skip = config.get("frame_skip", 1)
            record_video = config.get("record_video", False)
        else:
            lap_complete_percent = 0.95
            render_mode = None
            max_timesteps = None
            gray_scale = False
            frame_stack = 1
            frame_skip = 1
            record_video = False
        self.env = CarRacing(
            render_mode=render_mode,
            lap_complete_percent=lap_complete_percent,
            continuous=True,
            *args,
            **kwargs,
        )
        if record_video:
            self.env = wrappers.RecordVideo(
                self.env,
                video_folder="/tmp/videos",
                video_length=0,
                episode_trigger=lambda episode_id: episode_id
                % training_config.EVAL_DURATION
                == 0,
                name_prefix="car-racing-env",
            )
        if max_timesteps is not None:
            self.env = wrappers.TimeLimit(self.env, max_timesteps)
        if gray_scale:
            self.env = wrappers.GrayscaleObservation(self.env)
        if frame_stack > 1:
            self.env = wrappers.FrameStackObservation(self.env, frame_stack)
        if frame_skip > 1:
            self.env = wrappers.MaxAndSkipObservation(self.env, frame_skip)
        super().__init__(self.env)
        # Convert observation space to float32 and normalized
        self.observation_space = Box(
            low=0.0, high=1.0, shape=self.env.observation_space.shape, dtype=np.float32
        )
        self.env.t = FPS

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.env.t = FPS
        return self._preprocess_obs(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._preprocess_obs(obs), reward, terminated, truncated, info

    def _preprocess_obs(self, obs):
        """Convert uint8 observations to normalized float32."""
        return obs.astype(np.float32) / 255.0
