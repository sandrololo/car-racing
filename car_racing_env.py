from gymnasium.envs.box2d.car_racing import CarRacing
from gymnasium.spaces import Box
import gymnasium
import numpy as np


class CarRacingEnv(gymnasium.Wrapper):
    def __init__(self, config: dict = None, *args, **kwargs):
        if config:
            lap_complete_percent = config.get("lap_complete_percent", 0.95)
            render_mode = config.get("render_mode", None)
            max_timesteps = config.get("max_timesteps", None)
        else:
            lap_complete_percent = 0.95
            render_mode = None
            max_timesteps = None
        self.env = CarRacing(
            render_mode=render_mode,
            lap_complete_percent=lap_complete_percent,
            continuous=True,
            *args,
            **kwargs
        )
        super().__init__(self.env)
        # Convert observation space to float32 and normalized
        self.observation_space = Box(
            low=0.0, high=1.0, shape=self.env.observation_space.shape, dtype=np.float32
        )
        self.max_timesteps = max_timesteps
        self.current_step = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.current_step = 0
        return self._preprocess_obs(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.current_step += 1
        if self.max_timesteps is not None and self.current_step >= self.max_timesteps:
            truncated = True
        return self._preprocess_obs(obs), reward, terminated, truncated, info

    def _preprocess_obs(self, obs):
        """Convert uint8 observations to normalized float32."""
        return obs.astype(np.float32) / 255.0
