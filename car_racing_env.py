from gymnasium.envs.box2d.car_racing import CarRacing
from gymnasium.spaces import Box
import gymnasium
import numpy as np


class CarRacingEnv(gymnasium.Wrapper):
    def __init__(self, *args, **kwargs):
        self.env = CarRacing(*args, **kwargs)
        super().__init__(self.env)
        # Convert observation space to float32 and normalized
        self.observation_space = Box(
            low=0.0, high=1.0, shape=self.env.observation_space.shape, dtype=np.float32
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._preprocess_obs(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._preprocess_obs(obs), reward, terminated, truncated, info

    def _preprocess_obs(self, obs):
        """Convert uint8 observations to normalized float32."""
        return obs.astype(np.float32) / 255.0
