from typing import Union
import numpy as np
from gymnasium.envs.box2d.car_racing import (
    CarRacing,
    WINDOW_W,
    WINDOW_H,
    VIDEO_W,
    VIDEO_H,
    STATE_W,
    STATE_H,
    SCALE,
    ZOOM,
    PLAYFIELD,
    FPS,
)
from gymnasium.spaces import Box
from gymnasium import wrappers
from gymnasium.error import DependencyNotInstalled
import gymnasium

try:
    # As pygame is necessary for using the environment (reset and step) even without a render mode
    #   therefore, pygame is a necessary import for the environment.
    import pygame
except ImportError as e:
    raise DependencyNotInstalled(
        'pygame is not installed, run `pip install "gymnasium[box2d]"`'
    ) from e

import config as training_config


def _preprocess_obs(obs):
    """Convert uint8 observations to normalized float32."""
    return obs.astype(np.float32) / 255.0


class CustomCarRacingEnv(CarRacing):
    def __init__(
        self,
        render_mode=None,
        lap_complete_percent=0.95,
    ):
        super().__init__(
            render_mode=render_mode,
            lap_complete_percent=lap_complete_percent,
            continuous=True,
        )

    def step(self, action: Union[np.ndarray, int]):
        assert self.car is not None
        if action is not None:
            action = action.astype(np.float64)
            self.car.steer(-action[0])
            self.car.gas(action[1])
            self.car.brake(action[2])

        self.car.step(1.0 / FPS)
        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)
        self.t += 1.0 / FPS

        self.state = self._render("state_pixels")

        step_reward = 0
        terminated = False
        truncated = False
        info = {}
        if action is not None:  # First step without action, called from reset()
            self.reward -= training_config.REWARD_MINUS_PER_STEP
            # We actually don't want to count fuel spent, we want car to be faster.
            # self.reward -=  10 * self.car.fuel_spent / ENGINE_POWER
            self.car.fuel_spent = 0.0
            step_reward = self.reward - self.prev_reward
            self.prev_reward = self.reward
            if self.tile_visited_count == len(self.track) or self.new_lap:
                # Termination due to finishing lap
                terminated = True
                info["lap_finished"] = True
            x, y = self.car.hull.position
            if abs(x) > PLAYFIELD or abs(y) > PLAYFIELD:
                terminated = True
                info["lap_finished"] = False
                step_reward = -100

        if self.render_mode == "human":
            self.render()
        return self.state, step_reward, terminated, truncated, info

    def _render(self, mode: str):
        """
        Almost completely copied from gymnasium.envs.box2d.car_racing.CarRacing, except that zooming animation at the beginning is removed.
        """
        assert mode in self.metadata["render_modes"]

        pygame.font.init()
        if self.screen is None and mode == "human":
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        if "t" not in self.__dict__:
            return  # reset() not called yet

        self.surf = pygame.Surface((WINDOW_W, WINDOW_H))

        assert self.car is not None
        # computing transformations
        angle = -self.car.hull.angle
        # Animating first second zoom.
        zoom = ZOOM * SCALE
        scroll_x = -(self.car.hull.position[0]) * zoom
        scroll_y = -(self.car.hull.position[1]) * zoom
        trans = pygame.math.Vector2((scroll_x, scroll_y)).rotate_rad(angle)
        trans = (WINDOW_W / 2 + trans[0], WINDOW_H / 4 + trans[1])

        self._render_road(zoom, trans, angle)
        self.car.draw(
            self.surf,
            zoom,
            trans,
            angle,
            mode not in ["state_pixels_list", "state_pixels"],
        )

        self.surf = pygame.transform.flip(self.surf, False, True)

        # showing stats
        self._render_indicators(WINDOW_W, WINDOW_H)

        font = pygame.font.Font(pygame.font.get_default_font(), 42)
        text = font.render("%04i" % self.reward, True, (255, 255, 255), (0, 0, 0))
        text_rect = text.get_rect()
        text_rect.center = (60, WINDOW_H - WINDOW_H * 2.5 / 40.0)
        self.surf.blit(text, text_rect)

        if mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            assert self.screen is not None
            self.screen.fill(0)
            self.screen.blit(self.surf, (0, 0))
            pygame.display.flip()
        elif mode == "rgb_array":
            return self._create_image_array(self.surf, (VIDEO_W, VIDEO_H))
        elif mode == "state_pixels":
            return self._create_image_array(self.surf, (STATE_W, STATE_H))
        else:
            return self.isopen


class SingleAgentCarRacingEnv(gymnasium.Wrapper):
    def __init__(
        self,
        config: dict = None,
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
        self.env = CustomCarRacingEnv(render_mode, lap_complete_percent)
        if record_video:
            self.env = wrappers.RecordVideo(
                self.env,
                video_folder="/tmp/single-agent-videos",
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
        self.env = wrappers.NormalizeReward(self.env)
        self.env = wrappers.TransformObservation(
            self.env,
            _preprocess_obs,
            Box(
                low=0.0,
                high=1.0,
                shape=self.env.observation_space.shape,
                dtype=np.float32,
            ),
        )
        super().__init__(self.env)
