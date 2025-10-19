from typing import Optional, Union
import Box2D
from gymnasium.envs.box2d.car_racing import CarRacing, FrictionDetector
from gymnasium.envs.box2d.car_dynamics import Car
from gymnasium.error import InvalidAction
from gymnasium.spaces import Box
import gymnasium
from gymnasium import spaces
import pygame
import numpy as np

STATE_W = 96  # less than Atari 160x192
STATE_H = 96
WINDOW_W = 1000
WINDOW_H = 800

SCALE = 6.0  # Track scale
TRACK_RAD = 900 / SCALE  # Track is heavily morphed circle with this radius
PLAYFIELD = 2000 / SCALE  # Game over boundary
FPS = 50  # Frames per second
ZOOM = 2.7  # Camera zoom


class CarRacingEnv(gymnasium.Wrapper):
    def __init__(
        self,
        config: dict = None,
        *args,
        **kwargs,
    ):
        self.render_mode = render_mode
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
            **kwargs,
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


class CarInfo:
    def __init__(self, world, track):
        # TODO: initialize at different positions for multiple cars
        self.car = Car(world, *track[0][1:4])
        self.reward = 0.0
        self.prev_reward = 0.0
        self.tile_visited_count = 0
        self.fuel_spent = 0.0

    @property
    def position(self) -> tuple[float, float]:
        return self.car.hull.position

    @property
    def angle(self) -> float:
        return self.car.hull.angle

    def apply_action(self, action: np.ndarray):
        action = action.astype(np.float64)
        self.car.steer(-action[0])
        self.car.gas(action[1])
        self.car.brake(action[2])

    def step(self, dt: float):
        self.car.step(dt)

    def draw(
        self,
        surf: pygame.Surface,
        zoom: float,
        trans: tuple[float, float],
        draw_particles: bool = True,
    ):
        self.car.draw(surf, zoom, trans, draw_particles)


class MultiAgentCarRacingEnv(gymnasium.Wrapper):
    def __init__(self, config: dict = None, *args, **kwargs):
        if config:
            self.num_agents = config.get("num_agents", 4)
            lap_complete_percent = config.get("lap_complete_percent", 0.95)
            self.render_mode = config.get("render_mode", None)
            max_timesteps = config.get("max_timesteps", None)
        else:
            self.num_agents = 4
            lap_complete_percent = 0.95
            self.render_mode = None
            max_timesteps = None

        self.cars: list[CarInfo] = [CarInfo(None, None) for _ in range(self.num_agents)]

        self.road_color = np.array([102, 102, 102])
        self.bg_color = np.array([102, 204, 102])
        self.grass_color = np.array([102, 230, 102])

        self.world = Box2D.b2World((0, 0), contactListener=self.contactListener_keepref)
        self.screen: Optional[pygame.Surface] = None
        self.surf: list[pygame.Surface] = []
        self.clock: Optional[pygame.time.Clock] = None

        self.observation_space = spaces.Tuple(
            spaces.Box(low=0, high=255, shape=(STATE_H, STATE_W, 3), dtype=np.uint8)
            for _ in range(self.num_agents)
        )
        # do nothing, left, right, gas, brake
        self.action_space = spaces.Tuple(
            spaces.Discrete(5) for _ in range(self.num_agents)
        )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        self._destroy()
        self.world.contactListener_bug_workaround = FrictionDetector(
            self, self.lap_complete_percent
        )
        self.world.contactListener = self.world.contactListener_bug_workaround
        self.t = 0.0
        self.new_lap = False
        self.road_poly = []

        while True:
            success = self._create_track()
            if success:
                break
            if self.verbose:
                print(
                    "retry to generate track (normal if there are not many"
                    "instances of this message)"
                )
        for car in self.cars:
            car.reset(self.world, self.track)

        if self.render_mode == "human":
            self.render()
        return self.step(None)[0], {}

    def step(self, action: Union[list[np.ndarray], None]):
        assert self.car is not None
        if action is not None:
            for car, action in zip(self.cars, action):
                car.apply_action(action)

        for car in self.cars:
            car.step(1.0 / FPS)
        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)
        self.t += 1.0 / FPS

        self.state = self._render("state_pixels")

        step_rewards = [0.0 for _ in range(self.num_agents)]
        terminated = False
        truncated = False
        info = {"cars": [{} for _ in range(self.num_agents)]}
        if action is not None:  # First step without action, called from reset()
            for i, car in enumerate(self.cars):
                car.reward -= 0.1
                # We actually don't want to count fuel spent, we want car to be faster.
                # self.reward -=  10 * self.car.fuel_spent / ENGINE_POWER
                car.fuel_spent = 0.0
                step_rewards[i] = car.reward - car.prev_reward
                car.prev_reward = car.reward
                if car.tile_visited_count == len(self.track) or self.new_lap:
                    # Termination due to finishing lap
                    terminated = True
                    info["cars"][i]["lap_finished"] = True
                x, y = car.position
                if abs(x) > PLAYFIELD or abs(y) > PLAYFIELD:
                    terminated = True
                    info["cars"][i]["lap_finished"] = False
                    step_rewards[i] = -100

        if self.render_mode == "human":
            self.render()
        return self.state, step_rewards, terminated, truncated, info

    def _render(self, mode: str):
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

        self.surf = [
            pygame.Surface((WINDOW_W, WINDOW_H)) for _ in range(self.num_agents)
        ]

        assert len(self.cars) > 0
        for i, car in enumerate(self.cars):
            # computing transformations
            angle = -car.angle
            # Animating first second zoom.
            zoom = 0.1 * SCALE * max(1 - self.t, 0) + ZOOM * SCALE * min(self.t, 1)
            scroll_x = -(car.position[0]) * zoom
            scroll_y = -(car.position[1]) * zoom
            trans = pygame.math.Vector2((scroll_x, scroll_y)).rotate_rad(angle)
            trans = (WINDOW_W / 2 + trans[0], WINDOW_H / 4 + trans[1])

            self._render_road(self.surf[i], zoom, trans, angle)
            car.draw(
                self.surf[i],
                zoom,
                trans,
                angle,
                mode not in ["state_pixels_list", "state_pixels"],
            )
            self.surf[i] = pygame.transform.flip(self.surf[i], False, True)

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
        elif mode == "state_pixels":
            return self._create_image_arrays(self.surf, (STATE_W, STATE_H))
        else:
            return self.isopen

    def _create_image_arrays(
        self, surfaces: list[pygame.Surface], size
    ) -> list[np.ndarray]:
        image_arrays = []
        for surf in surfaces:
            scaled_screen = pygame.transform.smoothscale(surf, size)
            image_array = np.transpose(
                np.array(pygame.surfarray.pixels3d(scaled_screen)), axes=(1, 0, 2)
            )
            image_arrays.append(image_array)
        return image_arrays

    def _render_road(self, surface, zoom, translation, angle):
        bounds = PLAYFIELD
        field = [
            (bounds, bounds),
            (bounds, -bounds),
            (-bounds, -bounds),
            (-bounds, bounds),
        ]

        # draw background
        self._draw_colored_polygon(
            surface, field, self.bg_color, zoom, translation, angle, clip=False
        )

        # draw grass patches
        grass = []
        for x in range(-20, 20, 2):
            for y in range(-20, 20, 2):
                grass.append(
                    [
                        (GRASS_DIM * x + GRASS_DIM, GRASS_DIM * y + 0),
                        (GRASS_DIM * x + 0, GRASS_DIM * y + 0),
                        (GRASS_DIM * x + 0, GRASS_DIM * y + GRASS_DIM),
                        (GRASS_DIM * x + GRASS_DIM, GRASS_DIM * y + GRASS_DIM),
                    ]
                )
        for poly in grass:
            self._draw_colored_polygon(
                surface, poly, self.grass_color, zoom, translation, angle
            )

        # draw road
        for poly, color in self.road_poly:
            # converting to pixel coordinates
            poly = [(p[0], p[1]) for p in poly]
            color = [int(c) for c in color]
            self._draw_colored_polygon(surface, poly, color, zoom, translation, angle)
