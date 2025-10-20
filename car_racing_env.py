from typing import Optional, Union
import math
import numpy as np
from gymnasium.envs.box2d.car_racing import CarRacing, FrictionDetector
from gymnasium.envs.box2d.car_dynamics import Car
from gymnasium.error import InvalidAction, DependencyNotInstalled
from gymnasium.spaces import Box
from gymnasium import wrappers
import gymnasium
from gymnasium import spaces

try:
    import Box2D
except ImportError as e:
    raise DependencyNotInstalled(
        'Box2D is not installed, you can install it by run `pip install swig` followed by `pip install "gymnasium[box2d]"`'
    ) from e

try:
    # As pygame is necessary for using the environment (reset and step) even without a render mode
    #   therefore, pygame is a necessary import for the environment.
    import pygame
    from pygame import gfxdraw
except ImportError as e:
    raise DependencyNotInstalled(
        'pygame is not installed, run `pip install "gymnasium[box2d]"`'
    ) from e


STATE_W = 96  # less than Atari 160x192
STATE_H = 96
WINDOW_W = 1000
WINDOW_H = 800

SCALE = 6.0  # Track scale
TRACK_RAD = 900 / SCALE  # Track is heavily morphed circle with this radius
PLAYFIELD = 2000 / SCALE  # Game over boundary
FPS = 50  # Frames per second
ZOOM = 2.7  # Camera zoom

TRACK_DETAIL_STEP = 21 / SCALE
TRACK_TURN_RATE = 0.31
TRACK_WIDTH = 40 / SCALE
BORDER = 8 / SCALE
BORDER_MIN_COUNT = 4
GRASS_DIM = PLAYFIELD / 20.0
MAX_SHAPE_DIM = (
    max(GRASS_DIM, TRACK_WIDTH, TRACK_DETAIL_STEP) * math.sqrt(2) * ZOOM * SCALE
)


class CarRacingEnv(gymnasium.Wrapper):
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
        else:
            lap_complete_percent = 0.95
            render_mode = None
            max_timesteps = None
            gray_scale = False
            frame_stack = 1
            frame_skip = 1
        self.env = CarRacing(
            render_mode=render_mode,
            lap_complete_percent=lap_complete_percent,
            continuous=True,
            *args,
            **kwargs,
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

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._preprocess_obs(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
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
    def velocity(self) -> tuple[float, float]:
        return self.car.hull.linearVelocity

    @property
    def angle(self) -> float:
        return self.car.hull.angle

    @property
    def angular_velocity(self) -> float:
        return self.car.hull.angularVelocity

    @property
    def wheel_omega(self, index: int) -> float:
        return self.car.wheels[index].omega

    @property
    def wheel_joint_angle(self, index: int) -> float:
        return self.car.wheels[index].joint.angle

    def apply_action(self, action: np.ndarray):
        action = action.astype(np.float64)
        self.car.steer(-action[0])
        self.car.gas(action[1])
        self.car.brake(action[2])

    def step(self, dt: float):
        self.car.step(dt)

    def destroy(self):
        self.car.destroy()

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

        self.road = None
        self.cars: list[CarInfo] = [CarInfo(None, None) for _ in range(self.num_agents)]

        self.road_color = np.array([102, 102, 102])
        self.bg_color = np.array([102, 204, 102])
        self.grass_color = np.array([102, 230, 102])

        self.world = Box2D.b2World((0, 0), contactListener=self.contactListener_keepref)
        self.screen: Optional[pygame.Surface] = None
        self.surf: list[pygame.Surface] = []
        self.clock: Optional[pygame.time.Clock] = None
        self.fd_tile = Box2D.b2.fixtureDef(
            shape=Box2D.b2.polygonShape(vertices=[(0, 0), (1, 0), (1, -1), (0, -1)])
        )

        self.observation_space = spaces.Tuple(
            spaces.Box(low=0, high=255, shape=(STATE_H, STATE_W, 3), dtype=np.uint8)
            for _ in range(self.num_agents)
        )
        # do nothing, left, right, gas, brake
        self.action_space = spaces.Tuple(
            spaces.Discrete(5) for _ in range(self.num_agents)
        )

    def _destroy(self):
        if not self.road:
            return
        for t in self.road:
            self.world.DestroyBody(t)
        self.road = []
        assert len(self.cars) > 0
        for car in self.cars:
            car.destroy()

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
            self._render_indicators(self.surf[i], WINDOW_W, WINDOW_H)

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

    def _draw_colored_polygon(
        self, surface, poly, color, zoom, translation, angle, clip=True
    ):
        poly = [pygame.math.Vector2(c).rotate_rad(angle) for c in poly]
        poly = [
            (c[0] * zoom + translation[0], c[1] * zoom + translation[1]) for c in poly
        ]
        # This checks if the polygon is out of bounds of the screen, and we skip drawing if so.
        # Instead of calculating exactly if the polygon and screen overlap,
        # we simply check if the polygon is in a larger bounding box whose dimension
        # is greater than the screen by MAX_SHAPE_DIM, which is the maximum
        # diagonal length of an environment object
        if not clip or any(
            (-MAX_SHAPE_DIM <= coord[0] <= WINDOW_W + MAX_SHAPE_DIM)
            and (-MAX_SHAPE_DIM <= coord[1] <= WINDOW_H + MAX_SHAPE_DIM)
            for coord in poly
        ):
            gfxdraw.aapolygon(surface, poly, color)
            gfxdraw.filled_polygon(surface, poly, color)

    def _render_indicators(self, surface, car: CarInfo, W, H):
        s = W / 40.0
        h = H / 40.0
        color = (0, 0, 0)
        polygon = [(W, H), (W, H - 5 * h), (0, H - 5 * h), (0, H)]
        pygame.draw.polygon(surface, color=color, points=polygon)

        def vertical_ind(place, val):
            return [
                (place * s, H - (h + h * val)),
                ((place + 1) * s, H - (h + h * val)),
                ((place + 1) * s, H - h),
                ((place + 0) * s, H - h),
            ]

        def horiz_ind(place, val):
            return [
                ((place + 0) * s, H - 4 * h),
                ((place + val) * s, H - 4 * h),
                ((place + val) * s, H - 2 * h),
                ((place + 0) * s, H - 2 * h),
            ]

        assert car is not None
        true_speed = np.sqrt(np.square(car.velocity[0]) + np.square(car.velocity[1]))

        # simple wrapper to render if the indicator value is above a threshold
        def render_if_min(value, points, color):
            if abs(value) > 1e-4:
                pygame.draw.polygon(surface, points=points, color=color)

        render_if_min(true_speed, vertical_ind(5, 0.02 * true_speed), (255, 255, 255))
        # ABS sensors
        render_if_min(
            car.wheel_omega(0),
            vertical_ind(7, 0.01 * car.wheel_omega(0)),
            (0, 0, 255),
        )
        render_if_min(
            car.wheel_omega(1),
            vertical_ind(8, 0.01 * car.wheel_omega(1)),
            (0, 0, 255),
        )
        render_if_min(
            car.wheel_omega(2),
            vertical_ind(9, 0.01 * car.wheel_omega(2)),
            (51, 0, 255),
        )
        render_if_min(
            car.wheel_omega(3),
            vertical_ind(10, 0.01 * car.wheel_omega(3)),
            (51, 0, 255),
        )

        render_if_min(
            car.wheel_joint_angle(0),
            horiz_ind(20, -10.0 * car.wheel_joint_angle(0)),
            (0, 255, 0),
        )
        render_if_min(
            car.angular_velocity,
            horiz_ind(30, -0.8 * car.angular_velocity),
            (255, 0, 0),
        )

    def _create_track(self):
        CHECKPOINTS = 12

        # Create checkpoints
        checkpoints = []
        for c in range(CHECKPOINTS):
            noise = self.np_random.uniform(0, 2 * math.pi * 1 / CHECKPOINTS)
            alpha = 2 * math.pi * c / CHECKPOINTS + noise
            rad = self.np_random.uniform(TRACK_RAD / 3, TRACK_RAD)

            if c == 0:
                alpha = 0
                rad = 1.5 * TRACK_RAD
            if c == CHECKPOINTS - 1:
                alpha = 2 * math.pi * c / CHECKPOINTS
                self.start_alpha = 2 * math.pi * (-0.5) / CHECKPOINTS
                rad = 1.5 * TRACK_RAD

            checkpoints.append((alpha, rad * math.cos(alpha), rad * math.sin(alpha)))
        self.road = []

        # Go from one checkpoint to another to create track
        x, y, beta = 1.5 * TRACK_RAD, 0, 0
        dest_i = 0
        laps = 0
        track = []
        no_freeze = 2500
        visited_other_side = False
        while True:
            alpha = math.atan2(y, x)
            if visited_other_side and alpha > 0:
                laps += 1
                visited_other_side = False
            if alpha < 0:
                visited_other_side = True
                alpha += 2 * math.pi

            while True:  # Find destination from checkpoints
                failed = True

                while True:
                    dest_alpha, dest_x, dest_y = checkpoints[dest_i % len(checkpoints)]
                    if alpha <= dest_alpha:
                        failed = False
                        break
                    dest_i += 1
                    if dest_i % len(checkpoints) == 0:
                        break

                if not failed:
                    break

                alpha -= 2 * math.pi
                continue

            r1x = math.cos(beta)
            r1y = math.sin(beta)
            p1x = -r1y
            p1y = r1x
            dest_dx = dest_x - x  # vector towards destination
            dest_dy = dest_y - y
            # destination vector projected on rad:
            proj = r1x * dest_dx + r1y * dest_dy
            while beta - alpha > 1.5 * math.pi:
                beta -= 2 * math.pi
            while beta - alpha < -1.5 * math.pi:
                beta += 2 * math.pi
            prev_beta = beta
            proj *= SCALE
            if proj > 0.3:
                beta -= min(TRACK_TURN_RATE, abs(0.001 * proj))
            if proj < -0.3:
                beta += min(TRACK_TURN_RATE, abs(0.001 * proj))
            x += p1x * TRACK_DETAIL_STEP
            y += p1y * TRACK_DETAIL_STEP
            track.append((alpha, prev_beta * 0.5 + beta * 0.5, x, y))
            if laps > 4:
                break
            no_freeze -= 1
            if no_freeze == 0:
                break

        # Find closed loop range i1..i2, first loop should be ignored, second is OK
        i1, i2 = -1, -1
        i = len(track)
        while True:
            i -= 1
            if i == 0:
                return False  # Failed
            pass_through_start = (
                track[i][0] > self.start_alpha and track[i - 1][0] <= self.start_alpha
            )
            if pass_through_start and i2 == -1:
                i2 = i
            elif pass_through_start and i1 == -1:
                i1 = i
                break
        assert i1 != -1
        assert i2 != -1

        track = track[i1 : i2 - 1]

        first_beta = track[0][1]
        first_perp_x = math.cos(first_beta)
        first_perp_y = math.sin(first_beta)
        # Length of perpendicular jump to put together head and tail
        well_glued_together = np.sqrt(
            np.square(first_perp_x * (track[0][2] - track[-1][2]))
            + np.square(first_perp_y * (track[0][3] - track[-1][3]))
        )
        if well_glued_together > TRACK_DETAIL_STEP:
            return False

        # Red-white border on hard turns
        border = [False] * len(track)
        for i in range(len(track)):
            good = True
            oneside = 0
            for neg in range(BORDER_MIN_COUNT):
                beta1 = track[i - neg - 0][1]
                beta2 = track[i - neg - 1][1]
                good &= abs(beta1 - beta2) > TRACK_TURN_RATE * 0.2
                oneside += np.sign(beta1 - beta2)
            good &= abs(oneside) == BORDER_MIN_COUNT
            border[i] = good
        for i in range(len(track)):
            for neg in range(BORDER_MIN_COUNT):
                border[i - neg] |= border[i]

        # Create tiles
        for i in range(len(track)):
            alpha1, beta1, x1, y1 = track[i]
            alpha2, beta2, x2, y2 = track[i - 1]
            road1_l = (
                x1 - TRACK_WIDTH * math.cos(beta1),
                y1 - TRACK_WIDTH * math.sin(beta1),
            )
            road1_r = (
                x1 + TRACK_WIDTH * math.cos(beta1),
                y1 + TRACK_WIDTH * math.sin(beta1),
            )
            road2_l = (
                x2 - TRACK_WIDTH * math.cos(beta2),
                y2 - TRACK_WIDTH * math.sin(beta2),
            )
            road2_r = (
                x2 + TRACK_WIDTH * math.cos(beta2),
                y2 + TRACK_WIDTH * math.sin(beta2),
            )
            vertices = [road1_l, road1_r, road2_r, road2_l]
            self.fd_tile.shape.vertices = vertices
            t = self.world.CreateStaticBody(fixtures=self.fd_tile)
            t.userData = t
            c = 0.01 * (i % 3) * 255
            t.color = self.road_color + c
            t.road_visited = False
            t.road_friction = 1.0
            t.idx = i
            t.fixtures[0].sensor = True
            self.road_poly.append(([road1_l, road1_r, road2_r, road2_l], t.color))
            self.road.append(t)
            if border[i]:
                side = np.sign(beta2 - beta1)
                b1_l = (
                    x1 + side * TRACK_WIDTH * math.cos(beta1),
                    y1 + side * TRACK_WIDTH * math.sin(beta1),
                )
                b1_r = (
                    x1 + side * (TRACK_WIDTH + BORDER) * math.cos(beta1),
                    y1 + side * (TRACK_WIDTH + BORDER) * math.sin(beta1),
                )
                b2_l = (
                    x2 + side * TRACK_WIDTH * math.cos(beta2),
                    y2 + side * TRACK_WIDTH * math.sin(beta2),
                )
                b2_r = (
                    x2 + side * (TRACK_WIDTH + BORDER) * math.cos(beta2),
                    y2 + side * (TRACK_WIDTH + BORDER) * math.sin(beta2),
                )
                self.road_poly.append(
                    (
                        [b1_l, b1_r, b2_r, b2_l],
                        (255, 255, 255) if i % 2 == 0 else (255, 0, 0),
                    )
                )
        self.track = track
        return True
