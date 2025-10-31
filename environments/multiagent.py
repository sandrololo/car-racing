import sys
from typing import Optional, Union
import math
import numpy as np
from gymnasium.envs.box2d.car_dynamics import Car
from gymnasium.error import DependencyNotInstalled
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
WINDOW_W = 1600
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


class FrictionAndCrashDetector(Box2D.b2.contactListener):
    def __init__(self, env, lap_complete_percent):
        Box2D.b2.contactListener.__init__(self)
        self.env = env
        self.lap_complete_percent = lap_complete_percent

    def BeginContact(self, contact):
        self._crash_contact(contact)
        self._tile_contact(contact, True)

    def EndContact(self, contact):
        self._tile_contact(contact, False)

    def _crash_contact(self, contact):
        u1 = contact.fixtureA.body.userData
        u2 = contact.fixtureB.body.userData
        if u1 and "car_id" in u1.__dict__ and u2 and "car_id" in u2.__dict__:
            for car in self.env.cars:
                if car.id == u1.id or car.id == u2.id:
                    car.reward -= 1000.0

    def _tile_contact(self, contact, begin):
        tile = None
        obj = None
        u1 = contact.fixtureA.body.userData
        u2 = contact.fixtureB.body.userData
        if u1 and "road_friction" in u1.__dict__:
            tile = u1
            obj = u2
        if u2 and "road_friction" in u2.__dict__:
            tile = u2
            obj = u1
        if not tile:
            return

        # inherit tile color from env
        tile.color[:] = self.env.road_color
        # wheel has 'set' of tiles it is in contact with
        if not obj or "tiles" not in obj.__dict__:
            # check if the car object reached a new tile to update rewards
            if "car_id" in obj.__dict__:
                car_obj = obj
                for env_car in self.env.cars:
                    if env_car.id == car_obj.id:
                        env_car.tiles_visited.add(tile.idx)
                        env_car.reward += 1000.0 / len(self.env.track)
                        if (
                            tile.idx == 0
                            and len(env_car.tiles_visited) / len(self.env.track)
                            > self.lap_complete_percent
                        ):
                            env_car.new_lap()
            return
        wheel = obj
        # so that the wheel can keep track of which tiles it is touching to calculate friction
        if begin:
            wheel.tiles.add(tile)
            if not tile.road_visited:
                tile.road_visited = True
        else:
            wheel.tiles.remove(tile)


class CarInfo:
    car_count = 0

    def __init__(self):
        # TODO: initialize at different positions for multiple cars
        self.car = None
        self.reward = 0.0
        self.prev_reward = 0.0
        self.tiles_visited = set()
        self.fuel_spent = 0.0
        self.lap_count: int = 0
        self.id = CarInfo.car_count
        CarInfo.car_count += 1

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
    def wheels(self) -> float:
        return self.car.wheels

    @property
    def __dict__(self) -> dict:
        return {
            "car_id": self.id,
            "position": self.position,
            "velocity": self.velocity,
            "angle": self.angle,
            "angular_velocity": self.angular_velocity,
            "reward": self.reward,
            "tiles_visited": self.tiles_visited,
            "fuel_spent": self.fuel_spent,
            "lap_count": self.lap_count,
        }

    def reset(self, world, track):
        beta = track[0][1]
        x = track[0][2] + (-1) ** self.id * 2.5 - self.id * 5.0 * math.sin(beta)
        y = track[0][3] + self.id * 5.0 * math.cos(beta)
        self.car = Car(world, beta, x, y)
        self.car.hull.userData = self
        self.reward = 0.0
        self.prev_reward = 0.0
        self.tiles_visited.clear()
        self.fuel_spent = 0.0
        self.lap_count = 0

    def apply_action(self, action: np.ndarray):
        action = action.astype(np.float64)
        self.car.steer(-action[0])
        self.car.gas(action[1])
        self.car.brake(action[2])

    def step(self, dt: float):
        self.car.step(dt)

    def new_lap(self):
        self.lap_count += 1
        self.tiles_visited.clear()

    def destroy(self):
        self.car.destroy()

    def draw(
        self,
        surf: pygame.Surface,
        zoom: float,
        trans: tuple[float, float],
        angle: float,
        draw_particles: bool = True,
    ):
        self.car.draw(surf, zoom, trans, angle, draw_particles)


class MultiAgentCarRacingEnv(gymnasium.Env):
    """
    Multi-agent version of the CarRacing environment.
    It has the same mechanics. Each observation is the perspective of a car.
    The rewards are the same except that collisions between cars are punished with -1000.
    """

    metadata = {
        "render_modes": [
            "human",
            "state_pixels",
        ],
        "render_fps": FPS,
    }

    def __init__(self, config: dict = None, *args, **kwargs):
        if config:
            self.num_agents = config.get("num_agents", 4)
            self.lap_complete_percent = config.get("lap_complete_percent", 0.95)
            self.render_mode = config.get("render_mode", None)
            max_timesteps = config.get("max_timesteps", None)
        else:
            self.num_agents = 4
            self.lap_complete_percent = 0.95
            self.render_mode = None
            max_timesteps = None

        self.road = None

        self.road_color = np.array([102, 102, 102])
        self.bg_color = np.array([102, 204, 102])
        self.grass_color = np.array([102, 230, 102])

        self.contactListener_keepref = FrictionAndCrashDetector(
            self, self.lap_complete_percent
        )
        self.world = Box2D.b2World((0, 0), contactListener=self.contactListener_keepref)
        self.screen: Optional[pygame.Surface] = None
        self.surf: list[pygame.Surface] = []
        self.clock: Optional[pygame.time.Clock] = None
        self.fd_tile = Box2D.b2.fixtureDef(
            shape=Box2D.b2.polygonShape(vertices=[(0, 0), (1, 0), (1, -1), (0, -1)])
        )

        self.cars: list[CarInfo] = [CarInfo() for _ in range(self.num_agents)]

        self.observation_space = spaces.Tuple(
            spaces.Box(low=0, high=255, shape=(STATE_H, STATE_W, 3), dtype=np.uint8)
            for _ in range(self.num_agents)
        )
        # do nothing, left, right, gas, brake
        self.action_space = spaces.Tuple(
            [spaces.Discrete(5) for _ in range(self.num_agents)]
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
        self.world.contactListener_bug_workaround = FrictionAndCrashDetector(
            self, self.lap_complete_percent
        )
        self.world.contactListener = self.world.contactListener_bug_workaround
        self.t = 0.0
        self.road_poly = []

        while True:
            success = self._create_track()
            if success:
                break
            gymnasium.logger.warn("Failed to generate track, retrying...")
        for car in self.cars:
            car.reset(self.world, self.track)

        if self.render_mode == "human":
            self.render()
        return self.step(None)[0], {}

    def step(self, actions: Union[list[np.ndarray], None]):
        assert len(self.cars) > 0
        if actions is not None:
            for car, action in zip(self.cars, actions):
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
        if actions is not None:  # First step without action, called from reset()
            for i, car in enumerate(self.cars):
                car.reward -= 0.1
                # We actually don't want to count fuel spent, we want car to be faster.
                # self.reward -=  10 * self.car.fuel_spent / ENGINE_POWER
                car.fuel_spent = 0.0
                step_rewards[i] = car.reward - car.prev_reward
                car.prev_reward = car.reward
                if len(car.tiles_visited) == len(self.track) or car.lap_count >= 1:
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

    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return
        else:
            return self._render(self.render_mode)

    def _render(self, mode: str):
        assert mode in self.metadata["render_modes"]

        pygame.font.init()
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
            self._render_indicators(0, self.surf[i], car, WINDOW_W, WINDOW_H)

            font = pygame.font.Font(pygame.font.get_default_font(), 42)
            reward_text = font.render(
                "%04i" % car.reward, True, (255, 255, 255), (0, 0, 0)
            )
            reward_text_rect = reward_text.get_rect()
            reward_text_rect.center = (60, WINDOW_H - WINDOW_H * 2.5 / 40.0)
            self.surf[i].blit(reward_text, reward_text_rect)

        if mode == "human":
            if self.screen is None:
                pygame.init()
                pygame.display.init()
                self.screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
            surf = pygame.Surface((WINDOW_W, WINDOW_H))
            first_car = self.cars[0]
            last_car = self.cars[-1]
            min_x = sys.maxsize
            min_y = sys.maxsize
            max_x = -sys.maxsize
            max_y = -sys.maxsize
            for car in self.cars:
                x, y = car.position
                min_x = min(min_x, x)
                min_y = min(min_y, y)
                max_x = max(max_x, x)
                max_y = max(max_y, y)
                first_car = (
                    car
                    if len(car.tiles_visited) > len(first_car.tiles_visited)
                    else first_car
                )
                last_car = (
                    car
                    if len(car.tiles_visited) < len(last_car.tiles_visited)
                    else last_car
                )
            # computing transformations
            angle = -(first_car.angle + last_car.angle) / 2

            zoom = max((max_x - min_x), (max_y - min_y)) / 3.5
            scroll_x = -(max_x + min_x) / 2 * zoom
            scroll_y = -(max_y + min_y) / 2 * zoom
            trans = pygame.math.Vector2((scroll_x, scroll_y)).rotate_rad(angle)
            trans = (WINDOW_W / 2 + trans[0], WINDOW_H / 2 + trans[1])

            self._render_road(surf, zoom, trans, angle)
            for i, car in enumerate(self.cars):
                car.draw(
                    surf,
                    zoom,
                    trans,
                    angle,
                    mode not in ["state_pixels_list", "state_pixels"],
                )

                # showing stats
            surf = pygame.transform.flip(surf, False, True)
            for i, car in enumerate(self.cars):
                self._render_indicators(i, surf, car, WINDOW_W / 3, WINDOW_H)

                font = pygame.font.Font(pygame.font.get_default_font(), 21)
                car_id_text = font.render(f"Car {i}", True, (255, 255, 255), None)
                car_id_text_rect = car_id_text.get_rect()
                car_id_text_rect.center = (
                    40,
                    WINDOW_H - WINDOW_H * 4.2 / 40.0 - i * 5 * (WINDOW_H / 40.0),
                )
                surf.blit(car_id_text, car_id_text_rect)

                font = pygame.font.Font(pygame.font.get_default_font(), 42)
                reward_text = font.render(
                    "%04i" % car.reward, True, (255, 255, 255), None
                )
                reward_text_rect = reward_text.get_rect()
                reward_text_rect.center = (
                    60,
                    WINDOW_H - WINDOW_H * 2.5 / 40.0 - i * 5 * (WINDOW_H / 40.0),
                )
                surf.blit(reward_text, reward_text_rect)

                x_start = WINDOW_W * 4 / 6 + i % 2 * WINDOW_W / 6
                y_start = i // 2 * WINDOW_H / 4
                surf.blit(
                    pygame.transform.smoothscale(
                        self.surf[i], (WINDOW_W / 6, WINDOW_H / 4)
                    ),
                    (
                        x_start,
                        y_start,
                    ),
                )
                car_id_text_rect.center = (
                    x_start + 40,
                    y_start + 20,
                )
                surf.blit(car_id_text, car_id_text_rect)

            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            assert self.screen is not None
            self.screen.fill(0)
            self.screen.blit(surf, (0, 0))
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

    def _render_indicators(self, idx, surface, car: CarInfo, W, H):
        s = W / 40.0
        h = H / 40.0
        H -= idx * 5 * h
        indicator_box_surf = pygame.Surface((W, 5 * h), pygame.SRCALPHA)
        alpha = 128 if self.render_mode == "human" else 255
        indicator_box_surf.fill((100, 100, 100, alpha))
        polygon = [(W, h * 5), (W, 0), (0, 0), (0, h * 5)]
        pygame.draw.polygon(indicator_box_surf, (255, 255, 255, alpha), polygon, 1)
        surface.blit(indicator_box_surf, (0, H - 5 * h))

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
            car.wheels[0].omega,
            vertical_ind(7, 0.01 * car.wheels[0].omega),
            (0, 0, 255),
        )
        render_if_min(
            car.wheels[1].omega,
            vertical_ind(8, 0.01 * car.wheels[1].omega),
            (0, 0, 255),
        )
        render_if_min(
            car.wheels[2].omega,
            vertical_ind(9, 0.01 * car.wheels[2].omega),
            (51, 0, 255),
        )
        render_if_min(
            car.wheels[3].omega,
            vertical_ind(10, 0.01 * car.wheels[3].omega),
            (51, 0, 255),
        )

        render_if_min(
            car.wheels[0].joint.angle,
            horiz_ind(20, -10.0 * car.wheels[0].joint.angle),
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
