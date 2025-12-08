from typing import List, Optional
import math
import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils.typing import MultiAgentDict
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
from .cars import CarConfig, MultiAgentCars, LeaderBoard
from .config import (
    STATE_W,
    STATE_H,
    WINDOW_W,
    WINDOW_H,
    VIDEO_W,
    VIDEO_H,
    SCALE,
    TRACK_RAD,
    PLAYFIELD,
    FPS,
    ZOOM,
    TRACK_DETAIL_STEP,
    TRACK_TURN_RATE,
    TRACK_WIDTH,
    BORDER,
    BORDER_MIN_COUNT,
    GRASS_DIM,
    MAX_SHAPE_DIM,
)


class FrictionAndCrashDetector(Box2D.b2.contactListener):
    def __init__(self, env, lap_complete_percent, first_tile_visitor_reward_factor):
        Box2D.b2.contactListener.__init__(self)
        self.env = env
        self.lap_complete_percent = lap_complete_percent
        self.first_tile_visitor_reward_factor = first_tile_visitor_reward_factor
        self._tiles_visited = set()

    def BeginContact(self, contact):
        self._crash_contact(contact)
        self._tile_contact(contact, True)

    def EndContact(self, contact):
        self._tile_contact(contact, False)

    def _crash_contact(self, contact):
        u1 = contact.fixtureA.body.userData
        u2 = contact.fixtureB.body.userData
        if u1 and "car_id" in u1.__dict__ and u2 and "car_id" in u2.__dict__:
            car_1 = self.env.cars.get(u1.id)
            car_2 = self.env.cars.get(u2.id)
            if car_1.active and car_2.active:
                car_1.reward -= 100.0
                car_2.reward -= 100.0

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
                for env_car in self.env.cars.get_active():
                    if env_car.id == car_obj.id:
                        if not tile.idx in env_car.tiles_visited:
                            env_car.tiles_visited.add(tile.idx)
                            reward = 1000.0 / len(self.env.track)
                            if tile.idx not in self._tiles_visited:
                                self._tiles_visited.add(tile.idx)
                                reward *= self.first_tile_visitor_reward_factor
                            env_car.reward += reward
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


class MultiAgentCarRacingEnv(MultiAgentEnv):
    """
    Multi-agent version of the CarRacing environment.
    It has the same mechanics. Each observation is the perspective of a car.
    The rewards are the same except that collisions between cars are punished with -100.
    """

    metadata = {
        "render_modes": ["human", "state_pixels", "video"],
        "render_fps": FPS,
    }

    def __init__(self, config: dict = None, *args, **kwargs):
        self.num_cars = config.get("num_cars", 8)
        car_configs = config.get("car_configs", [CarConfig.default()] * 8)
        self.lap_complete_percent = config.get("lap_complete_percent", 0.95)
        self.render_mode = config.get("render_mode", None)
        self.first_tile_visitor_reward_factor = config.get(
            "first_tile_visitor_reward_factor", 1.0
        )

        self.isopen = True

        self.road = None

        self.road_color = np.array([102, 102, 102])
        self.bg_color = np.array([102, 204, 102])
        self.grass_color = np.array([102, 230, 102])

        self.contactListener_keepref = FrictionAndCrashDetector(
            self, self.lap_complete_percent, self.first_tile_visitor_reward_factor
        )
        self.world = Box2D.b2World((0, 0), contactListener=self.contactListener_keepref)
        self.screen: Optional[pygame.Surface] = None
        self.clock: Optional[pygame.time.Clock] = None
        self.fd_tile = Box2D.b2.fixtureDef(
            shape=Box2D.b2.polygonShape(vertices=[(0, 0), (1, 0), (1, -1), (0, -1)])
        )

        self.possible_agents: List[str] = [f"car_{i}" for i in range(8)]
        self.agents: List[str] = self.possible_agents.copy()[: self.num_cars]

        self.cars = MultiAgentCars(car_configs, self.possible_agents, self.agents)
        self.leaderboard = LeaderBoard(self.cars)
        self.observation_spaces = {
            agent: spaces.Box(
                low=0, high=255, shape=(STATE_H, STATE_W, 3), dtype=np.uint8
            )
            for agent in self.possible_agents
        }
        self.action_spaces = {
            agent: spaces.Box(
                np.array([-1, 0, 0]).astype(np.float32),
                np.array([+1, +1, +1]).astype(np.float32),
            )  # steer, gas, brake
            for agent in self.possible_agents
        }
        self.human_render_mode_camera_angle: Optional[float] = None
        pygame.font.init()
        self._obs_reward_font = pygame.font.Font(
            pygame.font.get_default_font(), STATE_H // 20
        )
        self._font_16 = pygame.font.Font(pygame.font.get_default_font(), 16)
        self._font_12 = pygame.font.Font(pygame.font.get_default_font(), 12)
        super().__init__()

    def _destroy(self):
        if not self.road:
            return
        for t in self.road:
            self.world.DestroyBody(t)
        self.road = []
        self.cars.destroy()

    def set_num_cars(self, num_cars: int):
        self.num_cars = num_cars
        self.agents = self.possible_agents.copy()[: self.num_cars]
        for car in self.cars.get_all():
            if car.id not in self.agents:
                car.active = False
            else:
                car.active = True

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        self._destroy()
        self.world.contactListener_bug_workaround = FrictionAndCrashDetector(
            self, self.lap_complete_percent, self.first_tile_visitor_reward_factor
        )
        self.world.contactListener = self.world.contactListener_bug_workaround
        self.t = 0.0
        self.road_poly = []

        self.agents = self.possible_agents.copy()[: self.num_cars]

        while True:
            success = self._create_track()
            if success:
                break
            gymnasium.logger.warn("Failed to generate track, retrying...")
        self.cars.reset(self.world, self.track)
        self.leaderboard.update()
        self.human_render_mode_camera_angle = None

        if self.render_mode == "human":
            self.render()
        info = {}
        for agent in self.agents:
            info[agent] = {}
        observations = self._render("state_pixels")
        return observations, info

    def step(self, actions: MultiAgentDict):
        self.cars.apply_actions(actions)
        self.leaderboard.update()
        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)

        observation_surfaces = self._render_observation_surfaces()
        observations = self._create_image_arrays(observation_surfaces)
        obs_d, rew_d, terminated_d, truncated_d, info_d = self.cars.step(
            self.track, actions, observations
        )
        for agent in self.agents:
            car = self.cars.get(agent)
            if car.active:
                if terminated_d[agent] or truncated_d[agent]:
                    self.agents.remove(agent)
        if self.render_mode == "human":
            self._render("human", observation_surfaces)
        return obs_d, rew_d, terminated_d, truncated_d, info_d

    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return
        else:
            return self._render(self.render_mode)

    def _render(
        self, mode: str, surfaces: Optional[MultiAgentDict] = None
    ) -> MultiAgentDict:
        if surfaces is None:
            surfaces = self._render_observation_surfaces()

        if mode == "human" or mode == "video":
            if mode == "human" and self.screen is None:
                pygame.init()
                pygame.display.init()
                self.screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
            if mode == "human":
                main_surface = pygame.Surface((WINDOW_W, WINDOW_H))
            else:
                main_surface = pygame.Surface((VIDEO_W, VIDEO_H))
            meain_surf_width, main_surf_height = main_surface.get_size()
            # computing transformations
            leading_car_track_tile_idx = (
                list(self.leaderboard[0].tiles_visited)[-1]
                if len(self.leaderboard[0].tiles_visited) > 0
                else 0
            )
            last_car_track_tile_idx = (
                list(self.leaderboard[-1].tiles_visited)[-1]
                if len(self.leaderboard[-1].tiles_visited) > 0
                else 0
            )
            leading_angle = self.track[leading_car_track_tile_idx][1]
            last_angle = self.track[last_car_track_tile_idx][1]
            current_angle = -(leading_angle + last_angle) / 2
            if self.human_render_mode_camera_angle is None:
                self.human_render_mode_camera_angle = current_angle
            self.human_render_mode_camera_angle += (
                current_angle - self.human_render_mode_camera_angle
            ) * 0.03
            min_x, min_y, width, height = self.cars.get_enclosing_rect()
            zoom = min(
                ZOOM * SCALE, main_surf_height / 1.3 / max(1, max((width), (height)))
            )
            scroll_x = -(min_x + width / 2) * zoom
            scroll_y = -(min_y + height / 2) * zoom
            trans = pygame.math.Vector2((scroll_x, scroll_y)).rotate_rad(
                self.human_render_mode_camera_angle
            )
            trans = (meain_surf_width / 2 + trans[0], main_surf_height / 2 + trans[1])

            self._render_road(
                main_surface, zoom, trans, self.human_render_mode_camera_angle
            )
            self.cars.draw(
                main_surface,
                zoom,
                trans,
                self.human_render_mode_camera_angle,
                True,
                draw_number=True,
            )

            # showing stats
            main_surface = pygame.transform.flip(main_surface, False, True)
            self.cars.render_indicators(
                self.render_mode, main_surface, meain_surf_width / 4, main_surf_height
            )

            for car in self.cars.get_active():
                agent = car.id
                i = car.count
                if mode == "human":
                    car_id_reward_text = self._font_16.render(
                        f"Car {car.count}: {car.reward:04.0f}",
                        True,
                        (255, 255, 255),
                        None,
                    )
                else:
                    car_id_reward_text = self._font_12.render(
                        f"Car {car.count}: {car.reward:04.0f}",
                        True,
                        (255, 255, 255),
                        None,
                    )
                car_id_reward_text_rect = car_id_reward_text.get_rect()
                car_id_reward_text_rect.center = (
                    meain_surf_width / 22,
                    main_surf_height
                    - main_surf_height * 4.2 / 40.0
                    - i * 5 * (main_surf_height / 40.0),
                )
                main_surface.blit(car_id_reward_text, car_id_reward_text_rect)

                if car.terminated or car.truncated:
                    if mode == "human":
                        pos_text = self._font_16.render(
                            "TERMINATED" if car.terminated else "TRUNCATED",
                            True,
                            (255, 255, 255),
                            None,
                        )
                    else:
                        pos_text = self._font_12.render(
                            "TERMINATED" if car.terminated else "TRUNCATED",
                            True,
                            (255, 255, 255),
                            None,
                        )
                    pos_text_rect = pos_text.get_rect()
                    pos_text_rect.center = (
                        meain_surf_width / 4 - main_surf_height / 13,
                        main_surf_height
                        - main_surf_height * 4.2 / 40.0
                        - i * 5 * (main_surf_height / 40.0),
                    )
                else:
                    if mode == "human":
                        pos_text = self._font_16.render(
                            f"Pos: {self.leaderboard.get_position(car) + 1}/{len(self.cars.get_active())}",
                            True,
                            (255, 255, 255),
                            None,
                        )
                    else:
                        pos_text = self._font_12.render(
                            f"Pos: {self.leaderboard.get_position(car) + 1}/{len(self.cars.get_active())}",
                            True,
                            (255, 255, 255),
                            None,
                        )
                    pos_text_rect = pos_text.get_rect()
                    pos_text_rect.center = (
                        meain_surf_width / 4 - meain_surf_width / 30,
                        main_surf_height
                        - main_surf_height * 4.2 / 40.0
                        - i * 5 * (main_surf_height / 40.0),
                    )
                main_surface.blit(pos_text, pos_text_rect)

                config_text_surf = car.config.get_surface(mode)
                config_text_rect = config_text_surf.get_rect()
                config_text_rect.center = (
                    meain_surf_width / 80 + config_text_rect.width / 2,
                    main_surf_height
                    - main_surf_height * 0.5 / 40.0
                    - i * 5 * (main_surf_height / 40.0),
                )
                main_surface.blit(config_text_surf, config_text_rect)

                x_start = meain_surf_width * 3 / 4 + i % 2 * main_surf_height / 4
                y_start = i // 2 * main_surf_height / 4
                if agent in surfaces:
                    surf = surfaces[agent]
                    main_surface.blit(
                        pygame.transform.smoothscale(
                            surf, (main_surf_height / 4, main_surf_height / 4)
                        ),
                        (x_start, y_start),
                    )
                    if mode == "human":
                        car_id_text = self._font_16.render(
                            f"Car {i}", True, (255, 255, 255), None
                        )
                    else:
                        car_id_text = self._font_12.render(
                            f"Car {i}", True, (255, 255, 255), None
                        )
                    car_id_text_rect = car_id_text.get_rect()
                    car_id_text_rect.center = (
                        x_start + main_surf_height / 23,
                        y_start + main_surf_height / 50,
                    )
                    main_surface.blit(car_id_text, car_id_text_rect)

            track_map_surf = self._create_track_map_surface(main_surf_height // 2.7)
            main_surface.blit(
                track_map_surf,
                (meain_surf_width / 4 + 20, main_surf_height - main_surf_height / 2.5),
            )

            self.clock.tick(self.metadata["render_fps"])
            if mode == "human":
                pygame.event.pump()
                assert self.screen is not None
                self.screen.fill(0)
                self.screen.blit(main_surface, (0, 0))
                pygame.display.flip()
            if mode == "video":
                return np.transpose(
                    np.array(pygame.surfarray.pixels3d(main_surface)), axes=(1, 0, 2)
                )
        elif mode == "state_pixels":
            return self._create_image_arrays(surfaces)
        else:
            return self.isopen

    def _render_observation_surfaces(self) -> MultiAgentDict:
        if self.clock is None:
            self.clock = pygame.time.Clock()

        if "t" not in self.__dict__:
            return  # reset() not called yet

        surfaces = {
            agent: pygame.Surface((STATE_W, STATE_H))
            for agent in self.agents
            if self.cars.get(agent).terminated is False
            and self.cars.get(agent).truncated is False
        }
        surf_width, surf_height = surfaces[self.agents[0]].get_size()
        surface_zoom = ZOOM * SCALE * surf_height / WINDOW_H
        assert len(self.cars.get_active()) > 0
        for agent, surface in surfaces.items():
            # computing transformations
            angle = -self.cars.get(agent).angle
            trans = self.cars.get(agent).get_translation(
                surface_zoom, surf_width, surf_height
            )

            self._render_road(surface, surface_zoom, trans, angle)

            self.cars.draw(
                surface, surface_zoom, trans, angle, tyre_marks=False, draw_number=False
            )
        for agent in surfaces.keys():
            surfaces[agent] = pygame.transform.flip(surfaces[agent], False, True)

        # showing stats
        self.cars.render_indicators(self.render_mode, surfaces, surf_width, surf_height)

        for agent, surface in surfaces.items():
            reward_text = self._obs_reward_font.render(
                "%04i" % self.cars.get(agent).reward, True, (255, 255, 255), (0, 0, 0)
            )
            reward_text_rect = reward_text.get_rect()
            reward_text_rect.center = (
                surf_height // 15,
                surf_height - surf_height * 2.5 / 40.0,
            )
            surface.blit(reward_text, reward_text_rect)
        return surfaces

    def _create_image_arrays(self, surfaces: MultiAgentDict) -> MultiAgentDict:
        image_arrays = {}
        for agent, surf in surfaces.items():
            image_array = np.transpose(
                np.array(pygame.surfarray.pixels3d(surf)), axes=(1, 0, 2)
            )
            image_arrays[agent] = image_array
        return image_arrays

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            self.isopen = False
            pygame.quit()

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
        self, surface: pygame.Surface, poly, color, zoom, translation, angle, clip=True
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
            (-MAX_SHAPE_DIM <= coord[0] <= surface.get_width() + MAX_SHAPE_DIM)
            and (-MAX_SHAPE_DIM <= coord[1] <= surface.get_height() + MAX_SHAPE_DIM)
            for coord in poly
        ):
            gfxdraw.aapolygon(surface, poly, color)
            gfxdraw.filled_polygon(surface, poly, color)

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

    def _create_track_map_surface(self, size: int) -> pygame.Surface:
        road_poly = self.get_wrapper_attr("road_poly")
        min_x = min(p[0] for p, c in road_poly)[0]
        max_x = max(p[0] for p, c in road_poly)[0]
        min_y = min(p[1] for p, c in road_poly)[0]
        max_y = max(p[1] for p, c in road_poly)[0]
        track_map_surf = pygame.Surface(
            (max_x - min_x + 80, max_y - min_y + 80), pygame.SRCALPHA
        )
        for poly, _ in road_poly:
            poly = [(p[0] - min_x + 40, p[1] - min_y + 40) for p in poly]
            gfxdraw.filled_polygon(track_map_surf, poly, (255, 255, 255, 255))

        for car in self.cars.get_active():
            pos = (
                int(car.position[0] - min_x + 40),
                int(car.position[1] - min_y + 40),
            )
            gfxdraw.filled_circle(
                track_map_surf,
                pos[0],
                pos[1],
                5,
                (255, 0, 0, 255),
            )
        track_map_surf = pygame.transform.flip(track_map_surf, False, True)
        return pygame.transform.smoothscale(
            track_map_surf,
            (
                size * track_map_surf.get_height() / track_map_surf.get_width(),
                size,
            ),
        )
