import math
from enum import Enum
from typing import Optional, Union
import numpy as np
from gymnasium.envs.box2d.car_dynamics import (
    Car,
    SIZE as CAR_SIZE,
    WHEEL_MOMENT_OF_INERTIA,
)
from gymnasium.error import DependencyNotInstalled
from ray.rllib.utils.typing import MultiAgentDict, AgentID

try:
    # As pygame is necessary for using the environment (reset and step) even without a render mode
    #   therefore, pygame is a necessary import for the environment.
    import pygame
except ImportError as e:
    raise DependencyNotInstalled(
        'pygame is not installed, run `pip install "gymnasium[box2d]"`'
    ) from e

from .config import FPS, PLAYFIELD


class EnginePower(Enum):
    LOW = 0.7
    MEDIUM = 1.0
    HIGH = 1.3


class TyreType(Enum):
    HARD = 0.8
    MEDIUM = 1.0
    SOFT = 1.2


class CarConfig:
    @staticmethod
    def default():
        return CarConfig(EnginePower.MEDIUM, TyreType.MEDIUM)

    def __init__(
        self,
        engine_power: EnginePower = EnginePower.MEDIUM,
        tyre_type: TyreType = TyreType.MEDIUM,
    ):
        self._engine_power: EnginePower = engine_power
        self._tyre_type: TyreType = tyre_type
        self._config_surface: Optional[pygame.Surface] = None

    @property
    def engine_power(self) -> EnginePower:
        return self._engine_power

    @property
    def tyre_type(self) -> TyreType:
        return self._tyre_type

    @property
    def surface(self) -> pygame.Surface:
        if self._config_surface is None:
            self._config_surface = self._render()
        return self._config_surface

    def _render(self) -> pygame.Surface:
        pygame.font.init()
        font = pygame.font.Font(pygame.font.get_default_font(), 9)
        power_color = (
            (255, 0, 0)
            if self.engine_power == EnginePower.HIGH
            else (
                (255, 200, 0)
                if self.engine_power == EnginePower.MEDIUM
                else (0, 255, 255)
            )
        )
        tyre_color = (
            (255, 0, 0)
            if self.tyre_type == TyreType.SOFT
            else ((255, 200, 0) if self.tyre_type == TyreType.MEDIUM else (0, 255, 255))
        )
        power_desc_text = font.render(f"Power:", True, (255, 255, 255), None)
        power_text = font.render(self.engine_power.name, True, power_color, None)
        tyre_desc_text = font.render(f"Tyres:", True, (255, 255, 255), None)
        tyre_text = font.render(self.tyre_type.name, True, tyre_color, None)
        power_text_offset = power_desc_text.get_width() + 10
        tyre_desc_text_offset = power_text_offset + power_text.get_width() + 10
        tyre_text_offset = tyre_desc_text_offset + tyre_desc_text.get_width() + 10
        config_text_surf = pygame.Surface(
            (tyre_text_offset + tyre_text.get_width(), tyre_text.get_height()),
            pygame.SRCALPHA,
        )
        config_text_surf.blits(
            [
                (power_desc_text, (0, 0)),
                (power_text, (power_text_offset, 0)),
                (tyre_desc_text, (tyre_desc_text_offset, 0)),
                (tyre_text, (tyre_text_offset, 0)),
            ],
        )
        return config_text_surf

    def to_dict(self) -> dict:
        return {
            "engine_power": self.engine_power.name,
            "tyre_type": self.tyre_type.name,
        }

    def __str__(self) -> str:
        return f"CarConfig(engine_power={self.engine_power.name}, tyre_type={self.tyre_type.name})"


class _Car:

    def __init__(
        self,
        id: int,
        config: CarConfig = CarConfig.default(),
    ):
        self.active = False
        self.car = None
        self.reward = 0.0
        self.prev_reward = 0.0
        self.terminated = False
        self.truncated = False
        self.tiles_visited = set()
        self.fuel_spent = 0.0
        self.lap_count: int = 0
        self.count = id
        self.id = f"car_{self.count}"
        self._config = config
        self.friction_limit = 1000000 * CAR_SIZE * CAR_SIZE * config.tyre_type.value
        self.engine_power = 100000000 * CAR_SIZE * CAR_SIZE * config.engine_power.value

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
    def config(self) -> CarConfig:
        return self._config

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
            "config": self.config.to_dict(),
        }

    def get_translation(
        self, zoom: float, surface_width: int, surface_height: int
    ) -> tuple[float, float]:
        scroll_x = -(self.position[0]) * zoom
        scroll_y = -(self.position[1]) * zoom
        trans = pygame.math.Vector2((scroll_x, scroll_y)).rotate_rad(-self.angle)
        trans = (surface_width / 2 + trans[0], surface_height / 4 + trans[1])
        return trans

    def reset(self, world, track):
        beta = track[0][1]
        x0 = track[0][2]
        y0 = track[0][3]
        x = (
            x0
            + (-1) ** self.count * 2.5 * math.cos(beta)
            - self.count * 5.0 * math.sin(beta)
        )
        y = (
            y0
            + (-1) ** self.count * 2.5 * math.sin(beta)
            + self.count * 5.0 * math.cos(beta)
        )
        self.car = Car(world, beta, x, y)
        self.car.hull.userData = self
        self.reward = 0.0
        self.terminated = False
        self.truncated = False
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
        for w in self.wheels:
            # Steer each wheel
            dir = np.sign(w.steer - w.joint.angle)
            val = abs(w.steer - w.joint.angle)
            w.joint.motorSpeed = dir * min(50.0 * val, 3.0)

            # Position => friction_limit
            grass = True
            friction_limit = self.friction_limit * 0.6  # Grass friction if no tile
            for tile in w.tiles:
                friction_limit = max(
                    friction_limit, self.friction_limit * tile.road_friction
                )
                grass = False

            # Force
            forw = w.GetWorldVector((0, 1))
            side = w.GetWorldVector((1, 0))
            v = w.linearVelocity
            vf = forw[0] * v[0] + forw[1] * v[1]  # forward speed
            vs = side[0] * v[0] + side[1] * v[1]  # side speed

            # add small coef not to divide by zero
            w.omega += (
                dt
                * self.engine_power
                * w.gas
                / WHEEL_MOMENT_OF_INERTIA
                / (abs(w.omega) + 5.0)
            )
            self.fuel_spent += dt * self.engine_power * w.gas

            if w.brake >= 0.9:
                w.omega = 0
            elif w.brake > 0:
                BRAKE_FORCE = 15  # radians per second
                dir = -np.sign(w.omega)
                val = BRAKE_FORCE * w.brake
                if abs(val) > abs(w.omega):
                    val = abs(w.omega)  # low speed => same as = 0
                w.omega += dir * val
            w.phase += w.omega * dt

            vr = w.omega * w.wheel_rad  # rotating wheel speed
            f_force = -vf + vr  # force direction is direction of speed difference
            p_force = -vs

            # Physically correct is to always apply friction_limit until speed is equal.
            # But dt is finite, that will lead to oscillations if difference is already near zero.

            # Random coefficient to cut oscillations in few steps (have no effect on friction_limit)
            f_force *= 205000 * CAR_SIZE * CAR_SIZE
            p_force *= 205000 * CAR_SIZE * CAR_SIZE
            force = np.sqrt(np.square(f_force) + np.square(p_force))

            # Skid trace
            if abs(force) > 2.0 * friction_limit:
                if (
                    w.skid_particle
                    and w.skid_particle.grass == grass
                    and len(w.skid_particle.poly) < 30
                ):
                    w.skid_particle.poly.append((w.position[0], w.position[1]))
                elif w.skid_start is None:
                    w.skid_start = w.position
                else:
                    w.skid_particle = self.car._create_particle(
                        w.skid_start, w.position, grass
                    )
                    w.skid_start = None
            else:
                w.skid_start = None
                w.skid_particle = None

            if abs(force) > friction_limit:
                f_force /= force
                p_force /= force
                force = friction_limit  # Correct physics here
                f_force *= force
                p_force *= force

            w.omega -= dt * f_force * w.wheel_rad / WHEEL_MOMENT_OF_INERTIA

            w.ApplyForceToCenter(
                (
                    p_force * side[0] + f_force * forw[0],
                    p_force * side[1] + f_force * forw[1],
                ),
                True,
            )

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

    def draw_tyre_marks(
        self,
        surface: pygame.Surface,
        zoom: float,
        angle: float,
        trans: tuple[float, float],
    ):
        for p in self.car.particles:
            poly = [pygame.math.Vector2(c).rotate_rad(angle) for c in p.poly]
            poly = [
                (
                    coords[0] * zoom + trans[0],
                    coords[1] * zoom + trans[1],
                )
                for coords in poly
            ]
            pygame.draw.lines(
                surface, color=p.color, points=poly, width=2, closed=False
            )

    def draw_number(
        self,
        surface: pygame.Surface,
        zoom: float,
        view_trans: tuple[float, float],
        view_angle: float,
    ):
        font_size = int(1.6 * zoom)
        font = pygame.font.Font(pygame.font.get_default_font(), font_size)
        number_surface = font.render(str(self.count), True, (255, 255, 255))
        number_surface = pygame.transform.rotate(
            number_surface, np.rad2deg(self.angle + view_angle)
        )
        number_surface = pygame.transform.flip(number_surface, False, True)

        pos_offset = pygame.math.Vector2(0, -0.6).rotate_rad(self.angle)
        pos_number = pygame.math.Vector2(self.position) + pos_offset
        pos_number = pos_number.rotate_rad(view_angle)
        number_text_rect = number_surface.get_rect()
        number_text_rect.center = (
            pos_number[0] * zoom + view_trans[0],
            pos_number[1] * zoom + view_trans[1],
        )
        surface.blit(number_surface, number_text_rect)

    def render_indicators(self, render_mode, idx, surface, W, H):
        s = W / 40.0
        h = H / 40.0
        H -= idx * 5 * h
        indicator_box_surf = pygame.Surface((W, 5 * h), pygame.SRCALPHA)
        alpha = 128 if render_mode == "human" else 255
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

        true_speed = np.sqrt(np.square(self.velocity[0]) + np.square(self.velocity[1]))

        # simple wrapper to render if the indicator value is above a threshold
        def render_if_min(value, points, color):
            if abs(value) > 1e-4:
                pygame.draw.polygon(surface, points=points, color=color)

        render_if_min(true_speed, vertical_ind(5, 0.02 * true_speed), (255, 255, 255))
        # ABS sensors
        render_if_min(
            self.wheels[0].omega,
            vertical_ind(7, 0.01 * self.wheels[0].omega),
            (0, 0, 255),
        )
        render_if_min(
            self.wheels[1].omega,
            vertical_ind(8, 0.01 * self.wheels[1].omega),
            (0, 0, 255),
        )
        render_if_min(
            self.wheels[2].omega,
            vertical_ind(9, 0.01 * self.wheels[2].omega),
            (51, 0, 255),
        )
        render_if_min(
            self.wheels[3].omega,
            vertical_ind(10, 0.01 * self.wheels[3].omega),
            (51, 0, 255),
        )

        render_if_min(
            self.wheels[0].joint.angle,
            horiz_ind(20, -10.0 * self.wheels[0].joint.angle),
            (0, 255, 0),
        )
        render_if_min(
            self.angular_velocity,
            horiz_ind(30, -0.8 * self.angular_velocity),
            (255, 0, 0),
        )


class MultiAgentCars:
    def __init__(
        self,
        car_configs: list[CarConfig],
        possible_agent_ids: list[AgentID],
        active_agent_ids: list[AgentID],
    ):
        self._cars: MultiAgentDict = {
            agent: _Car(i, car_configs[i]) for i, agent in enumerate(possible_agent_ids)
        }
        for id in active_agent_ids:
            self._cars.get(id).active = True

    def get(self, agent_id: AgentID) -> _Car:
        return self._cars[agent_id]

    def get_enclosing_rect(self) -> tuple[float, float, float, float]:
        active_cars = self.get_active()
        assert len(active_cars) > 0
        min_x = min(active_cars, key=lambda c: c.position[0]).position[0]
        min_y = min(active_cars, key=lambda c: c.position[1]).position[1]
        max_x = max(active_cars, key=lambda c: c.position[0]).position[0]
        max_y = max(active_cars, key=lambda c: c.position[1]).position[1]
        return (min_x, min_y, max_x - min_x, max_y - min_y)

    def reset(self, world, track):
        for car in self._cars.values():
            car.reset(world, track)

    def apply_actions(self, actions: Optional[MultiAgentDict] = None):
        assert len(self.get_active()) > 0
        if actions is not None:
            for agent, car in self._cars.items():
                action = actions.get(agent, None)
                if car.active and action is not None:
                    car.apply_action(action)
        for car in self._cars.values():
            if car.active:
                car.step(1.0 / FPS)

    def step(self, track, actions: MultiAgentDict, observations: MultiAgentDict):
        step_rewards = {agent: 0.0 for agent in self._cars.keys()}
        info_d = {}
        obs_d = {}
        rew_d = {}
        terminated_d = {}
        truncated_d = {}
        for agent, car in self._cars.items():
            car: _Car
            if car.active:
                if not car.terminated and not car.truncated:
                    info_d[agent] = {}
                    car.reward -= 0.1
                    # We actually don't want to count fuel spent, we want car to be faster.
                    # self.reward -=  10 * self.car.fuel_spent / ENGINE_POWER
                    car.fuel_spent = 0.0
                    step_rewards[agent] = car.reward - car.prev_reward
                    car.prev_reward = car.reward
                    if len(car.tiles_visited) == len(track) or car.lap_count >= 1:
                        # Termination due to finishing lap
                        car.terminated = True
                        info_d[agent]["lap_finished"] = True
                    x, y = car.position
                    if abs(x) > PLAYFIELD or abs(y) > PLAYFIELD:
                        car.terminated = True
                        info_d[agent]["lap_finished"] = False
                        step_rewards[agent] = -100
                    obs_d[agent] = observations[agent]
                    rew_d[agent] = step_rewards[agent]
                else:
                    obs_d[agent] = None
                terminated_d[agent] = car.terminated
                truncated_d[agent] = car.truncated
            else:
                obs_d[agent] = None
                rew_d[agent] = 0.0
                terminated_d[agent] = True
                truncated_d[agent] = True
                info_d[agent] = {}
        terminated_d["__all__"] = all(terminated_d.values())
        truncated_d["__all__"] = all(truncated_d.values())
        return obs_d, rew_d, terminated_d, truncated_d, info_d

    def destroy(self):
        for car in self._cars.values():
            car.destroy()

    def draw(
        self,
        surface: pygame.Surface,
        zoom: float,
        trans: tuple[float, float],
        angle: float,
        tyre_marks: bool = True,
        draw_number: bool = False,
    ):
        if tyre_marks:
            for car in self._cars.values():
                if car.active:
                    car.draw_tyre_marks(surface, zoom, angle, trans)
        for car in self._cars.values():
            if car.active:
                car.draw(surface, zoom, trans, angle, False)
                if draw_number:
                    car.draw_number(surface, zoom, trans, angle)

    def render_indicators(
        self,
        render_mode: str,
        surface: Union[pygame.Surface, MultiAgentDict],
        W: Union[int, float],
        H: Union[int, float],
    ):
        if isinstance(surface, dict):
            for agent, surf in surface.items():
                car = self.get(agent)
                if car.active:
                    car.render_indicators(render_mode, 0, surf, W, H)
        else:
            for idx, car in enumerate(self._cars.values()):
                if car.active:
                    car.render_indicators(render_mode, idx, surface, W, H)

    def get_all(self) -> list[_Car]:
        return list(self._cars.values())

    def get_active(self) -> list[_Car]:
        return [car for car in self._cars.values() if car.active]


class LeaderBoard:
    def __init__(self, cars: MultiAgentCars):
        self.cars = cars
        self.leaderboard = []

    def update(self):
        self.leaderboard = sorted(
            self.cars.get_active(),
            key=lambda car: (
                car.lap_count,
                max(list(car.tiles_visited)) if len(car.tiles_visited) > 0 else -1,
            ),
            reverse=True,
        )

    def get_position(self, car: _Car) -> int:
        for idx, c in enumerate(self.leaderboard):
            if c == car:
                return idx
        return -1

    def __iter__(self):
        return iter(self.leaderboard)

    def __getitem__(self, index: int) -> _Car:
        return self.leaderboard[index]
