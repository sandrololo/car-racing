import math
from typing import Union
import numpy as np
from gymnasium.envs.box2d.car_dynamics import Car
from gymnasium.error import DependencyNotInstalled

try:
    # As pygame is necessary for using the environment (reset and step) even without a render mode
    #   therefore, pygame is a necessary import for the environment.
    import pygame
    from pygame import gfxdraw
except ImportError as e:
    raise DependencyNotInstalled(
        'pygame is not installed, run `pip install "gymnasium[box2d]"`'
    ) from e

from .config import FPS, PLAYFIELD, WINDOW_H, WINDOW_W


class _CarInfo:
    car_count = 0

    def __init__(self):
        self.car = None
        self.reward = 0.0
        self.prev_reward = 0.0
        self.terminated = False
        self.truncated = False
        self.tiles_visited = set()
        self.fuel_spent = 0.0
        self.lap_count: int = 0
        self.count = _CarInfo.car_count
        self.id = f"car_{self.count}"
        _CarInfo.car_count += 1

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

    def get_translation(self, zoom: float) -> tuple[float, float]:
        scroll_x = -(self.position[0]) * zoom
        scroll_y = -(self.position[1]) * zoom
        trans = pygame.math.Vector2((scroll_x, scroll_y)).rotate_rad(-self.angle)
        trans = (WINDOW_W / 2 + trans[0], WINDOW_H / 4 + trans[1])
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
    def __init__(self, num_cars: int):
        self._cars: list[_CarInfo] = [_CarInfo() for _ in range(num_cars)]

    def get_leader(self) -> _CarInfo:
        assert len(self._cars) > 0
        return max(self._cars, key=lambda c: len(c.tiles_visited))

    def get_last(self) -> _CarInfo:
        assert len(self._cars) > 0
        return min(self._cars, key=lambda c: len(c.tiles_visited))

    def get_enclosing_rect(self) -> tuple[float, float, float, float]:
        assert len(self._cars) > 0
        min_x = min(self._cars, key=lambda c: c.position[0]).position[0]
        min_y = min(self._cars, key=lambda c: c.position[1]).position[1]
        max_x = max(self._cars, key=lambda c: c.position[0]).position[0]
        max_y = max(self._cars, key=lambda c: c.position[1]).position[1]
        return (min_x, min_y, max_x - min_x, max_y - min_y)

    def reset(self, world, track):
        for car in self._cars:
            car.reset(world, track)

    def apply_actions(self, actions: dict):
        assert len(self._cars) > 0
        if actions is not None:
            for car in self._cars:
                act = actions.get(car.id, None)
                if act is not None:
                    car.apply_action(act)
        for car in self._cars:
            car.step(1.0 / FPS)

    def step(self, track, actions: Union[dict, None], observations):
        step_rewards = [0.0 for _ in range(len(self._cars))]
        info_d = {}
        obs_d = {}
        rew_d = {}
        terminated_d = {}
        truncated_d = {}
        if actions is not None:  # First step without action, called from reset()
            for i, car in enumerate(self._cars):
                if not car.terminated or car.truncated:
                    info_d[car.id] = {}
                    car.reward -= 0.1
                    # We actually don't want to count fuel spent, we want car to be faster.
                    # self.reward -=  10 * self.car.fuel_spent / ENGINE_POWER
                    car.fuel_spent = 0.0
                    step_rewards[i] = car.reward - car.prev_reward
                    car.prev_reward = car.reward
                    if len(car.tiles_visited) == len(track) or car.lap_count >= 1:
                        # Termination due to finishing lap
                        car.terminated = True
                        info_d[car.id]["lap_finished"] = True
                    x, y = car.position
                    if abs(x) > PLAYFIELD or abs(y) > PLAYFIELD:
                        car.terminated = True
                        info_d[car.id]["lap_finished"] = False
                        step_rewards[i] = -100
                    obs_d[car.id] = observations[i]
                    rew_d[car.id] = step_rewards[i]
                    terminated_d[car.id] = car.terminated
                    truncated_d[car.id] = car.truncated
        terminated_d["__all__"] = all(terminated_d.values())
        truncated_d["__all__"] = all(truncated_d.values())
        return obs_d, rew_d, terminated_d, truncated_d, info_d

    def destroy(self):
        assert len(self._cars) > 0
        for car in self._cars:
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
            for car in self._cars:
                car.draw_tyre_marks(surface, zoom, angle, trans)
        for car in self._cars:
            car.draw(surface, zoom, trans, angle, False)
            if draw_number:
                car.draw_number(surface, zoom, trans, angle)

    def render_indicators(
        self,
        render_mode: str,
        surface: Union[pygame.Surface, list[pygame.Surface]],
        W: Union[int, float],
        H: Union[int, float],
    ):
        if isinstance(surface, list):
            for idx, car in enumerate(self._cars):
                self._cars[idx].render_indicators(render_mode, 0, surface[idx], W, H)
        else:
            for idx, car in enumerate(self._cars):
                car.render_indicators(render_mode, idx, surface, W, H)

    def __iter__(self):
        return iter(self._cars)

    def __getitem__(self, index: int) -> _CarInfo:
        return self._cars[index]

    def __len__(self):
        return len(self._cars)
