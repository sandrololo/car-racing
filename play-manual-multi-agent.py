import pygame
import numpy as np
from environments.multiagent import MultiAgentCarRacingEnv
from environments.multiagent.cars import CarConfig, EnginePower, TyreType

NUM_CARS = 8

CAR_CONFIGS = [
    CarConfig(EnginePower.HIGH, TyreType.SOFT),
    CarConfig(EnginePower.LOW, TyreType.HARD),
    CarConfig(EnginePower.LOW, TyreType.HARD),
    CarConfig(EnginePower.LOW, TyreType.HARD),
    CarConfig(EnginePower.MEDIUM, TyreType.MEDIUM),
    CarConfig(EnginePower.MEDIUM, TyreType.MEDIUM),
    CarConfig(EnginePower.HIGH, TyreType.SOFT),
    CarConfig(EnginePower.HIGH, TyreType.SOFT),
]

if __name__ == "__main__":
    a = np.array([0.0, 0.0, 0.0])

    def register_input():
        global quit, restart
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    a[0] = -1.0
                if event.key == pygame.K_RIGHT:
                    a[0] = +1.0
                if event.key == pygame.K_UP:
                    a[1] = +1.0
                if event.key == pygame.K_DOWN:
                    a[2] = +0.8  # set 1.0 for wheels to block to zero rotation
                if event.key == pygame.K_RETURN:
                    restart = True
                if event.key == pygame.K_ESCAPE:
                    quit = True

            if event.type == pygame.KEYUP:
                if event.key == pygame.K_LEFT:
                    a[0] = 0
                if event.key == pygame.K_RIGHT:
                    a[0] = 0
                if event.key == pygame.K_UP:
                    a[1] = 0
                if event.key == pygame.K_DOWN:
                    a[2] = 0

            if event.type == pygame.QUIT:
                quit = True

    env = MultiAgentCarRacingEnv(
        config={
            "first_tile_visitor_reward_factor": 1.5,
            "render_mode": "human",
            "num_cars": NUM_CARS,
            "car_configs": CAR_CONFIGS,
        }
    )

    actions = {}
    quit = False
    while not quit:
        env.reset()
        total_reward = 0.0
        steps = 0
        restart = False
        while True:
            register_input()
            actions["car_0"] = a
            for i in range(1, NUM_CARS):
                actions[f"car_{i}"] = np.random.rand(5) * 2 - 1
            s, r, terminated, truncated, info = env.step(actions)
            if steps % 200 == 0 or terminated["__all__"] or truncated["__all__"]:
                print(f"step {steps}")
                for car in env.cars.get_active():
                    print(f"reward={car.reward}, pos={car.car.hull.position}")
            steps += 1
            if terminated["__all__"] or truncated["__all__"] or restart or quit:
                break
    env.close()
