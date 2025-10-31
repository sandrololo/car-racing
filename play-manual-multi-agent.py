import pygame
import numpy as np
from environments import MultiAgentCarRacingEnv

NUM_CARS = 8


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
        config={"render_mode": "human", "num_agents": NUM_CARS}
    )

    actions = [None for _ in range(NUM_CARS)]
    quit = False
    while not quit:
        env.reset()
        total_reward = 0.0
        steps = 0
        restart = False
        while True:
            register_input()
            actions[0] = a
            for i in range(1, NUM_CARS):
                actions[i] = np.random.rand(5) * 2 - 1
            s, r, terminated, truncated, info = env.step(actions)
            if steps % 200 == 0 or terminated or truncated:
                print(f"step {steps}")
                for car in env.cars:
                    print(f"reward={car.reward}, pos={car.car.hull.position}")
            steps += 1
            if terminated or truncated or restart or quit:
                break
    env.close()
