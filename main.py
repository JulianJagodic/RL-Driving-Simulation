import pygame
import numpy as np
from car import Car
from rl_agent import QAgent

# Define possible actions: [accelerate, decelerate, turn left, turn right, do nothing]
ACTIONS = [
    [0.1, 0, 0],    # Accelerate
    [-0.1, 0, 0],   # Decelerate
    [0, -3, 0],     # Turn left
    [0, 3, 0],      # Turn right
    [0, 0, 0]       # Do nothing
]

def get_state(car):
    # Example state: [x, y, speed, angle]
    return np.array([car.x / 800, car.y / 600, car.speed / 10, car.angle / 360], dtype=np.float32)

def step(car, action, road_surface, lake_surface):
    # Apply action
    accel, turn, _ = ACTIONS[action]
    car.accelerate(accel)
    if turn != 0:
        car.turn(turn)
    car.update()

    # Restrict car to stay inside window
    car.x = max(0, min(car.x, 799))
    car.y = max(0, min(car.y, 599))

    # Get new state
    state = get_state(car)
    car_pos = (int(car.x), int(car.y))
    # Prevent out-of-bounds pixel access
    if not (0 <= car_pos[0] < 800 and 0 <= car_pos[1] < 600):
        return state, -5, True

    on_road = road_surface.get_at(car_pos)[3] > 0
    in_lake = lake_surface.get_at(car_pos)[3] > 0

    # Finish line coordinates (vertical line)
    finish_line_x = 400
    finish_line_y1 = 100
    finish_line_y2 = 200
    finish_line_thickness = 8

    # Check if car crosses finish line
    crossed_finish = (
        abs(car.x - finish_line_x) < finish_line_thickness and
        finish_line_y1 <= car.y <= finish_line_y2
    )

    # Reward logic
    reward = -0.01  # Small negative for time
    done = False
    if in_lake:
        reward = -10
        done = True
    elif not on_road:
        reward = -1
    elif crossed_finish:
        # Check direction: angle near 90 (down) is correct, up is wrong
        if 60 < car.angle % 360 < 120:
            reward = 10
            done = True
        else:
            reward = -5
            done = True

    return state, reward, done

def main():
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("Self-Driving Car RL")
    clock = pygame.time.Clock()

    state_dim = 4
    action_dim = len(ACTIONS)
    agent = QAgent(state_dim, action_dim)

    # Track geometry
    track_outer = [
        (150, 100), (650, 100), (750, 300), (650, 500), (150, 500), (50, 300)
    ]
    track_inner = [
        (250, 200), (550, 200), (600, 300), (550, 400), (250, 400), (200, 300)
    ]
    lake = [
        (300, 210), (500, 210), (530, 220), (550, 235), (590, 300), (500, 370), (425, 388), (380, 340), (260, 350), (230, 300), (250, 260)
    ]

    # Collision surfaces
    road_surface = pygame.Surface((800, 600), pygame.SRCALPHA)
    pygame.draw.polygon(road_surface, (255, 255, 255, 255), track_outer)
    pygame.draw.polygon(road_surface, (0, 0, 0, 0), track_inner)
    pygame.draw.polygon(road_surface, (0, 0, 0, 0), lake)

    lake_surface = pygame.Surface((800, 600), pygame.SRCALPHA)
    pygame.draw.polygon(lake_surface, (255, 255, 255, 255), lake)

    episodes = 1000
    for ep in range(episodes):
        car = Car(425, 150)
        state = get_state(car)
        total_reward = 0
        done = False
        steps = 0

        visualize = (ep % 200 == 0)  # Only visualize every 200th episode

        while not done and steps < 1000:
            # RL agent chooses action
            action = agent.select_action(state)
            next_state, reward, done = step(car, action, road_surface, lake_surface)
            agent.store((state, action, reward, next_state, done))
            agent.train()

            state = next_state
            total_reward += reward
            steps += 1

            # Draw everything only if visualize is True
            if visualize:
                screen.fill((34, 139, 34))
                pygame.draw.polygon(screen, (128, 128, 128), track_outer)
                pygame.draw.polygon(screen, (34, 139, 34), track_inner)
                pygame.draw.polygon(screen, (70, 130, 180), lake)
                pygame.draw.line(screen, (255, 255, 255), (400, 100), (400, 200), 8)
                car.draw(screen)
                pygame.display.flip()
                clock.tick(120)  # Lower tick for visualization
            else:
                # Run as fast as possible (no delay)
                pass

        print(f"Episode {ep+1}: Total Reward = {total_reward}")

    pygame.quit()

if __name__ == "__main__":
    main()
