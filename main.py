import pygame
from car import Car

def main():
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("Self-Driving Car")
    clock = pygame.time.Clock()

    car = Car(400, 300)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            car.accelerate(0.1)
        elif keys[pygame.K_DOWN]:
            car.accelerate(-0.1)
        else:
            # Geschwindigkeit langsam abbauen
            car.accelerate(-0.05 if car.speed > 0 else 0.05)

        if keys[pygame.K_LEFT]:
            car.turn(-3)
        if keys[pygame.K_RIGHT]:
            car.turn(3)

        car.update()

        screen.fill((30, 30, 30))
        car.draw(screen)
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()
