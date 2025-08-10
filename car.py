import math
import pygame

class Car:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.angle = 0  # in Grad, 0 = nach rechts
        self.speed = 0
        self.max_speed = 5
        self.length = 40
        self.width = 20

    def update(self):
        # Auto bewegt sich basierend auf speed & Winkel
        rad = math.radians(self.angle)
        self.x += self.speed * math.cos(rad)
        self.y += self.speed * math.sin(rad)

    def accelerate(self, amount):
        self.speed += amount
        self.speed = max(-self.max_speed, min(self.speed, self.max_speed))

    def turn(self, angle):
        self.angle += angle
        self.angle %= 360

    def draw(self, screen):
        # Rechteck als Auto zeichnen, rotiert nach self.angle
        rect = pygame.Rect(0, 0, self.length, self.width)
        rect.center = (self.x, self.y)
        rotated_surf = pygame.Surface((self.length, self.width), pygame.SRCALPHA)
        rotated_surf.fill((255, 0, 0))
        rotated_surf = pygame.transform.rotate(rotated_surf, -self.angle)
        rotated_rect = rotated_surf.get_rect(center=rect.center)
        screen.blit(rotated_surf, rotated_rect.topleft)
