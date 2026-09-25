import pygame
from .settings import *

class Canoe:
    def __init__(self, x, y, boat_img=None):
        self.x = x
        self.y = y
        self.width = CANOE_WIDTH
        self.height = CANOE_HEIGHT
        self.speed = CANOE_SPEED
        self.boat_img = boat_img

    def move(self, direction):
        """Move canoe based on paddle input"""
        if direction == "LEFT":
            self.x -= self.speed
        elif direction == "RIGHT":
            self.x += self.speed
        # No boundary check here - will be checked against river bounds in game

    def draw(self, screen):
        if self.boat_img:
            # Draw boat image
            screen.blit(self.boat_img, (self.x, self.y))
        else:
            # Fallback to original drawing
            pygame.draw.rect(screen, (139, 69, 19),
                            (self.x, self.y, self.width, self.height))
            pygame.draw.rect(screen, BLACK,
                            (self.x, self.y, self.width, self.height), 3)
            points = [
                (self.x, self.y),
                (self.x + self.width, self.y),
                (self.x + self.width // 2, self.y - 15)
            ]
            pygame.draw.polygon(screen, (139, 69, 19), points)
            pygame.draw.polygon(screen, BLACK, points, 3)

    def get_rect(self):
        return pygame.Rect(self.x, self.y, self.width, self.height)

    def get_collision_rect(self):
        """Smaller hitbox for more accurate collision detection"""
        # Shrink hitbox by 30% on all sides
        margin_x = self.width * 0.15
        margin_y = self.height * 0.15
        return pygame.Rect(self.x + margin_x, self.y + margin_y,
                          self.width - margin_x * 2, self.height - margin_y * 2)
