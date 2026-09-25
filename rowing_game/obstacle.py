import pygame
from .settings import *
class Obstacle:
    def __init__(self, x, y, rock_img=None):
        self.x = x
        self.y = y
        self.width = OBSTACLE_WIDTH
        self.height = OBSTACLE_HEIGHT
        self.rock_img = rock_img

    def update(self, river_scroll_speed=0):
        # Move only with river scroll - obstacles are fixed to the river map
        self.y -= river_scroll_speed

    def draw(self, screen):
        if self.rock_img:
            # Draw rock image
            screen.blit(self.rock_img, (self.x, self.y))
        else:
            # Fallback to original drawing
            pygame.draw.rect(screen, GRAY,
                            (self.x, self.y, self.width, self.height))
            pygame.draw.rect(screen, (50, 50, 50),
                            (self.x, self.y, self.width, self.height), 3)
            # Add some texture lines
            pygame.draw.line(screen, (70, 70, 70),
                            (self.x + 10, self.y + 10),
                            (self.x + 30, self.y + 25), 2)
            pygame.draw.line(screen, (70, 70, 70),
                            (self.x + 50, self.y + 15),
                            (self.x + 70, self.y + 35), 2)

    def get_rect(self):
        return pygame.Rect(self.x, self.y, self.width, self.height)

    def is_off_screen(self):
        return self.y > SCREEN_HEIGHT
