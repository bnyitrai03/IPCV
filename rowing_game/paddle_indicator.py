import pygame
from .settings import *
class PaddleIndicator:
    def __init__(self, x, y, label):
        self.x = x
        self.y = y
        self.label = label
        self.active = False
        self.width = 80
        self.height = 40
        
    def set_active(self, active):
        self.active = active
        
    def draw(self, screen):
        color = GREEN if self.active else GRAY
        pygame.draw.rect(screen, color, 
                        (self.x, self.y, self.width, self.height))
        pygame.draw.rect(screen, BLACK, 
                        (self.x, self.y, self.width, self.height), 2)
        
        # Draw label
        font = pygame.font.Font(None, 24)
        text = font.render(self.label, True, BLACK)
        text_rect = text.get_rect(center=(self.x + self.width // 2, 
                                          self.y + self.height // 2))
        screen.blit(text, text_rect)

