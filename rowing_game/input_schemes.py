import pygame
from rowing_game.direction import Direction

class InputScheme:
    def get_direction(self) -> Direction:
        pass

class KeyboardScheme(InputScheme):
    def __init__(self, left_key: int, right_key:int):
        self.left_key = left_key
        self.right_key = right_key

    def get_direction(self) -> Direction:
        keys = pygame.key.get_pressed()
        direction = "STOP"
        if keys[self.left_key] & keys[self.right_key]:
            direction = "STRAIGHT"
        elif keys[self.left_key]:
            direction = "RIGHT"
        elif keys[self.right_key]:
            direction = "LEFT"
        return Direction(keys[self.left_key], keys[self.right_key], direction)