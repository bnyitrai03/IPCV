import pygame
from rowing_game.ble_message import Message
from rowing_game.direction import Direction

class InputScheme:
    def get_direction(self) -> Direction:
        pass

class BLEScheme(InputScheme):
    def __init__(self, ble_message: Message):
        self.ble_message = ble_message
        ble_message.CONFIGURED.set()

    def get_direction(self) -> Direction:
        return self.ble_message.get_direction()
    
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