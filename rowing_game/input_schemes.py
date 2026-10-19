import pygame
from rowing_game.ble_message import Message
from rowing_game.direction import Direction

class InputScheme:
    def get_direction(self) -> Direction:
        pass

    def get_debug_info(self) -> str | None:
        return None

    def get_preview(self):
        return None

class YoloScheme(InputScheme):
    def __init__(self, message: Message):
        self.message = message
        message.CONFIGURED.set()

    def get_direction(self) -> Direction:
        return self.message.get_direction()

    def get_debug_info(self) -> str:
        if not self.message.camera_ok:
            return "Camera not connected"
        return f"Cam {self.message.fps:.0f} FPS | YOLO {self.message.inference_ms:.0f} ms"

    def get_preview(self):
        return self.message.preview

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