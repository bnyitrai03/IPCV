import logging
import threading
from rowing_game.direction import Direction

logger = logging.getLogger("ble_message")

class Message:
    def __init__(self):
        self.LEFT = False
        self.RIGHT = False
        self.CONFIGURED = threading.Event()
        
    def get_direction(self) -> Direction:
        direction = "STOP"
        if self.LEFT and self.RIGHT:
            direction = "STRAIGHT"
        elif self.RIGHT:
            direction = "LEFT"
        elif self.LEFT:
            direction = "RIGHT"
        return Direction(self.LEFT, self.RIGHT, direction)