from rowing_game.input_schemes import InputScheme
from rowing_game.direction import Direction

class Player:
    def __init__(self, input_scheme: InputScheme, id = 0):
        self.input_scheme = input_scheme
        self.id = id
        self.score = 0

    def get_direction(self) -> Direction:
        return self.input_scheme.get_direction()