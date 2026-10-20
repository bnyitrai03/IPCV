"""Tiny CPU opponent used for the menu attract mode, the automated smoke
test and single-player practice (``--cpu``). Not part of the CV pipeline."""
from __future__ import annotations

import random

from game.controls import PlayerInput
from game.fighter import Fighter, FighterState


class SimpleAI:
    def __init__(self, seed: int = 0, aggression: float = 0.5) -> None:
        self.rng = random.Random(seed)
        self.aggression = aggression
        self.cooldown = 0.0
        self.block_time = 0.0

    def decide(self, me: Fighter, opp: Fighter, dt: float) -> PlayerInput:
        self.cooldown -= dt
        self.block_time -= dt
        inp = PlayerInput()
        dist = abs(opp.x - me.x)
        toward = 1.0 if opp.x > me.x else -1.0
        if self.block_time > 0:
            inp.block = True
            return inp
        if opp.state == FighterState.PUNCH and dist < 150 and self.rng.random() < 0.35:
            self.block_time = 0.35
            inp.block = True
            return inp
        if dist > 150:
            inp.move = toward
        elif dist < 100 and self.rng.random() < 0.02:
            inp.move = -toward
        if self.cooldown <= 0:
            if me.energy >= me.cfg.special_cost and dist > 250 and self.rng.random() < 0.3:
                inp.special = True
                self.cooldown = 1.0
            elif dist < 145 and self.rng.random() < self.aggression:
                if self.rng.random() < 0.5:
                    inp.punch_left = True
                else:
                    inp.punch_right = True
                self.cooldown = self.rng.uniform(0.25, 0.7)
            else:
                self.cooldown = 0.15
        return inp
