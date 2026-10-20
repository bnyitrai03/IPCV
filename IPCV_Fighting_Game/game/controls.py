"""Input abstraction shared by keyboard and webcam controllers.

Both controllers produce a :class:`PlayerInput` per logic tick, so the
game logic never knows (or cares) where an action came from.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PlayerInput:
    """Actions requested by one player during one logic tick."""

    move: float = 0.0              # -1 (left) .. +1 (right), screen space
    block: bool = False            # held action
    punch_left: bool = False       # one-shot events
    punch_right: bool = False
    special: bool = False
    # perf_counter() timestamps of the camera frames that produced each
    # one-shot event; used to measure command latency. Empty for keyboard.
    source_ts: dict[str, float] = field(default_factory=dict)

    def merge(self, other: "PlayerInput") -> "PlayerInput":
        """Combine two inputs (e.g. keyboard + webcam)."""
        move = self.move if abs(self.move) >= abs(other.move) else other.move
        ts = dict(other.source_ts)
        ts.update(self.source_ts)
        return PlayerInput(
            move=max(-1.0, min(1.0, move)),
            block=self.block or other.block,
            punch_left=self.punch_left or other.punch_left,
            punch_right=self.punch_right or other.punch_right,
            special=self.special or other.special,
            source_ts=ts,
        )

    @property
    def any_attack(self) -> bool:
        return self.punch_left or self.punch_right or self.special


# Keyboard bindings, by pygame key name (layout independent of macOS/Windows).
KEYMAP = {
    0: {"left": "a", "right": "d", "block": "s",
        "punch_left": "f", "punch_right": "g", "special": "r"},
    1: {"left": "left", "right": "right", "block": "down",
        "punch_left": "k", "punch_right": "l", "special": "i"},
}


class KeyboardController:
    """Translates pygame keyboard state/events into PlayerInputs."""

    def __init__(self) -> None:
        import pygame  # local import keeps game logic importable without a display

        self._pg = pygame
        self._codes = {
            pid: {action: pygame.key.key_code(name) for action, name in binds.items()}
            for pid, binds in KEYMAP.items()
        }
        self._pending: dict[int, PlayerInput] = {0: PlayerInput(), 1: PlayerInput()}

    def handle_event(self, event) -> None:
        if event.type != self._pg.KEYDOWN:
            return
        for pid, codes in self._codes.items():
            pending = self._pending[pid]
            if event.key == codes["punch_left"]:
                pending.punch_left = True
            elif event.key == codes["punch_right"]:
                pending.punch_right = True
            elif event.key == codes["special"]:
                pending.special = True

    def poll(self, player_id: int) -> PlayerInput:
        keys = self._pg.key.get_pressed()
        codes = self._codes[player_id]
        pending = self._pending[player_id]
        move = float(keys[codes["right"]]) - float(keys[codes["left"]])
        result = PlayerInput(
            move=move,
            block=bool(keys[codes["block"]]),
            punch_left=pending.punch_left,
            punch_right=pending.punch_right,
            special=pending.special,
        )
        self._pending[player_id] = PlayerInput()
        return result
