"""Input contract between the webcam adapter and the game logic.

The webcam session produces a :class:`PlayerInput` per logic tick, so the
game logic never knows how an action was recognised.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PlayerInput:
    """Actions requested by one player during one logic tick."""

    move: float = 0.0              # -1 (left) .. +1 (right), screen space
    block: bool = False            # held action
    hook_left: bool = False        # one-shot events: bent-arm swing across the body
    hook_right: bool = False
    uppercut_left: bool = False    # rising punch, fist ends above the head
    uppercut_right: bool = False
    special: bool = False
    # perf_counter() timestamps of the camera frames that produced each
    # one-shot event; used to measure command latency.
    source_ts: dict[str, float] = field(default_factory=dict)

    def merge(self, other: "PlayerInput") -> "PlayerInput":
        """Combine two inputs (e.g. held state + newly recognised gestures)."""
        move = self.move if abs(self.move) >= abs(other.move) else other.move
        ts = dict(other.source_ts)
        ts.update(self.source_ts)
        return PlayerInput(
            move=max(-1.0, min(1.0, move)),
            block=self.block or other.block,
            hook_left=self.hook_left or other.hook_left,
            hook_right=self.hook_right or other.hook_right,
            uppercut_left=self.uppercut_left or other.uppercut_left,
            uppercut_right=self.uppercut_right or other.uppercut_right,
            special=self.special or other.special,
            source_ts=ts,
        )


# Order = priority when several are requested in the same tick.
ATTACK_ACTIONS = ("uppercut_left", "uppercut_right", "hook_left", "hook_right")
ONE_SHOT_ACTIONS = ATTACK_ACTIONS + ("special",)


def scancode(name: str) -> int:
    import pygame
    return getattr(pygame, f"KSCAN_{name}")


def key_is(event, name: str) -> bool:
    """True if a KEYDOWN/KEYUP event is the physical key ``name`` (layout independent)."""
    import pygame
    if getattr(event, "scancode", None) == scancode(name):
        return True
    return event.key == pygame.key.key_code(name.lower()) if len(name) == 1 else False
