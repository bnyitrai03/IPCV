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


# Order = priority when several are requested in the same tick.
ATTACK_ACTIONS = ("uppercut_left", "uppercut_right", "hook_left", "hook_right")
ONE_SHOT_ACTIONS = ATTACK_ACTIONS + ("special",)
