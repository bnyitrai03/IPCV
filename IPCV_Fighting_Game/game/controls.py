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
    hook_left: bool = False        # one-shot events: bent-arm swing across the body
    hook_right: bool = False
    uppercut_left: bool = False    # rising punch, fist ends above the head
    uppercut_right: bool = False
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
            hook_left=self.hook_left or other.hook_left,
            hook_right=self.hook_right or other.hook_right,
            uppercut_left=self.uppercut_left or other.uppercut_left,
            uppercut_right=self.uppercut_right or other.uppercut_right,
            special=self.special or other.special,
            source_ts=ts,
        )

    @property
    def any_attack(self) -> bool:
        return any(getattr(self, a) for a in ONE_SHOT_ACTIONS)


# Order = priority when several are requested in the same tick.
ATTACK_ACTIONS = ("uppercut_left", "uppercut_right", "hook_left", "hook_right")
ONE_SHOT_ACTIONS = ATTACK_ACTIONS + ("special",)


# Keyboard bindings by *physical key position* (US-QWERTY names). Scancodes are
# used so the controls work with any keyboard layout (Greek, AZERTY, QWERTZ...):
# a layout-dependent keycode would turn the "A" key into e.g. "α" and be ignored.
KEYMAP = {
    0: {"left": "A", "right": "D", "block": "S",
        "hook_left": "F", "hook_right": "G", "uppercut_left": "T", "uppercut_right": "Y",
        "special": "R"},
    1: {"left": "LEFT", "right": "RIGHT", "block": "DOWN",
        "hook_left": "K", "hook_right": "L", "uppercut_left": "O", "uppercut_right": "P",
        "special": "I"},
}


def scancode(name: str) -> int:
    import pygame
    return getattr(pygame, f"KSCAN_{name}")


def key_is(event, name: str) -> bool:
    """True if a KEYDOWN/KEYUP event is the physical key ``name`` (layout independent)."""
    import pygame
    if getattr(event, "scancode", None) == scancode(name):
        return True
    return event.key == pygame.key.key_code(name.lower()) if len(name) == 1 else False


class KeyboardController:
    """Translates pygame key events into PlayerInputs.

    Held keys are tracked from KEYDOWN/KEYUP *scancodes* (physical positions),
    with the layout keycode accepted as a fallback.
    """

    def __init__(self) -> None:
        import pygame  # local import keeps game logic importable without a display

        self._pg = pygame
        self._binds = KEYMAP
        self._held: set[tuple[int, str]] = set()      # (player, action) currently held
        self._pending: dict[int, PlayerInput] = {0: PlayerInput(), 1: PlayerInput()}

    def _actions_for(self, event) -> list[tuple[int, str]]:
        out = []
        for pid, binds in self._binds.items():
            for action, name in binds.items():
                if key_is(event, name):
                    out.append((pid, action))
        return out

    def handle_event(self, event) -> None:
        pg = self._pg
        if event.type in (pg.WINDOWFOCUSLOST,):
            self._held.clear()            # avoid "stuck" keys after alt-tab
            return
        if event.type not in (pg.KEYDOWN, pg.KEYUP):
            return
        for pid, action in self._actions_for(event):
            if event.type == pg.KEYDOWN:
                self._held.add((pid, action))
                if action in ONE_SHOT_ACTIONS:
                    setattr(self._pending[pid], action, True)
            else:
                self._held.discard((pid, action))

    def reset(self) -> None:
        self._held.clear()
        self._pending = {0: PlayerInput(), 1: PlayerInput()}

    def poll(self, player_id: int) -> PlayerInput:
        held = lambda a: (player_id, a) in self._held
        pending = self._pending[player_id]
        result = PlayerInput(
            move=float(held("right")) - float(held("left")),
            block=held("block"),
            **{a: getattr(pending, a) for a in ONE_SHOT_ACTIONS},
        )
        self._pending[player_id] = PlayerInput()
        return result
