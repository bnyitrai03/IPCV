"""Round/match flow: intro -> fight -> round over -> ... -> match over.

Pure game logic (no rendering, no pygame). Rendering reads the public
attributes; the main loop feeds PlayerInputs and consumes GameEvents.
"""
from __future__ import annotations

from enum import Enum, auto

from config import GameConfig
from game.collision import separate_pushboxes
from game.combat import GameEvent, Projectile, resolve_melee, resolve_projectiles
from game.controls import PlayerInput
from game.fighter import PUSH_GAP, Fighter, FighterState

ROUND_INTRO_S = 2.2
ROUND_OVER_S = 3.2
SLOWMO_S = 0.9           # brief slow motion after a KO


class Phase(Enum):
    ROUND_INTRO = auto()
    FIGHT = auto()
    ROUND_OVER = auto()
    MATCH_OVER = auto()


class GameState:
    def __init__(self, cfg: GameConfig, names: tuple[str, str] = ("BLAZE", "VOLT"),
                 arena_left: float = 60.0, arena_right: float | None = None,
                 ground_y: float = 500.0) -> None:
        self.cfg = cfg
        self.arena_left = arena_left
        self.arena_right = arena_right if arena_right is not None else cfg.window_width - 60.0
        self.ground_y = ground_y
        width = self.arena_right - self.arena_left
        self.start_x = (self.arena_left + width * 0.3, self.arena_left + width * 0.7)
        self.fighters = [Fighter(0, names[0], self.start_x[0], 1, cfg),
                         Fighter(1, names[1], self.start_x[1], -1, cfg)]
        for f in self.fighters:
            f.ground_y = ground_y
        self.projectiles: list[Projectile] = []
        self.round_wins = [0, 0]
        self.round_number = 0
        self.match_winner: int | None = None     # 0/1, or -1 for draw
        self.round_winner: int | None = None
        self.new_round()

    # ------------------------------------------------------------------ flow
    def new_round(self) -> None:
        self.round_number += 1
        self.phase = Phase.ROUND_INTRO
        self.phase_time = 0.0
        self.time_left = float(self.cfg.round_time_s)
        self.projectiles.clear()
        self.round_winner = None
        self.slowmo = 0.0
        for f, x, facing in zip(self.fighters, self.start_x, (1, -1)):
            f.reset_for_round(x, facing)
        self._pending_events = [GameEvent("round_intro", {"round": self.round_number})]

    @property
    def time_scale(self) -> float:
        return 0.35 if self.slowmo > 0 else 1.0

    # ---------------------------------------------------------------- update
    def update(self, dt: float, inputs: tuple[PlayerInput, PlayerInput]) -> list[GameEvent]:
        events, self._pending_events = self._pending_events, []
        self.phase_time += dt
        if self.slowmo > 0:
            self.slowmo -= dt
        sdt = dt * self.time_scale

        if self.phase == Phase.ROUND_INTRO:
            if self.phase_time >= ROUND_INTRO_S:
                self.phase = Phase.FIGHT
                self.phase_time = 0.0
                events.append(GameEvent("fight"))
            # Fighters idle-animate but cannot act.
            for f in self.fighters:
                f.update(sdt, self.arena_left, self.arena_right)
            return events

        if self.phase == Phase.FIGHT:
            for f, inp in zip(self.fighters, inputs):
                for action in f.apply_input(inp):
                    events.append(GameEvent("action", {"player": f.player_id, "action": action,
                                                       "source_ts": inp.source_ts.get(action)}))
            self._step_world(sdt, events)
            self.time_left = max(0.0, self.time_left - dt)
            self._check_round_end(events)
        elif self.phase in (Phase.ROUND_OVER, Phase.MATCH_OVER):
            neutral = PlayerInput()
            for f in self.fighters:
                if f.state not in (FighterState.KO, FighterState.VICTORY):
                    f.apply_input(neutral)
            self._step_world(sdt, events)
            if self.phase == Phase.ROUND_OVER and self.phase_time >= ROUND_OVER_S:
                if self.match_winner is not None:
                    self.phase = Phase.MATCH_OVER
                    self.phase_time = 0.0
                    events.append(GameEvent("match_over", {"winner": self.match_winner}))
                else:
                    self.new_round()
                    events.extend(self._pending_events)
                    self._pending_events = []
        return events

    def _step_world(self, dt: float, events: list[GameEvent]) -> None:
        f0, f1 = self.fighters
        f0.face_towards(f1.x)
        f1.face_towards(f0.x)
        for f in self.fighters:
            f.update(dt, self.arena_left, self.arena_right)
            if f.pending_special:
                f.pending_special = False
                p = Projectile(owner=f.player_id, x=f.x + f.facing * 70,
                               y=self.ground_y - 150, direction=f.facing,
                               damage=self.cfg.special_damage)
                self.projectiles.append(p)
                events.append(GameEvent("special_fire", {"player": f.player_id,
                                                         "x": p.x, "y": p.y}))
        # Keep bodies from overlapping (only while both are standing).
        if f0.state != FighterState.KO and f1.state != FighterState.KO:
            f0.x, f1.x = separate_pushboxes(f0.x, f1.x, PUSH_GAP)
            for f in self.fighters:
                f.x = max(self.arena_left + 38, min(self.arena_right - 38, f.x))
        for p in self.projectiles:
            p.update(dt, self.arena_left, self.arena_right)
        events.extend(resolve_melee(self.fighters, self.cfg))
        events.extend(resolve_projectiles(self.projectiles, self.fighters, self.cfg))
        self.projectiles = [p for p in self.projectiles if p.alive]

    def _check_round_end(self, events: list[GameEvent]) -> None:
        f0, f1 = self.fighters
        ko0, ko1 = not f0.is_alive, not f1.is_alive
        if not (ko0 or ko1 or self.time_left <= 0):
            return
        if ko0 and ko1:
            winner, reason = -1, "double_ko"
        elif ko0 or ko1:
            winner, reason = (1 if ko0 else 0), "ko"
        else:
            reason = "time"
            if abs(f0.health - f1.health) < 1e-6:
                winner = -1
            else:
                winner = 0 if f0.health > f1.health else 1
        self.round_winner = winner
        if winner >= 0:
            self.round_wins[winner] += 1
            self.fighters[winner].set_victory()
        if reason in ("ko", "double_ko"):
            self.slowmo = SLOWMO_S
        self.phase = Phase.ROUND_OVER
        self.phase_time = 0.0
        events.append(GameEvent("round_over", {"winner": winner, "reason": reason,
                                               "round": self.round_number}))
        for pid in (0, 1):
            if self.round_wins[pid] >= self.cfg.rounds_to_win:
                self.match_winner = pid
        # Safety net: a long run of draws ends the match as a draw.
        if self.match_winner is None and self.round_number >= 2 * self.cfg.rounds_to_win + 1:
            if self.round_wins[0] == self.round_wins[1]:
                self.match_winner = -1
            else:
                self.match_winner = 0 if self.round_wins[0] > self.round_wins[1] else 1
