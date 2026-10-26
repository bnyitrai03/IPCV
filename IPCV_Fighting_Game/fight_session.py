"""Webcam-only fight session: calibration, input adaptation and match phases.

This is the single seam between the vision pipeline and the pure game core.
It owns the ``GameState``, decides when calibration is complete, and converts
the latest tracked player state plus recognised gestures into ``PlayerInput``
values. No pygame, rendering or evaluation code lives here.
"""
from __future__ import annotations

from enum import Enum, auto

from config import Config
from game.combat import GameEvent
from game.controls import ONE_SHOT_ACTIONS, PlayerInput
from game.game_state import GameState, Phase
from vision.gesture_recognition import GestureEvent
from vision.player_tracking import TrackState
from vision.pipeline import VisionSnapshot

CALIBRATION_HOLD_S = 2.5


class FightPhase(Enum):
    CALIBRATING = auto()
    FIGHT = auto()
    PAUSED = auto()
    MATCH_OVER = auto()


class FightSession:
    def __init__(self, cfg: Config, names: tuple[str, str], arena_left: float,
                 arena_right: float, ground_y: float) -> None:
        self.cfg = cfg
        self.names = names
        self.arena_left = arena_left
        self.arena_right = arena_right
        self.ground_y = ground_y
        self.phase = FightPhase.CALIBRATING
        self.state = self._new_state()
        self.pending = [PlayerInput(), PlayerInput()]
        self.pending_vision: list[GestureEvent] = []
        self._calibration_since: float | None = None
        self._start_requested = False

    # --------------------------------------------------------------- setup
    def _new_state(self) -> GameState:
        return GameState(self.cfg.game, self.names, arena_left=self.arena_left,
                         arena_right=self.arena_right, ground_y=self.ground_y)

    def start(self) -> None:
        self.state = self._new_state()
        self.pending = [PlayerInput(), PlayerInput()]
        self.pending_vision.clear()
        self._calibration_since = None
        self._start_requested = False
        self.phase = FightPhase.FIGHT

    def restart(self) -> None:
        self.start()

    def request_start(self) -> None:
        """Start as soon as at least one player is registered (solo start)."""
        if self.phase == FightPhase.CALIBRATING:
            self._start_requested = True

    def can_start_early(self, snapshot: VisionSnapshot | None) -> bool:
        """True when at least one player is already registered and startable."""
        return (self.phase == FightPhase.CALIBRATING and snapshot is not None
                and any(p.state == TrackState.ACTIVE for p in snapshot.players))

    def reset_calibration(self) -> None:
        self.phase = FightPhase.CALIBRATING
        self._calibration_since = None
        self._start_requested = False
        self.pending_vision.clear()

    # -------------------------------------------------------------- update
    def observe(self, snapshot: VisionSnapshot | None, events: list[GestureEvent],
                now: float) -> bool:
        """Consume one camera snapshot. Returns True when calibration completes."""
        if self.phase == FightPhase.CALIBRATING:
            return self._observe_calibration(snapshot, now)
        if self.phase == FightPhase.FIGHT:
            self._collect_inputs(snapshot, events)
        return False

    def _observe_calibration(self, snapshot: VisionSnapshot | None, now: float) -> bool:
        if self._start_requested:
            if snapshot is not None and any(p.state == TrackState.ACTIVE
                                            for p in snapshot.players):
                self.start()
                return True
            return False
        if snapshot is None or not snapshot.registered:
            self._calibration_since = None
            return False
        if self._calibration_since is None:
            self._calibration_since = now
            return False
        if now - self._calibration_since < CALIBRATION_HOLD_S:
            return False
        self.start()
        return True

    def _collect_inputs(self, snapshot: VisionSnapshot | None,
                        events: list[GestureEvent]) -> None:
        for pid in (0, 1):
            p = snapshot.players[pid] if snapshot is not None and pid < len(snapshot.players) else None
            active = p is not None and p.state == TrackState.ACTIVE
            # Held values must reflect the latest snapshot, not an OR over frames.
            self.pending[pid].move = max(-1.0, min(1.0, p.move)) if active else 0.0
            self.pending[pid].block = bool(p.block) if active else False
        for ev in events:
            if ev.gesture in ONE_SHOT_ACTIONS:
                inp = self.pending[ev.player_id]
                setattr(inp, ev.gesture, True)
                inp.source_ts.setdefault(ev.gesture, ev.capture_ts)
            self.pending_vision.append(ev)

    def step(self, dt: float) -> list[GameEvent]:
        inputs = (self.pending[0], self.pending[1])
        events = self.state.update(dt, inputs)
        self.pending = [PlayerInput(move=i.move, block=i.block) for i in inputs]
        if self.state.phase == Phase.MATCH_OVER:
            self.phase = FightPhase.MATCH_OVER
        return events

    def take_vision_events(self) -> list[GestureEvent]:
        events, self.pending_vision = self.pending_vision, []
        return events

    def resolve_vision_events(self, game_events: list[GameEvent],
                              phase_before: Phase) -> list[tuple[GestureEvent, str]]:
        """Pair drained gesture events with their game outcome (for logging)."""
        executed = {e.data.get("source_ts"): e.data.get("action")
                    for e in game_events
                    if e.kind == "action" and e.data.get("source_ts") is not None}
        results = []
        for ev in self.take_vision_events():
            if ev.gesture in ("block_start", "block_end"):
                result = "held_state"
            elif phase_before != Phase.FIGHT:
                result = "ignored_not_fighting"
            else:
                act = executed.get(ev.capture_ts)
                result = ("executed" if act and act != "special_denied"
                          else "denied_no_energy" if act == "special_denied"
                          else "not_executed_busy")
            results.append((ev, result))
        return results

    # ------------------------------------------------------------- control
    def pause(self) -> None:
        if self.phase == FightPhase.FIGHT:
            self.phase = FightPhase.PAUSED

    def resume(self) -> None:
        if self.phase == FightPhase.PAUSED:
            self.phase = FightPhase.FIGHT

    def calibration_remaining(self, now: float) -> float | None:
        if self.phase != FightPhase.CALIBRATING or self._calibration_since is None:
            return None
        return max(0.0, CALIBRATION_HOLD_S - (now - self._calibration_since))
