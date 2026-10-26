"""Tests for the webcam-only fight session (calibration + input adapter)."""
from types import SimpleNamespace

import pygame

from application import App
from config import Config
from fight_session import CALIBRATION_HOLD_S, FightPhase, FightSession
from game.game_state import Phase
from vision.player_tracking import TrackState


class FakePlayer:
    def __init__(self, state, move=0.0, block=False):
        self.state = state
        self.move = move
        self.block = block


class FakeSnapshot:
    def __init__(self, registered, players):
        self.registered = registered
        self.players = players


def make_session() -> FightSession:
    cfg = Config()
    return FightSession(cfg, ("BLAZE", "VOLT"), 60, 1220, 496)


def registered_snapshot(p0=None, p1=None) -> FakeSnapshot:
    return FakeSnapshot(True, [p0 or FakePlayer(TrackState.ACTIVE),
                               p1 or FakePlayer(TrackState.ACTIVE)])


def gesture(player_id, name, ts=1.0):
    return SimpleNamespace(player_id=player_id, gesture=name, capture_ts=ts,
                           onset_ts=ts - 0.1, processed_ts=ts + 0.02)


def test_calibration_requires_both_players_and_hold_time():
    session = make_session()
    snap = registered_snapshot()
    assert not session.observe(snap, [], 0.0)
    assert session.phase == FightPhase.CALIBRATING
    assert session.calibration_remaining(0.0) == CALIBRATION_HOLD_S

    assert not session.observe(snap, [], CALIBRATION_HOLD_S - 0.1)
    assert session.observe(snap, [], CALIBRATION_HOLD_S)
    assert session.phase == FightPhase.FIGHT


def test_calibration_resets_when_a_player_disappears():
    session = make_session()
    assert not session.observe(registered_snapshot(), [], 0.0)
    assert not session.observe(FakeSnapshot(False, []), [], 1.0)
    assert session.calibration_remaining(1.0) is None
    assert not session.observe(registered_snapshot(), [], 2.0)
    assert not session.observe(registered_snapshot(), [], 3.0)
    assert session.observe(registered_snapshot(), [], 2.0 + CALIBRATION_HOLD_S)


def test_can_start_early_with_one_active_player():
    session = make_session()
    assert not session.can_start_early(None)
    assert not session.can_start_early(
        FakeSnapshot(False, [FakePlayer(TrackState.UNREGISTERED),
                             FakePlayer(TrackState.UNREGISTERED)]))
    assert session.can_start_early(
        FakeSnapshot(False, [FakePlayer(TrackState.ACTIVE),
                             FakePlayer(TrackState.UNREGISTERED)]))
    session.start()
    assert not session.can_start_early(
        FakeSnapshot(True, [FakePlayer(TrackState.ACTIVE),
                            FakePlayer(TrackState.ACTIVE)]))


def test_second_player_can_join_after_a_solo_start():
    session = make_session()
    session.start()
    solo = FakeSnapshot(False, [FakePlayer(TrackState.ACTIVE, move=0.5),
                                FakePlayer(TrackState.UNREGISTERED)])
    session.observe(solo, [], 0.0)
    assert session.pending[0].move == 0.5
    assert session.pending[1].move == 0.0

    joined = FakeSnapshot(True, [FakePlayer(TrackState.ACTIVE, move=0.5),
                                 FakePlayer(TrackState.ACTIVE, move=-0.5, block=True)])
    session.observe(joined, [], 0.1)
    assert session.pending[1].move == -0.5
    assert session.pending[1].block


def test_webcam_inputs_forward_held_state_and_one_shot_gestures():
    session = make_session()
    session.start()
    snap = registered_snapshot(
        FakePlayer(TrackState.ACTIVE, move=1.0, block=True),
        FakePlayer(TrackState.LOST, move=-1.0, block=True))
    events = [gesture(0, "hook_right", ts=10.0), gesture(0, "block_start"),
              gesture(1, "special", ts=11.0)]
    session.observe(snap, events, 12.0)

    assert session.pending[0].move == 1.0
    assert session.pending[0].block
    assert session.pending[0].hook_right
    assert session.pending[0].source_ts["hook_right"] == 10.0
    assert session.pending[1].move == 0.0      # LOST is neutral
    assert not session.pending[1].block
    assert session.pending[1].special

    logged = session.take_vision_events()
    assert logged == events
    assert session.take_vision_events() == []


def test_held_values_use_the_latest_snapshot():
    session = make_session()
    session.start()
    session.observe(registered_snapshot(FakePlayer(TrackState.ACTIVE, move=1.0, block=True)), [], 0.0)
    session.observe(registered_snapshot(FakePlayer(TrackState.ACTIVE, move=-1.0, block=False)), [], 0.1)
    assert session.pending[0].move == -1.0
    assert not session.pending[0].block


def test_one_shots_are_consumed_by_step_but_held_state_survives():
    session = make_session()
    session.start()
    session.observe(registered_snapshot(FakePlayer(TrackState.ACTIVE, move=1.0, block=True)),
                    [gesture(0, "uppercut_left")], 0.0)
    session.step(1 / 60)
    assert not session.pending[0].uppercut_left
    assert session.pending[0].move == 1.0
    assert session.pending[0].block


def test_pause_resume_restart_and_match_over_phase():
    session = make_session()
    session.start()
    session.pause()
    assert session.phase == FightPhase.PAUSED
    session.resume()
    assert session.phase == FightPhase.FIGHT
    session.state.round_wins = [1, 1]
    session.state.round_number = 5
    session.pause()
    session.restart()
    assert session.phase == FightPhase.FIGHT
    assert session.state.fighters[0].health == session.cfg.game.max_health
    assert session.state.round_wins == [0, 0]
    assert session.state.round_number == 1

    session.state.phase = Phase.MATCH_OVER
    session.step(1 / 60)
    assert session.phase == FightPhase.MATCH_OVER


def test_reset_calibration_returns_to_registration():
    session = make_session()
    session.start()
    session.reset_calibration()
    assert session.phase == FightPhase.CALIBRATING
    assert session.calibration_remaining(0.0) is None


class StubVision:
    error = None

    def __init__(self, snapshot):
        self.snap = snapshot
        self.requests = []

    def snapshot(self):
        return self.snap

    def request(self, command):
        self.requests.append(command)


def test_enter_starts_a_solo_fight_from_registration():
    pygame.init()
    app = App(Config(), vision_source="synthetic")
    try:
        active = FakeSnapshot(
            False, [FakePlayer(TrackState.ACTIVE), FakePlayer(TrackState.UNREGISTERED)])
        app.vision = StubVision(active)
        app._handle_event(pygame.event.Event(
            pygame.KEYDOWN, key=pygame.K_RETURN, scancode=pygame.KSCAN_RETURN,
            mod=0, unicode="\r"))
        assert app.session.phase == FightPhase.CALIBRATING
        assert app.session.observe(active, [], 0.0)
        assert app.session.phase == FightPhase.FIGHT
        app._on_fight_start()
        assert app.vision.requests == ["recalibrate"]

        app.session.reset_calibration()
        nobody = FakeSnapshot(
            False, [FakePlayer(TrackState.UNREGISTERED), FakePlayer(TrackState.UNREGISTERED)])
        app.vision.snap = nobody
        app._handle_event(pygame.event.Event(
            pygame.KEYDOWN, key=pygame.K_RETURN, scancode=pygame.KSCAN_RETURN,
            mod=0, unicode="\r"))
        assert not app.session.observe(nobody, [], 0.0)
        app.vision.snap = active
        assert app.session.observe(active, [], 0.1)
        assert app.session.phase == FightPhase.FIGHT
    finally:
        pygame.quit()
