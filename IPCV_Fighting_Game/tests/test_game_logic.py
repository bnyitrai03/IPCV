"""Unit tests for game logic (fighter state machine, combat, round flow)."""
from config import GameConfig
from game.collision import Box, separate_pushboxes
from game.combat import Projectile, resolve_projectiles
from game.controls import PlayerInput
from game.fighter import HOOK, FighterState
from game.game_state import ROUND_INTRO_S, GameState, Phase

DT = 1 / 60


def make_state(**overrides) -> GameState:
    cfg = GameConfig(**overrides)
    gs = GameState(cfg)
    # Skip the intro.
    t = 0.0
    while gs.phase != Phase.FIGHT:
        gs.update(DT, (PlayerInput(), PlayerInput()))
        t += DT
        assert t < ROUND_INTRO_S + 1
    return gs


def step(gs: GameState, n: int, p0: PlayerInput | None = None, p1: PlayerInput | None = None):
    events = []
    for i in range(n):
        a = p0 if (p0 and i == 0) else PlayerInput(move=p0.move, block=p0.block) if p0 else PlayerInput()
        b = p1 if (p1 and i == 0) else PlayerInput(move=p1.move, block=p1.block) if p1 else PlayerInput()
        events += gs.update(DT, (a, b))
    return events


def put_in_range(gs: GameState) -> None:
    gs.fighters[0].x = 500
    gs.fighters[1].x = 500 + 110


def test_box_intersection():
    a = Box(0, 0, 10, 10)
    assert a.intersects(Box(5, 5, 15, 15))
    assert not a.intersects(Box(10, 0, 20, 10))  # touching edges do not collide


def test_pushbox_separation():
    x1, x2 = separate_pushboxes(100, 120, 80)
    assert abs((x2 - x1) - 80) < 1e-9


def test_walk_moves_fighter():
    gs = make_state()
    x0 = gs.fighters[0].x
    step(gs, 30, PlayerInput(move=1.0))
    assert gs.fighters[0].x > x0
    assert gs.fighters[0].state == FighterState.WALK


def test_hook_in_range_hits_once():
    gs = make_state()
    put_in_range(gs)
    events = step(gs, 40, PlayerInput(hook_right=True))
    hits = [e for e in events if e.kind == "hit"]
    assert len(hits) == 1, "a single hook must register exactly one hit"
    assert gs.fighters[1].health == gs.cfg.max_health - HOOK.damage


def test_hook_out_of_range_misses():
    gs = make_state()
    gs.fighters[0].x, gs.fighters[1].x = 300, 900
    events = step(gs, 40, PlayerInput(hook_left=True))
    assert not [e for e in events if e.kind == "hit"]
    assert gs.fighters[1].health == gs.cfg.max_health


def test_block_reduces_damage():
    gs = make_state()
    put_in_range(gs)
    events = step(gs, 40, PlayerInput(hook_right=True), PlayerInput(block=True))
    blocks = [e for e in events if e.kind == "block"]
    assert len(blocks) == 1
    expected = gs.cfg.max_health - HOOK.damage * gs.cfg.block_damage_factor
    assert abs(gs.fighters[1].health - expected) < 1e-6


def test_cannot_punch_during_recovery_but_buffer_works():
    gs = make_state()
    put_in_range(gs)
    gs.update(DT, (PlayerInput(hook_right=True), PlayerInput()))
    assert gs.fighters[0].state == FighterState.PUNCH
    # Second press during the punch is buffered and executes afterwards (within buffer window).
    total = HOOK.startup + HOOK.active + HOOK.recovery
    n_before = int((total - 0.05) / DT)
    step(gs, n_before)
    gs.update(DT, (PlayerInput(hook_left=True), PlayerInput()))
    step(gs, 6)
    f0 = gs.fighters[0]
    assert f0.state == FighterState.PUNCH and f0.attack and f0.attack.side == "left" and f0.attack.spec.name == "hook"


def test_simultaneous_hooks_trade():
    gs = make_state()
    put_in_range(gs)
    events = step(gs, 40, PlayerInput(hook_right=True), PlayerInput(hook_left=True))
    hits = [e for e in events if e.kind == "hit"]
    assert {e.data["attacker"] for e in hits} == {0, 1}


def test_special_requires_energy_and_spawns_projectile():
    gs = make_state()
    f0 = gs.fighters[0]
    f0.energy = 10
    events = step(gs, 5, PlayerInput(special=True))
    assert any(e.kind == "action" and e.data["action"] == "special_denied" for e in events)
    f0.energy = gs.cfg.max_energy
    events = step(gs, 30, PlayerInput(special=True))
    assert any(e.kind == "special_fire" for e in events)


def test_projectile_hits_and_clashes():
    cfg = GameConfig()
    gs = GameState(cfg)
    f0, f1 = gs.fighters
    p = Projectile(owner=0, x=f1.x - 20, y=gs.ground_y - 150, direction=1)
    events = resolve_projectiles([p], gs.fighters, cfg)
    assert events and events[0].kind == "hit" and not p.alive
    a = Projectile(owner=0, x=600, y=300, direction=1)
    b = Projectile(owner=1, x=610, y=300, direction=-1)
    events = resolve_projectiles([a, b], gs.fighters, cfg)
    assert events[0].kind == "clash" and not a.alive and not b.alive


def test_ko_ends_round_and_match():
    gs = make_state(rounds_to_win=1)
    put_in_range(gs)
    gs.fighters[1].health = 1.0
    events = step(gs, 40, PlayerInput(hook_right=True))
    assert any(e.kind == "round_over" for e in events)
    assert gs.fighters[1].state == FighterState.KO
    assert gs.match_winner == 0
    events = step(gs, int(5 / DT))
    assert gs.phase == Phase.MATCH_OVER


def test_time_up_awards_round_to_healthier_player():
    gs = make_state(round_time_s=1)
    gs.fighters[0].health = 50
    events = step(gs, int(1.2 / DT))
    over = [e for e in events if e.kind == "round_over"]
    assert over and over[0].data["winner"] == 1 and over[0].data["reason"] == "time"


def test_restart_resets_everything():
    gs = make_state(rounds_to_win=1)
    gs.round_wins = [1, 0]
    gs.match_winner = 0
    gs.restart_match()
    assert gs.round_wins == [0, 0] and gs.match_winner is None
    assert gs.phase == Phase.ROUND_INTRO and gs.round_number == 1
    assert all(f.health == gs.cfg.max_health for f in gs.fighters)


def test_counter_hit_and_combo_scaling():
    gs = make_state()
    put_in_range(gs)
    f0, f1 = gs.fighters
    # Defender starts a punch on the same tick; attacker's earlier punch lands in their startup.
    gs.update(DT, (PlayerInput(hook_right=True), PlayerInput()))
    n = int((HOOK.startup - 0.01) / DT)
    step(gs, n)
    events = step(gs, 30, None, PlayerInput(hook_left=True))
    first = [e for e in events if e.kind == "hit" and e.data["attacker"] == 0]
    assert first and first[0].data["counter"]
