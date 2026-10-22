"""Keyboard controls must work regardless of the OS keyboard layout."""
import pygame

from config import GameConfig
from game.controls import KeyboardController, PlayerInput
from game.fighter import FighterState
from game.game_state import GameState, Phase

pygame.init()


def ev(kind, scan_name, key):
    return pygame.event.Event(kind, key=key, scancode=getattr(pygame, f"KSCAN_{scan_name}"),
                              mod=0, unicode="")


GREEK_ALPHA = 945   # keycode produced by the physical "A" key on a Greek layout


def test_physical_keys_work_with_non_us_layout():
    kb = KeyboardController()
    kb.handle_event(ev(pygame.KEYDOWN, "A", GREEK_ALPHA))
    assert kb.poll(0).move == -1.0
    kb.handle_event(ev(pygame.KEYUP, "A", GREEK_ALPHA))
    assert kb.poll(0).move == 0.0
    kb.handle_event(ev(pygame.KEYDOWN, "F", 966))         # "φ" on Greek layout
    assert kb.poll(0).hook_left
    assert not kb.poll(0).hook_left                        # one-shot, consumed


def test_keyboard_drives_the_fighters_end_to_end():
    kb = KeyboardController()
    gs = GameState(GameConfig())
    while gs.phase != Phase.FIGHT:
        gs.update(1 / 60, (PlayerInput(), PlayerInput()))
    x0 = gs.fighters[1].x
    kb.handle_event(ev(pygame.KEYDOWN, "LEFT", pygame.K_LEFT))      # P2 walks left
    for _ in range(30):
        gs.update(1 / 60, (kb.poll(0), kb.poll(1)))
    assert gs.fighters[1].x < x0
    kb.handle_event(ev(pygame.KEYUP, "LEFT", pygame.K_LEFT))
    kb.handle_event(ev(pygame.KEYDOWN, "G", 947))                  # P1 right hook
    gs.update(1 / 60, (kb.poll(0), kb.poll(1)))
    assert gs.fighters[0].state == FighterState.PUNCH
    kb.handle_event(ev(pygame.KEYDOWN, "DOWN", pygame.K_DOWN))      # P2 holds block
    for _ in range(40):
        gs.update(1 / 60, (kb.poll(0), kb.poll(1)))
    assert gs.fighters[1].state == FighterState.BLOCK


def test_focus_loss_releases_held_keys():
    kb = KeyboardController()
    kb.handle_event(ev(pygame.KEYDOWN, "D", 948))
    kb.handle_event(pygame.event.Event(pygame.WINDOWFOCUSLOST))
    assert kb.poll(0).move == 0.0


def test_hook_and_uppercut_keys():
    kb = KeyboardController()
    kb.handle_event(ev(pygame.KEYDOWN, "T", 964))         # P1 left uppercut ("τ")
    kb.handle_event(ev(pygame.KEYDOWN, "P", pygame.K_p))  # P2 right uppercut
    assert kb.poll(0).uppercut_left
    assert kb.poll(1).uppercut_right
