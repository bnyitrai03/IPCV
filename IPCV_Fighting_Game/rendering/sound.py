"""Procedurally synthesised sound effects (no audio files required).

If the audio device cannot be opened (headless machine, no output device),
all calls become no-ops; the game never depends on audio.
"""
from __future__ import annotations

import logging

import numpy as np
import pygame

log = logging.getLogger(__name__)
RATE = 22050


def _env(n: int, attack: float = 0.005, decay: float = 8.0) -> np.ndarray:
    t = np.arange(n) / RATE
    a = np.clip(t / attack, 0, 1)
    return a * np.exp(-t * decay)


def _to_sound(wave: np.ndarray, volume: float = 0.6) -> pygame.mixer.Sound:
    wave = np.clip(wave * volume, -1, 1)
    data = (wave * 32767).astype(np.int16)
    channels = pygame.mixer.get_init()[2]
    if channels == 2:
        data = np.column_stack([data, data])
    return pygame.sndarray.make_sound(np.ascontiguousarray(data))


class SoundBank:
    def __init__(self, enabled: bool = True) -> None:
        self.sounds: dict[str, pygame.mixer.Sound] = {}
        self.enabled = False
        if not enabled:
            return
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init(frequency=RATE, size=-16, channels=2, buffer=512)
            self._build()
            self.enabled = True
        except Exception as exc:   # noqa: BLE001 - audio is optional
            log.warning("Audio disabled: %s", exc)

    def _build(self) -> None:
        rng = np.random.default_rng(0)
        def noise(sec):
            return rng.uniform(-1, 1, int(RATE * sec))
        n = int(RATE * 0.18)
        t = np.arange(n) / RATE
        thump = np.sin(2 * np.pi * (120 - 300 * t) * t)
        self.sounds["hit"] = _to_sound((0.7 * noise(0.18) * _env(n, decay=30) + thump * _env(n, decay=18)), 0.7)
        self.sounds["block"] = _to_sound(np.sin(2 * np.pi * 900 * t) * _env(n, decay=40)
                                         + 0.3 * noise(0.18) * _env(n, decay=50), 0.45)
        n2 = int(RATE * 0.12)
        self.sounds["whoosh"] = _to_sound(noise(0.12) * np.sin(np.linspace(0, np.pi, n2)) * 0.5, 0.35)
        n3 = int(RATE * 0.6)
        t3 = np.arange(n3) / RATE
        sweep = np.sin(2 * np.pi * (200 + 900 * t3) * t3)
        self.sounds["special"] = _to_sound(sweep * _env(n3, attack=0.05, decay=4) +
                                           0.3 * noise(0.6) * _env(n3, decay=6), 0.5)
        n4 = int(RATE * 1.2)
        t4 = np.arange(n4) / RATE
        self.sounds["ko"] = _to_sound(np.sin(2 * np.pi * (70 - 30 * t4) * t4) * _env(n4, decay=2.5)
                                      + 0.4 * noise(1.2) * _env(n4, decay=5), 0.9)
        n5 = int(RATE * 1.0)
        t5 = np.arange(n5) / RATE
        bell = sum(np.sin(2 * np.pi * f * t5) / (k + 1) for k, f in enumerate((660, 1320, 1990)))
        self.sounds["bell"] = _to_sound(bell * _env(n5, decay=3.0), 0.35)

    def play(self, name: str) -> None:
        if self.enabled and name in self.sounds:
            self.sounds[name].play()
