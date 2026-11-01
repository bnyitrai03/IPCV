"""Procedurally synthesised sound effects (no audio files required).

If the audio device cannot be opened (headless machine, no output device),
all calls become no-ops; the game never depends on audio.
"""
from __future__ import annotations

import logging
import math

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
    mixer_rate = pygame.mixer.get_init()[0]
    if mixer_rate != RATE:
        n = int(round(len(wave) * mixer_rate / RATE))
        wave = np.interp(np.linspace(0.0, len(wave) - 1, n), np.arange(len(wave)), wave)
    data = (wave * 32767).astype(np.int16)
    channels = pygame.mixer.get_init()[2]
    if channels == 2:
        data = np.column_stack([data, data])
    return pygame.sndarray.make_sound(np.ascontiguousarray(data))


def _glide(start: float, end: float, n: int, curve: float = 1.0) -> np.ndarray:
    return start + (end - start) * np.linspace(0.0, 1.0, n) ** curve


def _resonator(x: np.ndarray, a1: np.ndarray, a2: float, r: float) -> np.ndarray:
    y = np.empty_like(x)
    y1 = y2 = 0.0
    gain = 1.0 - r
    for i in range(len(x)):
        v = gain * x[i] + a1[i] * y1 + a2 * y2
        y[i] = v
        y2, y1 = y1, v
    return y


def _formants(source: np.ndarray, f1, f2, f3) -> np.ndarray:
    out = np.zeros_like(source)
    for freq, bw, gain in ((f1, 90.0, 1.0), (f2, 120.0, 0.6), (f3, 180.0, 0.25)):
        r = math.exp(-math.pi * bw / RATE)
        freq = np.broadcast_to(np.asarray(freq, dtype=float), source.shape)
        a1 = 2 * r * np.cos(2 * math.pi * freq / RATE)
        out += gain * _resonator(source, a1, -r * r, r)
    return out


def _syllable(f0_start: float, f0_end: float, dur: float, f1, f2, f3,
              decay: float, vibrato: float, rng: np.random.Generator) -> np.ndarray:
    n = int(RATE * dur)
    t = np.arange(n) / RATE
    f0 = _glide(f0_start, f0_end, n, 0.8) * (1 + vibrato * np.sin(2 * math.pi * 5.5 * t))
    phase = 2 * math.pi * np.cumsum(f0) / RATE
    source = sum(np.sin(k * phase) / k for k in range(1, 26)) * 0.45
    source += 0.015 * rng.uniform(-1, 1, n)
    wave = _formants(source, f1, f2, f3)
    wave *= _env(n, attack=0.018, decay=decay)
    return wave


def _announcer_ko() -> np.ndarray:
    """Procedural arcade-announcer voice shouting "K.O."."""
    rng = np.random.default_rng(4)
    n_kay = int(RATE * 0.32)
    kay = _syllable(185, 155, 0.32, _glide(650, 430, n_kay), _glide(1750, 2250, n_kay, 0.8),
                    np.full(n_kay, 2750.0), decay=0.9, vibrato=0.012, rng=rng)
    n_burst = int(RATE * 0.028)
    burst = np.diff(rng.uniform(-1, 1, n_burst + 1)) * np.exp(-np.arange(n_burst) / RATE * 90)
    kay[:n_burst] += 0.7 * burst
    n_asp = int(RATE * 0.03)
    asp = np.diff(rng.uniform(-1, 1, n_asp + 1)) * np.linspace(0.5, 0.0, n_asp)
    start = int(RATE * 0.012)
    kay[start:start + n_asp] += 0.5 * asp
    n_release = int(RATE * 0.02)
    kay[-n_release:] *= np.linspace(1.0, 0.0, n_release)
    kay = np.concatenate([np.zeros(start), kay])
    n_oh = int(RATE * 0.62)
    oh = _syllable(165, 100, 0.62, _glide(540, 390, n_oh), _glide(820, 980, n_oh, 0.7),
                   np.full(n_oh, 2450.0), decay=2.4, vibrato=0.02, rng=rng)
    wave = np.concatenate([kay, np.zeros(int(RATE * 0.07)), oh])
    wave = np.tanh(wave * 1.4)
    n_tail = int(RATE * 0.05)
    wave[-n_tail:] *= np.linspace(1.0, 0.0, n_tail)
    return wave / (np.max(np.abs(wave)) + 1e-9)


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
        self.sounds["ko_voice"] = _to_sound(_announcer_ko(), 0.85)
        n5 = int(RATE * 1.0)
        t5 = np.arange(n5) / RATE
        bell = sum(np.sin(2 * np.pi * f * t5) / (k + 1) for k, f in enumerate((660, 1320, 1990)))
        self.sounds["bell"] = _to_sound(bell * _env(n5, decay=3.0), 0.35)

    def play(self, name: str) -> None:
        if self.enabled and name in self.sounds:
            self.sounds[name].play()
