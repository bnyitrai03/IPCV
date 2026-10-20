"""Combat visual effects: particles, impact bursts, projectiles, floating
damage numbers, screen shake and flashes. All effects are time-based and
pooled in plain lists; there is a hard cap on live particles so effects
can never degrade the frame rate during long sessions.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

import pygame

from game.combat import Projectile

MAX_PARTICLES = 600


@dataclass
class Particle:
    x: float
    y: float
    vx: float
    vy: float
    life: float
    max_life: float
    size: float
    color: tuple[int, int, int]
    gravity: float = 0.0
    kind: str = "spark"     # spark | dust | ring | glow


@dataclass
class FloatingText:
    text: str
    x: float
    y: float
    color: tuple[int, int, int]
    life: float
    size: int


class EffectsManager:
    def __init__(self) -> None:
        self.particles: list[Particle] = []
        self.texts: list[FloatingText] = []
        self.shake_time = 0.0
        self.shake_mag = 0.0
        self.flash_alpha = 0.0
        self.flash_color = (255, 255, 255)
        self.hitstop = 0.0           # brief freeze on impact ("hit stop")
        self.rng = random.Random(3)
        self._fonts: dict[int, pygame.font.Font] = {}
        self._glow_cache: dict[tuple, pygame.Surface] = {}

    def _font(self, size: int) -> pygame.font.Font:
        if size not in self._fonts:
            self._fonts[size] = pygame.font.Font(None, size)
        return self._fonts[size]

    def _glow(self, radius: int, color: tuple[int, int, int]) -> pygame.Surface:
        key = (radius, color)
        if key not in self._glow_cache:
            s = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
            for k in range(6, 0, -1):
                r = int(radius * k / 6)
                a = int(28 + (6 - k) * 30)
                pygame.draw.circle(s, (*color, min(255, a)), (radius, radius), r)
            self._glow_cache[key] = s
        return self._glow_cache[key]

    def clear(self) -> None:
        self.particles.clear()
        self.texts.clear()
        self.shake_time = self.flash_alpha = self.hitstop = 0.0

    # ------------------------------------------------------------- spawners
    def _add(self, p: Particle) -> None:
        if len(self.particles) < MAX_PARTICLES:
            self.particles.append(p)

    def impact(self, x: float, y: float, color, strength: float = 1.0, blocked: bool = False) -> None:
        r = self.rng
        n = int((10 if blocked else 22) * strength)
        for _ in range(n):
            a = r.uniform(0, math.tau)
            sp = r.uniform(180, 520) * strength
            self._add(Particle(x, y, math.cos(a) * sp, math.sin(a) * sp, r.uniform(0.15, 0.35), 0.35,
                               r.uniform(2, 4.5), (255, 255, 220) if r.random() < 0.4 else color,
                               gravity=600))
        self._add(Particle(x, y, 0, 0, 0.22, 0.22, 60 * strength,
                           (120, 200, 255) if blocked else color, kind="ring"))
        self._add(Particle(x, y, 0, 0, 0.12, 0.12, 50 * strength, (255, 255, 230), kind="glow"))
        self.shake(6 * strength if not blocked else 3, 0.18)
        self.hitstop = max(self.hitstop, 0.05 * strength if not blocked else 0.03)

    def dust(self, x: float, y: float, direction: int) -> None:
        for _ in range(6):
            self._add(Particle(x, y - 4, -direction * self.rng.uniform(40, 140),
                               -self.rng.uniform(10, 60), 0.5, 0.5, self.rng.uniform(5, 10),
                               (170, 130, 100), gravity=60, kind="dust"))

    def projectile_trail(self, p: Projectile, color) -> None:
        r = self.rng
        for _ in range(2):
            self._add(Particle(p.x - p.direction * r.uniform(0, 20), p.y + r.uniform(-14, 14),
                               -p.direction * r.uniform(30, 120), r.uniform(-40, 40),
                               0.3, 0.3, r.uniform(3, 7), color, kind="spark"))

    def charge(self, x: float, y: float, color) -> None:
        a = self.rng.uniform(0, math.tau)
        d = self.rng.uniform(40, 80)
        self._add(Particle(x + math.cos(a) * d, y + math.sin(a) * d, -math.cos(a) * d * 4,
                           -math.sin(a) * d * 4, 0.22, 0.22, 3, color))

    def ko_burst(self, x: float, y: float, color) -> None:
        self.impact(x, y, color, strength=2.0)
        self.flash(255, (255, 255, 255))
        self.shake(14, 0.5)

    def text(self, text: str, x: float, y: float, color=(255, 240, 120), size: int = 36,
             life: float = 0.9) -> None:
        self.texts.append(FloatingText(text, x, y, color, life, size))

    def shake(self, magnitude: float, duration: float) -> None:
        self.shake_mag = max(self.shake_mag, magnitude)
        self.shake_time = max(self.shake_time, duration)

    def flash(self, alpha: float, color=(255, 255, 255)) -> None:
        self.flash_alpha = max(self.flash_alpha, alpha)
        self.flash_color = color

    # -------------------------------------------------------------- update
    def offset(self) -> tuple[float, float]:
        if self.shake_time <= 0:
            return (0.0, 0.0)
        m = self.shake_mag * min(1.0, self.shake_time * 6)
        return (self.rng.uniform(-m, m), self.rng.uniform(-m, m))

    def update(self, dt: float) -> None:
        self.shake_time = max(0.0, self.shake_time - dt)
        if self.shake_time == 0:
            self.shake_mag = 0.0
        self.flash_alpha = max(0.0, self.flash_alpha - dt * 900)
        self.hitstop = max(0.0, self.hitstop - dt)
        alive = []
        for p in self.particles:
            p.life -= dt
            if p.life <= 0:
                continue
            p.vy += p.gravity * dt
            p.x += p.vx * dt
            p.y += p.vy * dt
            p.vx *= 0.96 if p.kind == "dust" else 0.99
            alive.append(p)
        self.particles = alive
        for t in self.texts:
            t.life -= dt
            t.y -= 50 * dt
        self.texts = [t for t in self.texts if t.life > 0]

    # ---------------------------------------------------------------- draw
    def draw_projectile(self, surf: pygame.Surface, p: Projectile, color, clock: float,
                        offset=(0, 0)) -> None:
        x, y = p.x + offset[0], p.y + offset[1]
        pulse = 1.0 + 0.12 * math.sin(clock * 30)
        r = int(p.radius * 1.9 * pulse)
        surf.blit(self._glow(r, color), (x - r, y - r), special_flags=pygame.BLEND_RGBA_ADD)
        pygame.draw.circle(surf, (255, 255, 255), (int(x), int(y)), int(p.radius * 0.55))
        for k in range(3):
            a = clock * 14 + k * math.tau / 3
            pygame.draw.circle(surf, color, (int(x + math.cos(a) * p.radius * 0.7),
                                             int(y + math.sin(a) * p.radius * 0.5)), 5)

    def draw(self, surf: pygame.Surface, offset=(0, 0)) -> None:
        ox, oy = offset
        for p in self.particles:
            k = p.life / p.max_life
            x, y = p.x + ox, p.y + oy
            if p.kind == "ring":
                r = int(p.size * (1.2 - k))
                if r > 2:
                    pygame.draw.circle(surf, p.color, (int(x), int(y)), r, max(1, int(5 * k)))
            elif p.kind == "glow":
                r = max(2, int(p.size * (0.6 + 0.4 * k)))
                surf.blit(self._glow(r, p.color), (x - r, y - r), special_flags=pygame.BLEND_RGBA_ADD)
            elif p.kind == "dust":
                s = max(1, int(p.size * (1.4 - k)))
                pygame.draw.circle(surf, p.color, (int(x), int(y)), s)
            else:
                # Sparks are drawn as short streaks along their velocity.
                ex, ey = x - p.vx * 0.02, y - p.vy * 0.02
                pygame.draw.line(surf, p.color, (x, y), (ex, ey), max(1, int(p.size * k + 1)))
        for t in self.texts:
            font = self._font(t.size)
            a = min(1.0, t.life * 3)
            img = font.render(t.text, True, t.color)
            shadow = font.render(t.text, True, (20, 10, 20))
            img.set_alpha(int(255 * a))
            shadow.set_alpha(int(255 * a))
            rect = img.get_rect(center=(t.x + ox, t.y + oy))
            surf.blit(shadow, rect.move(2, 2))
            surf.blit(img, rect)

    def draw_flash(self, surf: pygame.Surface) -> None:
        if self.flash_alpha > 1:
            overlay = pygame.Surface(surf.get_size())
            overlay.fill(self.flash_color)
            overlay.set_alpha(int(min(255, self.flash_alpha)))
            surf.blit(overlay, (0, 0))
