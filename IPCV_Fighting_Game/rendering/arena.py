"""Animated fighting arena: a rooftop dojo at dusk.

Static layers (sky gradient, mountains, city, temple, floor) are rendered
once into cached surfaces; only light-weight dynamic layers (sun glow,
drifting clouds, flickering lanterns, bobbing crowd, floor reflections)
are drawn every frame, so the background costs very little frame time.
"""
from __future__ import annotations

import math
import random

import pygame


def _vgradient(size: tuple[int, int], top, bottom) -> pygame.Surface:
    w, h = size
    surf = pygame.Surface(size)
    for y in range(h):
        t = y / max(h - 1, 1)
        c = tuple(int(top[i] + (bottom[i] - top[i]) * t) for i in range(3))
        pygame.draw.line(surf, c, (0, y), (w, y))
    return surf


class Arena:
    def __init__(self, width: int, height: int, ground_y: int) -> None:
        self.w, self.h, self.ground_y = width, height, ground_y
        rng = random.Random(7)
        self.sky = _vgradient((width, ground_y), (28, 20, 58), (236, 120, 84))
        self.far = pygame.Surface((width, ground_y), pygame.SRCALPHA)
        self._mountains(self.far, rng)
        self.mid = pygame.Surface((width, ground_y), pygame.SRCALPHA)
        self._city(self.mid, rng)
        self.near = pygame.Surface((width, height), pygame.SRCALPHA)
        self._temple(self.near)
        self.floor = pygame.Surface((width, height - ground_y + 40), pygame.SRCALPHA)
        self._floor(self.floor)
        self.clouds = [(rng.uniform(0, width), rng.uniform(40, 200), rng.uniform(80, 200),
                        rng.uniform(6, 18)) for _ in range(7)]
        self.crowd = [(rng.uniform(0, width), rng.uniform(0, math.tau), rng.choice((0, 1, 2)))
                      for _ in range(46)]
        self.lanterns = [(140, 170), (420, 150), (width - 420, 150), (width - 140, 170)]
        self.intensity = 0.0     # pulses on big hits (arena reacts to the fight)
        # Pre-rendered glow sprites (never allocate surfaces per frame).
        self.sun_glow = pygame.Surface((420, 420), pygame.SRCALPHA)
        for r, a in ((200, 18), (150, 26), (110, 40), (80, 70)):
            pygame.draw.circle(self.sun_glow, (255, 190, 120, a), (210, 210), r)
        self.lantern_glow = pygame.Surface((120, 120), pygame.SRCALPHA)
        pygame.draw.circle(self.lantern_glow, (255, 150, 60, 60), (60, 60), 55)
        pygame.draw.circle(self.lantern_glow, (255, 190, 90, 90), (60, 60), 30)
        self.cloud_surfs = []
        for _x, _y, cw, _s in self.clouds:
            c = pygame.Surface((int(cw), 30), pygame.SRCALPHA)
            pygame.draw.ellipse(c, (255, 170, 160, 70), c.get_rect())
            self.cloud_surfs.append(c)
        self.floor_flash = pygame.Surface((width, 60), pygame.SRCALPHA)

    # ---------------------------------------------------------- static art
    def _mountains(self, surf, rng) -> None:
        w, gy = self.w, self.ground_y
        for layer, (col, base, amp) in enumerate((((84, 54, 98), gy - 170, 110),
                                                  ((64, 40, 80), gy - 120, 80))):
            pts = [(0, gy)]
            x = 0
            while x <= w:
                y = base - amp * (0.5 + 0.5 * math.sin(x * 0.006 + layer * 2.1)) \
                    - rng.uniform(0, 25)
                pts.append((x, y))
                x += 40
            pts.append((w, gy))
            pygame.draw.polygon(surf, col, pts)

    def _city(self, surf, rng) -> None:
        w, gy = self.w, self.ground_y
        x = 0
        while x < w:
            bw = rng.randint(30, 70)
            bh = rng.randint(50, 150)
            rect = pygame.Rect(x, gy - 60 - bh, bw, bh + 60)
            pygame.draw.rect(surf, (40, 28, 56), rect)
            for wy in range(rect.top + 8, gy - 70, 14):
                for wx in range(rect.left + 5, rect.right - 6, 11):
                    if rng.random() < 0.28:
                        pygame.draw.rect(surf, (255, 200, 120), (wx, wy, 4, 6))
            x += bw + rng.randint(2, 12)

    def _temple(self, surf) -> None:
        w, gy = self.w, self.ground_y
        dark, beam = (30, 18, 28), (120, 32, 36)
        # Torii-style gate framing the stage.
        for px in (70, w - 70):
            pygame.draw.rect(surf, dark, (px - 14, 110, 28, gy - 110))
            pygame.draw.rect(surf, beam, (px - 10, 110, 20, gy - 110))
        pygame.draw.polygon(surf, dark, [(20, 92), (w - 20, 92), (w - 50, 122), (50, 122)])
        pygame.draw.polygon(surf, beam, [(30, 96), (w - 30, 96), (w - 54, 118), (54, 118)])
        pygame.draw.rect(surf, dark, (60, 140, w - 120, 16))
        pygame.draw.rect(surf, beam, (64, 143, w - 128, 10))
        # Railing behind the floor.
        rail_y = gy - 34
        pygame.draw.rect(surf, dark, (0, rail_y, w, 8))
        for x in range(0, w, 48):
            pygame.draw.rect(surf, dark, (x, rail_y, 6, 34))

    def _floor(self, surf) -> None:
        w, h = surf.get_size()
        grad = _vgradient((w, h), (122, 76, 52), (58, 34, 28))
        surf.blit(grad, (0, 0))
        cx = w / 2
        for i in range(-14, 15):
            x_top = cx + i * 60
            x_bot = cx + i * 130
            pygame.draw.line(surf, (88, 52, 38), (x_top, 0), (x_bot, h), 2)
        y, gap = 8.0, 10.0
        while y < h:
            pygame.draw.line(surf, (96, 58, 42), (0, int(y)), (w, int(y)), 1)
            gap *= 1.35
            y += gap
        pygame.draw.line(surf, (190, 130, 90), (0, 0), (w, 0), 3)

    # -------------------------------------------------------------- frame
    def pulse(self, amount: float) -> None:
        self.intensity = min(1.0, self.intensity + amount)

    def draw(self, surf: pygame.Surface, clock: float, shake: tuple[float, float],
             dt: float) -> None:
        self.intensity = max(0.0, self.intensity - dt * 1.5)
        sx, sy = shake
        surf.blit(self.sky, (0, 0))
        # Sun with soft glow.
        sun = (self.w // 2, self.ground_y - 150)
        surf.blit(self.sun_glow, (sun[0] - 210, sun[1] - 210))
        pygame.draw.circle(surf, (255, 222, 170), sun, 62)
        # Clouds drift.
        for (x, y, _cw, speed), cloud in zip(self.clouds, self.cloud_surfs):
            cx = (x + clock * speed) % (self.w + 300) - 150
            surf.blit(cloud, (cx, y))
        surf.blit(self.far, (sx * 0.2, sy * 0.2))
        surf.blit(self.mid, (sx * 0.4, sy * 0.4))
        # Crowd silhouettes behind the railing, bouncing more when the fight heats up.
        excite = 2 + self.intensity * 10
        for x, ph, kind in self.crowd:
            bob = abs(math.sin(clock * (3 + kind) + ph)) * excite
            base_y = self.ground_y - 30 - bob
            col = (22 + kind * 6, 14, 26)
            pygame.draw.circle(surf, col, (int(x + sx * 0.6), int(base_y - 30)), 9)
            pygame.draw.rect(surf, col, (int(x - 11 + sx * 0.6), int(base_y - 22), 22, 26),
                             border_radius=6)
            if kind == 2 and self.intensity > 0.3:
                pygame.draw.line(surf, col, (x + 8 + sx * 0.6, base_y - 18),
                                 (x + 14 + sx * 0.6, base_y - 44), 4)
        surf.blit(self.near, (sx * 0.8, sy * 0.8))
        # Lanterns flicker.
        for i, (lx, ly) in enumerate(self.lanterns):
            flick = 0.85 + 0.15 * math.sin(clock * 9 + i * 1.7) + 0.05 * math.sin(clock * 23 + i)
            self.lantern_glow.set_alpha(int(255 * min(1.0, flick)))
            surf.blit(self.lantern_glow, (lx - 60 + sx * 0.8, ly - 60 + sy * 0.8))
            pygame.draw.line(surf, (30, 18, 28), (lx + sx * 0.8, 122), (lx + sx * 0.8, ly - 18), 2)
            pygame.draw.ellipse(surf, (210, 60, 40), (lx - 14 + sx * 0.8, ly - 18 + sy * 0.8, 28, 36))
            pygame.draw.ellipse(surf, (255, 200, 120), (lx - 7 + sx * 0.8, ly - 10 + sy * 0.8, 14, 20))
        surf.blit(self.floor, (sx, self.ground_y + sy))
        # Floor glow reacting to big hits.
        if self.intensity > 0.02:
            self.floor_flash.fill((int(90 * self.intensity), int(55 * self.intensity),
                                   int(25 * self.intensity), 0))
            surf.blit(self.floor_flash, (0, self.ground_y), special_flags=pygame.BLEND_RGB_ADD)
