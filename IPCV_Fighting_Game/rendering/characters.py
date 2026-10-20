"""Procedural 2D fighter graphics (no external assets needed).

Each fighter is drawn from the animated skeleton as layered, outlined,
shaded body parts — tapered limbs, a gi with collar and belt, gloves, hair
and a fluttering headband — so the characters read as arcade fighters, not
stick figures. Two original characters: BLAZE (fire) and VOLT (lightning).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import pygame

from game.fighter import Fighter, FighterState
from rendering.animations import Animator, Pose

Color = tuple[int, int, int]


@dataclass(frozen=True)
class FighterStyle:
    name: str
    skin: Color
    gi: Color
    gi_shade: Color
    pants: Color
    belt: Color
    glove: Color
    hair: Color
    band: Color
    accent: Color          # effects / UI colour
    outline: Color = (18, 14, 24)
    spiky: bool = False


BLAZE = FighterStyle("BLAZE", skin=(232, 178, 136), gi=(206, 46, 42), gi_shade=(150, 28, 32),
                     pants=(186, 38, 36), belt=(30, 26, 30), glove=(120, 26, 26),
                     hair=(34, 26, 30), band=(245, 245, 240), accent=(255, 120, 40))
VOLT = FighterStyle("VOLT", skin=(196, 140, 104), gi=(38, 110, 210), gi_shade=(26, 70, 150),
                    pants=(34, 60, 120), belt=(250, 210, 60), glove=(235, 235, 240),
                    hair=(250, 232, 140), band=(250, 210, 60), accent=(90, 210, 255), spiky=True)
STYLES = (BLAZE, VOLT)


def _shade(c: Color, k: float) -> Color:
    return tuple(max(0, min(255, int(v * k))) for v in c)


def _mix(a: Color, b: Color, t: float) -> Color:
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def _capsule(surf: pygame.Surface, p1, p2, r1: float, r2: float, color: Color) -> None:
    x1, y1 = p1
    x2, y2 = p2
    dx, dy = x2 - x1, y2 - y1
    length = math.hypot(dx, dy) or 1.0
    nx, ny = -dy / length, dx / length
    pts = [(x1 + nx * r1, y1 + ny * r1), (x2 + nx * r2, y2 + ny * r2),
           (x2 - nx * r2, y2 - ny * r2), (x1 - nx * r1, y1 - ny * r1)]
    pygame.draw.polygon(surf, color, pts)
    pygame.draw.circle(surf, color, (int(x1), int(y1)), int(r1))
    pygame.draw.circle(surf, color, (int(x2), int(y2)), int(r2))


class FighterRenderer:
    def __init__(self, style: FighterStyle) -> None:
        self.style = style
        self.animator = Animator()
        self.flash = 0.0

    def _col(self, c: Color) -> Color:
        return _mix(c, (255, 255, 255), self.flash) if self.flash > 0 else c

    def draw(self, surf: pygame.Surface, f: Fighter, clock: float, dt: float,
             offset: tuple[float, float] = (0, 0)) -> Pose:
        pose = self.animator.update(f, clock, dt)
        if f.state == FighterState.HITSTUN and f.state_time < 0.09:
            self.flash = 0.75
        else:
            self.flash = max(0.0, self.flash - dt * 6)
        ox, oy = f.x + offset[0], f.ground_y + offset[1]
        fx = f.facing
        P = {j: (ox + fx * x, oy + y) for j, (x, y) in pose.items()}
        s = self.style
        O = s.outline

        # Shadow.
        shadow = pygame.Surface((150, 26), pygame.SRCALPHA)
        pygame.draw.ellipse(shadow, (0, 0, 0, 90), shadow.get_rect())
        surf.blit(shadow, (f.x + offset[0] - 75, f.ground_y + offset[1] - 13))

        def limb(a, b, r1, r2, color, outline=True):
            if outline:
                _capsule(surf, P[a], P[b], r1 + 2.5, r2 + 2.5, O)
            _capsule(surf, P[a], P[b], r1, r2, self._col(color))

        # Back leg and back arm (darker: further from the viewer).
        back_pants = _shade(s.pants, 0.72)
        limb("hip", "kn_b", 15, 12, back_pants)
        limb("kn_b", "an_b", 12, 9, back_pants)
        self._foot(surf, P["an_b"], fx, _shade(s.skin, 0.75))
        back_sleeve = _shade(s.gi, 0.7)
        limb("sh_b", "el_b", 11, 9, back_sleeve)
        limb("el_b", "wr_b", 9, 8, _shade(s.skin, 0.78))
        self._glove(surf, P["wr_b"], P["el_b"], _shade(s.glove, 0.75))

        # Torso.
        self._torso(surf, P, fx)

        # Front leg.
        limb("hip", "kn_f", 16, 13, s.pants)
        limb("kn_f", "an_f", 13, 10, s.pants)
        self._foot(surf, P["an_f"], fx, s.skin)
        self._belt(surf, P, fx, clock)

        # Head.
        self._head(surf, P, fx, clock, f)

        # Front arm.
        limb("sh_f", "el_f", 12, 10, s.gi)
        limb("el_f", "wr_f", 10, 8.5, s.skin)
        self._glove(surf, P["wr_f"], P["el_f"], s.glove)
        return pose

    # ------------------------------------------------------------- parts
    def _torso(self, surf, P, fx) -> None:
        s = self.style
        hip, chest, neck = P["hip"], P["chest"], P["neck"]
        sf, sb = P["sh_f"], P["sh_b"]
        hip_f = (hip[0] + fx * 22, hip[1])
        hip_b = (hip[0] - fx * 22, hip[1])
        chest_f = (chest[0] + fx * 26, chest[1])
        chest_b = (chest[0] - fx * 24, chest[1] + 4)
        body = [neck, (sf[0] + fx * 8, sf[1]), chest_f, hip_f, hip_b, chest_b, (sb[0] - fx * 8, sb[1])]
        pygame.draw.polygon(surf, s.outline, body, 0)
        inner = [(x + (chest[0] - x) * 0.07, y + (chest[1] - y) * 0.07) for x, y in body]
        pygame.draw.polygon(surf, self._col(s.gi), inner)
        # Shading on the back half of the torso.
        shade = [neck, chest, hip, hip_b, chest_b, (sb[0] - fx * 8, sb[1])]
        shade = [(x + (chest[0] - x) * 0.07, y + (chest[1] - y) * 0.07) for x, y in shade]
        pygame.draw.polygon(surf, self._col(s.gi_shade), shade)
        # V collar showing skin + lapel line.
        v = [(neck[0] - fx * 9, neck[1] - 2), (neck[0] + fx * 11, neck[1] - 2),
             (chest[0] + fx * 6, chest[1] - 6)]
        pygame.draw.polygon(surf, self._col(s.skin), v)
        pygame.draw.line(surf, s.outline, v[1], v[2], 3)
        pygame.draw.line(surf, _shade(s.gi, 1.25), (chest[0] + fx * 6, chest[1] - 6),
                         (hip[0] + fx * 4, hip[1] - 4), 3)

    def _belt(self, surf, P, fx, clock) -> None:
        s = self.style
        hip = P["hip"]
        rect = pygame.Rect(0, 0, 50, 11)
        rect.center = (int(hip[0]), int(hip[1] - 4))
        pygame.draw.rect(surf, s.outline, rect.inflate(4, 4), border_radius=4)
        pygame.draw.rect(surf, self._col(s.belt), rect, border_radius=4)
        knot = (hip[0] + fx * 8, hip[1] - 2)
        sway = math.sin(clock * 6) * 3
        for k, dx in enumerate((4, 12)):
            end = (knot[0] + fx * dx - fx * sway, knot[1] + 20 + k * 3)
            pygame.draw.line(surf, s.outline, knot, end, 7)
            pygame.draw.line(surf, self._col(s.belt), knot, end, 4)

    def _foot(self, surf, ankle, fx, color) -> None:
        x, y = ankle
        pts = [(x - fx * 8, y - 10), (x + fx * 22, y - 4), (x + fx * 22, y + 2), (x - fx * 10, y + 2)]
        pygame.draw.polygon(surf, self.style.outline, pts, 0)
        inner = [(px + (x - px) * 0.18, py + (y - 4 - py) * 0.18) for px, py in pts]
        pygame.draw.polygon(surf, self._col(color), inner)

    def _glove(self, surf, wrist, elbow, color) -> None:
        dx, dy = wrist[0] - elbow[0], wrist[1] - elbow[1]
        d = math.hypot(dx, dy) or 1
        c = (wrist[0] + dx / d * 7, wrist[1] + dy / d * 7)
        pygame.draw.circle(surf, self.style.outline, (int(c[0]), int(c[1])), 15)
        pygame.draw.circle(surf, self._col(color), (int(c[0]), int(c[1])), 12)
        hl = _mix(color, (255, 255, 255), 0.45)
        pygame.draw.circle(surf, self._col(hl), (int(c[0] - 4), int(c[1] - 4)), 4)
        # Cuff.
        cuff = (wrist[0] - dx / d * 2, wrist[1] - dy / d * 2)
        pygame.draw.circle(surf, self.style.outline, (int(cuff[0]), int(cuff[1])), 10)
        pygame.draw.circle(surf, self._col(_shade(color, 0.7)), (int(cuff[0]), int(cuff[1])), 8)

    def _head(self, surf, P, fx, clock, f: Fighter) -> None:
        s = self.style
        hx, hy = P["head"]
        neck = P["neck"]
        _capsule(surf, neck, (hx - fx * 2, hy + 10), 9, 8, s.outline)
        _capsule(surf, neck, (hx - fx * 2, hy + 10), 7, 6, self._col(_shade(s.skin, 0.85)))
        # Hair behind the head.
        if s.spiky:
            spikes = []
            for k in range(7):
                a = math.pi * (0.55 + k * 0.17)
                r = 30 if k % 2 == 0 else 20
                ax = -math.cos(a) * fx
                spikes.append((hx + ax * r - fx * 4, hy - math.sin(a) * r - 4))
            spikes = [(hx + fx * 14, hy - 4)] + spikes + [(hx - fx * 8, hy + 14)]
            pygame.draw.polygon(surf, s.outline, spikes)
            pygame.draw.polygon(surf, self._col(s.hair),
                                [(x + (hx - x) * 0.1, y + (hy - y) * 0.1) for x, y in spikes])
        else:
            pygame.draw.circle(surf, s.outline, (int(hx - fx * 3), int(hy - 3)), 23)
            pygame.draw.circle(surf, self._col(s.hair), (int(hx - fx * 3), int(hy - 3)), 21)
        # Face.
        pygame.draw.circle(surf, s.outline, (int(hx), int(hy)), 19)
        pygame.draw.circle(surf, self._col(s.skin), (int(hx), int(hy)), 17)
        if not s.spiky:
            # Short hair cap on top.
            pygame.draw.arc(surf, s.hair, pygame.Rect(hx - 18, hy - 20, 36, 30), 0.3, 2.85, 8)
        # Jaw shading.
        pygame.draw.arc(surf, _shade(s.skin, 0.8), pygame.Rect(hx - 16, hy - 14, 32, 30),
                        math.pi * 1.1, math.pi * 1.9, 3)
        # Eye + brow (look towards the opponent; X eyes when KO).
        ex, ey = hx + fx * 9, hy - 3
        if f.state == FighterState.KO:
            for d in (-3, 3):
                pygame.draw.line(surf, s.outline, (ex - 3, ey - 3 + d * 0), (ex + 3, ey + 3), 2)
                pygame.draw.line(surf, s.outline, (ex + 3, ey - 3), (ex - 3, ey + 3), 2)
        else:
            pygame.draw.ellipse(surf, (250, 250, 250), pygame.Rect(ex - 4, ey - 3, 8, 6))
            pygame.draw.circle(surf, (20, 20, 30), (int(ex + fx * 1.5), int(ey)), 2)
            pygame.draw.line(surf, s.outline, (ex - fx * 5, ey - 7), (ex + fx * 6, ey - 5), 3)
        # Mouth.
        pygame.draw.line(surf, _shade(s.skin, 0.55), (hx + fx * 6, hy + 9), (hx + fx * 12, hy + 8), 2)
        # Headband + fluttering tails.
        band_y = hy - 9
        pygame.draw.line(surf, s.outline, (hx - 18, band_y), (hx + 18, band_y), 8)
        pygame.draw.line(surf, self._col(s.band), (hx - 17, band_y), (hx + 17, band_y), 5)
        knot = (hx - fx * 18, band_y)
        for k in range(2):
            pts = [knot]
            for i in range(1, 6):
                wave = math.sin(clock * 12 + i * 0.9 + k) * (2 + i)
                pts.append((knot[0] - fx * i * 8, knot[1] + i * (2.5 + k * 2) + wave))
            pygame.draw.lines(surf, s.outline, False, pts, 6)
            pygame.draw.lines(surf, self._col(s.band), False, pts, 3)
