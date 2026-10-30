"""Task 4 — augmented effects on the players' REAL bodies (gestures/actions).

Drawn into each player's webcam panel, anchored to that player's tracked
landmarks, so the effect sits on the correct person and follows the motion:

* hook / uppercut  -> glowing trail following the real fist + action label,
* block            -> translucent energy shield in front of the forearms,
* special charging -> energy ball growing between the raised hands, burst on release,
* lean (movement)  -> arrow at the shoulders showing the walking direction.

Face effects (Task 1) are in ``face_effects.py``; this module only uses body
landmarks, so the two stay independent.
"""
from __future__ import annotations

import math
import time
from collections import deque

import cv2
import numpy as np

from vision.pipeline import PlayerSnapshot
from vision.pose_tracking import L_SHOULDER, L_WRIST, R_SHOULDER, R_WRIST

AA = cv2.LINE_AA
ATTACK_SHOW_S = 0.6        # how long an attack effect stays visible
LABELS = {"hook": "HOOK!", "uppercut": "UPPERCUT!"}


def _bgr(rgb: tuple[int, int, int]) -> tuple[int, int, int]:
    return (rgb[2], rgb[1], rgb[0])


def _blend(base: np.ndarray, overlay: np.ndarray, alpha: float) -> None:
    if alpha > 0:
        cv2.addWeighted(overlay, alpha, base, 1 - alpha, 0, dst=base)


class BodyEffects:
    """Keeps a short history of each wrist (for trails) per player panel."""

    def __init__(self, accent_rgb: tuple[int, int, int]) -> None:
        self.color = _bgr(accent_rgb)
        self.trail = {"left": deque(maxlen=10), "right": deque(maxlen=10)}
        self._last_frame = -1

    def update(self, p: PlayerSnapshot, frame_id: int) -> None:
        """Record wrist positions once per camera frame (normalised image coords)."""
        if frame_id == self._last_frame:
            return
        self._last_frame = frame_id
        for side, idx in (("left", L_WRIST), ("right", R_WRIST)):
            pt = p.landmarks[idx]
            if np.isnan(pt).any():
                self.trail[side].clear()
            else:
                self.trail[side].append((float(pt[0]), float(pt[1])))

    def draw(self, img: np.ndarray, p: PlayerSnapshot, to_panel, clock: float) -> None:
        lm = p.landmarks
        sw_px = None
        ls, rs = lm[L_SHOULDER], lm[R_SHOULDER]
        if not (np.isnan(ls).any() or np.isnan(rs).any()):
            a, b = np.array(to_panel(ls)), np.array(to_panel(rs))
            sw_px = float(np.linalg.norm(a - b))
        if not sw_px or sw_px < 4:
            return
        self._lean_arrow(img, p, lm, to_panel, sw_px, clock)
        if p.block:
            self._shield(img, lm, to_panel, sw_px, clock)
        now = time.perf_counter()        # same clock as the gesture timestamps
        special_age = now - p.special_t
        attack_age = now - p.last_attack_t
        if p.special_progress > 0 or special_age < 0.4:
            self._charge(img, p, lm, to_panel, sw_px, clock, special_age)
        if p.last_attack and attack_age < ATTACK_SHOW_S:
            self._attack(img, p, to_panel, sw_px, attack_age)

    # ---------------------------------------------------------------- effects
    def _attack(self, img, p: PlayerSnapshot, to_panel, sw_px: float, age: float) -> None:
        kind, side = p.last_attack.split("_")
        pts = [to_panel(q) for q in self.trail[side]]
        if len(pts) < 2:
            return
        fade = 1.0 - age / ATTACK_SHOW_S
        overlay = img.copy()
        n = len(pts)
        for i in range(1, n):          # comet: thicker and brighter towards the fist
            k = i / (n - 1)
            th = max(2, int(sw_px * 0.06 + sw_px * 0.22 * k))
            col = tuple(int(c * (0.5 + 0.5 * k)) for c in self.color)
            cv2.line(overlay, tuple(map(int, pts[i - 1])), tuple(map(int, pts[i])), col, th, AA)
        fist = tuple(map(int, pts[-1]))
        cv2.circle(overlay, fist, max(4, int(sw_px * 0.22)), (255, 255, 255), -1, AA)
        cv2.circle(overlay, fist, max(6, int(sw_px * 0.32)), self.color, max(2, int(sw_px * 0.05)), AA)
        _blend(img, overlay, 0.75 * fade)
        label = LABELS.get(kind, kind.upper())
        scale = max(0.5, sw_px / 90)
        org = (fist[0] - int(40 * scale), fist[1] - int(sw_px * 0.45))
        cv2.putText(img, label, org, cv2.FONT_HERSHEY_DUPLEX, scale, (20, 20, 20), 4, AA)
        cv2.putText(img, label, org, cv2.FONT_HERSHEY_DUPLEX, scale, self.color, 2, AA)

    def _shield(self, img, lm, to_panel, sw_px: float, clock: float) -> None:
        lw, rw = lm[L_WRIST], lm[R_WRIST]
        if np.isnan(lw).any() or np.isnan(rw).any():
            return
        a, b = np.array(to_panel(lw)), np.array(to_panel(rw))
        c = (a + b) / 2
        r = max(sw_px * 0.75, float(np.linalg.norm(a - b)) * 0.75)
        overlay = img.copy()
        pulse = 1.0 + 0.04 * math.sin(clock * 10)
        hexagon = np.array([(c[0] + math.cos(t) * r * pulse, c[1] + math.sin(t) * r * 1.1 * pulse)
                            for t in np.linspace(0, math.tau, 7)[:-1] + math.pi / 6], np.int32)
        cv2.fillConvexPoly(overlay, hexagon, (255, 170, 60), AA)
        cv2.polylines(overlay, [hexagon], True, (255, 240, 200), max(2, int(sw_px * 0.04)), AA)
        _blend(img, overlay, 0.35)

    def _charge(self, img, p: PlayerSnapshot, lm, to_panel, sw_px: float, clock: float,
                special_age: float) -> None:
        lw, rw = lm[L_WRIST], lm[R_WRIST]
        if np.isnan(lw).any() or np.isnan(rw).any():
            return
        c = (np.array(to_panel(lw)) + np.array(to_panel(rw))) / 2
        overlay = img.copy()
        if p.special_progress > 0:
            r = sw_px * (0.15 + 0.45 * p.special_progress) * (1 + 0.08 * math.sin(clock * 25))
            cv2.circle(overlay, tuple(map(int, c)), int(r * 1.6), self.color, -1, AA)
            cv2.circle(overlay, tuple(map(int, c)), int(r), (255, 255, 255), -1, AA)
            for k in range(8):           # sparks pulled into the ball
                t = clock * 5 + k * math.tau / 8
                d = r * (2.2 + 0.6 * math.sin(clock * 9 + k))
                q = (int(c[0] + math.cos(t) * d), int(c[1] + math.sin(t) * d))
                cv2.circle(overlay, q, max(2, int(sw_px * 0.05)), self.color, -1, AA)
            _blend(img, overlay, 0.7)
        else:                            # just fired: expanding ring
            k = special_age / 0.4
            cv2.circle(overlay, tuple(map(int, c)), int(sw_px * (0.6 + 2.0 * k)), self.color,
                       max(2, int(sw_px * 0.1 * (1 - k))), AA)
            _blend(img, overlay, 0.8 * (1 - k))

    def _lean_arrow(self, img, p: PlayerSnapshot, lm, to_panel, sw_px: float, clock: float) -> None:
        if abs(p.move) < 0.1:
            return
        # Anchored just below the shoulder line (the panel crop cuts off the hips).
        ls, rs = lm[L_SHOULDER], lm[R_SHOULDER]
        c = (np.array(to_panel(ls)) + np.array(to_panel(rs))) / 2 + np.array([0.0, -sw_px * 0.05])
        d = 1 if p.move > 0 else -1
        length = sw_px * (0.5 + 0.7 * abs(p.move))
        start = (int(c[0]), int(c[1]))
        end = (int(c[0] + d * length), int(c[1]))
        cv2.arrowedLine(img, start, end, (20, 20, 20), max(4, int(sw_px * 0.12)), AA, tipLength=0.4)
        cv2.arrowedLine(img, start, end, (60, 220, 255), max(2, int(sw_px * 0.07)), AA, tipLength=0.4)
