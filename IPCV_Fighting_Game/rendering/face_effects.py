"""Task 1 — augmented-reality effects drawn on the *real* players' faces.

Effects are defined in a face-local coordinate frame (origin between the
eyes, x along the eye line, unit = yaw-corrected face scale) and mapped into
the webcam panel with the tracked position, scale, roll and yaw. Because the
frame is built from the tracked face parameters, the effects translate,
scale and rotate with the head, and are horizontally foreshortened when the
head turns (yaw).

Per player identity:
  * P1 (BLAZE): fighter's headband on the forehead with tails that trail
    behind the head turn direction.
  * P2 (VOLT): translucent visor over the eyes with lightning horns.
Status effects (the "additional visual effect indicating player status"):
  * a health ring gauge floating above the head (green -> red),
  * blue shield while blocking, red flash when hit, energy sparks while the
    special is charging, dizzy stars on KO, a crown for the round winner.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from vision.face_tracking import FaceStatus
from vision.pipeline import FaceSnapshot


@dataclass
class PlayerStatus:
    health: float = 1.0          # 0..1
    blocking: bool = False
    hit_flash: float = 0.0       # 0..1, decays after being hit
    special_charge: float = 0.0  # 0..1
    ko: bool = False
    winner: bool = False


class FaceFrame:
    """Face-local -> panel pixel transform."""

    def __init__(self, center: tuple[float, float], scale: float, roll: float, yaw: float) -> None:
        self.c = np.array(center, dtype=np.float64)
        self.s = scale
        self.roll = roll
        self.yaw = yaw
        self.R = np.array([[math.cos(roll), -math.sin(roll)], [math.sin(roll), math.cos(roll)]])
        self.squash = max(0.35, math.cos(yaw))

    def pt(self, u: float, v: float, depth: float = 0.0) -> tuple[int, int]:
        """(u, v) in face units; depth > 0 shifts with yaw (points in front of the face)."""
        x = u * self.squash * self.s + math.sin(self.yaw) * depth * self.s
        y = v * self.s
        p = self.c + self.R @ np.array([x, y])
        return int(round(p[0])), int(round(p[1]))

    def poly(self, pts, depth: float = 0.0) -> np.ndarray:
        return np.array([self.pt(u, v, depth) for u, v in pts], dtype=np.int32)


def _blend(base: np.ndarray, overlay: np.ndarray, alpha: float) -> None:
    if alpha <= 0:
        return
    cv2.addWeighted(overlay, alpha, base, 1 - alpha, 0, dst=base)


_HEXAGON = [(math.cos(a) * 1.9, math.sin(a) * 2.1 + 0.4)
            for a in np.linspace(0, math.tau, 7)[:-1] + math.pi / 6]


def health_color(h: float) -> tuple[int, int, int]:
    """BGR: green (full) -> yellow -> red (low)."""
    h = max(0.0, min(1.0, h))
    if h > 0.5:
        k = (h - 0.5) * 2
        return (40, 220, int(255 * (1 - k) + 60 * k))
    k = h * 2
    return (40, int(60 + 160 * k), 255)


def draw_face_effects(img: np.ndarray, face: FaceSnapshot, to_panel, panel_scale: float,
                      pid: int, status: PlayerStatus, clock: float) -> None:
    """Draw AR effects for one player's face into ``img`` (BGR, modified in place).

    ``to_panel`` maps normalised image coords -> panel pixels; ``panel_scale``
    converts normalised-x lengths to panel pixels.
    """
    if face.center is None or face.status == FaceStatus.LOST:
        return
    c = to_panel(face.center)
    s = face.scale * panel_scale
    if s < 4:
        return
    F = FaceFrame(c, s, face.roll, face.yaw)
    alpha = float(face.alpha)
    overlay = img.copy()
    AA = cv2.LINE_AA

    # ---------------------------------------------------- identity effect
    if pid == 0:
        # Headband anchored on the measured forehead point when available.
        if face.forehead is not None:
            fh = np.array(to_panel(face.forehead), dtype=np.float64)
            # Express the anchor in face units so the band follows head pitch.
            d = fh - F.c
            local_y = -d[0] * math.sin(F.roll) + d[1] * math.cos(F.roll)
            v0 = float(local_y / s) * 0.62
        else:
            v0 = -0.75
        band = F.poly([(-1.25, v0 - 0.16), (1.25, v0 - 0.16), (1.22, v0 + 0.16), (-1.22, v0 + 0.16)])
        cv2.fillConvexPoly(overlay, band, (245, 245, 245), AA)
        cv2.polylines(overlay, [band], True, (30, 30, 40), max(1, int(s * 0.05)), AA)
        cv2.circle(overlay, F.pt(0, v0), max(2, int(s * 0.13)), (40, 40, 220), -1, AA)
        # Tails trail opposite the head turn (and flutter).
        side = -1.0 if face.yaw >= 0 else 1.0
        knot = (side * 1.25, v0)
        for k in range(2):
            pts = [knot]
            for i in range(1, 6):
                wave = math.sin(clock * 10 + i + k * 1.3) * 0.08 * i
                pts.append((knot[0] + side * 0.32 * i, knot[1] + (0.12 + 0.1 * k) * i + wave))
            cv2.polylines(overlay, [F.poly(pts)], False, (245, 245, 245),
                          max(2, int(s * 0.14)), AA)
    else:
        visor = F.poly([(-1.35, -0.28), (1.35, -0.28), (1.15, 0.3), (-1.15, 0.3)], depth=0.3)
        cv2.fillConvexPoly(overlay, visor, (255, 200, 60), AA)
        cv2.polylines(overlay, [visor], True, (60, 230, 255), max(1, int(s * 0.07)), AA)
        # Lightning horns.
        for sg in (-1, 1):
            bolt = [(sg * 0.7, -1.0), (sg * 1.0, -1.6), (sg * 0.85, -1.55), (sg * 1.25, -2.25),
                    (sg * 0.95, -1.75), (sg * 1.1, -1.8), (sg * 0.75, -1.1)]
            cv2.fillPoly(overlay, [F.poly(bolt)], (60, 230, 255), AA)

    # ---------------------------------------------------- status effects
    # Health ring gauge above the head.
    ring_c = F.pt(0, -2.2)
    rad = max(4, int(s * 0.55))
    cv2.ellipse(overlay, ring_c, (rad, int(rad * 0.45)), math.degrees(face.roll), 0, 360,
                (40, 40, 40), max(2, int(s * 0.16)), AA)
    if status.health > 0:
        cv2.ellipse(overlay, ring_c, (rad, int(rad * 0.45)), math.degrees(face.roll), -90,
                    -90 + 360 * status.health, health_color(status.health),
                    max(2, int(s * 0.12)), AA)
    if status.blocking:
        cv2.fillConvexPoly(overlay, F.poly(_HEXAGON, depth=0.8), (255, 170, 60), AA)
    if status.special_charge > 0:
        for k in range(10):
            a = clock * 6 + k * math.tau / 10
            r = 2.0 + 0.4 * math.sin(clock * 15 + k)
            col = (40, 160, 255) if pid == 0 else (255, 230, 80)
            cv2.circle(overlay, F.pt(math.cos(a) * r, math.sin(a) * r * 1.2),
                       max(2, int(s * 0.12 * status.special_charge)), col, -1, AA)
    if status.ko:
        for k in range(4):
            a = clock * 4 + k * math.tau / 4
            p = F.pt(math.cos(a) * 1.6, -1.6 + math.sin(a) * 0.4)
            _star(overlay, p, max(3, int(s * 0.22)), (60, 230, 255))
    if status.winner:
        crown = [(-0.9, -1.5), (-0.9, -2.2), (-0.45, -1.8), (0, -2.4), (0.45, -1.8),
                 (0.9, -2.2), (0.9, -1.5)]
        cv2.fillPoly(overlay, [F.poly(crown)], (40, 200, 255), AA)
    _blend(img, overlay, 0.78 * max(alpha, 0.5))

    if status.hit_flash > 0:
        x0, y0 = F.pt(-1.8, -1.5)
        x1, y1 = F.pt(1.8, 2.4)
        x0, x1 = sorted((max(0, x0), min(img.shape[1], x1)))
        y0, y1 = sorted((max(0, y0), min(img.shape[0], y1)))
        if x1 > x0 and y1 > y0:
            roi = img[y0:y1, x0:x1]
            tint = np.empty_like(roi)
            tint[:] = (40, 40, 255)
            _blend(roi, tint, 0.35 * status.hit_flash)


def _star(img, center, r, color) -> None:
    pts = []
    for k in range(10):
        a = -math.pi / 2 + k * math.pi / 5
        rr = r if k % 2 == 0 else r * 0.45
        pts.append((int(center[0] + math.cos(a) * rr), int(center[1] + math.sin(a) * rr)))
    cv2.fillPoly(img, [np.array(pts, np.int32)], color, cv2.LINE_AA)
