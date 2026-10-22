"""HUD, menus, webcam panels and debug overlay."""
from __future__ import annotations

import math

import cv2
import numpy as np
import pygame

from game.fighter import Fighter, FighterState
from game.game_state import GameState, Phase
from rendering.characters import STYLES
from rendering.face_effects import PlayerStatus, draw_face_effects
from vision.face_tracking import FaceStatus
from vision.pipeline import PlayerSnapshot, VisionSnapshot
from vision.player_tracking import TrackState
from vision.pose_tracking import SKELETON_EDGES

WHITE = (245, 245, 245)
DARK = (16, 12, 22)
GOLD = (255, 205, 70)
PANEL_BG = (24, 20, 34)


class Fonts:
    def __init__(self) -> None:
        self._cache: dict[int, pygame.font.Font] = {}

    def get(self, size: int) -> pygame.font.Font:
        if size not in self._cache:
            self._cache[size] = pygame.font.Font(None, size)
        return self._cache[size]


FONTS: Fonts | None = None


def fonts() -> Fonts:
    global FONTS
    if FONTS is None:
        FONTS = Fonts()
    return FONTS


def text(surf, msg, size, pos, color=WHITE, anchor="center", outline=2, ocolor=DARK,
         alpha: int = 255) -> pygame.Rect:
    font = fonts().get(size)
    img = font.render(msg, True, color)
    rect = img.get_rect(**{anchor: pos})
    if outline:
        o = font.render(msg, True, ocolor)
        if alpha < 255:
            o.set_alpha(alpha)
        for dx in (-outline, 0, outline):
            for dy in (-outline, 0, outline):
                if dx or dy:
                    surf.blit(o, rect.move(dx, dy))
    if alpha < 255:
        img.set_alpha(alpha)
    surf.blit(img, rect)
    return rect


# ======================================================================= HUD
class HUD:
    def __init__(self, width: int) -> None:
        self.w = width
        self.trail = [1.0, 1.0]      # delayed (white) health for damage indication
        self.shown = [1.0, 1.0]
        self.trail_hold = [0.0, 0.0]

    def reset(self) -> None:
        self.trail = [1.0, 1.0]
        self.shown = [1.0, 1.0]

    def update(self, gs: GameState, dt: float) -> None:
        for i, f in enumerate(gs.fighters):
            h = f.health / f.cfg.max_health
            if h < self.shown[i] - 1e-6:
                self.trail_hold[i] = 0.45
            self.shown[i] = h
            if self.trail_hold[i] > 0:
                self.trail_hold[i] -= dt
            elif self.trail[i] > h:
                self.trail[i] = max(h, self.trail[i] - dt * 0.6)
            if self.trail[i] < h:
                self.trail[i] = h

    def _bar(self, surf, x0, x1, y, h, frac, trail, anchor_left, low_pulse) -> None:
        """Slanted health bar. Health is anchored at the outer edge, so damage
        appears next to the timer (classic arcade convention)."""
        slant = 14

        def shape(a: float, b: float):
            xa, xb = x0 + (x1 - x0) * a, x0 + (x1 - x0) * b
            return [(xa + slant, y), (xb + slant, y), (xb, y + h), (xa, y + h)]

        def span(f: float):
            return shape(0, f) if anchor_left else shape(1 - f, 1)

        frame = shape(0, 1)
        outer = [(frame[0][0] - 3, y - 4), (frame[1][0] + 5, y - 4),
                 (frame[2][0] + 3, y + h + 4), (frame[3][0] - 5, y + h + 4)]
        pygame.draw.polygon(surf, DARK, outer)
        pygame.draw.polygon(surf, (70, 20, 30), frame)
        if trail > 0:
            pygame.draw.polygon(surf, (255, 240, 220), span(trail))
        if frac > 0:
            col = (255, 210, 40) if frac > 0.25 else (255, 70 + int(60 * low_pulse), 40)
            pts = span(frac)
            pygame.draw.polygon(surf, col, pts)
            hl = [pts[0], pts[1], (pts[1][0] - 5, y + h * 0.38), (pts[0][0] - 5, y + h * 0.38)]
            pygame.draw.polygon(surf, (255, 245, 170) if frac > 0.25 else (255, 160, 120), hl)
        pygame.draw.polygon(surf, (250, 250, 250), frame, 2)

    def draw(self, surf, gs: GameState, clock: float, control_labels: tuple[str, str]) -> None:
        w = self.w
        pulse = 0.5 + 0.5 * math.sin(clock * 10)
        bar_w = 470
        # P1 bar: anchored left, drains towards the centre.
        self._bar(surf, 40, 40 + bar_w, 26, 30, self.shown[0], self.trail[0], anchor_left=True,
                  low_pulse=pulse)
        self._bar(surf, w - 40 - bar_w, w - 40, 26, 30, self.shown[1], self.trail[1], anchor_left=False,
                  low_pulse=pulse)
        for i, f in enumerate(gs.fighters):
            style = STYLES[i]
            left = i == 0
            x_name = 52 if left else w - 52
            anchor = "topleft" if left else "topright"
            text(surf, style.name, 34, (x_name, 62), WHITE, anchor)
            text(surf, control_labels[i], 20, (x_name + (110 if left else -110), 70),
                 style.accent, anchor, outline=1)
            # Round win pips.
            for r in range(gs.cfg.rounds_to_win):
                cx = (40 + bar_w - 16 - r * 26) if left else (w - 40 - bar_w + 16 + r * 26)
                won = gs.round_wins[i] > r
                pygame.draw.circle(surf, DARK, (cx, 78), 10)
                pygame.draw.circle(surf, GOLD if won else (70, 60, 80), (cx, 78), 8)
            # Energy meter.
            e = f.energy / f.cfg.max_energy
            ready = f.energy >= f.cfg.special_cost
            ex0 = 40 if left else w - 40 - 260
            pygame.draw.rect(surf, DARK, (ex0 - 2, 96, 264, 14), border_radius=4)
            fill = int(260 * e)
            col = style.accent if ready else (90, 110, 160)
            if ready:
                col = tuple(min(255, int(c * (0.8 + 0.2 * pulse))) for c in col)
            if left:
                pygame.draw.rect(surf, col, (ex0, 98, fill, 10), border_radius=3)
            else:
                pygame.draw.rect(surf, col, (ex0 + 260 - fill, 98, fill, 10), border_radius=3)
            # Mark the special cost threshold.
            mark = ex0 + int(260 * f.cfg.special_cost / f.cfg.max_energy)
            if not left:
                mark = ex0 + 260 - int(260 * f.cfg.special_cost / f.cfg.max_energy)
            pygame.draw.line(surf, WHITE, (mark, 96), (mark, 110), 2)
            if ready:
                text(surf, "SPECIAL READY", 20, (ex0 + (268 if left else -8), 103), style.accent,
                     "midleft" if left else "midright", outline=1)
        # Timer.
        tc = (w // 2, 50)
        pygame.draw.circle(surf, DARK, tc, 40)
        pygame.draw.circle(surf, (60, 40, 70), tc, 36)
        pygame.draw.circle(surf, GOLD, tc, 36, 3)
        secs = int(math.ceil(gs.time_left))
        col = (255, 90, 70) if secs <= 10 and gs.phase == Phase.FIGHT else WHITE
        text(surf, f"{secs:02d}", 52, tc, col)
        text(surf, f"ROUND {gs.round_number}", 20, (w // 2, 100), GOLD, outline=1)


# ================================================================== banners
class Banner:
    def __init__(self) -> None:
        self.msg = ""
        self.sub = ""
        self.time = 0.0
        self.duration = 0.0
        self.color = WHITE

    def show(self, msg: str, duration: float = 1.4, sub: str = "", color=WHITE) -> None:
        self.msg, self.sub, self.duration, self.time, self.color = msg, sub, duration, 0.0, color

    def update(self, dt: float) -> None:
        self.time += dt

    def draw(self, surf, center) -> None:
        if not self.msg or self.time > self.duration:
            return
        t = self.time
        scale = 1.6 - 0.6 * min(1.0, t / 0.18)
        alpha = int(255 * min(1.0, (self.duration - t) / 0.3))
        size = int(110 * scale)
        band = pygame.Surface((surf.get_width(), 150), pygame.SRCALPHA)
        band.fill((10, 6, 16, int(140 * alpha / 255)))
        surf.blit(band, (0, center[1] - 75))
        text(surf, self.msg, size, center, self.color, outline=4, alpha=alpha)
        if self.sub:
            text(surf, self.sub, 34, (center[0], center[1] + 55), GOLD, alpha=alpha)


# ============================================================ webcam panels
class WebcamPanel:
    """Shows one tracked player (crop that follows their identity) with face AR."""

    def __init__(self, pid: int, rect: pygame.Rect) -> None:
        self.pid = pid
        self.rect = rect
        self.crop_c: np.ndarray | None = None
        self.crop_w = 0.5
        self.show_pose = True
        self._last_surface: pygame.Surface | None = None
        self._last_frame_id = -1

    def _crop_target(self, p: PlayerSnapshot, aspect: float) -> tuple[np.ndarray, float]:
        """Crop centre (normalised image coords) and width (normalised-x): head and
        shoulders, so the face AR effects are large enough to inspect."""
        sw = max(p.shoulder_width, 0.04)
        width = float(np.clip(sw * 3.4, 0.15, 1.0))
        cx, cy = p.center
        head = p.face.center
        if head is not None:
            # Face ~38 % from the top of the panel (room for the status ring above it).
            cx = 0.5 * cx + 0.5 * head[0]
            panel_h_norm = width * aspect / (self.rect.width / self.rect.height)
            cy = head[1] + 0.12 * panel_h_norm
        return np.array([cx, cy]), width

    def render(self, surf: pygame.Surface, snap: VisionSnapshot | None, status: PlayerStatus,
               clock: float, mode_label: str) -> None:
        r = self.rect
        style = STYLES[self.pid]
        pygame.draw.rect(surf, DARK, r.inflate(8, 8), border_radius=10)
        pygame.draw.rect(surf, PANEL_BG, r, border_radius=8)
        player = snap.players[self.pid] if snap and self.pid < len(snap.players) else None
        if snap is None or snap.image is None:
            text(surf, mode_label, 30, r.center, (170, 160, 190))
        elif player is None or player.state == TrackState.UNREGISTERED:
            self._blit_frame(surf, snap, None, status, clock)
            text(surf, "WAITING FOR PLAYER", 26, r.center, GOLD)
        else:
            self._blit_frame(surf, snap, player, status, clock)
        # Frame + label.
        pygame.draw.rect(surf, style.accent, r, 3, border_radius=8)
        label = pygame.Rect(r.left, r.top - 2, 150, 26)
        pygame.draw.rect(surf, style.accent, label, border_top_left_radius=8,
                         border_bottom_right_radius=10)
        text(surf, f"P{self.pid + 1}  {style.name}", 24, (label.left + 10, label.centery), DARK,
             "midleft", outline=0)
        if player is not None and snap is not None and snap.image is not None:
            self._status_line(surf, player)

    def _status_line(self, surf, p: PlayerSnapshot) -> None:
        r = self.rect
        state = p.state.value.upper()
        col = {TrackState.ACTIVE: (90, 230, 120), TrackState.LOST: (255, 190, 60)}.get(
            p.state, (255, 90, 90))
        face_col = {FaceStatus.TRACKED: (90, 230, 120), FaceStatus.PREDICTED: (255, 190, 60),
                    FaceStatus.POSE_FALLBACK: (120, 190, 255)}.get(p.face.status, (255, 90, 90))
        bar = pygame.Surface((r.width, 24), pygame.SRCALPHA)
        bar.fill((10, 8, 16, 190))
        surf.blit(bar, (r.left, r.bottom - 24))
        text(surf, f"BODY {state} {p.confidence:.2f}", 19, (r.left + 8, r.bottom - 12), col,
             "midleft", outline=0)
        text(surf, f"FACE {p.face.status.value.upper()}", 19, (r.right - 8, r.bottom - 12),
             face_col, "midright", outline=0)
        # Action lights (visual feedback of the recognised gesture state).
        x = r.right - 10
        if p.block:
            text(surf, "BLOCK", 20, (x, r.top + 14), (120, 200, 255), "midright", outline=1)
        if p.special_progress > 0:
            w = int(70 * p.special_progress)
            pygame.draw.rect(surf, GOLD, (r.right - 80, r.top + 26, w, 6))
        if abs(p.move) > 0.05:
            arrow = ">>" if p.move > 0 else "<<"
            text(surf, arrow, 26, (r.centerx, r.top + 14), GOLD, outline=1)

    def _blit_frame(self, surf, snap: VisionSnapshot, p: PlayerSnapshot | None,
                    status: PlayerStatus, clock: float) -> None:
        img = snap.image
        H, W = img.shape[:2]
        aspect = W / H
        r = self.rect
        panel_aspect = r.width / r.height
        if p is not None and p.state in (TrackState.ACTIVE, TrackState.LOST):
            tc, tw = self._crop_target(p, aspect)
            if self.crop_c is None:
                self.crop_c, self.crop_w = tc, tw
            else:  # smooth camera framing
                self.crop_c = self.crop_c + (tc - self.crop_c) * 0.2
                self.crop_w += (tw - self.crop_w) * 0.1
        elif self.crop_c is None:
            self.crop_c, self.crop_w = np.array([0.25 + 0.5 * self.pid, 0.5]), 0.5
        cw_px = self.crop_w * W
        ch_px = cw_px / panel_aspect
        cw_px = min(cw_px, W)
        ch_px = min(ch_px, H)
        cw_px = min(cw_px, ch_px * panel_aspect)
        x0 = float(np.clip(self.crop_c[0] * W - cw_px / 2, 0, W - cw_px))
        y0 = float(np.clip(self.crop_c[1] * H - ch_px / 2, 0, H - ch_px))
        crop = img[int(y0):int(y0 + ch_px), int(x0):int(x0 + cw_px)]
        if crop.size == 0:
            return
        k = r.width / crop.shape[1]
        panel = cv2.resize(crop, (r.width, r.height), interpolation=cv2.INTER_LINEAR)

        def to_panel(pt):
            return ((pt[0] * W - x0) * k, (pt[1] * H - y0) * k)

        if p is not None and p.state == TrackState.ACTIVE:
            if self.show_pose:
                self._draw_pose(panel, p, to_panel)
            draw_face_effects(panel, p.face, to_panel, W * k, self.pid, status, clock)
        if p is not None and p.state == TrackState.LOST:
            cv2.putText(panel, "TRACKING LOST", (10, r.height // 2), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (60, 190, 255), 2, cv2.LINE_AA)
        rgb = cv2.cvtColor(panel, cv2.COLOR_BGR2RGB)
        frame = pygame.image.frombuffer(rgb.tobytes(), (r.width, r.height), "RGB")
        surf.blit(frame, r.topleft)

    @staticmethod
    def _draw_pose(panel, p: PlayerSnapshot, to_panel) -> None:
        lm = p.landmarks
        col = (90, 220, 255) if p.pid == 0 else (255, 200, 90)
        for a, b in SKELETON_EDGES:
            if np.isnan(lm[a]).any() or np.isnan(lm[b]).any():
                continue
            pa, pb = to_panel(lm[a]), to_panel(lm[b])
            cv2.line(panel, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])), col, 2, cv2.LINE_AA)
        for i in range(lm.shape[0]):
            if np.isnan(lm[i]).any():
                continue
            q = to_panel(lm[i])
            # Filled = observed this frame, hollow = held through a short dropout.
            cv2.circle(panel, (int(q[0]), int(q[1])), 3, col, -1 if p.observed[i] else 1, cv2.LINE_AA)


# ============================================================ debug overlay
def draw_debug(surf: pygame.Surface, snap: VisionSnapshot | None, perf_lines: list[str],
               rect: pygame.Rect) -> None:
    bg = pygame.Surface(rect.size, pygame.SRCALPHA)
    bg.fill((8, 6, 14, 200))
    surf.blit(bg, rect.topleft)
    y = rect.top + 10
    for line in perf_lines:
        text(surf, line, 20, (rect.left + 10, y), (200, 255, 200), "topleft", outline=0)
        y += 18
    if snap is None or snap.image is None:
        return
    for p in snap.players:
        s = p.signals
        line = (f"P{p.pid + 1} {p.state.value:<8} conf {p.confidence:.2f} move {p.move:+.2f}"
                f" off {p.move_offset:+.2f}")
        text(surf, line, 20, (rect.left + 10, y), (255, 230, 160), "topleft", outline=0)
        y += 18
        st = {0.0: "armed", 2.0: "recover"}
        line = (f"   L {st.get(s.get('state_left'), '-'):<7} elbow {s.get('elbow_left', float('nan')):3.0f}"
                f" hook {s.get('hook_left', 0.0):+.2f} rise {s.get('rise_left', 0.0):+.2f}"
                f"   R {st.get(s.get('state_right'), '-'):<7} elbow {s.get('elbow_right', float('nan')):3.0f}"
                f" hook {s.get('hook_right', 0.0):+.2f} rise {s.get('rise_right', 0.0):+.2f}")
        text(surf, line, 20, (rect.left + 10, y), (230, 230, 230), "topleft", outline=0)
        y += 18
        line = (f"   block {s.get('block_dist', float('nan')):.2f} raise {s.get('raise', float('nan')):+.2f}"
                f" face {p.face.status.value} miss {','.join(p.missing_joints) or '-'}")
        text(surf, line, 20, (rect.left + 10, y), (230, 230, 230), "topleft", outline=0)
        y += 18
    # Full camera view with both skeletons and identities.
    img = snap.image.copy()
    H, W = img.shape[:2]
    for p in snap.players:
        if p.state != TrackState.ACTIVE:
            continue
        col = (90, 220, 255) if p.pid == 0 else (255, 200, 90)
        WebcamPanel._draw_pose(img, p, lambda pt: (pt[0] * W, pt[1] * H))
        cx, cy = int(p.center[0] * W), int(p.center[1] * H)
        cv2.putText(img, f"P{p.pid + 1}", (cx - 15, cy), cv2.FONT_HERSHEY_SIMPLEX, 1.0, col, 3, cv2.LINE_AA)
    tw = rect.width - 20
    th = int(tw * H / W)
    if y + th > rect.bottom:
        th = rect.bottom - y - 10
        tw = int(th * W / H)
    if th > 20:
        small = cv2.resize(img, (tw, th))
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        surf.blit(pygame.image.frombuffer(rgb.tobytes(), (tw, th), "RGB"), (rect.left + 10, y + 6))


def status_for(f: Fighter, hit_flash: float, winner: bool, special_progress: float) -> PlayerStatus:
    return PlayerStatus(health=f.health / f.cfg.max_health, blocking=f.is_blocking,
                        hit_flash=hit_flash, special_charge=special_progress,
                        ko=f.state == FighterState.KO, winner=winner)
