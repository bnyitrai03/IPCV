"""Task 5 — real-time integration for the two-player webcam fight.

The game starts directly in player registration, then fights. The main loop
runs at the display rate (60 Hz target) with a fixed-step game-logic
accumulator. Vision runs in its own thread; the loop only reads the latest
snapshot and drains gesture events, so slow inference can never stall
rendering.
"""
from __future__ import annotations

import logging
import time

import cv2
import numpy as np
import pygame

from config import Config
from evaluation.gesture_metrics import GestureLogger
from evaluation.performance import PerformanceMonitor
from evaluation.session import Session
from evaluation.tracking_metrics import TrackingLogger
from fight_session import FightPhase, FightSession
from game.controls import key_is
from game.fighter import SPECIAL_CHARGE, FighterState
from game.game_state import Phase
from rendering.arena import Arena
from rendering.characters import STYLES, FighterRenderer
from rendering.effects import EffectsManager
from rendering.interface import (DARK, GOLD, HUD, WHITE, Banner, WebcamPanel, draw_debug,
                                 status_for, text)
from rendering.sound import SoundBank
from vision.pipeline import VisionError, VisionPipeline, VisionSnapshot
from vision.player_tracking import TrackState
from vision.pose_tracking import SKELETON_EDGES

log = logging.getLogger(__name__)


class App:
    def __init__(self, cfg: Config, vision_source: str = "camera", log_session: bool = False,
                 debug: bool = False, max_frames: int | None = None,
                 screenshot: str | None = None) -> None:
        self.cfg = cfg
        g = cfg.game
        pygame.init()
        pygame.display.set_caption("Webcam Warriors - IPCV 2026")
        flags = pygame.SCALED | pygame.RESIZABLE
        if g.fullscreen:
            flags |= pygame.FULLSCREEN
        try:
            self.screen = pygame.display.set_mode((g.window_width, g.window_height), flags, vsync=1)
        except pygame.error:
            self.screen = pygame.display.set_mode((g.window_width, g.window_height))
        self.W, self.H = g.window_width, g.window_height
        self.arena_h = g.arena_height
        self.ground_y = self.arena_h - 64
        self.arena_surf = pygame.Surface((self.W, self.arena_h))
        self.arena = Arena(self.W, self.arena_h, self.ground_y)
        self.effects = EffectsManager()
        self.hud = HUD(self.W)
        self.banner = Banner()
        self.sound = SoundBank(g.sound)
        self.renderers = [FighterRenderer(STYLES[0]), FighterRenderer(STYLES[1])]
        pw, ph = 320, 180
        top = self.arena_h + 10
        self.panels = [WebcamPanel(0, pygame.Rect(24, top, pw, ph)),
                       WebcamPanel(1, pygame.Rect(self.W - 24 - pw, top, pw, ph))]
        self.perf = PerformanceMonitor()
        self.session = FightSession(self.cfg, (STYLES[0].name, STYLES[1].name),
                                    arena_left=60, arena_right=self.W - 60,
                                    ground_y=self.ground_y)
        self.vision_source = vision_source
        self.vision: VisionPipeline | None = None
        self.vision_error: str | None = None
        self.debug = debug
        self.pause_index = 0
        self.clock_t = 0.0
        self.hit_flash = [0.0, 0.0]
        self.eval_session: Session | None = None
        self.tracking_logger: TrackingLogger | None = None
        self.gesture_logger: GestureLogger | None = None
        self.max_frames = max_frames
        self.screenshot = screenshot
        self.frame_count = 0
        self.running = True
        self._winner_banner_at: float | None = None
        self._winner_text = ""
        if log_session:
            self._open_session("game")

    # ------------------------------------------------------------ helpers
    def _open_session(self, tag: str) -> None:
        if self.eval_session is None:
            self.eval_session = Session(self.cfg, tag)
            self.gesture_logger = GestureLogger(self.eval_session)
            self._perf_writer = self.eval_session.writer("performance", PerformanceMonitor.HEADER)
            log.info("Logging to %s", self.eval_session.dir)
            if self.vision is not None:
                self._attach_tracking_logger()

    def _attach_tracking_logger(self) -> None:
        if self.eval_session and self.vision and self.tracking_logger is None:
            self.tracking_logger = TrackingLogger(self.eval_session)
            self.vision.frame_listeners.append(self.tracking_logger.on_frame)

    def _ensure_vision(self) -> bool:
        if self.vision is not None:
            return self.vision.error is None
        vp = VisionPipeline(self.cfg, self.vision_source)
        try:
            vp.start()
        except VisionError as exc:
            self.vision_error = str(exc)
            log.error("Webcam control unavailable: %s", exc)
            print("\n" + "=" * 70 + "\nWEBCAM MODE COULD NOT START:\n  " + str(exc) +
                  "\nRun:  python scripts/check_setup.py   to find and fix the problem.\n" + "=" * 70,
                  flush=True)
            return False
        self.vision = vp
        self.vision_error = None
        self._attach_tracking_logger()
        return True

    def _on_fight_start(self) -> None:
        self.hud.reset()
        self.effects.clear()
        self.hit_flash = [0.0, 0.0]

    # --------------------------------------------------------------- loop
    def run(self) -> None:
        clock = pygame.time.Clock()
        step = 1.0 / self.cfg.game.logic_hz
        acc = 0.0
        last = time.perf_counter()
        self._ensure_vision()
        while self.running:
            now = time.perf_counter()
            dt = min(now - last, 0.1)
            last = now
            self.perf.frame(dt)
            self.clock_t += dt
            for event in pygame.event.get():
                self._handle_event(event)
            if self.vision and self.vision.error:
                self.vision_error = self.vision.error
            snap = self.vision.snapshot() if self.vision else None
            events = self.vision.drain_events() if self.vision else []
            if self.session.observe(snap, events, now):
                self._on_fight_start()
            if self.session.phase == FightPhase.FIGHT:
                if self.effects.hitstop > 0:
                    acc = 0.0
                else:
                    acc += dt
                    while acc >= step:
                        self._logic_step(step)
                        acc -= step
                        if self.session.phase != FightPhase.FIGHT:
                            acc = 0.0
                            break
            self.effects.update(dt)
            self.banner.update(dt)
            self.hud.update(self.session.state, dt)
            self.hit_flash = [max(0.0, h - dt * 2.5) for h in self.hit_flash]
            self._render(dt, snap)
            pygame.display.flip()
            flip_ts = time.perf_counter()
            self.perf.on_display(snap, flip_ts)
            row = self.perf.tick(self.vision.vision_fps if self.vision else 0.0,
                                 {"particles": len(self.effects.particles)})
            if row and self.eval_session:
                self._perf_writer.write({**row, "t": self.eval_session.rel(row["t"])})
            self.frame_count += 1
            if self.max_frames and self.frame_count >= self.max_frames:
                if self.screenshot:
                    pygame.image.save(self.screen, self.screenshot)
                self.running = False
            clock.tick(self.cfg.game.target_fps)
        self.shutdown()

    def shutdown(self) -> None:
        if self.vision:
            self.vision.stop()
        if self.eval_session:
            self.eval_session.close()
            log.info("Session logs: %s", self.eval_session.dir)
            print(f"\nLogs written to: {self.eval_session.dir}\n"
                  f"Analyse with:   python scripts/analyze_session.py \"{self.eval_session.dir}\"")
        pygame.quit()

    # -------------------------------------------------------------- input
    def _handle_event(self, event) -> None:
        if event.type == pygame.QUIT:
            self.running = False
            return
        if event.type != pygame.KEYDOWN:
            return
        key = event.key
        if key == pygame.K_F1:
            self.debug = not self.debug
        elif key == pygame.K_F2:
            for p in self.panels:
                p.show_pose = not p.show_pose
        elif key in (pygame.K_F5, pygame.K_F6, pygame.K_F7, pygame.K_F8):
            self._annotate(key)

        phase = self.session.phase
        if phase == FightPhase.CALIBRATING:
            if key == pygame.K_ESCAPE:
                self.running = False
            elif key == pygame.K_RETURN:
                self.session.request_start()
            elif key_is(event, "R") and self.vision:
                self.vision.request("reregister")
                self.session.reset_calibration()
        elif phase == FightPhase.PAUSED:
            if key in (pygame.K_UP, pygame.K_DOWN):
                self.pause_index = (self.pause_index + (1 if key == pygame.K_DOWN else -1)) % 3
            elif key == pygame.K_RETURN:
                choice = self.pause_index
                if choice == 0:
                    self.session.resume()
                elif choice == 1:
                    self.session.restart()
                    self._on_fight_start()
                else:
                    self.running = False
            elif key == pygame.K_ESCAPE:
                self.session.resume()
        elif phase == FightPhase.MATCH_OVER:
            if key == pygame.K_RETURN:
                self.session.restart()
                self._on_fight_start()
            elif key == pygame.K_ESCAPE:
                self.running = False
        elif phase == FightPhase.FIGHT:
            if key == pygame.K_ESCAPE:
                self.session.pause()
                self.pause_index = 0

    def _annotate(self, key: int) -> None:
        if self.tracking_logger is None:
            return
        now = time.perf_counter()
        if key == pygame.K_F5:
            label = "stationary_end" if self.tracking_logger.stationary else "stationary_start"
        else:
            label = {pygame.K_F6: "identity_switch_observed", pygame.K_F7: "crossing_ok",
                     pygame.K_F8: "crossing_failed"}[key]
        self.tracking_logger.annotate(now, pygame.key.name(key), label)
        self.banner.show(label.replace("_", " ").upper(), 0.7, color=GOLD)

    def _logic_step(self, step: float) -> None:
        applied_ts = time.perf_counter()
        phase_before = self.session.state.phase
        events = self.session.step(step)
        vision_events = self.session.take_vision_events()
        if vision_events and self.gesture_logger:
            executed = {e.data.get("source_ts"): e.data["action"] for e in events
                        if e.kind == "action" and e.data.get("source_ts") is not None}
            for ev in vision_events:
                if ev.gesture in ("block_start", "block_end"):
                    result = "held_state"
                elif phase_before != Phase.FIGHT:
                    result = "ignored_not_fighting"
                else:
                    act = executed.get(ev.capture_ts)
                    result = ("executed" if act and act != "special_denied"
                              else "denied_no_energy" if act == "special_denied"
                              else "not_executed_busy")
                self.gesture_logger.log(ev, applied_ts, result)
        self._game_events(events)
        for f in self.session.state.fighters:
            style = STYLES[f.player_id]
            if f.state == FighterState.SPECIAL and f.state_time < SPECIAL_CHARGE:
                self.effects.charge(f.x + f.facing * 30, f.ground_y - 140, style.accent)
            if f.state == FighterState.WALK and int(f.total_time * 6) != int((f.total_time - step) * 6):
                self.effects.dust(f.x, f.ground_y, 1 if f.walk_dir > 0 else -1)
        for p in self.session.state.projectiles:
            self.effects.projectile_trail(p, STYLES[p.owner].accent)

    def _game_events(self, events) -> None:
        for e in events:
            d = e.data
            if e.kind == "round_intro":
                self.banner.show(f"ROUND {d['round']}", 1.6, color=GOLD)
                self.sound.play("bell")
            elif e.kind == "fight":
                self.banner.show("FIGHT!", 0.9, color=(255, 90, 60))
            elif e.kind == "action" and d["action"].split("_")[0] in ("hook", "uppercut"):
                self.sound.play("whoosh")
            elif e.kind == "action" and d["action"] == "special_denied":
                f = self.session.state.fighters[d["player"]]
                self.effects.text("NO ENERGY", f.x, f.ground_y - 250, (160, 170, 200), 28, 0.6)
            elif e.kind == "special_fire":
                self.sound.play("special")
                self.effects.flash(60, STYLES[d["player"]].accent)
            elif e.kind in ("hit", "block"):
                atk = d["attacker"]
                blocked = e.kind == "block"
                color = STYLES[atk].accent
                strength = {"special": 1.6, "uppercut": 1.4, "hook": 1.15}.get(
                    d["attack"].split("_")[0], 1.0)
                self.effects.impact(d["x"], d["y"], color, strength, blocked)
                self.sound.play("block" if blocked else "hit")
                if not blocked:
                    self.hit_flash[d["defender"]] = 1.0
                    self.arena.pulse(0.25 * strength)
                    self.effects.text(f"-{d['damage']:.0f}", d["x"], d["y"] - 30, (255, 240, 120), 34)
                    if d["attack"].startswith("uppercut"):
                        self.effects.text("UPPERCUT!", d["x"], d["y"] - 100, (255, 200, 60), 44, 0.9)
                        self.effects.shake(9, 0.25)
                    elif d["attack"].startswith("hook"):
                        self.effects.text("HOOK!", d["x"], d["y"] - 100, (255, 230, 120), 38, 0.7)
                    if d["counter"]:
                        self.effects.text("COUNTER!", d["x"], d["y"] - 70, (255, 120, 60), 40, 1.0)
                    if d["combo"] >= 2:
                        cx = 200 if atk == 0 else self.W - 200
                        self.effects.text(f"{d['combo']} HITS", cx, 200, GOLD, 56, 1.0)
                else:
                    self.effects.text("BLOCK", d["x"], d["y"] - 40, (140, 200, 255), 30, 0.6)
                if d.get("ko"):
                    self.effects.ko_burst(d["x"], d["y"], color)
            elif e.kind == "clash":
                self.effects.impact(d["x"], d["y"], (255, 255, 255), 1.8)
                self.sound.play("hit")
            elif e.kind == "round_over":
                if d["reason"] in ("ko", "double_ko"):
                    self.banner.show("K.O.", 2.0, color=(255, 60, 50))
                    self.sound.play("ko")
                else:
                    self.banner.show("TIME", 1.2, color=WHITE)
                w = d["winner"]
                self._winner_banner_at = self.clock_t + 1.4
                self._winner_text = "DRAW" if w < 0 else f"{STYLES[w].name} WINS"
            elif e.kind == "match_over":
                w = d["winner"]
                msg = "DRAW GAME" if w < 0 else f"{STYLES[w].name} WINS THE MATCH"
                self.banner.show(msg, 1e9, sub="ENTER: rematch    ESC: quit", color=GOLD)
        if self._winner_banner_at is not None and self.clock_t >= self._winner_banner_at:
            self.banner.show(self._winner_text, 1.6, color=GOLD)
            self._winner_banner_at = None

    # ------------------------------------------------------------- render
    def _render(self, dt: float, snap: VisionSnapshot | None) -> None:
        if self.session.phase == FightPhase.CALIBRATING:
            self._render_calibration(snap)
        else:
            self._render_arena(dt)
            self.hud.draw(self.arena_surf, self.session.state, self.clock_t,
                          self._control_labels(snap))
            self.banner.draw(self.arena_surf, (self.W // 2, self.arena_h // 2 - 20))
            self.screen.blit(self.arena_surf, (0, 0))
            self._render_bottom(snap)
            if self.session.phase == FightPhase.PAUSED:
                self._render_pause()
        if self.debug:
            r = pygame.Rect(self.W - 520, 120, 500, 430)
            lines = self.perf.overlay_lines()
            if self.vision and snap:
                lines.append(f"source {snap.source}  poses {snap.num_poses}  faces {snap.num_faces}"
                             f"  overlap {snap.overlap}")
            draw_debug(self.screen, snap, lines, r)
        self.effects.draw_flash(self.screen)

    def _render_arena(self, dt: float) -> None:
        surf = self.arena_surf
        shake = self.effects.offset()
        self.arena.draw(surf, self.clock_t, shake, dt)
        fighters = sorted(self.session.state.fighters,
                          key=lambda f: f.state in (FighterState.PUNCH, FighterState.SPECIAL))
        for f in fighters:
            r = self.renderers[f.player_id]
            r.draw(surf, f, self.clock_t, dt, shake)
            self._attack_trail(f, r)
        for p in self.session.state.projectiles:
            self.effects.draw_projectile(surf, p, STYLES[p.owner].accent, self.clock_t, shake)
        self.effects.draw(surf, shake)

    def _attack_trail(self, f, renderer) -> None:
        """Motion trail on the striking fist so hook and uppercut read differently."""
        a = f.attack
        if f.state != FighterState.PUNCH or a is None:
            return
        spec = a.spec
        if not (spec.startup * 0.6 <= f.state_time <= spec.startup + spec.active + 0.06):
            return
        fist = renderer.joints.get("wr_f" if a.side == "left" else "wr_b")
        if fist is None:
            return
        color = STYLES[f.player_id].accent
        self.effects.swing_trail(fist[0], fist[1], spec.name, color, f.facing)
        if spec.name == "uppercut" and abs(f.state_time - spec.startup) < 0.02:
            self.effects.dust(f.x, f.ground_y, f.facing)
            self.effects.dust(f.x, f.ground_y, -f.facing)

    def _control_labels(self, snap: VisionSnapshot | None) -> tuple[str, str]:
        labels = []
        for pid in (0, 1):
            state = None
            if snap is not None and pid < len(snap.players):
                state = snap.players[pid].state
            if state == TrackState.ACTIVE:
                labels.append("WEBCAM")
            elif state == TrackState.LOST:
                labels.append("WEBCAM: LOST")
            else:
                labels.append("NO TRACK")
        return labels[0], labels[1]

    def _render_bottom(self, snap: VisionSnapshot | None) -> None:
        top = self.arena_h
        strip = pygame.Rect(0, top, self.W, self.H - top)
        pygame.draw.rect(self.screen, (20, 16, 28), strip)
        pygame.draw.line(self.screen, GOLD, (0, top), (self.W, top), 3)
        for pid, panel in enumerate(self.panels):
            f = self.session.state.fighters[pid]
            sp = 0.0
            if snap and pid < len(snap.players) and snap.players[pid].special_progress > 0:
                sp = snap.players[pid].special_progress
            elif f.state == FighterState.SPECIAL and f.state_time < SPECIAL_CHARGE:
                sp = f.state_time / SPECIAL_CHARGE
            winner = self.session.state.phase in (Phase.ROUND_OVER, Phase.MATCH_OVER) and \
                self.session.state.round_winner == pid
            status = status_for(f, self.hit_flash[pid], winner, sp)
            label = "NO CAMERA" if snap is None or snap.image is None else ""
            panel.render(self.screen, snap, status, self.clock_t, label)
        cx = self.W // 2
        y = top + 26
        hints = ["HOOK: elbow bent 90°, swing fist across your body",
                 "UPPERCUT: one fist up and forward to head height",
                 "BLOCK: both hands to face",
                 "SPECIAL: both arms up & hold",
                 "MOVE: step or lean left / right"]
        for h in hints:
            text(self.screen, h, 24, (cx, y), (215, 205, 230), outline=0)
            y += 26
        text(self.screen, "ESC pause   F1 debug   F2 skeleton", 20,
             (cx, self.H - 20), (140, 130, 160), outline=0)

    def _render_pause(self) -> None:
        overlay = pygame.Surface((self.W, self.H), pygame.SRCALPHA)
        overlay.fill((8, 6, 14, 170))
        self.screen.blit(overlay, (0, 0))
        text(self.screen, "PAUSED", 90, (self.W // 2, 180), GOLD, outline=4)
        for i, label in enumerate(("RESUME", "RESTART MATCH", "QUIT")):
            sel = i == self.pause_index
            text(self.screen, ("> " if sel else "") + label, 44, (self.W // 2, 280 + i * 56),
                 GOLD if sel else WHITE)

    def _camera_view(self, snap: VisionSnapshot | None, rect: pygame.Rect) -> None:
        pygame.draw.rect(self.screen, DARK, rect.inflate(10, 10), border_radius=10)
        if snap is None or snap.image is None:
            text(self.screen, "Starting camera...", 40, rect.center, WHITE)
            return
        img = snap.image.copy()
        H, W = img.shape[:2]
        colors = [(90, 220, 255), (255, 200, 90)]
        for p in snap.players:
            if p.state != TrackState.ACTIVE:
                continue
            lm = p.landmarks
            for a, b in SKELETON_EDGES:
                if np.isnan(lm[a]).any() or np.isnan(lm[b]).any():
                    continue
                cv2.line(img, (int(lm[a, 0] * W), int(lm[a, 1] * H)),
                         (int(lm[b, 0] * W), int(lm[b, 1] * H)), colors[p.pid], 3, cv2.LINE_AA)
            cx = int(p.center[0] * W)
            top = int(p.bbox[1] * H) if p.bbox else int(p.center[1] * H)
            label = f"P{p.pid + 1} {STYLES[p.pid].name}"
            (tw_, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            org = (cx - tw_ // 2, max(24, top - 30))
            cv2.putText(img, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 5, cv2.LINE_AA)
            cv2.putText(img, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.8, colors[p.pid], 2, cv2.LINE_AA)
        th = rect.height
        tw = int(th * W / H)
        if tw > rect.width:
            tw = rect.width
            th = int(tw * H / W)
        img = cv2.resize(img, (tw, th))
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        surf = pygame.image.frombuffer(rgb.tobytes(), (tw, th), "RGB")
        self.screen.blit(surf, (rect.centerx - tw // 2, rect.top))

    def _render_calibration(self, snap: VisionSnapshot | None) -> None:
        self.screen.fill((18, 14, 26))
        text(self.screen, "PLAYER REGISTRATION", 64, (self.W // 2, 50), GOLD, outline=3)
        rect = pygame.Rect(0, 0, 840, 472)
        rect.midtop = (self.W // 2, 95)
        self._camera_view(snap, rect)
        y = rect.bottom + 30
        can_start = self.session.can_start_early(snap)
        if self.vision_error:
            for k, line in enumerate(_wrap(self.vision_error, 100)[:3]):
                text(self.screen, line, 26, (self.W // 2, y + k * 26), (255, 140, 120))
        else:
            now = time.perf_counter()
            remaining = self.session.calibration_remaining(now)
            if remaining is not None:
                text(self.screen, f"BOTH PLAYERS REGISTERED - starting in {remaining:.1f}s", 34,
                     (self.W // 2, y), (120, 240, 140))
            else:
                n = snap.num_poses if snap else 0
                hint = ("Player 1 stands on the LEFT, Player 2 on the RIGHT. Face the camera,"
                        " upper body and hips visible, arms relaxed.")
                text(self.screen, hint, 26, (self.W // 2, y), WHITE)
                col = GOLD if n < 2 else (120, 240, 140)
                status = f"People detected: {n}/2"
                if n == 0:
                    status += "   - step back so your whole upper body is in the picture"
                elif n == 1:
                    status += "   - press ENTER to start solo; P2 can join later"
                else:
                    status += "   - hold still..."
                text(self.screen, status, 28, (self.W // 2, y + 34), col)
        footer = "ENTER: start now    R: re-register    ESC: quit"
        footer_col = (120, 240, 140) if can_start else (170, 160, 190)
        text(self.screen, footer, 22, (self.W // 2, self.H - 24), footer_col, outline=0)


def _wrap(msg: str, width: int) -> list[str]:
    words, lines, cur = msg.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return lines
