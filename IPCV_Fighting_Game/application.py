"""Task 5 — real-time integration: scenes, main loop, input routing, rendering.

Scenes: MENU -> CALIBRATION (webcam) -> FIGHT, and GESTURE_TEST.

The main loop runs at the display rate (60 Hz target) with a fixed-step
game-logic accumulator. Vision runs in its own thread; the loop only reads
the latest snapshot and drains gesture events, so slow inference can never
stall rendering.
"""
from __future__ import annotations

import logging
import math
import time
from enum import Enum, auto

import cv2
import numpy as np
import pygame

from config import Config
from evaluation.gesture_metrics import GestureLogger, GestureTrialRunner
from evaluation.performance import PerformanceMonitor
from evaluation.session import Session
from evaluation.tracking_metrics import TrackingLogger
from game.ai import SimpleAI
from game.controls import KeyboardController, PlayerInput
from game.fighter import SPECIAL_CHARGE, FighterState
from game.game_state import GameState, Phase
from rendering.arena import Arena
from rendering.characters import STYLES, FighterRenderer
from rendering.effects import EffectsManager
from rendering.face_effects import PlayerStatus
from rendering.interface import (DARK, GOLD, HUD, WHITE, Banner, WebcamPanel, draw_debug,
                                 status_for, text)
from rendering.sound import SoundBank
from vision.gesture_recognition import GestureEvent
from vision.pipeline import VisionError, VisionPipeline, VisionSnapshot
from vision.player_tracking import TrackState
from vision.pose_tracking import SKELETON_EDGES

log = logging.getLogger(__name__)


class Scene(Enum):
    MENU = auto()
    CALIBRATION = auto()
    FIGHT = auto()
    GESTURE_TEST = auto()


MENU_ITEMS = [("webcam", "FIGHT  -  WEBCAM CONTROL"), ("keyboard", "FIGHT  -  KEYBOARD"),
              ("cpu", "PRACTICE  -  VS CPU"), ("gesture_test", "GESTURE TEST (EVALUATION)"),
              ("quit", "QUIT")]


class App:
    def __init__(self, cfg: Config, vision_source: str = "camera", log_session: bool = False,
                 debug: bool = False, max_frames: int | None = None,
                 auto_start: str | None = None, screenshot: str | None = None) -> None:
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
        self.keyboard = KeyboardController()
        self.renderers = [FighterRenderer(STYLES[0]), FighterRenderer(STYLES[1])]
        pw, ph = 320, 180
        top = self.arena_h + 10
        self.panels = [WebcamPanel(0, pygame.Rect(24, top, pw, ph)),
                       WebcamPanel(1, pygame.Rect(self.W - 24 - pw, top, pw, ph))]
        self.perf = PerformanceMonitor()
        self.state = self._new_state()
        self.scene = Scene.MENU
        self.menu_index = 0
        self.vision_source = vision_source
        self.vision: VisionPipeline | None = None
        self.vision_error: str | None = None
        self.use_vision = False
        self.cpu_players: dict[int, SimpleAI] = {}
        self.attract_ai = [SimpleAI(1, 0.4), SimpleAI(2, 0.4)]
        self.debug = debug
        self.paused = False
        self.pause_index = 0
        self.clock_t = 0.0
        self.hit_flash = [0.0, 0.0]
        self.pending = [PlayerInput(), PlayerInput()]
        self.pending_vision: list[GestureEvent] = []
        self.log_session = log_session
        self.session: Session | None = None
        self.tracking_logger: TrackingLogger | None = None
        self.gesture_logger: GestureLogger | None = None
        self.trials: GestureTrialRunner | None = None
        self.after_calibration = Scene.FIGHT
        self.calib_ready_since: float | None = None
        self.max_frames = max_frames
        self.auto_start = auto_start
        self.screenshot = screenshot
        self.frame_count = 0
        self.running = True
        if log_session:
            self._open_session("game")

    # ------------------------------------------------------------ helpers
    def _new_state(self) -> GameState:
        gs = GameState(self.cfg.game, (STYLES[0].name, STYLES[1].name), arena_left=60,
                       arena_right=self.W - 60, ground_y=self.ground_y)
        return gs

    def _open_session(self, tag: str) -> None:
        if self.session is None:
            self.session = Session(self.cfg, tag)
            self.gesture_logger = GestureLogger(self.session)
            self._perf_writer = self.session.writer("performance", PerformanceMonitor.HEADER)
            log.info("Logging to %s", self.session.dir)
            if self.vision is not None:
                self._attach_tracking_logger()

    def _attach_tracking_logger(self) -> None:
        if self.session and self.vision and self.tracking_logger is None:
            self.tracking_logger = TrackingLogger(self.session)
            self.vision.frame_listeners.append(self.tracking_logger.on_frame)
            self.vision.gesture_listeners.append(self._on_gesture_thread)

    def _on_gesture_thread(self, ev: GestureEvent) -> None:
        # Gesture test runner gets events straight from the vision thread
        # (it only appends to lists; evaluated in the main thread).
        if self.trials is not None:
            self.trials.on_gesture(ev)

    def _ensure_vision(self) -> bool:
        if self.vision is not None:
            return self.vision.error is None
        vp = VisionPipeline(self.cfg, self.vision_source)
        try:
            vp.start()
        except VisionError as exc:
            self.vision_error = str(exc)
            log.error("Webcam control unavailable: %s", exc)
            return False
        self.vision = vp
        self.vision_error = None
        self._attach_tracking_logger()
        return True

    def _start_fight(self, use_vision: bool, cpu: bool = False) -> None:
        self.use_vision = use_vision
        self.cpu_players = {1: SimpleAI(5, 0.45)} if cpu else {}
        self.state = self._new_state()
        self.hud.reset()
        self.effects.clear()
        self.paused = False
        self.scene = Scene.FIGHT
        self.pending = [PlayerInput(), PlayerInput()]
        if self.vision:
            self.vision.drain_events()

    # --------------------------------------------------------------- loop
    def run(self) -> None:
        clock = pygame.time.Clock()
        step = 1.0 / self.cfg.game.logic_hz
        acc = 0.0
        last = time.perf_counter()
        if self.auto_start:
            self._activate(self.auto_start)
        while self.running:
            now = time.perf_counter()
            dt = min(now - last, 0.1)
            last = now
            self.perf.frame(dt)
            self.clock_t += dt
            for event in pygame.event.get():
                self._handle_event(event)
            if self.scene == Scene.FIGHT and not self.paused:
                self._collect_inputs()
                if self.effects.hitstop > 0:
                    acc = 0.0
                else:
                    acc += dt
                    while acc >= step:
                        self._logic_step(step)
                        acc -= step
            elif self.scene == Scene.MENU:
                self._attract_step(dt)
            elif self.scene == Scene.CALIBRATION:
                self._calibration_step()
            elif self.scene == Scene.GESTURE_TEST and self.trials:
                self.trials.update(time.perf_counter())
            self.effects.update(dt)
            self.banner.update(dt)
            self.hud.update(self.state, dt)
            self.hit_flash = [max(0.0, h - dt * 2.5) for h in self.hit_flash]
            self._render(dt)
            pygame.display.flip()
            flip_ts = time.perf_counter()
            snap = self.vision.snapshot() if self.vision else None
            self.perf.on_display(snap, flip_ts)
            row = self.perf.tick(self.vision.vision_fps if self.vision else 0.0,
                                 {"particles": len(self.effects.particles)})
            if row and self.session:
                self._perf_writer.write({**row, "t": self.session.rel(row["t"])})
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
        if self.session:
            self.session.close()
            log.info("Session logs: %s", self.session.dir)
            print(f"\nLogs written to: {self.session.dir}\n"
                  f"Analyse with:   python scripts/analyze_session.py \"{self.session.dir}\"")
        pygame.quit()

    # -------------------------------------------------------------- input
    def _activate(self, item: str) -> None:
        self.sound.play("select")
        if item == "quit":
            self.running = False
        elif item == "keyboard":
            self._start_fight(use_vision=False)
        elif item == "cpu":
            self._start_fight(use_vision=False, cpu=True)
        elif item in ("webcam", "gesture_test"):
            if item == "gesture_test":
                self._open_session("gesture_test")
            if self._ensure_vision():
                self.after_calibration = Scene.FIGHT if item == "webcam" else Scene.GESTURE_TEST
                self.scene = Scene.CALIBRATION
                self.calib_ready_since = None

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

        if self.scene == Scene.MENU:
            if key in (pygame.K_UP, pygame.K_w):
                self.menu_index = (self.menu_index - 1) % len(MENU_ITEMS)
            elif key in (pygame.K_DOWN, pygame.K_s):
                self.menu_index = (self.menu_index + 1) % len(MENU_ITEMS)
            elif key in (pygame.K_RETURN, pygame.K_SPACE):
                self._activate(MENU_ITEMS[self.menu_index][0])
            elif key == pygame.K_ESCAPE:
                self.running = False
        elif self.scene == Scene.CALIBRATION:
            if key == pygame.K_ESCAPE:
                self.scene = Scene.MENU
            elif key == pygame.K_r and self.vision:
                self.vision.request("reregister")
                self.calib_ready_since = None
            elif key == pygame.K_RETURN:
                self._finish_calibration()
        elif self.scene == Scene.GESTURE_TEST:
            if key == pygame.K_ESCAPE:
                self.trials = None
                self.scene = Scene.MENU
        elif self.scene == Scene.FIGHT:
            self.keyboard.handle_event(event)
            if self.paused:
                if key in (pygame.K_UP, pygame.K_DOWN):
                    self.pause_index = (self.pause_index + (1 if key == pygame.K_DOWN else -1)) % 3
                elif key == pygame.K_RETURN:
                    choice = self.pause_index
                    self.paused = False
                    if choice == 1:
                        self._start_fight(self.use_vision, bool(self.cpu_players))
                    elif choice == 2:
                        self.scene = Scene.MENU
                elif key == pygame.K_ESCAPE:
                    self.paused = False
                return
            if key == pygame.K_ESCAPE:
                if self.state.phase == Phase.MATCH_OVER:
                    self.scene = Scene.MENU
                else:
                    self.paused, self.pause_index = True, 0
            elif key == pygame.K_RETURN and self.state.phase == Phase.MATCH_OVER:
                self._start_fight(self.use_vision, bool(self.cpu_players))
            elif key == pygame.K_c and self.vision:
                self.vision.request("recalibrate")
                self.banner.show("RECALIBRATED", 0.8, color=GOLD)

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

    def _collect_inputs(self) -> None:
        """Merge keyboard, CPU and webcam inputs into pending per-player inputs."""
        snap = self.vision.snapshot() if (self.vision and self.use_vision) else None
        events = self.vision.drain_events() if (self.vision and self.use_vision) else []
        for pid in (0, 1):
            inp = self.keyboard.poll(pid)
            if pid in self.cpu_players:
                inp = self.cpu_players[pid].decide(self.state.fighters[pid],
                                                   self.state.fighters[1 - pid], 1 / 60)
            if snap is not None and pid < len(snap.players):
                p = snap.players[pid]
                if p.state == TrackState.ACTIVE:
                    inp = inp.merge(PlayerInput(move=p.move, block=p.block))
            self.pending[pid] = self.pending[pid].merge(inp)
            # Held values must reflect the *latest* state, not an OR over frames.
            self.pending[pid].move = inp.move
            self.pending[pid].block = inp.block
        for ev in events:
            pid = ev.player_id
            if ev.gesture in ("punch_left", "punch_right", "special"):
                vi = PlayerInput(**{ev.gesture: True})
                vi.source_ts[ev.gesture] = ev.capture_ts
                self.pending[pid] = self.pending[pid].merge(vi)
            self.pending_vision.append(ev)

    def _logic_step(self, step: float) -> None:
        inputs = (self.pending[0], self.pending[1])
        applied_ts = time.perf_counter()
        phase_before = self.state.phase
        events = self.state.update(step, inputs)
        # Keep held inputs, clear one-shot ones.
        self.pending = [PlayerInput(move=i.move, block=i.block) for i in inputs]
        if self.pending_vision:
            executed = {e.data.get("source_ts"): e.data["action"] for e in events
                        if e.kind == "action" and e.data.get("source_ts") is not None}
            for ev in self.pending_vision:
                if self.gesture_logger:
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
            self.pending_vision = []
        self._game_events(events)
        # Continuous effects.
        for f in self.state.fighters:
            style = STYLES[f.player_id]
            if f.state == FighterState.SPECIAL and f.state_time < SPECIAL_CHARGE:
                self.effects.charge(f.x + f.facing * 30, f.ground_y - 140, style.accent)
            if f.state == FighterState.WALK and int(f.total_time * 6) != int((f.total_time - step) * 6):
                self.effects.dust(f.x, f.ground_y, 1 if f.walk_dir > 0 else -1)
        for p in self.state.projectiles:
            self.effects.projectile_trail(p, STYLES[p.owner].accent)

    def _game_events(self, events) -> None:
        for e in events:
            d = e.data
            if e.kind == "round_intro":
                self.banner.show(f"ROUND {d['round']}", 1.6, color=GOLD)
                self.sound.play("bell")
            elif e.kind == "fight":
                self.banner.show("FIGHT!", 0.9, color=(255, 90, 60))
            elif e.kind == "action" and d["action"] in ("punch_left", "punch_right"):
                self.sound.play("whoosh")
            elif e.kind == "action" and d["action"] == "special_denied":
                f = self.state.fighters[d["player"]]
                self.effects.text("NO ENERGY", f.x, f.ground_y - 250, (160, 170, 200), 28, 0.6)
            elif e.kind == "special_fire":
                self.sound.play("special")
                self.effects.flash(60, STYLES[d["player"]].accent)
            elif e.kind in ("hit", "block"):
                atk = d["attacker"]
                blocked = e.kind == "block"
                color = STYLES[atk].accent
                strength = 1.6 if d["attack"] == "special" else 1.0
                self.effects.impact(d["x"], d["y"], color, strength, blocked)
                self.sound.play("block" if blocked else "hit")
                if not blocked:
                    self.hit_flash[d["defender"]] = 1.0
                    self.arena.pulse(0.25 * strength)
                    self.effects.text(f"-{d['damage']:.0f}", d["x"], d["y"] - 30, (255, 240, 120), 34)
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
                self.banner.show(msg, 1e9, sub="ENTER: rematch    ESC: menu", color=GOLD)
        if getattr(self, "_winner_banner_at", None) and self.clock_t >= self._winner_banner_at:
            self.banner.show(self._winner_text, 1.6, color=GOLD)
            self._winner_banner_at = None

    def _attract_step(self, dt: float) -> None:
        gs = self.state
        if gs.phase == Phase.MATCH_OVER or (gs.phase == Phase.ROUND_OVER and gs.phase_time > 2):
            self.state = self._new_state()
            gs = self.state
        inputs = (self.attract_ai[0].decide(gs.fighters[0], gs.fighters[1], dt),
                  self.attract_ai[1].decide(gs.fighters[1], gs.fighters[0], dt))
        events = gs.update(dt, inputs)
        for e in events:
            if e.kind in ("hit", "block"):
                self.effects.impact(e.data["x"], e.data["y"], STYLES[e.data["attacker"]].accent,
                                    1.0, e.kind == "block")
        self.effects.shake_time = 0.0   # keep the menu calm

    def _calibration_step(self) -> None:
        snap = self.vision.snapshot() if self.vision else None
        if self.vision and self.vision.error:
            self.vision_error = self.vision.error
        if snap and snap.registered:
            if self.calib_ready_since is None:
                self.calib_ready_since = time.perf_counter()
            elif time.perf_counter() - self.calib_ready_since > 2.5:
                self._finish_calibration()
        else:
            self.calib_ready_since = None

    def _finish_calibration(self) -> None:
        if self.vision:
            self.vision.request("recalibrate")
        if self.after_calibration == Scene.GESTURE_TEST:
            self._open_session("gesture_test")
            self.trials = GestureTrialRunner(self.session)
            self.scene = Scene.GESTURE_TEST
        else:
            self._start_fight(use_vision=True)

    # ------------------------------------------------------------- render
    def _render(self, dt: float) -> None:
        snap = self.vision.snapshot() if self.vision else None
        if self.scene in (Scene.FIGHT, Scene.MENU):
            self._render_arena(dt)
        if self.scene == Scene.FIGHT:
            self.hud.draw(self.arena_surf, self.state, self.clock_t, self._control_labels(snap))
            self.banner.draw(self.arena_surf, (self.W // 2, self.arena_h // 2 - 20))
            self.screen.blit(self.arena_surf, (0, 0))
            self._render_bottom(snap)
            if self.paused:
                self._render_pause()
        elif self.scene == Scene.MENU:
            self.screen.blit(self.arena_surf, (0, 0))
            self._render_bottom(None)
            self._render_menu()
        elif self.scene == Scene.CALIBRATION:
            self._render_calibration(snap)
        elif self.scene == Scene.GESTURE_TEST:
            self._render_gesture_test(snap)
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
        fighters = sorted(self.state.fighters, key=lambda f: f.state in (FighterState.PUNCH,
                                                                         FighterState.SPECIAL))
        for f in fighters:
            self.renderers[f.player_id].draw(surf, f, self.clock_t, dt, shake)
        for p in self.state.projectiles:
            self.effects.draw_projectile(surf, p, STYLES[p.owner].accent, self.clock_t, shake)
        self.effects.draw(surf, shake)

    def _control_labels(self, snap: VisionSnapshot | None) -> tuple[str, str]:
        labels = []
        for pid in (0, 1):
            if pid in self.cpu_players:
                labels.append("CPU")
            elif self.use_vision and snap and snap.players[pid].state == TrackState.ACTIVE:
                labels.append("WEBCAM")
            elif self.use_vision and snap and snap.players[pid].state == TrackState.LOST:
                labels.append("WEBCAM: LOST")
            else:
                labels.append("KEYBOARD")
        return labels[0], labels[1]

    def _render_bottom(self, snap: VisionSnapshot | None) -> None:
        top = self.arena_h
        strip = pygame.Rect(0, top, self.W, self.H - top)
        pygame.draw.rect(self.screen, (20, 16, 28), strip)
        pygame.draw.line(self.screen, GOLD, (0, top), (self.W, top), 3)
        for pid, panel in enumerate(self.panels):
            f = self.state.fighters[pid]
            sp = 0.0
            if snap and snap.players[pid].special_progress > 0:
                sp = snap.players[pid].special_progress
            elif f.state == FighterState.SPECIAL and f.state_time < SPECIAL_CHARGE:
                sp = f.state_time / SPECIAL_CHARGE
            winner = self.state.phase in (Phase.ROUND_OVER, Phase.MATCH_OVER) and \
                self.state.round_winner == pid
            status = status_for(f, self.hit_flash[pid], winner, sp) if self.scene == Scene.FIGHT \
                else PlayerStatus()
            label = "KEYBOARD MODE" if not self.use_vision or self.scene != Scene.FIGHT else "NO CAMERA"
            if self.scene == Scene.MENU:
                label = "WEBCAM PANEL"
            panel.render(self.screen, snap if (self.use_vision and self.scene == Scene.FIGHT) else None,
                         status, self.clock_t, label)
        cx = self.W // 2
        y = top + 26
        if self.scene == Scene.FIGHT:
            if self.use_vision:
                hints = ["PUNCH: fast arm extension", "BLOCK: both hands to face",
                         "SPECIAL: both arms up & hold", "MOVE: step left / right"]
            else:
                hints = ["P1  A/D move  S block  F/G punch  R special",
                         "P2  arrows move  DOWN block  K/L punch  I special"]
            for h in hints:
                text(self.screen, h, 24, (cx, y), (215, 205, 230), outline=0)
                y += 26
            text(self.screen, "ESC pause   F1 debug   F2 skeleton   C recalibrate", 20,
                 (cx, self.H - 20), (140, 130, 160), outline=0)
        else:
            text(self.screen, "IPCV 2026  -  University of Twente", 22, (cx, y + 40),
                 (150, 140, 170), outline=0)

    def _render_menu(self) -> None:
        overlay = pygame.Surface((self.W, self.arena_h), pygame.SRCALPHA)
        overlay.fill((10, 6, 18, 150))
        self.screen.blit(overlay, (0, 0))
        pulse = 0.5 + 0.5 * math.sin(self.clock_t * 3)
        text(self.screen, "WEBCAM WARRIORS", 104, (self.W // 2, 120), (255, 200 + int(40 * pulse), 70),
             outline=5)
        text(self.screen, "a body-controlled two-player fighting game", 30, (self.W // 2, 178),
             (230, 220, 240))
        y = 238
        for i, (key, label) in enumerate(MENU_ITEMS):
            sel = i == self.menu_index
            if key == "webcam" and self.vision_source == "synthetic":
                label += "  (SYNTHETIC DEMO)"
            if sel:
                bar = pygame.Rect(0, 0, 640, 46)
                bar.center = (self.W // 2, y)
                pygame.draw.rect(self.screen, (255, 120, 40), bar, border_radius=10)
                pygame.draw.rect(self.screen, GOLD, bar, 3, border_radius=10)
            text(self.screen, label, 42, (self.W // 2, y), DARK if sel else WHITE,
                 outline=0 if sel else 2)
            y += 50
        if self.vision_error:
            box = pygame.Rect(0, 0, 1000, 76)
            box.midtop = (self.W // 2, y - 14)
            pygame.draw.rect(self.screen, (40, 10, 16), box, border_radius=8)
            for k, line in enumerate(_wrap(self.vision_error, 100)[:3]):
                text(self.screen, line, 23, (self.W // 2, y + 1 + k * 22), (255, 150, 130), outline=0)

    def _render_pause(self) -> None:
        overlay = pygame.Surface((self.W, self.H), pygame.SRCALPHA)
        overlay.fill((8, 6, 14, 170))
        self.screen.blit(overlay, (0, 0))
        text(self.screen, "PAUSED", 90, (self.W // 2, 180), GOLD, outline=4)
        for i, label in enumerate(("RESUME", "RESTART MATCH", "MAIN MENU")):
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
        if self.vision_error:
            for k, line in enumerate(_wrap(self.vision_error, 100)[:3]):
                text(self.screen, line, 26, (self.W // 2, y + k * 26), (255, 140, 120))
            return
        if snap and snap.registered:
            remain = 2.5 - (time.perf_counter() - (self.calib_ready_since or time.perf_counter()))
            text(self.screen, f"BOTH PLAYERS REGISTERED - starting in {max(0, remain):.1f}s", 34,
                 (self.W // 2, y), (120, 240, 140))
        else:
            n = snap.num_poses if snap else 0
            text(self.screen, "Player 1 stands on the LEFT, Player 2 on the RIGHT. Face the camera,"
                 " upper body and hips visible, arms relaxed.", 26, (self.W // 2, y), WHITE)
            text(self.screen, f"People detected: {n}/2", 30, (self.W // 2, y + 34), GOLD)
        text(self.screen, "ENTER: start now (missing player uses keyboard)    R: re-register    ESC: back",
             22, (self.W // 2, self.H - 24), (170, 160, 190), outline=0)

    def _render_gesture_test(self, snap: VisionSnapshot | None) -> None:
        self.screen.fill((18, 14, 26))
        rect = pygame.Rect(0, 0, 760, 428)
        rect.midtop = (self.W // 2, 150)
        self._camera_view(snap, rect)
        if self.trials is None:
            return
        title, sub = self.trials.prompt_text()
        tr = self.trials.trial
        color = GOLD if tr is None else (255, 120, 60)
        text(self.screen, title, 60, (self.W // 2, 60), color, outline=3)
        text(self.screen, sub, 28, (self.W // 2, 110), WHITE)
        if tr is not None:
            k = min(1.0, (time.perf_counter() - tr.start) / tr.duration)
            pygame.draw.rect(self.screen, (60, 50, 70), (self.W // 2 - 300, 128, 600, 10))
            pygame.draw.rect(self.screen, color, (self.W // 2 - 300, 128, int(600 * (1 - k)), 10))
        res = self.trials.results
        if res:
            ok = sum(int(r["correct"]) for r in res)
            text(self.screen, f"Completed trials: {len(res)}   correct: {ok}", 26,
                 (self.W // 2, rect.bottom + 40), (200, 230, 200))
        if self.trials.done and self.session:
            text(self.screen, f"Saved to {self.session.dir.name}", 24,
                 (self.W // 2, rect.bottom + 74), GOLD)
        text(self.screen, "ESC: stop test", 22, (self.W // 2, self.H - 24), (170, 160, 190), outline=0)


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
