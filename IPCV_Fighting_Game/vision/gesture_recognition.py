"""Task 4 — gesture recognition: body motion -> game commands.

All measurements are expressed in **shoulder widths** (sw), which makes the
thresholds invariant to how far the player stands from the camera.

Punch (left / right) — motion-based state machine per arm::

    ARMED --(reach rising fast AND reach > extend AND arm roughly horizontal
             or forward)--> FIRE (event) --> EXTENDED
    EXTENDED --(reach < retract, i.e. arm pulled back)--> ARMED (after cooldown)

  *reach* = wrist-to-shoulder distance / shoulder width. Because a monocular
  camera cannot measure depth directly, two approximations are combined:
  (a) 2D image reach (works for punches towards the side / opponent), and
  (b) reach from MediaPipe's estimated metric world landmarks (captures punches
  towards the camera, which barely change the 2D projection). The larger of the
  two is used. Holding the arm extended never re-fires: the arm must be
  retracted to re-arm, and a cooldown applies.

Block — both wrists within ``block_enter_dist`` sw of the nose for
``block_confirm_s`` (confirmation), released with hysteresis at ``block_exit_dist``.

Special — both wrists above the head (``special_raise`` sw above the nose) held
for ``special_hold_s``; re-arms only after both arms are lowered below the
shoulders, plus a cooldown.

Movement — horizontal offset of the shoulder midpoint from the player's
calibrated neutral position, with a dead zone and EMA smoothing.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from config import GestureConfig
from vision.pose_tracking import (L_ELBOW, L_SHOULDER, L_WRIST, NOSE, R_ELBOW, R_SHOULDER,
                                  R_WRIST, PlayerPose)


class ArmState(Enum):
    ARMED = "armed"
    EXTENDED = "extended"


@dataclass
class GestureEvent:
    player_id: int
    gesture: str              # punch_left, punch_right, special, block_start, block_end
    capture_ts: float         # camera timestamp of the frame that confirmed it
    onset_ts: float           # camera timestamp of the frame where the motion started
    processed_ts: float       # perf_counter when the recogniser emitted it
    values: dict = field(default_factory=dict)


@dataclass
class ArmTracker:
    side: str                                  # "left" / "right"
    state: ArmState = ArmState.ARMED
    last_fire: float = -math.inf
    onset_ts: float | None = None
    history: deque = field(default_factory=lambda: deque(maxlen=8))   # (t, reach)
    reach: float = math.nan
    rate: float = 0.0


class GestureRecognizer:
    def __init__(self, player_id: int, cfg: GestureConfig) -> None:
        self.player_id = player_id
        self.cfg = cfg
        self.arms = {"left": ArmTracker("left"), "right": ArmTracker("right")}
        self.block_active = False
        self._block_candidate_since: float | None = None
        self._special_since: float | None = None
        self._special_armed = True
        self._special_last = -math.inf
        self.special_progress = 0.0
        self.neutral_x: float | None = None
        self.move = 0.0
        self.move_offset = 0.0
        self.signals: dict[str, float] = {}
        self._sw_world_hist: deque[float] = deque(maxlen=15)

    # ------------------------------------------------------------ calibration
    def calibrate(self, pose: PlayerPose) -> bool:
        """Store the neutral standing position. Returns False if not possible."""
        ls, rs = pose.point(L_SHOULDER), pose.point(R_SHOULDER)
        if ls is None or rs is None:
            return False
        self.neutral_x = float((ls[0] + rs[0]) / 2)
        return True

    def reset(self) -> None:
        for arm in self.arms.values():
            arm.state, arm.onset_ts = ArmState.ARMED, None
            arm.history.clear()
        self.block_active = False
        self._block_candidate_since = None
        self._special_since = None
        self._special_armed = True
        self.move = 0.0

    # ----------------------------------------------------------------- helpers
    def _world_reach(self, pose: PlayerPose, s_idx: int, w_idx: int) -> float:
        ls, rs = pose.world_point(L_SHOULDER), pose.world_point(R_SHOULDER)
        s, w = pose.world_point(s_idx), pose.world_point(w_idx)
        if ls is None or rs is None or s is None or w is None:
            return math.nan
        self._sw_world_hist.append(float(np.linalg.norm(ls - rs)))
        sw = max(float(np.median(self._sw_world_hist)), 0.15)
        return float(np.linalg.norm(w - s) / sw)

    # ------------------------------------------------------------------ update
    def update(self, pose: PlayerPose, capture_ts: float, now: float) -> list[GestureEvent]:
        events: list[GestureEvent] = []
        sw = pose.shoulder_width
        nose = pose.point(NOSE)
        lw, rw = pose.point(L_WRIST), pose.point(R_WRIST)
        ls, rs = pose.point(L_SHOULDER), pose.point(R_SHOULDER)

        # ---------------- block (both hands near the face) ----------------
        block_now = False
        if nose is not None and lw is not None and rw is not None:
            dl = np.linalg.norm(lw - nose) / sw
            dr = np.linalg.norm(rw - nose) / sw
            self.signals["block_dist"] = float(max(dl, dr))
            limit = self.cfg.block_exit_dist if self.block_active else self.cfg.block_enter_dist
            block_now = dl < limit and dr < limit
        else:
            self.signals["block_dist"] = math.nan
        if block_now and not self.block_active:
            if self._block_candidate_since is None:
                self._block_candidate_since = capture_ts
            if capture_ts - self._block_candidate_since >= self.cfg.block_confirm_s:
                self.block_active = True
                events.append(GestureEvent(self.player_id, "block_start", capture_ts,
                                           self._block_candidate_since, now))
        elif not block_now:
            self._block_candidate_since = None
            if self.block_active:
                self.block_active = False
                events.append(GestureEvent(self.player_id, "block_end", capture_ts, capture_ts, now))

        # ---------------- special (both arms above the head) ----------------
        raised = False
        if nose is not None and lw is not None and rw is not None:
            top = nose[1] - self.cfg.special_raise * sw
            raised = lw[1] < top and rw[1] < top
            self.signals["raise"] = float((nose[1] - max(lw[1], rw[1])) / sw)
        if not self._special_armed and ls is not None and rs is not None \
                and lw is not None and rw is not None:
            if lw[1] > ls[1] and rw[1] > rs[1]:
                self._special_armed = True        # arms lowered -> can fire again
        if raised and self._special_armed and now - self._special_last >= self.cfg.special_cooldown_s:
            if self._special_since is None:
                self._special_since = capture_ts
            held = capture_ts - self._special_since
            self.special_progress = min(1.0, held / self.cfg.special_hold_s)
            if held >= self.cfg.special_hold_s:
                events.append(GestureEvent(self.player_id, "special", capture_ts,
                                           self._special_since, now))
                self._special_last = now
                self._special_armed = False
                self._special_since = None
                self.special_progress = 0.0
        else:
            self._special_since = None
            self.special_progress = 0.0

        # ---------------- punches (per arm state machine) ----------------
        for side, s_idx, e_idx, w_idx in (("left", L_SHOULDER, L_ELBOW, L_WRIST),
                                          ("right", R_SHOULDER, R_ELBOW, R_WRIST)):
            ev = self._update_arm(pose, self.arms[side], s_idx, e_idx, w_idx, sw,
                                  capture_ts, now, suppress=self.block_active or raised)
            if ev:
                events.append(ev)

        # ---------------- movement ----------------
        target = 0.0
        if ls is not None and rs is not None and self.neutral_x is not None:
            offset = ((ls[0] + rs[0]) / 2 - self.neutral_x) / sw
            self.move_offset = float(offset)
            mag = abs(offset)
            if mag > self.cfg.move_dead_zone:
                target = math.copysign(min(1.0, (mag - self.cfg.move_dead_zone) /
                                           (self.cfg.move_full_scale - self.cfg.move_dead_zone)),
                                       offset)
        a = self.cfg.move_smoothing
        self.move = (1 - a) * self.move + a * target
        if abs(self.move) < 0.02:
            self.move = 0.0
        return events

    def _update_arm(self, pose: PlayerPose, arm: ArmTracker, s_idx: int, e_idx: int,
                    w_idx: int, sw: float, capture_ts: float, now: float,
                    suppress: bool) -> GestureEvent | None:
        s, w = pose.point(s_idx), pose.point(w_idx)
        if s is None or w is None:
            # Missing landmarks: never fire on incomplete data, forget the motion.
            arm.history.clear()
            arm.onset_ts = None
            arm.reach = math.nan
            return None
        reach_2d = float(np.linalg.norm(w - s) / sw)
        reach_3d = self._world_reach(pose, s_idx, w_idx)
        reach = reach_2d if math.isnan(reach_3d) else max(reach_2d, reach_3d)
        arm.reach = reach
        arm.history.append((capture_ts, reach))
        # Reach rate over a ~100 ms window (robust to single-frame noise).
        rate = 0.0
        if len(arm.history) >= 2:
            t0, r0 = arm.history[0]
            for t_old, r_old in arm.history:
                if capture_ts - t_old <= 0.12:
                    t0, r0 = t_old, r_old
                    break
            if capture_ts - t0 > 1e-3:
                rate = (reach - r0) / (capture_ts - t0)
        arm.rate = rate
        self.signals[f"reach_{arm.side}"] = reach
        self.signals[f"rate_{arm.side}"] = rate

        # Height gate: wrist roughly at chest/head height (rejects arms dropping
        # down and arms being raised overhead).
        dy = (w[1] - s[1]) / sw
        height_ok = -0.9 <= dy <= 0.75
        # Reject fast upward swings (raising arms for the special).
        vy = pose.velocity(w_idx)[1] / sw
        upward_swing = vy < -2.5 and dy < -0.3

        if arm.state == ArmState.EXTENDED:
            if reach < self.cfg.punch_retract_ratio or dy > 1.0:
                arm.state = ArmState.ARMED
                arm.onset_ts = None
            return None

        # ARMED
        if rate > 0.5 * self.cfg.punch_min_speed:
            if arm.onset_ts is None:
                arm.onset_ts = capture_ts
        elif rate <= 0:
            arm.onset_ts = None
        if (not suppress and height_ok and not upward_swing
                and reach > self.cfg.punch_extend_ratio
                and rate > self.cfg.punch_min_speed
                and now - arm.last_fire >= self.cfg.punch_cooldown_s):
            arm.state = ArmState.EXTENDED
            arm.last_fire = now
            onset = arm.onset_ts if arm.onset_ts is not None else capture_ts
            arm.onset_ts = None
            return GestureEvent(self.player_id, f"punch_{arm.side}", capture_ts, onset, now,
                                {"reach": reach, "rate": rate})
        # Arm extended slowly (e.g. reaching, stretching): consume the extension
        # so it cannot fire later just by twitching.
        if reach > self.cfg.punch_extend_ratio and rate < 0.5 * self.cfg.punch_min_speed:
            arm.state = ArmState.EXTENDED
        return None
