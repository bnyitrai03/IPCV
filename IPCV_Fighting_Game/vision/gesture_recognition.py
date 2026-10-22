"""Task 4 — gesture recognition: body motion -> game commands.

All distances are in **shoulder widths** (sw), so thresholds do not depend on
how far the player stands from the camera. Image y points down.

Attacks (per arm)
-----------------
* **Hook** — the elbow is bent at roughly 90 degrees and the fist sweeps
  fast *horizontally* towards / across the body midline at chest-head height.
  The sweep is measured from the furthest-out point of the recent wind-up
  (a hook from the guard first moves out, then in). The elbow angle is
  computed in 3D from MediaPipe's estimated world landmarks (a bent arm whose
  forearm points at the camera looks straight in the 2D image), with the 2D
  image angle as fallback; the median angle over the sweep must lie in
  ``[hook_min_elbow_deg, hook_max_elbow_deg]``. A straight-arm swing is
  therefore not a hook.
* **Uppercut** — one fist driven fast upwards *and forwards*, ending at about
  head height with the arm reaching out (``uppercut_min_reach``). It is
  confirmed after ``uppercut_confirm_s`` so that raising *both* arms (the
  special) or both hands (the guard) is not mistaken for an uppercut. A hand
  that only comes up next to the face (guard) does not reach far enough.

Both attacks are found in a short trail (``hook_window_s``) of recent fist
positions, measured from the point that gives the fastest motion (the turning
point of a wind-up or the bottom of a dip). State machine per arm::

    ARMED ──hook──▶ fire ──▶ RECOVER
    ARMED ──uppercut──▶ confirm (not a two-arm raise) ──▶ fire ──▶ RECOVER
    RECOVER ──fist back (or arm pulled in) + slowed down + cooldown──▶ ARMED

Holding a pose never re-fires: the arm has to come back and slow down before
it re-arms, plus a per-arm cooldown. Downward motions never fire.

Block — both wrists within ``block_enter_dist`` sw of the nose for
``block_confirm_s`` (confirmation), released with hysteresis. An attack thrown
out of the guard ends the block (attacks take priority, as in the game logic).

Special — both wrists above the head held for ``special_hold_s``; re-arms
only after both arms are lowered below the shoulders, plus a cooldown.

Movement — horizontal offset of the shoulder midpoint from the calibrated
neutral position, with dead zone and EMA smoothing.
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

STRIKE_KINDS = ("hook", "uppercut")
ARM_JOINTS = {"left": (L_SHOULDER, L_ELBOW, L_WRIST), "right": (R_SHOULDER, R_ELBOW, R_WRIST)}


class ArmState(Enum):
    ARMED = "armed"
    RECOVER = "recover"


@dataclass
class GestureEvent:
    player_id: int
    gesture: str              # hook_*, uppercut_*, special, block_start, block_end
    capture_ts: float         # camera timestamp of the frame that confirmed it
    onset_ts: float           # camera timestamp of the frame where the motion started
    processed_ts: float       # perf_counter when the recogniser emitted it
    values: dict = field(default_factory=dict)


@dataclass
class ArmSample:
    t: float
    pos: np.ndarray           # wrist, isotropic image coords / sw
    depth: float              # towards-camera depth of wrist rel. shoulder, sw (nan if unknown)
    reach: float              # max(2D, 3D) wrist-shoulder distance / sw
    elbow_deg: float          # elbow angle (180 = straight arm), nan if elbow not visible


@dataclass
class ArmTracker:
    side: str                                  # "left" / "right" (anatomical)
    state: ArmState = ArmState.ARMED
    last_fire: float = -math.inf
    onset: ArmSample | None = None
    prev: ArmSample | None = None
    pending_uppercut: tuple | None = None      # (since, onset, sample, values)
    trail: deque = field(default_factory=lambda: deque(maxlen=30))   # recent ArmSamples
    last_kind: str = ""


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-6 or nb < 1e-6:
        return math.nan
    return math.degrees(math.acos(float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))))


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
            arm.state, arm.onset, arm.prev, arm.pending_uppercut = ArmState.ARMED, None, None, None
            arm.trail.clear()
        self.block_active = False
        self._block_candidate_since = None
        self._special_since = None
        self._special_armed = True
        self.move = 0.0

    # ----------------------------------------------------------------- helpers
    def _world_sw(self, pose: PlayerPose) -> float | None:
        ls, rs = pose.world_point(L_SHOULDER), pose.world_point(R_SHOULDER)
        if ls is None or rs is None:
            return None
        self._sw_world_hist.append(float(np.linalg.norm(ls - rs)))
        return max(float(np.median(self._sw_world_hist)), 0.15)

    def _sample(self, pose: PlayerPose, side: str, t: float, sw: float,
                sw_world: float | None) -> ArmSample | None:
        s_idx, e_idx, w_idx = ARM_JOINTS[side]
        s, e, w = pose.point(s_idx), pose.point(e_idx), pose.point(w_idx)
        if s is None or w is None:
            return None
        reach = float(np.linalg.norm(w - s) / sw)
        depth = math.nan
        elbow = math.nan
        ws, we, ww = pose.world_point(s_idx), pose.world_point(e_idx), pose.world_point(w_idx)
        if sw_world is not None and ws is not None and ww is not None:
            depth = float(-(ww[2] - ws[2]) / sw_world)          # + = towards the camera
            reach = max(reach, float(np.linalg.norm(ww - ws) / sw_world))
        if ws is not None and we is not None and ww is not None:
            # 3D angle: a bent arm with the forearm pointing at the camera looks
            # straight in 2D, so the estimated world landmarks are preferred.
            elbow = _angle_deg(ws - we, ww - we)
        elif e is not None:
            elbow = _angle_deg(s - e, w - e)
        return ArmSample(t, w / sw, depth, reach, elbow)

    # ------------------------------------------------------------------ update
    def update(self, pose: PlayerPose, capture_ts: float, now: float) -> list[GestureEvent]:
        events: list[GestureEvent] = []
        cfg = self.cfg
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
            limit = cfg.block_exit_dist if self.block_active else cfg.block_enter_dist
            block_now = dl < limit and dr < limit
        else:
            self.signals["block_dist"] = math.nan
        if block_now and not self.block_active:
            if self._block_candidate_since is None:
                self._block_candidate_since = capture_ts
            if capture_ts - self._block_candidate_since >= cfg.block_confirm_s:
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
            top = nose[1] - cfg.special_raise * sw
            raised = lw[1] < top and rw[1] < top
            self.signals["raise"] = float((nose[1] - max(lw[1], rw[1])) / sw)
        if not self._special_armed and ls is not None and rs is not None \
                and lw is not None and rw is not None:
            if lw[1] > ls[1] and rw[1] > rs[1]:
                self._special_armed = True        # arms lowered -> can fire again
        if raised and self._special_armed and now - self._special_last >= cfg.special_cooldown_s:
            if self._special_since is None:
                self._special_since = capture_ts
            held = capture_ts - self._special_since
            self.special_progress = min(1.0, held / cfg.special_hold_s)
            if held >= cfg.special_hold_s:
                events.append(GestureEvent(self.player_id, "special", capture_ts,
                                           self._special_since, now))
                self._special_last = now
                self._special_armed = False
                self._special_since = None
                self.special_progress = 0.0
        else:
            self._special_since = None
            self.special_progress = 0.0

        # ---------------- attacks (per arm) ----------------
        sw_world = self._world_sw(pose)
        geo = {"nose": None if nose is None else nose / sw,
               "mid_x": None if ls is None or rs is None else float((ls[0] + rs[0]) / 2 / sw),
               "shoulder_y": None if ls is None or rs is None else float((ls[1] + rs[1]) / 2 / sw),
               "sh_x": {"left": None if ls is None else float(ls[0] / sw),
                        "right": None if rs is None else float(rs[0] / sw)}}
        samples = {side: self._sample(pose, side, capture_ts, sw, sw_world)
                   for side in ("left", "right")}
        # Only the two-arm special pose suppresses attacks. A guard (= block) does
        # NOT: a hook travels around the face, so the "hands near face" test stays
        # true during it. An attack out of the guard ends the block instead; a
        # still guard never fires because attacks need fast motion.
        suppress = raised
        for side in ("left", "right"):
            other = "right" if side == "left" else "left"
            ev = self._update_arm(self.arms[side], samples[side], samples[other],
                                  self.arms[other], geo, capture_ts, now, suppress)
            if ev:
                if self.block_active:
                    self.block_active = False
                    events.append(GestureEvent(self.player_id, "block_end", capture_ts,
                                               capture_ts, now))
                self._block_candidate_since = capture_ts   # re-confirm the guard afterwards
                events.append(ev)

        # ---------------- movement ----------------
        target = 0.0
        if ls is not None and rs is not None and self.neutral_x is not None:
            offset = ((ls[0] + rs[0]) / 2 - self.neutral_x) / sw
            self.move_offset = float(offset)
            mag = abs(offset)
            if mag > cfg.move_dead_zone:
                target = math.copysign(min(1.0, (mag - cfg.move_dead_zone) /
                                           (cfg.move_full_scale - cfg.move_dead_zone)), offset)
        a = cfg.move_smoothing
        self.move = (1 - a) * self.move + a * target
        if abs(self.move) < 0.02:
            self.move = 0.0
        return events

    # ------------------------------------------------------------ attack logic
    def _detect_uppercut(self, arm: ArmTracker, cur: ArmSample,
                         geo: dict) -> tuple[ArmSample, dict] | None:
        """Fast rise (up and forward) ending at about head height with the arm
        reaching out. Measured from the start of the rise = the recent point from
        which the fist rose FASTEST (for a dip-then-rise: the bottom of the dip)."""
        cfg = self.cfg
        nose = geo["nose"]
        if nose is None or len(arm.trail) < 3:
            return None
        past = list(arm.trail)[:-1]

        def rise_speed(p: ArmSample) -> float:
            return (p.pos[1] - cur.pos[1]) / max(cur.t - p.t, 1e-3)

        start = max(past, key=rise_speed)
        rise = float(start.pos[1] - cur.pos[1])
        dx = float(cur.pos[0] - start.pos[0])
        dt = cur.t - start.t
        if dt <= 1e-3 or dt > cfg.uppercut_max_duration_s:
            return None
        speed = math.hypot(dx, rise) / dt
        self.signals[f"rise_{arm.side}"] = rise
        if (speed >= cfg.attack_min_speed
                and rise >= cfg.uppercut_min_rise and rise >= abs(dx)          # mostly upwards
                and cur.pos[1] <= nose[1] + cfg.uppercut_end_below_nose        # ends at head height
                and cur.reach >= cfg.uppercut_min_reach):                      # arm reaches out
            return start, {"dx": dx, "dy": -rise, "speed": speed, "reach": cur.reach, "dt": dt}
        return None

    @staticmethod
    def _recent_rise(arm: ArmTracker, cur: ArmSample | None) -> float:
        if cur is None or not arm.trail:
            return 0.0
        return float(max(p.pos[1] for p in arm.trail) - cur.pos[1])

    def _update_arm(self, arm: ArmTracker, cur: ArmSample | None, other: ArmSample | None,
                    other_arm: ArmTracker, geo: dict, capture_ts: float, now: float,
                    suppress: bool) -> GestureEvent | None:
        cfg = self.cfg
        if cur is None:
            # Missing landmarks: never fire on incomplete data, forget the motion.
            arm.onset, arm.prev, arm.pending_uppercut = None, None, None
            arm.trail.clear()
            return None
        prev = arm.prev
        arm.prev = cur
        arm.trail.append(cur)
        while arm.trail and cur.t - arm.trail[0].t > cfg.hook_window_s:
            arm.trail.popleft()
        s = self.signals
        s[f"reach_{arm.side}"] = cur.reach
        s[f"elbow_{arm.side}"] = cur.elbow_deg
        s[f"state_{arm.side}"] = {"armed": 0.0, "recover": 2.0}[arm.state.value]

        inst = 0.0
        if prev is not None and cur.t - prev.t > 1e-4:
            inst = float(np.linalg.norm(cur.pos - prev.pos)) / (cur.t - prev.t)

        # ---- pending uppercut: cancel if it turns into a two-arm raise ----
        if arm.pending_uppercut is not None:
            since, onset, sample, pvals = arm.pending_uppercut
            # Other fist ALSO above the head = special pose; a guard hand is fine.
            other_up = other is not None and geo["nose"] is not None and \
                other.pos[1] < geo["nose"][1] - 0.1
            other_rising = self._recent_rise(other_arm, other) > 0.6
            if suppress or other_up or other_rising:
                arm.pending_uppercut = None
                arm.state = ArmState.RECOVER          # consume: it was a two-arm raise
                arm.onset = onset
                return None
            if capture_ts - since >= cfg.uppercut_confirm_s:
                arm.pending_uppercut = None
                return self._fire(arm, "uppercut", onset, sample, now,
                                  {**pvals, "confirm_ms": (capture_ts - since) * 1000})
            return None

        if arm.state == ArmState.RECOVER:
            back = arm.onset is None or np.linalg.norm(cur.pos - arm.onset.pos) < cfg.rearm_dist
            retracted = cur.reach < cfg.rearm_reach
            settled = inst < cfg.strike_onset_speed       # pull-back finished, not mid-swing
            if (back or retracted) and settled and now - arm.last_fire >= cfg.attack_cooldown_s:
                arm.state = ArmState.ARMED
                arm.onset = None
                arm.trail.clear()        # the pull-back must not count as a hook wind-up
            return None

        # ARMED: look for either attack in the recent motion trail.
        hook = self._detect_hook(arm, cur, geo)
        if hook is not None and not suppress:
            onset, vals = hook
            return self._fire(arm, "hook", onset, cur, now, vals)
        upper = self._detect_uppercut(arm, cur, geo)
        if upper is not None and not suppress:
            onset, vals = upper
            arm.pending_uppercut = (capture_ts, onset, cur, vals)
        return None

    def _detect_hook(self, arm: ArmTracker, cur: ArmSample,
                     geo: dict) -> tuple[ArmSample, dict] | None:
        """Hook = elbow bent ~90 deg + fast inward horizontal sweep, measured from the
        start of the sweep (for a hook from the guard: the turning point out -> in)."""
        cfg = self.cfg
        mid, sh_x, sh_y = geo["mid_x"], geo["sh_x"][arm.side], geo["shoulder_y"]
        if mid is None or sh_x is None or sh_y is None or len(arm.trail) < 3:
            return None
        side_sign = 1.0 if sh_x >= mid else -1.0          # which side of the body this arm is on

        def outward(p: ArmSample) -> float:
            return (p.pos[0] - mid) * side_sign

        # Start of the sweep = the recent point from which the fist moved inward
        # FASTEST. This finds the turning point of an out-then-in wind-up and is
        # not fooled by jitter while the fist rested before the swing.
        past = list(arm.trail)[:-1]
        now_out = outward(cur)

        def sweep_speed(p: ArmSample) -> float:
            return (outward(p) - now_out) / max(cur.t - p.t, 1e-3)

        apex_i = max(range(len(past)), key=lambda i: sweep_speed(past[i]))
        apex = past[apex_i]
        inward = outward(apex) - outward(cur)
        dy = float(cur.pos[1] - apex.pos[1])
        dt = cur.t - apex.t
        sweep = past[apex_i:] + [cur]
        angles = [p.elbow_deg for p in sweep if p.elbow_deg == p.elbow_deg]
        elbow = float(np.median(angles)) if angles else math.nan
        self.signals[f"hook_{arm.side}"] = inward
        if dt <= 1e-3 or dt > cfg.hook_max_duration_s:
            return None
        speed = math.hypot(inward, dy) / dt
        apex_h, end_h = apex.pos[1] - sh_y, cur.pos[1] - sh_y
        bent = cfg.hook_min_elbow_deg <= elbow <= cfg.hook_max_elbow_deg
        if (inward >= cfg.hook_min_dx and inward >= cfg.hook_dominance * abs(dy)
                and speed >= cfg.attack_min_speed and bent
                and -1.3 <= apex_h <= 0.9 and -1.1 <= end_h <= 0.7):   # chest-head height
            return apex, {"dx": inward, "dy": dy, "speed": speed, "dt": dt,
                          "elbow_deg": elbow, "reach": cur.reach}
        return None

    def _fire(self, arm: ArmTracker, kind: str, onset: ArmSample, cur: ArmSample,
              now: float, vals: dict) -> GestureEvent:
        arm.state = ArmState.RECOVER
        arm.last_fire = now
        arm.onset = onset          # re-arm reference: where the attack started
        arm.last_kind = kind
        arm.trail.clear()
        arm.pending_uppercut = None
        return GestureEvent(self.player_id, f"{kind}_{arm.side}", cur.t, onset.t, now,
                            {k: float(v) for k, v in vals.items() if isinstance(v, (int, float))})
