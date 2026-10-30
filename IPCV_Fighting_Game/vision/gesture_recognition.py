"""Task 4 — gesture recognition: body motion -> game commands.

All distances are in **shoulder widths** (sw), so thresholds do not depend on
how far the player stands from the camera. Image y points down.

Attacks (per arm)
-----------------
* **Hook** — the elbow is bent and the fist sweeps fast *horizontally*
  towards / across the body midline, ending around head height (it may start
  from chest height). The sweep is measured from the start of the fastest inward
  motion, at least two frames back (for a hook from the guard or from the arm
  out to the side: the turning point out -> in). The arm counts as bent if
  EITHER the 3D elbow angle from MediaPipe's estimated world landmarks OR the
  2D image angle lies in ``[hook_min_elbow_deg, hook_max_elbow_deg]`` (most-bent
  quarter of the sweep). The 2D angle is only used when neither arm segment is
  foreshortened, so a straight-arm swing is still not a hook.
  ``hook_direction="both"`` also accepts outward swings (off by default: the
  wind-up of a hook would fire as a second hook).
* **Uppercut** — one fist driven fast upwards *and forwards*, ending at least
  at chin height, roughly above its own shoulder (``uppercut_max_side``: raising
  the arm out wide is the wind-up of a hook, not an uppercut), with the fist
  away from the shoulder (bent arm is fine). The rise is measured from the
  lowest point of the last 0.4 s (a dip first is fine) and must be clearly more
  up than sideways; "fast" = peak fist speed. It is confirmed after
  ``uppercut_confirm_s``: cancelled if both arms go up (= special), if the fist
  drops again (tracking glitch) or if it starts sweeping inward (= a hook, which
  then fires instead). A guard hand next to the face does not cancel it.

Robustness to real webcam tracking (tuned on a recorded session)
---------------------------------------------------------------
Wrist/elbow are taken from the RAW detection of each frame, not the smoothed
landmarks: a wrist outside the picture (only guessed by the model, often onto
a hand-like object such as a ball on a shelf) counts as missing, and a
physically impossible jump (> ``max_fist_speed``) is dropped as a glitch. The
smoothing would blend such a glitch into a fake fast punch. Short gaps
(``arm_gap_tolerance_s``, motion blur) keep the motion; a hook must be tracked
without gaps. After the special, both arms re-arm only once lowered.

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

Movement — leaning the upper body left/right (``move_mode="lean"``): the
tilt of the shoulder line plus the head's sideways offset, in sw, minus the lean measured
while standing upright at registration. Dead zone + linear ramp + EMA
smoothing; frozen briefly after an attack (a punch twists the torso) and while
blocking. Players can therefore stand anywhere in the picture. The previous
"step away from your spot" behaviour is ``move_mode="position"``.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from config import GestureConfig
from vision.pose_tracking import (L_ELBOW, L_HIP, L_SHOULDER, L_WRIST, NOSE, R_ELBOW, R_HIP,
                                  R_SHOULDER, R_WRIST, PlayerPose)

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


@dataclass
class ArmSample:
    t: float
    pos: np.ndarray           # wrist, isotropic image coords / sw
    reach: float              # max(2D, 3D) wrist-shoulder distance / sw
    elbow_deg: float          # elbow angle (180 = straight arm), nan if elbow not visible
    elbow2d: float = math.nan  # image-plane elbow angle (nan if an arm segment is foreshortened)


@dataclass
class ArmTracker:
    side: str                                  # "left" / "right" (anatomical)
    state: ArmState = ArmState.ARMED
    last_fire: float = -math.inf
    onset: ArmSample | None = None
    prev: ArmSample | None = None
    pending_uppercut: tuple | None = None      # (since, onset, sample, values)
    trail: deque = field(default_factory=lambda: deque(maxlen=30))   # recent ArmSamples
    last_kind: str = ""                        # last attack fired ("hook"/"uppercut"), for effects
    missing_since: float | None = None         # wrist lost since (motion-blur gap tolerance)
    inst_speed: float = 0.0                    # current fist speed, sw/s
    glitches: int = 0                          # consecutive rejected landmark jumps
    wait_low: bool = False                     # after a special: re-arm only once lowered


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
        self.neutral_lean = 0.0
        self.move = 0.0
        self.move_offset = 0.0     # lean (or step) relative to neutral, sw
        self._lean_hist: deque[float] = deque(maxlen=5)
        self._walking = False      # hysteresis: start at the dead zone, stop a bit below it
        self.signals: dict[str, float] = {}
        self._sw_world_hist: deque[float] = deque(maxlen=15)

    # ------------------------------------------------------------ calibration
    def calibrate(self, pose: PlayerPose) -> bool:
        """Store the neutral standing position. Returns False if not possible."""
        ls, rs = pose.point(L_SHOULDER), pose.point(R_SHOULDER)
        if ls is None or rs is None:
            return False
        self.neutral_x = float((ls[0] + rs[0]) / 2)
        lean = self._raw_lean(pose)
        # Standing upright at registration defines "no lean" (people stand a bit crooked).
        self.neutral_lean = lean if lean is not None and abs(lean) < 0.3 else 0.0
        return True

    def _raw_lean(self, pose: PlayerPose) -> float | None:
        """Sideways lean of the upper body (+ = towards image right), in sw.

        = tilt of the shoulder line (the shoulder on the leaning side drops)
        + sideways offset of the head from the shoulder midpoint.
        Both only use the shoulders and the head, so the measure does not
        depend on where the player stands and works when the hips are out of
        view (players close to the camera: the usual webcam setup). Using the
        hips was unreliable: they flicker in and out at the image border."""
        sw = pose.shoulder_width
        ls, rs = pose.point(L_SHOULDER), pose.point(R_SHOULDER)
        if ls is None or rs is None:
            return None
        a, b = (ls, rs) if ls[0] <= rs[0] else (rs, ls)          # image-left, image-right
        tilt = float((b[1] - a[1]) / sw)                          # > 0: right shoulder lower
        nose = pose.point(NOSE)
        if nose is None:
            return tilt * 1.6
        return tilt + float((nose[0] - (a[0] + b[0]) / 2) / sw)

    def reset(self) -> None:
        for arm in self.arms.values():
            arm.state, arm.onset, arm.prev, arm.pending_uppercut = ArmState.ARMED, None, None, None
            arm.trail.clear()
        self.block_active = False
        self._block_candidate_since = None
        self._special_since = None
        self._special_armed = True
        self.move = 0.0
        self._lean_hist.clear()
        self._walking = False

    # ----------------------------------------------------------------- helpers
    def _world_sw(self, pose: PlayerPose) -> float | None:
        ls, rs = pose.world_point(L_SHOULDER), pose.world_point(R_SHOULDER)
        if ls is None or rs is None:
            return None
        self._sw_world_hist.append(float(np.linalg.norm(ls - rs)))
        return max(float(np.median(self._sw_world_hist)), 0.15)

    def _raw_point(self, pose: PlayerPose, idx: int) -> np.ndarray | None:
        """Raw landmark if it is confidently detected AND inside the picture. A hand
        outside the image is only guessed by the model (often onto another object,
        e.g. a ball on a shelf), so it counts as missing."""
        if pose.raw_visibility[idx] < self.cfg.raw_min_visibility:
            return None
        p = pose.raw_points[idx]
        m = self.cfg.frame_margin
        if not np.isfinite(p).all() or not (m <= p[0] <= 1 - m and m <= p[1] <= 1 / pose.aspect - m):
            return None
        return p

    def _sample(self, pose: PlayerPose, side: str, t: float, sw: float,
                sw_world: float | None) -> ArmSample | None:
        s_idx, e_idx, w_idx = ARM_JOINTS[side]
        s = pose.point(s_idx)
        # Wrist and elbow: the RAW detection of this frame (not the smoothed value).
        # The smoothing holds and blends points, which turns a one-frame tracking
        # glitch into a fake fast punch; on raw points such a glitch is a single
        # impossible jump that _update_arm drops (max_fist_speed).
        e, w = self._raw_point(pose, e_idx), self._raw_point(pose, w_idx)
        if s is None or w is None:
            return None
        reach = float(np.linalg.norm(w - s) / sw)
        elbow = math.nan
        ws, we, ww = pose.world_point(s_idx), pose.world_point(e_idx), pose.world_point(w_idx)
        if sw_world is not None and ws is not None and ww is not None:
            reach = max(reach, float(np.linalg.norm(ww - ws) / sw_world))
        if ws is not None and we is not None and ww is not None:
            # 3D angle: a bent arm with the forearm pointing at the camera looks
            # straight in 2D, so the estimated world landmarks are preferred.
            elbow = _angle_deg(ws - we, ww - we)
        elif e is not None:
            elbow = _angle_deg(s - e, w - e)
        elbow2d = math.nan
        if e is not None and min(np.linalg.norm(s - e), np.linalg.norm(w - e)) >= \
                self.cfg.elbow2d_min_segment * sw:
            # Image angle, only when neither arm segment is foreshortened (an arm
            # pointing at the camera looks short and its 2D angle is meaningless).
            elbow2d = _angle_deg(s - e, w - e)
        return ArmSample(t, w / sw, reach, elbow, elbow2d)

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
                # Lowering both arms afterwards must not count as attacks: both
                # arms wait until they have come to rest (normal re-arm rule).
                for arm in self.arms.values():
                    arm.state, arm.onset, arm.last_fire = ArmState.RECOVER, None, now
                    arm.pending_uppercut, arm.wait_low = None, True
                    arm.trail.clear()
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
        # Is the OTHER hand clearly above the head? (both hands up = special / lowering
        # from it, not a one-arm hook)
        geo["other_high"] = {
            side: (geo["nose"] is not None and samples[o] is not None
                   and samples[o].pos[1] < geo["nose"][1] - 0.3)
            for side, o in (("left", "right"), ("right", "left"))}
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
        offset, dead, full = None, cfg.move_dead_zone, cfg.move_full_scale
        if cfg.move_mode == "lean":
            lean = self._raw_lean(pose)
            if lean is not None:
                # Median of the last frames: a single jittery frame or the quick
                # torso twist of a punch does not start walking.
                self._lean_hist.append(lean - self.neutral_lean)
                offset = float(np.median(self._lean_hist))
                dead, full = cfg.lean_dead_zone, cfg.lean_full_scale
        elif ls is not None and rs is not None and self.neutral_x is not None:
            offset = ((ls[0] + rs[0]) / 2 - self.neutral_x) / sw
        # Freeze while punching: the torso twists during the swing (before the
        # attack fires) and for a moment after it.
        punching = any(now - arm.last_fire < cfg.move_freeze_after_attack_s
                       or arm.inst_speed >= cfg.attack_min_speed
                       for arm in self.arms.values())
        if offset is not None:
            self.move_offset = float(offset)
            mag = abs(offset)
            stop_at = dead * cfg.lean_release_ratio
            self._walking = mag > dead or (self._walking and mag > stop_at)
            if self._walking and not punching and not self.block_active:
                target = math.copysign(min(1.0, max(0.0, mag - stop_at) / (full - stop_at)), offset)
        a = cfg.move_smoothing
        self.move = (1 - a) * self.move + a * target
        if abs(self.move) < 0.02:
            self.move = 0.0
        return events

    # ------------------------------------------------------------ attack logic
    def _detect_uppercut(self, arm: ArmTracker, cur: ArmSample,
                         geo: dict) -> ArmSample | None:
        """Fast rise (up and forward) ending at least at chin height. The rise is
        measured from the LOWEST point of the last ``uppercut_max_duration_s``
        (for a dip-then-rise: the bottom of the dip), and "fast" means the peak
        fist speed during the rise, so a slow start does not hide a fast punch."""
        cfg = self.cfg
        nose = geo["nose"]
        if nose is None or len(arm.trail) < 3:
            return None
        window = [p for p in list(arm.trail)[:-1] if cur.t - p.t <= cfg.uppercut_max_duration_s]
        if not window:
            return None
        start_i = max(range(len(window)), key=lambda i: window[i].pos[1])   # lowest point
        start = window[start_i]
        seg = window[start_i:] + [cur]
        peak = max((float(np.linalg.norm(b.pos - a.pos)) / max(b.t - a.t, 1e-3)
                    for a, b in zip(seg, seg[1:])), default=0.0)
        rise = float(start.pos[1] - cur.pos[1])
        dx = float(cur.pos[0] - start.pos[0])
        uph = float(cur.pos[1] - (nose[1] + cfg.uppercut_end_below_nose))  # <= 0: high enough
        sh_x = geo["sh_x"][arm.side]
        side_off = math.nan if sh_x is None else abs(float(cur.pos[0]) - sh_x)
        self._hold_peak(arm.side, "up", rise, {"rise": rise, "h": uph, "reach": cur.reach,
                                               "speed": peak, "side": side_off}, cur.t)
        for name, v in (("up_rise", rise), ("up_h", uph), ("up_reach", cur.reach), ("up_peak", peak)):
            self.signals[f"now_{arm.side}_{name}"] = v
        self.signals[f"now_{arm.side}_up_side"] = side_off
        sweeping = self._inward_speed(arm, cur, geo) > cfg.uppercut_max_inward_speed
        self.signals[f"now_{arm.side}_up_inward"] = self._inward_speed(arm, cur, geo)
        if (peak >= cfg.uppercut_min_peak_speed
                and rise >= cfg.uppercut_min_rise
                and rise >= cfg.uppercut_dominance * abs(dx)          # clearly upwards
                and uph <= 0.0                                        # ends at chin height+
                and cur.reach >= cfg.uppercut_min_reach               # fist away from shoulder
                # in front of / above the own shoulder, NOT far out to the side
                # (raising the arm out wide is the wind-up of a hook)
                and side_off <= cfg.uppercut_max_side
                and not sweeping):                                    # turning into a hook
            return start
        return None

    @staticmethod
    def _inward_speed(arm: ArmTracker, cur: ArmSample, geo: dict) -> float:
        """Horizontal fist speed towards the body midline over the last ~0.1 s (sw/s)."""
        mid, sh_x = geo["mid_x"], geo["sh_x"][arm.side]
        past = [p for p in arm.trail if 0.05 <= cur.t - p.t <= 0.12]
        if mid is None or sh_x is None or not past:
            return 0.0
        p = past[-1]
        side_sign = 1.0 if sh_x >= mid else -1.0
        return float((p.pos[0] - cur.pos[0]) * side_sign) / (cur.t - p.t)

    @staticmethod
    def _max_gap(seg: list[ArmSample]) -> float:
        return max((b.t - a.t for a, b in zip(seg, seg[1:])), default=0.0)

    def _hold_peak(self, side: str, key: str, score: float, values: dict, t: float) -> None:
        """Keep the strongest recent attempt for 2 s so it can be read in the F1 panel."""
        k = f"{key}_{side}"
        if score >= self.signals.get(k + "_score", -1e9) or t - self.signals.get(k + "_t", -1e9) > 2.0:
            self.signals[k + "_score"] = score
            self.signals[k + "_t"] = t
            for name, v in values.items():
                self.signals[f"{k}_{name}"] = float(v)

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
            # Missing wrist (often motion blur during a fast punch): never fire on
            # incomplete data, but keep the motion for a short gap so the punch
            # can still be recognised when the fist reappears.
            arm.inst_speed = 0.0
            if arm.missing_since is None:
                arm.missing_since = capture_ts
            if capture_ts - arm.missing_since > cfg.arm_gap_tolerance_s:
                arm.onset, arm.prev, arm.pending_uppercut = None, None, None
                arm.trail.clear()
            return None
        arm.missing_since = None
        prev = arm.prev
        if prev is not None and cur.t - prev.t > 1e-4:
            step = float(np.linalg.norm(cur.pos - prev.pos)) / (cur.t - prev.t)
            if step > cfg.max_fist_speed and arm.glitches < 3:
                # Impossible jump (MediaPipe briefly put the wrist somewhere else,
                # e.g. on the other hand or at the image border): ignore this
                # sample instead of reading it as a super-fast punch.
                arm.glitches += 1
                arm.pending_uppercut = None   # unreliable tracking: do not confirm an uppercut
                return None
            if step > cfg.max_fist_speed:       # it stayed there: real relocation
                arm.trail.clear()
                arm.pending_uppercut = None
        arm.glitches = 0
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
        arm.inst_speed = inst

        # ---- pending uppercut: cancel if it turns into a two-arm raise ----
        if arm.pending_uppercut is not None:
            since, onset, sample = arm.pending_uppercut
            # Cancel only if the other fist is ALSO going overhead (= special pose).
            # A guard hand next to the face - even at eye level - is fine.
            nose_y = geo["nose"][1] if geo["nose"] is not None else None
            other_up = other is not None and nose_y is not None and other.pos[1] < nose_y - 0.3
            other_rising = (other is not None and nose_y is not None and other.pos[1] < nose_y
                            and self._recent_rise(other_arm, other) > 0.6)
            if suppress or other_up or other_rising:
                arm.pending_uppercut = None
                arm.state = ArmState.RECOVER          # consume: it was a two-arm raise
                arm.onset = onset
                return None
            if self._inward_speed(arm, cur, geo) > cfg.uppercut_max_inward_speed:
                arm.pending_uppercut = None           # it turns into a hook: let that fire
                onset = self._detect_hook(arm, cur, geo)
                if onset is not None and not suppress:
                    return self._fire(arm, "hook", onset, cur, now)
                return None
            # The fist must STAY up while confirming: a tracking glitch (the wrist
            # point flickering to another spot for a frame or two) jumps back down.
            if nose_y is not None and cur.pos[1] > nose_y + cfg.uppercut_end_below_nose + 0.35:
                arm.pending_uppercut = None
                return None
            if capture_ts - since >= cfg.uppercut_confirm_s:
                arm.pending_uppercut = None
                return self._fire(arm, "uppercut", onset, sample, now)
            return None

        if arm.state == ArmState.RECOVER:
            back = arm.onset is None or np.linalg.norm(cur.pos - arm.onset.pos) < cfg.rearm_dist
            retracted = cur.reach < cfg.rearm_reach
            settled = inst < cfg.strike_onset_speed       # pull-back finished, not mid-swing
            if arm.wait_low:
                if geo["shoulder_y"] is not None and cur.pos[1] > geo["shoulder_y"]:
                    arm.wait_low = False                  # arm lowered below the shoulders
                else:
                    return None
            if (back or retracted) and settled and now - arm.last_fire >= cfg.attack_cooldown_s:
                arm.state = ArmState.ARMED
                arm.onset = None
                arm.trail.clear()        # the pull-back must not count as a hook wind-up
            return None

        # ARMED: look for either attack in the recent motion trail.
        onset = self._detect_hook(arm, cur, geo)
        if onset is not None and not suppress:
            return self._fire(arm, "hook", onset, cur, now)
        onset = self._detect_uppercut(arm, cur, geo)
        if onset is not None and not suppress:
            arm.pending_uppercut = (capture_ts, onset, cur)
        return None

    def _detect_hook(self, arm: ArmTracker, cur: ArmSample,
                     geo: dict) -> ArmSample | None:
        """Hook = bent elbow + fast, mostly HORIZONTAL fist sweep at chest-head height.

        ``hook_direction="both"`` (default) accepts a sweep across the body AND a
        sweep away from it (players side by side naturally swing towards the
        opponent, which is outward for one arm). ``"inward"`` = only across the body.
        The sweep is measured from the point the fist moved away from fastest
        (for a hook from the guard: the turning point of the wind-up)."""
        cfg = self.cfg
        mid, sh_x, sh_y = geo["mid_x"], geo["sh_x"][arm.side], geo["shoulder_y"]
        if mid is None or sh_x is None or sh_y is None or len(arm.trail) < 3:
            return None
        side_sign = 1.0 if sh_x >= mid else -1.0          # which side of the body this arm is on
        # At least two frame intervals back: a one-frame jump of the landmark is
        # not a swing (a real hook is visible over several frames).
        past = list(arm.trail)[:-2]
        if not past:
            return None
        directions = (1.0,) if cfg.hook_direction == "inward" else (1.0, -1.0)

        best = None
        for d in directions:                               # +1 = inward, -1 = outward
            def travel(p: ArmSample) -> float:             # horizontal travel p -> cur in direction d
                return float(p.pos[0] - cur.pos[0]) * side_sign * d

            i = max(range(len(past)), key=lambda k: travel(past[k]) / max(cur.t - past[k].t, 1e-3))
            dx = travel(past[i])
            if best is None or dx > best[1]:
                best = (i, dx, d)
        apex_i, horiz, direction = best
        apex = past[apex_i]
        dy = float(cur.pos[1] - apex.pos[1])
        dt = cur.t - apex.t
        sweep = past[apex_i:] + [cur]
        # Most-bent angle during the sweep: the arm is clearly bent at some point
        # of a hook, while the per-frame estimate of a fast arm is noisy.
        a3 = [p.elbow_deg for p in sweep if p.elbow_deg == p.elbow_deg]
        a2 = [p.elbow2d for p in sweep if p.elbow2d == p.elbow2d]
        elbow = float(np.percentile(a3, 25)) if a3 else math.nan
        elbow2d = float(np.percentile(a2, 25)) if a2 else math.nan
        speed = math.hypot(horiz, dy) / max(dt, 1e-3)
        self._hold_peak(arm.side, "hk", horiz, {"in": horiz * direction, "elbow": elbow,
                                                "elbow2d": elbow2d, "dy": dy, "speed": speed},
                        cur.t)
        now_ = self.signals                     # per-frame values (gesture_features.csv)
        now_[f"now_{arm.side}_hook_in"], now_[f"now_{arm.side}_hook_speed"] = horiz * direction, speed
        now_[f"now_{arm.side}_elbow3d"], now_[f"now_{arm.side}_elbow2d"] = elbow, elbow2d
        if dt <= 1e-3 or dt > cfg.hook_max_duration_s:
            return None
        nose = geo["nose"]
        # Bent arm if EITHER the 3D estimate OR the (unforeshortened) image angle says so.
        bent = (cfg.hook_min_elbow_deg <= elbow <= cfg.hook_max_elbow_deg
                or cfg.hook_min_elbow_deg <= elbow2d <= cfg.hook_max_elbow_deg)
        if (horiz >= cfg.hook_min_dx and horiz >= cfg.hook_dominance * abs(dy)
                and speed >= cfg.attack_min_speed and bent
                and nose is not None                                   # around head height:
                and cfg.hook_min_y <= apex.pos[1] - nose[1] <= cfg.hook_max_start_y
                and cfg.hook_min_y <= cur.pos[1] - nose[1] <= cfg.hook_max_y
                and self._max_gap(sweep) <= cfg.max_sample_gap_s       # tracked continuously
                and not geo.get("other_high", {}).get(arm.side, False)):  # not both hands up
            return apex
        return None

    def _fire(self, arm: ArmTracker, kind: str, onset: ArmSample, cur: ArmSample,
              now: float) -> GestureEvent:
        arm.state = ArmState.RECOVER
        arm.last_fire = now
        arm.onset = onset          # re-arm reference: where the attack started
        arm.last_kind = kind
        arm.trail.clear()
        arm.pending_uppercut = None
        return GestureEvent(self.player_id, f"{kind}_{arm.side}", cur.t, onset.t, now)
