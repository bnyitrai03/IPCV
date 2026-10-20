"""Game-interaction metrics: gesture logging, prompted trial protocol, analysis.

Two sources of evidence:

1. ``gestures.csv`` — every gesture recognised during play, with
   * recognition latency  = confirm frame capture - motion-onset frame capture
                            + processing time (how long the recogniser needed
                            to be sure, excluding human reaction),
   * command latency      = game tick that applied it - capture of the
                            confirming frame,
   * whether the fighter executed it (or it was ignored: stunned, no energy...).

2. ``gesture_trials.csv`` — the GESTURE TEST mode prompts each player to
   perform a specific gesture (or to stay idle/move freely without attacking).
   The prompt is the ground truth label, so precision / recall / false
   positives can be computed without manual video annotation. Response time
   from the prompt *includes human reaction time* and is reported separately.
"""
from __future__ import annotations

import csv
import math
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from evaluation.session import Session
from vision.gesture_recognition import GestureEvent

ATTACK_GESTURES = ("punch_left", "punch_right", "block_start", "special")
GESTURE_LABEL = {"punch_left": "LEFT PUNCH", "punch_right": "RIGHT PUNCH",
                 "block_start": "BLOCK (hands to face)", "special": "SPECIAL (arms up, hold)",
                 "idle": "STAY IDLE / MOVE, NO ATTACKS", "move_left": "STEP LEFT",
                 "move_right": "STEP RIGHT"}


class GestureLogger:
    HEADER = ["t_capture", "player", "gesture", "onset_rel_ms", "recognition_latency_ms",
              "processing_ms", "command_latency_ms", "applied", "game_result"]

    def __init__(self, session: Session) -> None:
        self.session = session
        self.w = session.writer("gestures", self.HEADER)

    def log(self, ev: GestureEvent, applied_ts: float | None, result: str) -> None:
        self.w.write({
            "t_capture": self.session.rel(ev.capture_ts), "player": ev.player_id + 1,
            "gesture": ev.gesture, "onset_rel_ms": (ev.onset_ts - ev.capture_ts) * 1000,
            "recognition_latency_ms": (ev.processed_ts - ev.onset_ts) * 1000,
            "processing_ms": (ev.processed_ts - ev.capture_ts) * 1000,
            "command_latency_ms": (applied_ts - ev.capture_ts) * 1000 if applied_ts else math.nan,
            "applied": int(applied_ts is not None), "game_result": result})


@dataclass
class Trial:
    index: int
    players: tuple[int, ...]       # 0-based ids prompted
    gesture: str                   # expected gesture or "idle"
    start: float = 0.0
    duration: float = 2.5
    detections: dict[int, list[tuple[str, float]]] = field(default_factory=dict)


class GestureTrialRunner:
    """Prompted protocol. Drive with ``update(now)`` and ``on_gesture(ev)``."""

    HEADER = ["trial", "players", "expected", "player", "detected_first", "correct",
              "response_ms", "recognition_latency_ms", "wrong_detections", "duration_s"]

    def __init__(self, session: Session | None, players: tuple[int, ...] = (0, 1),
                 repeats: int = 5, seed: int | None = None) -> None:
        self.session = session
        self.writer = session.writer("gesture_trials", self.HEADER) if session else None
        rng = random.Random(seed)
        trials: list[Trial] = []
        gestures = ["punch_left", "punch_right", "block_start", "special"]
        for _ in range(repeats):
            block = []
            for g in gestures:
                for p in players:
                    block.append(((p,), g))
            if len(players) > 1:
                block.append((tuple(players), rng.choice(["punch_left", "punch_right"])))
            block.append((tuple(players), "idle"))
            rng.shuffle(block)
            trials += [Trial(0, ps, g, duration=6.0 if g == "idle" else 2.5) for ps, g in block]
        for i, t in enumerate(trials):
            t.index = i + 1
        self.trials = trials
        self.current: int | None = None       # index of the running trial
        self.next_index = 0
        self.gap_until = time.perf_counter() + 3.0      # get-ready pause
        self.results: list[dict] = []
        self.done = False

    @property
    def trial(self) -> Trial | None:
        return self.trials[self.current] if self.current is not None else None

    def update(self, now: float) -> None:
        if self.done:
            return
        tr = self.trial
        if tr is not None:
            if now - tr.start >= tr.duration:
                self._finish(tr)
                self.current = None
                self.gap_until = now + 1.5
            return
        if now >= self.gap_until:
            if self.next_index >= len(self.trials):
                self.done = True
                return
            self.current = self.next_index
            self.next_index += 1
            self.trials[self.current].start = now

    def on_gesture(self, ev: GestureEvent) -> None:
        tr = self.trial
        if tr is None or ev.gesture == "block_end":
            return
        # Only events whose motion began after the prompt count for this trial.
        if ev.onset_ts < tr.start - 0.1:
            return
        tr.detections.setdefault(ev.player_id, []).append((ev.gesture, ev.processed_ts, ev.onset_ts))

    def _finish(self, tr: Trial) -> None:
        for p in (0, 1):
            dets = tr.detections.get(p, [])
            if p in tr.players:
                expected = tr.gesture
            else:
                expected = "idle"     # the non-prompted player is an implicit idle trial
                if not dets:
                    continue          # nothing to report for a quiet bystander
            first = dets[0] if dets else None
            correct_hits = [d for d in dets if d[0] == expected]
            wrong = [d[0] for d in dets if d[0] != expected]
            row = {
                "trial": tr.index, "players": "+".join(str(x + 1) for x in tr.players),
                "expected": expected if p in tr.players else "idle(bystander)",
                "player": p + 1,
                "detected_first": first[0] if first else "",
                "correct": int(bool(correct_hits)) if expected != "idle" else int(not dets),
                "response_ms": (correct_hits[0][1] - tr.start) * 1000 if correct_hits else math.nan,
                "recognition_latency_ms": (correct_hits[0][1] - correct_hits[0][2]) * 1000
                if correct_hits else math.nan,
                "wrong_detections": ";".join(wrong), "duration_s": tr.duration,
            }
            self.results.append(row)
            if self.writer:
                self.writer.write(row)

    def prompt_text(self) -> tuple[str, str]:
        tr = self.trial
        if self.done:
            return "TEST COMPLETE", "Results saved. Press ESC."
        if tr is None:
            return "GET READY", f"Next: trial {self.next_index + 1} / {len(self.trials)}"
        who = " + ".join(f"P{p + 1}" for p in tr.players)
        return f"{who}: {GESTURE_LABEL[tr.gesture]}", f"Trial {tr.index} / {len(self.trials)}"


# ====================================================================== analysis
def _read(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _f(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return math.nan


def analyze_gestures(session_dir: Path) -> dict:
    out: dict = {}
    g = _read(session_dir / "gestures.csv")
    if g:
        per = defaultdict(list)
        for r in g:
            per[r["gesture"]].append(r)
        out["recognised_counts"] = {k: len(v) for k, v in per.items()}
        # Per gesture: block and special include their deliberate confirmation /
        # hold times, so pooling them with punches would be misleading.
        for gname in ATTACK_GESTURES:
            rows = per.get(gname, [])
            if not rows:
                continue
            stats = {}
            for key in ("recognition_latency_ms", "command_latency_ms", "processing_ms"):
                vals = np.array([_f(r[key]) for r in rows])
                vals = vals[np.isfinite(vals)]
                if vals.size:
                    stats[f"{key}_mean"] = float(vals.mean())
                    stats[f"{key}_p95"] = float(np.percentile(vals, 95))
            out[f"latency_{gname}"] = stats
        executed = [r for r in g if r["gesture"] in ("punch_left", "punch_right", "special")]
        if executed:
            out["executed_by_fighter_pct"] = float(
                np.mean([r["game_result"] == "executed" for r in executed]) * 100)

    trials = _read(session_dir / "gesture_trials.csv")
    if trials:
        res: dict = {}
        by_expected = defaultdict(list)
        for r in trials:
            by_expected[r["expected"]].append(r)
        predicted_counts = defaultdict(int)
        true_pos = defaultdict(int)
        for r in trials:
            exp = r["expected"]
            dets = [d for d in r["wrong_detections"].split(";") if d]
            if exp in ATTACK_GESTURES and r["correct"] == "1":
                true_pos[exp] += 1
                predicted_counts[exp] += 1
            for d in dets:
                predicted_counts[d] += 1
        for gname in ATTACK_GESTURES:
            rows = by_expected.get(gname, [])
            if not rows:
                continue
            recall = np.mean([r["correct"] == "1" for r in rows])
            prec = true_pos[gname] / predicted_counts[gname] if predicted_counts[gname] else math.nan
            lat = np.array([_f(r["recognition_latency_ms"]) for r in rows])
            resp = np.array([_f(r["response_ms"]) for r in rows])
            res[gname] = {"trials": len(rows), "recall_pct": float(recall * 100),
                          "missed": int(sum(r["correct"] != "1" for r in rows)),
                          "precision_pct": float(prec * 100) if prec == prec else math.nan,
                          "recognition_latency_ms_mean": float(np.nanmean(lat)) if np.isfinite(lat).any() else math.nan,
                          "response_from_prompt_ms_mean_INCLUDES_REACTION": float(np.nanmean(resp)) if np.isfinite(resp).any() else math.nan}
        idle = [r for r in trials if r["expected"] == "idle"]
        if idle:
            fp = sum(len([d for d in r["wrong_detections"].split(";") if d]) for r in idle)
            minutes = sum(_f(r["duration_s"]) for r in idle) / 60
            res["idle_trials"] = len(idle)
            res["idle_false_positives"] = fp
            res["idle_false_positives_per_min"] = fp / minutes if minutes else math.nan
        bystander = [r for r in trials if r["expected"] == "idle(bystander)"]
        res["bystander_false_positives"] = sum(
            len([d for d in r["wrong_detections"].split(";") if d]) for r in bystander)
        simul = defaultdict(list)
        for r in trials:
            if "+" in r["players"] and not r["expected"].startswith("idle"):
                simul[r["trial"]].append(r["correct"] == "1")
        if simul:
            res["simultaneous_trials"] = len(simul)
            res["simultaneous_success_pct"] = float(np.mean([all(v) and len(v) == 2
                                                             for v in simul.values()]) * 100)
        out["trials"] = res
    return out
