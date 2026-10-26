"""Game-interaction metrics: gesture logging and analysis.

``gestures.csv`` records every gesture recognised during play, with
* recognition latency  = confirm frame capture - motion-onset frame capture
                         + processing time (how long the recogniser needed
                         to be sure, excluding human reaction),
* command latency      = game tick that applied it - capture of the
                         confirming frame,
* whether the fighter executed it (or it was ignored: stunned, no energy...).
"""
from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from evaluation.session import Session, parse_float, read_csv
from vision.gesture_recognition import GestureEvent

ATTACK_GESTURES = ("hook_left", "hook_right",
                   "uppercut_left", "uppercut_right", "block_start", "special")


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


# ====================================================================== analysis
def analyze_gestures(session_dir: Path) -> dict:
    out: dict = {}
    g = read_csv(session_dir / "gestures.csv")
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
                vals = np.array([parse_float(r[key]) for r in rows])
                vals = vals[np.isfinite(vals)]
                if vals.size:
                    stats[f"{key}_mean"] = float(vals.mean())
                    stats[f"{key}_p95"] = float(np.percentile(vals, 95))
            out[f"latency_{gname}"] = stats
        executed = [r for r in g if r["gesture"] not in ("block_start", "block_end")]
        if executed:
            out["executed_by_fighter_pct"] = float(
                np.mean([r["game_result"] == "executed" for r in executed]) * 100)
    return out
