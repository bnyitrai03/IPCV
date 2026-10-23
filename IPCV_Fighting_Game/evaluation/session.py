"""Experiment session: one folder of CSV logs per run.

    logs/session_YYYYmmdd_HHMMSS/
        performance.csv     one row per second (FPS, frame-time percentiles, latency, CPU, RAM)
        vision_frames.csv   one row per processed camera frame (tracking + face + timings)
        tracker_events.csv  identity events (registered, lost, recovered, absent, reentry, overlap)
        gestures.csv        every recognised gesture + when the game applied it
        annotations.csv     operator ground-truth key presses (see README, "Evaluation")
        gesture_trials.csv  prompted trials from the gesture test mode
        meta.json           config + platform info

Writes are thread-safe (the vision thread and the game loop both log).
"""
from __future__ import annotations

import csv
import json
import platform
import threading
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from config import LOGS_DIR, Config


class CsvWriter:
    def __init__(self, path: Path, header: list[str]) -> None:
        self.path = path
        self.header = header
        self._fh = path.open("w", newline="", encoding="utf-8")
        self._w = csv.writer(self._fh)
        self._w.writerow(header)
        self._lock = threading.Lock()
        self.rows = 0

    def write(self, row: dict) -> None:
        with self._lock:
            self._w.writerow([_fmt(row.get(k, "")) for k in self.header])
            self.rows += 1
            if self.rows % 50 == 0:
                self._fh.flush()

    def close(self) -> None:
        with self._lock:
            self._fh.flush()
            self._fh.close()


def _fmt(v):
    if isinstance(v, float):
        return f"{v:.6f}" if v == v else ""   # NaN -> empty cell
    return v


class Session:
    def __init__(self, cfg: Config, tag: str = "", root: Path | None = None) -> None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"session_{stamp}" + (f"_{tag}" if tag else "")
        self.dir = (root or LOGS_DIR) / name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.t0 = time.perf_counter()
        self.writers: dict[str, CsvWriter] = {}
        meta = {"started": datetime.now().isoformat(), "platform": platform.platform(),
                "python": platform.python_version(), "machine": platform.machine(),
                "processor": platform.processor(), "config": asdict(cfg), "tag": tag,
                "perf_counter_t0": self.t0}
        (self.dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str))

    def writer(self, name: str, header: list[str]) -> CsvWriter:
        if name not in self.writers:
            self.writers[name] = CsvWriter(self.dir / f"{name}.csv", header)
        return self.writers[name]

    def rel(self, ts: float) -> float:
        """perf_counter timestamp -> seconds since session start."""
        return ts - self.t0

    def close(self) -> None:
        for w in self.writers.values():
            w.close()
