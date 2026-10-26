"""System performance monitoring: FPS, frame-time percentiles, latency, CPU/RAM.

Definitions (all measured, nothing assumed):
  * render FPS         frames presented per second by the game loop
  * 1% low FPS         1 / 99th-percentile frame time over the window
  * vision FPS         camera frames fully processed per second
  * capture->result    camera read() returned -> tracking/gestures finished
  * end-to-end (e2e)   camera read() returned -> first display flip showing that
                       frame's result. Excludes sensor exposure and USB/driver
                       latency before read() returns (not measurable in software).
  * command latency    capture of the frame that confirmed a gesture -> the game
                       logic tick that applied it (logged in gestures.csv).
"""
from __future__ import annotations

import time

import numpy as np

try:
    import psutil
except ImportError:      # optional dependency
    psutil = None


class PerformanceMonitor:
    HEADER = ["t", "render_fps", "frame_ms_mean", "frame_ms_p99", "low1_fps", "vision_fps",
              "pose_ms", "face_ms", "track_ms", "capture_to_result_ms", "e2e_ms",
              "cpu_percent", "rss_mb", "particles"]

    def __init__(self, window_s: float = 1.0) -> None:
        self.window_s = window_s
        self._win_frames: list[float] = []
        self._win_vision: list[dict] = []
        self._win_e2e: list[float] = []
        self._win_start = time.perf_counter()
        self._last_snapshot_id = -1
        self.proc = psutil.Process() if psutil else None
        if self.proc:
            self.proc.cpu_percent(None)
        self.latest_row: dict = {}

    def frame(self, dt: float) -> None:
        if dt <= 0:
            return
        self._win_frames.append(dt)

    def on_display(self, snapshot, flip_ts: float) -> None:
        """Call right after display.flip(); counts each vision frame once."""
        if snapshot is None or snapshot.frame_id == self._last_snapshot_id:
            return
        self._last_snapshot_id = snapshot.frame_id
        e2e = (flip_ts - snapshot.capture_ts) * 1000
        self._win_e2e.append(e2e)
        self._win_vision.append(snapshot.timings_ms)

    def tick(self, vision_fps: float, extra: dict | None = None) -> dict | None:
        """Close the window once per second; returns a CSV row or None."""
        now = time.perf_counter()
        if now - self._win_start < self.window_s:
            return None
        frames = np.array(self._win_frames) if self._win_frames else np.array([np.nan])
        span = now - self._win_start
        def vmean(key):
            vals = [v[key] for v in self._win_vision if key in v]
            return float(np.mean(vals)) if vals else float("nan")
        p99 = float(np.percentile(frames, 99)) if self._win_frames else float("nan")
        row = {
            "t": now,
            "render_fps": len(self._win_frames) / span,
            "frame_ms_mean": float(np.nanmean(frames) * 1000),
            "frame_ms_p99": p99 * 1000,
            "low1_fps": 1.0 / p99 if p99 == p99 and p99 > 0 else float("nan"),
            "vision_fps": vision_fps,
            "pose_ms": vmean("pose_ms"), "face_ms": vmean("face_ms"),
            "track_ms": vmean("track_ms"),
            "capture_to_result_ms": vmean("capture_to_result_ms"),
            "e2e_ms": float(np.mean(self._win_e2e)) if self._win_e2e else float("nan"),
            "cpu_percent": self.proc.cpu_percent(None) if self.proc else float("nan"),
            "rss_mb": self.proc.memory_info().rss / 1e6 if self.proc else float("nan"),
        }
        if extra:
            row.update(extra)
        self._win_frames, self._win_vision, self._win_e2e = [], [], []
        self._win_start = now
        self.latest_row = row
        return row

    def overlay_lines(self) -> list[str]:
        r = self.latest_row
        if not r:
            return ["measuring..."]
        def f(k, fmt="{:.1f}"):
            v = r.get(k, float("nan"))
            return "n/a" if v != v else fmt.format(v)
        return [
            f"render {f('render_fps')} fps  (1% low {f('low1_fps')})  frame {f('frame_ms_mean')} ms",
            f"vision {f('vision_fps')} fps  pose {f('pose_ms')} ms  face {f('face_ms')} ms"
            f"  track {f('track_ms')} ms",
            f"capture->result {f('capture_to_result_ms')} ms   end-to-end {f('e2e_ms')} ms",
            f"CPU {f('cpu_percent')} %   RAM {f('rss_mb', '{:.0f}')} MB",
        ]
