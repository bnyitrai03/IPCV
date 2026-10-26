"""Temporal filtering for landmarks and face parameters.

Uses the One Euro filter (Casiez et al., CHI 2012): a low-pass filter whose
cutoff frequency adapts to speed. Slow motion -> strong smoothing (low
jitter); fast motion -> little smoothing (low lag). This trade-off is exactly
what a fighting game needs: steady guard poses, responsive punches.

All filters are vectorised over numpy arrays and take explicit timestamps,
so they stay correct when the vision frame rate fluctuates.
"""
from __future__ import annotations

import math

import numpy as np


def _alpha(cutoff: float | np.ndarray, dt: float) -> float | np.ndarray:
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter:
    """Vectorised One Euro filter. Also exposes the filtered derivative (velocity)."""

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.0, d_cutoff: float = 1.0) -> None:
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev: np.ndarray | None = None
        self.dx_prev: np.ndarray | None = None
        self.t_prev: float | None = None

    def reset(self) -> None:
        self.x_prev = self.dx_prev = self.t_prev = None

    def __call__(self, x, t: float) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if self.x_prev is None or self.t_prev is None:
            self.x_prev, self.dx_prev, self.t_prev = x.copy(), np.zeros_like(x), t
            return x.copy()
        dt = t - self.t_prev
        if dt <= 1e-6:
            return self.x_prev.copy()
        dx = (x - self.x_prev) / dt
        a_d = _alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = _alpha(cutoff, dt)
        x_hat = a * x + (1 - a) * self.x_prev
        self.x_prev, self.dx_prev, self.t_prev = x_hat, dx_hat, t
        return x_hat.copy()

    def scalar(self, x: float, t: float) -> float:
        """Filter one scalar without allocating 1-element arrays."""
        if self.x_prev is None or self.t_prev is None:
            self.x_prev = np.array([x], dtype=np.float64)
            self.dx_prev = np.zeros_like(self.x_prev)
            self.t_prev = t
            return float(x)
        prev = float(self.x_prev[0])
        dt = t - self.t_prev
        if dt <= 1e-6:
            return prev
        dx = (x - prev) / dt
        a_d = _alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1 - a_d) * float(self.dx_prev[0])
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = _alpha(cutoff, dt)
        x_hat = a * x + (1 - a) * prev
        self.x_prev[0] = x_hat
        self.dx_prev[0] = dx_hat
        self.t_prev = t
        return float(x_hat)

    @property
    def velocity(self) -> np.ndarray | None:
        return None if self.dx_prev is None else self.dx_prev.copy()


class AngleFilter(OneEuroFilter):
    """One Euro filter for angles (radians) that handles wrap-around at +-pi."""

    def __call__(self, x, t: float) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if self.x_prev is not None:
            # Unwrap the new measurement close to the previous estimate.
            x = self.x_prev + (x - self.x_prev + np.pi) % (2 * np.pi) - np.pi
        return super().__call__(x, t)

    def scalar(self, x: float, t: float) -> float:
        if self.x_prev is not None:
            prev = float(self.x_prev[0])
            x = prev + (x - prev + math.pi) % (2 * math.pi) - math.pi
        return super().scalar(x, t)


class LandmarkSmoother:
    """Filters an (N, D) landmark array with per-landmark visibility handling.

    * Landmarks with visibility >= threshold are filtered normally.
    * Missing landmarks keep their last filtered value for ``hold_s`` seconds
      (flagged as *held*), then become invalid (NaN) until seen again.
    * When a landmark reappears after being invalid, its filter restarts so
      it doesn't "slide" in from a stale position.
    """

    def __init__(self, n: int, dims: int, vis_threshold: float, hold_s: float,
                 min_cutoff: float, beta: float, d_cutoff: float = 1.0) -> None:
        self.n, self.dims = n, dims
        self.vis_threshold = vis_threshold
        self.hold_s = hold_s
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.value = np.full((n, dims), np.nan)
        self.last_seen = np.full(n, -np.inf)
        self.valid = np.zeros(n, dtype=bool)      # value usable (seen or held)
        self.observed = np.zeros(n, dtype=bool)   # actually seen this frame
        self.recoveries = 0                        # count of missing -> seen events
        self._dx_prev = np.zeros((n, dims))
        self._t_prev = np.full(n, np.nan)

    def update(self, points: np.ndarray, visibility: np.ndarray, t: float) -> None:
        seen = visibility >= self.vis_threshold
        reappeared = seen & ~self.valid
        if np.any(reappeared & np.isfinite(self.last_seen)):
            self.recoveries += int(np.count_nonzero(reappeared & np.isfinite(self.last_seen)))

        continuing = seen & self.valid & np.isfinite(self._t_prev)
        if np.any(continuing):
            dt = t - self._t_prev[continuing]
            active = dt > 1e-6
            idx = np.flatnonzero(continuing)[active]
            if idx.size:
                dtv = dt[active][:, None]
                x = points[idx]
                x_prev = self.value[idx]
                dx = (x - x_prev) / dtv
                a_d = _alpha(self.d_cutoff, dtv)
                dx_hat = a_d * dx + (1 - a_d) * self._dx_prev[idx]
                cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
                a = _alpha(cutoff, dtv)
                self.value[idx] = a * x + (1 - a) * x_prev
                self._dx_prev[idx] = dx_hat
                self._t_prev[idx] = t

        if np.any(reappeared):
            self.value[reappeared] = points[reappeared]
            self._dx_prev[reappeared] = 0.0
            self._t_prev[reappeared] = t

        self.last_seen[seen] = t
        self.valid[seen] = True

        missing = ~seen
        if np.any(missing):
            miss_idx = np.flatnonzero(missing)
            held = (t - self.last_seen[miss_idx]) <= self.hold_s
            held_idx = miss_idx[held]
            invalid_idx = miss_idx[~held]
            self.valid[held_idx] = True
            self.valid[invalid_idx] = False
            self.value[invalid_idx] = np.nan
        self.observed = seen.copy()

    def mark_all_missing(self, t: float) -> None:
        self.update(np.zeros((self.n, self.dims)), np.zeros(self.n), t)
