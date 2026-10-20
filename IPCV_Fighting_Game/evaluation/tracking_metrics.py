"""Face / body / multiplayer tracking metrics: logging + offline analysis.

Logged automatically (no ground truth needed):
  * face detection rate, body keypoint reliability (observed fraction per joint)
  * stationary jitter (during operator-marked "stationary" windows, F5)
  * dynamic jitter (second-difference RMS while moving) — raw vs filtered
  * face-effect alignment error (smoothed anchor vs measured landmark)
  * missing-landmark recoveries, occlusion recovery gaps, re-entry times
  * processing latency per stage

Require MANUAL ground truth (operator key presses, see README):
  * identity switches (F6 when you SEE the labels swap)
  * crossing success rate (F7 = crossing kept identities, F8 = crossing failed)
  * identity accuracy, which is derived from the above
"""
from __future__ import annotations

import csv
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from evaluation.session import Session
from vision.pipeline import VisionSnapshot
from vision.pose_tracking import KEY_JOINTS

PLAYERS = (1, 2)


def frame_header() -> list[str]:
    cols = ["t", "frame_id", "n_poses", "n_faces", "pose_ms", "face_ms", "track_ms", "total_ms",
            "capture_to_result_ms", "overlap", "stationary"]
    for p in PLAYERS:
        cols += [f"p{p}_{c}" for c in (
            "state", "conf", "cx", "cy", "sw", "face_status", "face_x", "face_y", "face_raw_x",
            "face_raw_y", "face_scale", "face_roll", "face_yaw", "align_err", "n_obs_key",
            "n_held_key", "recoveries")]
        for j in KEY_JOINTS:
            cols += [f"p{p}_{j}_x", f"p{p}_{j}_y", f"p{p}_{j}_rx", f"p{p}_{j}_ry", f"p{p}_{j}_obs"]
    return cols


class TrackingLogger:
    """Registered as a VisionPipeline frame listener (runs in the vision thread)."""

    def __init__(self, session: Session) -> None:
        self.session = session
        self.frames = session.writer("vision_frames", frame_header())
        self.events = session.writer("tracker_events", ["t", "player", "event", "details"])
        self.annotations = session.writer("annotations", ["t", "key", "label"])
        self.stationary = False

    def on_frame(self, snap: VisionSnapshot) -> None:
        s = self.session
        row = {"t": s.rel(snap.capture_ts), "frame_id": snap.frame_id, "n_poses": snap.num_poses,
               "n_faces": snap.num_faces, "overlap": int(snap.overlap),
               "stationary": int(self.stationary)}
        row.update(snap.timings_ms)
        for p in snap.players:
            k = f"p{p.pid + 1}_"
            f = p.face
            row.update({k + "state": p.state.value, k + "conf": p.confidence,
                        k + "cx": p.center[0], k + "cy": p.center[1], k + "sw": p.shoulder_width,
                        k + "face_status": f.status.value,
                        k + "face_x": f.center[0] if f.center else math.nan,
                        k + "face_y": f.center[1] if f.center else math.nan,
                        k + "face_raw_x": p.face_raw[0] if p.face_raw else math.nan,
                        k + "face_raw_y": p.face_raw[1] if p.face_raw else math.nan,
                        k + "face_scale": f.scale, k + "face_roll": f.roll, k + "face_yaw": f.yaw,
                        k + "align_err": f.alignment_error, k + "recoveries": p.landmark_recoveries})
            n_obs = n_held = 0
            for name, idx in KEY_JOINTS.items():
                obs = bool(p.observed[idx])
                valid = not np.isnan(p.landmarks[idx]).any()
                n_obs += obs
                n_held += valid and not obs
                row[f"{k}{name}_x"], row[f"{k}{name}_y"] = (float(p.landmarks[idx, 0]),
                                                            float(p.landmarks[idx, 1]))
                row[f"{k}{name}_rx"], row[f"{k}{name}_ry"] = (float(p.raw_landmarks[idx, 0]),
                                                              float(p.raw_landmarks[idx, 1]))
                row[f"{k}{name}_obs"] = int(obs)
            row[k + "n_obs_key"], row[k + "n_held_key"] = n_obs, n_held
        self.frames.write(row)
        for ev in snap.tracker_events:
            self.events.write({"t": s.rel(ev.ts), "player": ev.player + 1 if ev.player >= 0 else 0,
                               "event": ev.kind, "details": ev.details})

    def annotate(self, ts: float, key: str, label: str) -> None:
        if label in ("stationary_start", "stationary_end"):
            self.stationary = label == "stationary_start"
        self.annotations.write({"t": self.session.rel(ts), "key": key, "label": label})


# ====================================================================== analysis
def _read(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _num(v: str) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return math.nan


def _segments(mask: np.ndarray) -> list[tuple[int, int]]:
    segs, start = [], None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        elif not m and start is not None:
            segs.append((start, i))
            start = None
    if start is not None:
        segs.append((start, len(mask)))
    return segs


def _jitter(xs: np.ndarray, ys: np.ndarray, scale: np.ndarray, mask: np.ndarray,
            mode: str) -> float:
    """Jitter in units of ``scale`` over contiguous valid segments.

    stationary: RMS distance to the segment mean.
    dynamic:    RMS of the 2nd difference (acceleration) — removes smooth motion,
                keeps frame-to-frame noise. Divided by sqrt(6) so white noise of
                std s gives ~s, comparable to the stationary measure.
    """
    vals = []
    for a, b in _segments(mask & np.isfinite(xs) & np.isfinite(ys) & (scale > 0)):
        if b - a < 5:
            continue
        x, y, s = xs[a:b], ys[a:b], scale[a:b]
        if mode == "stationary":
            d = np.hypot(x - x.mean(), y - y.mean()) / s
        else:
            d = np.hypot(np.diff(x, 2), np.diff(y, 2)) / s[2:] / math.sqrt(6)
        vals.append(d)
    if not vals:
        return math.nan
    allv = np.concatenate(vals)
    return float(np.sqrt(np.mean(allv ** 2)))


def analyze_tracking(session_dir: Path) -> dict:
    rows = _read(session_dir / "vision_frames.csv")
    events = _read(session_dir / "tracker_events.csv")
    ann = _read(session_dir / "annotations.csv")
    out: dict = {"frames": len(rows)}
    if not rows:
        return out
    col = lambda k: np.array([_num(r.get(k)) for r in rows])
    stationary = col("stationary") > 0.5
    duration = col("t")[-1] - col("t")[0] if len(rows) > 1 else math.nan
    out["duration_s"] = duration
    for stage in ("pose_ms", "face_ms", "track_ms", "total_ms", "capture_to_result_ms"):
        v = col(stage)
        out[f"latency_{stage}_mean"] = float(np.nanmean(v))
        out[f"latency_{stage}_p95"] = float(np.nanpercentile(v, 95))
    out["frames_with_2_poses_pct"] = float(np.mean(col("n_poses") >= 2) * 100)

    for p in PLAYERS:
        k = f"p{p}_"
        state = np.array([r[k + "state"] for r in rows])
        active = state == "active"
        face_status = np.array([r[k + "face_status"] for r in rows])
        res: dict = {"active_frames": int(active.sum()),
                     "active_pct": float(active.mean() * 100)}
        if active.any():
            tracked = (face_status == "tracked") & active
            res["face_detection_rate_pct"] = float(tracked.sum() / active.sum() * 100)
            res["face_pose_fallback_pct"] = float(((face_status == "pose") & active).sum()
                                                  / active.sum() * 100)
            scale = col(k + "face_scale")
            fx, fy = col(k + "face_x"), col(k + "face_y")
            rx, ry = col(k + "face_raw_x"), col(k + "face_raw_y")
            for name, m in (("stationary", stationary & tracked), ("dynamic", ~stationary & tracked)):
                res[f"face_{name}_jitter_filtered"] = _jitter(fx, fy, scale, m, name)
                res[f"face_{name}_jitter_raw"] = _jitter(rx, ry, scale, m, name)
            al = col(k + "align_err")[tracked]
            al = al[np.isfinite(al)]
            if al.size:
                res["face_alignment_err_mean"] = float(al.mean())
                res["face_alignment_err_p95"] = float(np.percentile(al, 95))
            sw = col(k + "sw")
            rel = {}
            jit = {}
            for j in KEY_JOINTS:
                obs = col(f"{k}{j}_obs")
                rel[j] = float(np.nanmean(obs[active]) * 100)
                if j in ("l_wrist", "r_wrist", "head", "l_shoulder", "r_shoulder"):
                    m = stationary & active & (obs > 0.5)
                    jit[j] = {"filtered": _jitter(col(f"{k}{j}_x"), col(f"{k}{j}_y"), sw, m, "stationary"),
                              "raw": _jitter(col(f"{k}{j}_rx"), col(f"{k}{j}_ry"), sw, m, "stationary")}
            res["keypoint_observed_pct"] = rel
            res["landmark_stationary_jitter_sw"] = jit
            res["held_landmark_frames_pct"] = float(np.mean(col(k + "n_held_key")[active] > 0) * 100)
            rec = col(k + "recoveries")
            res["landmark_recoveries"] = int(np.nanmax(rec)) if np.isfinite(rec).any() else 0
        out[f"player{p}"] = res

    ev_counts = Counter((e["player"], e["event"]) for e in events)
    out["tracker_event_counts"] = {f"P{p}:{e}" if p != "0" else e: n
                                   for (p, e), n in sorted(ev_counts.items())}
    gaps = defaultdict(list)
    for e in events:
        d = e.get("details", "")
        for key in ("gap_s", "absent_s"):
            if key in d:
                try:
                    val = float(d.split(f"'{key}': ")[1].split(",")[0].split("}")[0])
                    gaps[key].append(val)
                except (IndexError, ValueError):
                    pass
    out["occlusion_recovery_gap_s_mean"] = float(np.mean(gaps["gap_s"])) if gaps["gap_s"] else math.nan
    out["occlusion_recoveries"] = len(gaps["gap_s"])
    out["reentries"] = len(gaps["absent_s"])
    out["reentry_recovery_s"] = _reentry_times(rows)

    labels = Counter(a["label"] for a in ann)
    switches = labels.get("identity_switch_observed", 0)
    ok, fail = labels.get("crossing_ok", 0), labels.get("crossing_failed", 0)
    out["MANUAL_identity_switches_observed"] = switches
    out["MANUAL_crossings_ok"] = ok
    out["MANUAL_crossings_failed"] = fail
    out["MANUAL_crossing_success_rate_pct"] = (ok / (ok + fail) * 100) if ok + fail else math.nan
    if duration and duration == duration and duration > 0:
        out["MANUAL_identity_switches_per_min"] = switches / (duration / 60)
    return out


def _reentry_times(rows: list[dict]) -> list[float]:
    """Time from the first frame where a missing player is visible again
    (more poses detected than active tracks) to the identity being restored."""
    times = []
    waiting: dict[int, float | None] = {1: None, 2: None}
    prev_state = {1: "", 2: ""}
    for r in rows:
        t = _num(r["t"])
        n_poses = _num(r["n_poses"])
        states = {p: r[f"p{p}_state"] for p in PLAYERS}
        n_active = sum(s == "active" for s in states.values())
        for p in PLAYERS:
            s = states[p]
            if s in ("lost", "absent"):
                if waiting[p] is None and n_poses > n_active:
                    waiting[p] = t
            elif s == "active" and prev_state[p] in ("lost", "absent"):
                if waiting[p] is not None:
                    times.append(round(t - waiting[p], 3))
                waiting[p] = None
            prev_state[p] = s
    return times
