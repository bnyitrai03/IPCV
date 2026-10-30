"""Task 5 (vision side) — the real-time computer-vision pipeline.

Threads::

    camera thread  --latest frame-->  vision thread  --snapshot + events-->  game loop
    (CameraThread)                    (this module)                         (main thread)

The vision thread always takes the *newest* camera frame (stale frames are
dropped), runs pose + face inference, identity tracking and gesture
recognition, then publishes an immutable :class:`VisionSnapshot` and pushes
gesture events into a thread-safe queue. The game loop never waits for
inference: it renders at its own rate using the latest snapshot.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable

import cv2
import numpy as np

from config import Config
from vision.face_tracking import FaceStatus
from vision.gesture_recognition import GestureEvent
from vision.player_tracking import PlayerTracker, TrackerEvent, TrackState

log = logging.getLogger(__name__)


@dataclass
class FaceSnapshot:
    status: FaceStatus
    center: tuple[float, float] | None       # normalised image coords
    forehead: tuple[float, float] | None
    scale: float                             # in normalised-x units
    roll: float
    yaw: float
    alpha: float
    alignment_error: float


@dataclass
class PlayerSnapshot:
    pid: int
    state: TrackState
    landmarks: np.ndarray            # (33, 2) normalised image coords (NaN = missing)
    observed: np.ndarray             # (33,) bool, seen this frame
    confidence: float
    missing_joints: list[str]
    bbox: tuple[float, float, float, float] | None
    center: tuple[float, float]      # normalised image coords
    shoulder_width: float            # normalised-x units
    face: FaceSnapshot
    move: float
    move_offset: float
    block: bool
    special_progress: float
    last_attack: str             # most recent recognised attack ("hook_left", ...), "" if none
    last_attack_t: float         # time.perf_counter() when it was recognised (-inf if none)
    special_t: float             # time.perf_counter() of the last recognised special
    signals: dict[str, float]
    landmark_recoveries: int
    raw_landmarks: np.ndarray        # (33, 2) unfiltered detector output (NaN = not visible)
    face_raw: tuple[float, float] | None   # unfiltered face centre (normalised image coords)


@dataclass
class VisionSnapshot:
    frame_id: int
    capture_ts: float
    image: np.ndarray | None                 # mirrored BGR at inference resolution
    players: list[PlayerSnapshot]
    num_poses: int
    num_faces: int
    timings_ms: dict[str, float]
    overlap: bool
    registered: bool
    source: str
    tracker_events: list[TrackerEvent] = field(default_factory=list)


class VisionError(RuntimeError):
    pass


class VisionPipeline:
    def __init__(self, cfg: Config, source: str = "camera") -> None:
        self.cfg = cfg
        self.source = source
        self._lock = threading.Lock()
        self._snapshot: VisionSnapshot | None = None
        self._events: deque[GestureEvent] = deque(maxlen=256)
        self._running = False
        self._thread: threading.Thread | None = None
        self.error: str | None = None
        self.camera = None
        self.pose_backend = None
        self.face_backend = None
        self.synthetic = None
        self.tracker: PlayerTracker | None = None
        self.frame_listeners: list[Callable[[VisionSnapshot], None]] = []
        self.vision_fps = 0.0
        self._fps_times: deque[float] = deque(maxlen=60)
        self._commands: deque[str] = deque()

    # ----------------------------------------------------------------- setup
    def start(self) -> None:
        """Open camera + models. Raises VisionError with a readable message."""
        try:
            if self.source == "synthetic":
                from vision.synthetic import SyntheticScenario
                self.synthetic = SyntheticScenario()
                aspect = self.synthetic.aspect
            else:
                from vision.camera import CameraError, CameraThread
                from vision.face_tracking import MediaPipeFaceBackend
                from vision.pose_tracking import MediaPipePoseBackend
                try:
                    self.pose_backend = MediaPipePoseBackend(
                        self.cfg.pose, self.cfg.model_path(self.cfg.pose.model_file))
                    self.face_backend = MediaPipeFaceBackend(
                        self.cfg.face, self.cfg.model_path(self.cfg.face.model_file))
                except FileNotFoundError as exc:
                    raise VisionError(str(exc)) from exc
                except Exception as exc:   # MediaPipe init errors (e.g. GPU/delegate issues)
                    raise VisionError(f"MediaPipe initialisation failed: {exc}") from exc
                self.camera = CameraThread(self.cfg.camera)
                try:
                    self.camera.start()
                except CameraError as exc:
                    raise VisionError(str(exc)) from exc
                w, h = self.camera.actual_size
                aspect = w / h
            self.tracker = PlayerTracker(self.cfg, aspect)
            self._running = True
            self._thread = threading.Thread(target=self._loop, name="vision", daemon=True)
            self._thread.start()
        except Exception:
            # Release anything acquired before the failure (camera, MediaPipe graphs).
            self.stop()
            raise

    def stop(self) -> None:
        self._running = False
        if self.camera:
            try:
                self.camera.stop()
            except Exception:   # noqa: BLE001 - best effort on shutdown
                log.exception("Error while stopping the camera thread")
        thread, self._thread = self._thread, None
        joined = True
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=5.0)
            joined = not thread.is_alive()
            if not joined:
                # Never close MediaPipe graphs under a running detect() call:
                # that can deadlock in native code. The thread is a daemon, so
                # leaving its resources to process exit is the safe option.
                log.warning("Vision thread did not stop within timeout; "
                            "skipping native backend close")
        if joined:
            for b in (self.pose_backend, self.face_backend):
                if b is not None:
                    try:
                        b.close()
                    except Exception:   # noqa: BLE001 - best effort on shutdown
                        pass
            self.pose_backend = self.face_backend = None

    # -------------------------------------------------------- public access
    def snapshot(self) -> VisionSnapshot | None:
        with self._lock:
            return self._snapshot

    def drain_events(self) -> list[GestureEvent]:
        with self._lock:
            out = list(self._events)
            self._events.clear()
        return out

    def request(self, command: str) -> None:
        """Thread-safe commands executed by the vision thread: 'reregister'."""
        with self._lock:
            self._commands.append(command)

    # ------------------------------------------------------------------ loop
    def _loop(self) -> None:
        last_id = 0
        frame_period = 1.0 / 30.0
        t0 = time.perf_counter()
        face_counter = 0
        while self._running:
            try:
                self._run_commands()
                if self.synthetic is not None:
                    # Emulate a 30 fps camera.
                    target = t0 + (last_id + 1) * frame_period
                    delay = target - time.perf_counter()
                    if delay > 0:
                        time.sleep(delay)
                    capture_ts = time.perf_counter()
                    last_id += 1
                    t_start = time.perf_counter()
                    display, poses, faces = self.synthetic.frame(capture_ts - t0)
                    pose_ms = face_ms = 0.0
                    faces_ran = True
                else:
                    frame = self.camera.latest(newer_than=last_id)
                    if frame is None:
                        time.sleep(0.002)
                        continue
                    last_id, capture_ts = frame.frame_id, frame.capture_ts
                    t_start = time.perf_counter()
                    small = self._resize(frame.image)
                    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                    ts_ms = int(capture_ts * 1000)
                    a = time.perf_counter()
                    poses = self.pose_backend.detect(rgb, ts_ms)
                    pose_ms = (time.perf_counter() - a) * 1000
                    face_counter += 1
                    faces_ran = face_counter % max(1, self.cfg.face.run_every_n_frames) == 0
                    faces, face_ms = None, 0.0
                    if faces_ran:
                        a = time.perf_counter()
                        faces = self.face_backend.detect(rgb, ts_ms)
                        face_ms = (time.perf_counter() - a) * 1000
                    display = cv2.flip(small, 1) if self.cfg.camera.mirror else small
                a = time.perf_counter()
                n_events_before = len(self.tracker.events)
                events = self.tracker.update(poses, faces if faces_ran else None, display,
                                             capture_ts, time.perf_counter())
                track_ms = (time.perf_counter() - a) * 1000
                processed = time.perf_counter()
                timings = {"pose_ms": pose_ms, "face_ms": face_ms, "track_ms": track_ms,
                           "total_ms": (processed - t_start) * 1000,
                           "queue_ms": (t_start - capture_ts) * 1000,
                           "capture_to_result_ms": (processed - capture_ts) * 1000}
                self._fps_times.append(processed)
                if len(self._fps_times) > 1:
                    span = self._fps_times[-1] - self._fps_times[0]
                    self.vision_fps = (len(self._fps_times) - 1) / span if span > 0 else 0.0
                snap = self._make_snapshot(last_id, capture_ts, display, poses,
                                           faces, timings,
                                           self.tracker.events[n_events_before:])
                with self._lock:
                    self._snapshot = snap
                    self._events.extend(events)
                for fn in self.frame_listeners:
                    fn(snap)
                # Keep the tracker's event list bounded during long sessions.
                if len(self.tracker.events) > 500:
                    del self.tracker.events[:250]
            except Exception as exc:   # never let the vision thread die silently
                log.exception("Vision loop error")
                self.error = f"Vision error: {exc}"
                time.sleep(0.2)

    def _run_commands(self) -> None:
        with self._lock:
            cmds = list(self._commands)
            self._commands.clear()
        for c in cmds:
            if c == "reregister" and self.tracker:
                self.tracker.reset_registration()

    def _resize(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        target_w = self.cfg.camera.inference_width
        if w <= target_w:
            return img
        return cv2.resize(img, (target_w, int(round(h * target_w / w))),
                          interpolation=cv2.INTER_AREA)

    def _make_snapshot(self, frame_id, capture_ts, display, poses, faces, timings,
                       new_events) -> VisionSnapshot:
        assert self.tracker is not None
        aspect = self.tracker.aspect
        players = []
        for tr in self.tracker.tracks:
            lm = tr.pose.img.value.copy()
            lm[:, 1] *= aspect
            lm[~tr.pose.img.valid] = np.nan
            raw = tr.pose.raw_points.copy()
            raw[:, 1] *= aspect
            raw[tr.pose.raw_visibility < self.cfg.pose.landmark_visibility_threshold] = np.nan
            if not tr.matched_this_frame:
                raw[:] = np.nan
            f = tr.face
            arms = [a for a in tr.gestures.arms.values() if a.last_kind]
            recent = max(arms, key=lambda a: a.last_fire, default=None)
            last_kind = f"{recent.last_kind}_{recent.side}" if recent else ""
            last_t = recent.last_fire if recent else float("-inf")

            def img_pt(p):
                return None if p is None else (float(p[0]), float(p[1] * aspect))
            face = FaceSnapshot(
                status=f.status,
                center=img_pt(f.center) if f.visible else None,
                forehead=img_pt(f.forehead) if f.visible else None,
                scale=f.scale, roll=f.roll, yaw=f.yaw, alpha=f.alpha,
                alignment_error=f.alignment_error)
            players.append(PlayerSnapshot(
                pid=tr.id, state=tr.state, landmarks=lm,
                observed=tr.pose.img.observed.copy(),
                confidence=tr.pose.confidence() if tr.state == TrackState.ACTIVE else 0.0,
                missing_joints=tr.pose.missing_key_joints(),
                bbox=tr.bbox,
                center=(float(tr.center[0]), float(tr.center[1] * aspect)),
                shoulder_width=float(tr.pose.shoulder_width),
                face=face, move=tr.gestures.move, move_offset=tr.gestures.move_offset,
                block=tr.gestures.block_active,
                special_progress=tr.gestures.special_progress,
                last_attack=last_kind, last_attack_t=last_t,
                special_t=tr.gestures._special_last,
                signals=dict(tr.gestures.signals),
                landmark_recoveries=tr.pose.img.recoveries,
                raw_landmarks=raw,
                face_raw=img_pt(f.raw.center) if f.raw is not None else None))
        return VisionSnapshot(
            frame_id=frame_id, capture_ts=capture_ts,
            image=display, players=players, num_poses=len(poses),
            num_faces=len(faces) if faces is not None else -1, timings_ms=timings,
            overlap=self.tracker.overlap, registered=self.tracker.all_registered,
            source=self.source, tracker_events=list(new_events))
