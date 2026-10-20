"""Threaded webcam acquisition.

A dedicated thread reads frames as fast as the camera delivers them and
keeps only the newest one (stale frames are dropped). Consumers therefore
always process the most recent image, which bounds latency even when
inference is slower than the camera.
"""
from __future__ import annotations

import logging
import platform
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np

from config import CameraConfig

log = logging.getLogger(__name__)


@dataclass
class Frame:
    image: np.ndarray        # BGR, original orientation
    capture_ts: float        # time.perf_counter() right after read() returned
    frame_id: int


def _backend_flag(name: str) -> int:
    name = name.lower()
    if name == "auto":
        system = platform.system()
        if system == "Darwin":
            return cv2.CAP_AVFOUNDATION
        if system == "Windows":
            return cv2.CAP_DSHOW      # opens faster than MSMF on most laptops
        return cv2.CAP_ANY
    return {"avfoundation": cv2.CAP_AVFOUNDATION, "dshow": cv2.CAP_DSHOW,
            "msmf": cv2.CAP_MSMF, "v4l2": cv2.CAP_V4L2}.get(name, cv2.CAP_ANY)


class CameraError(RuntimeError):
    pass


class CameraThread:
    def __init__(self, cfg: CameraConfig) -> None:
        self.cfg = cfg
        self._cap: cv2.VideoCapture | None = None
        self._lock = threading.Lock()
        self._latest: Frame | None = None
        self._running = False
        self._thread: threading.Thread | None = None
        self.frames_read = 0
        self.read_failures = 0
        self.actual_size: tuple[int, int] = (0, 0)
        self.actual_fps = 0.0

    def start(self) -> None:
        cap = cv2.VideoCapture(self.cfg.index, _backend_flag(self.cfg.backend))
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(self.cfg.index)        # fallback: default backend
        if not cap.isOpened():
            raise CameraError(
                f"Could not open camera index {self.cfg.index}. On macOS, grant camera "
                "permission to your terminal/IDE (System Settings > Privacy & Security > "
                "Camera) and try --camera 1 if you have several cameras.")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.height)
        cap.set(cv2.CAP_PROP_FPS, self.cfg.fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        ok, img = cap.read()
        if not ok or img is None:
            cap.release()
            raise CameraError(f"Camera {self.cfg.index} opened but returned no frames.")
        self.actual_size = (img.shape[1], img.shape[0])
        self.actual_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        log.info("Camera %d opened at %dx%d (reported %.1f fps)", self.cfg.index,
                 *self.actual_size, self.actual_fps)
        self._cap = cap
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="camera", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        assert self._cap is not None
        while self._running:
            ok, img = self._cap.read()
            ts = time.perf_counter()
            if not ok or img is None:
                self.read_failures += 1
                time.sleep(0.01)
                continue
            self.frames_read += 1
            with self._lock:
                self._latest = Frame(img, ts, self.frames_read)

    def latest(self, newer_than: int = 0) -> Frame | None:
        """Newest frame, or None if there is nothing newer than ``newer_than``."""
        with self._lock:
            f = self._latest
        if f is None or f.frame_id <= newer_than:
            return None
        return f

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        if self._cap:
            self._cap.release()
        self._cap = None
