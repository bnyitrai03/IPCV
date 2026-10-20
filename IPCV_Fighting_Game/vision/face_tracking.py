"""Task 1 — multi-face tracking and face-pose estimation for AR effects.

Backend: MediaPipe Tasks ``FaceLandmarker`` (478-point mesh, ``num_faces=2``).

Per frame we estimate, for every detected face:
  * position  — midpoint between the eyes and a forehead anchor,
  * scale     — inter-ocular distance corrected for yaw foreshortening,
  * orientation — roll (eye-line angle), yaw (nose/cheek asymmetry) and a
                  pitch ratio (nose position between forehead and chin).

These are 2D geometric approximations from a monocular image, not a full
6-DoF head pose; they are robust, cheap and sufficient for aligning 2D
overlays.

Faces are NOT tracked by themselves: each detection is associated to a
body identity (see ``player_tracking.py``), so an effect can never jump to
the other player even if the two faces swap order in the detector output.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np

from config import FaceConfig
from vision.smoothing import AngleFilter, OneEuroFilter

log = logging.getLogger(__name__)

# Face-mesh indices.
EYE_A = (33, 133)        # one eye: outer, inner corner
EYE_B = (362, 263)       # other eye: inner, outer corner
NOSE_TIP = 1
FOREHEAD = 10
CHIN = 152
CHEEK_A, CHEEK_B = 234, 454


@dataclass
class FaceDetection:
    landmarks: np.ndarray    # (N, 3) normalised image coords, mirrored to display space


@dataclass
class FaceGeometry:
    center: np.ndarray       # eye midpoint, isotropic normalised coords
    forehead: np.ndarray
    chin: np.ndarray
    eye_dist: float
    scale: float             # yaw-corrected face scale (≈ frontal eye distance)
    roll: float              # radians, image plane
    yaw: float               # radians, + = turned towards image right
    pitch: float             # ratio, ~0 neutral, + = looking down
    bbox: tuple[float, float, float, float]


def compute_face_geometry(lms: np.ndarray, aspect: float) -> FaceGeometry:
    """Geometry from 478 face-mesh landmarks (normalised image coords)."""
    p = lms[:, :2].astype(np.float64).copy()
    p[:, 1] /= aspect                       # isotropic coordinates
    ea = p[list(EYE_A)].mean(axis=0)
    eb = p[list(EYE_B)].mean(axis=0)
    left, right = (ea, eb) if ea[0] <= eb[0] else (eb, ea)
    center = (left + right) / 2
    eye_vec = right - left
    eye_dist = float(max(np.linalg.norm(eye_vec), 1e-4))
    roll = math.atan2(eye_vec[1], eye_vec[0])

    nose = p[NOSE_TIP]
    ca, cb = p[CHEEK_A], p[CHEEK_B]
    c_left, c_right = (ca, cb) if ca[0] <= cb[0] else (cb, ca)
    d_l = np.linalg.norm(nose - c_left)
    d_r = np.linalg.norm(nose - c_right)
    yaw_ratio = float(np.clip((d_l - d_r) / max(d_l + d_r, 1e-6), -0.95, 0.95))
    yaw = math.asin(yaw_ratio)

    fh, ch = p[FOREHEAD], p[CHIN]
    d_up = np.linalg.norm(nose - fh)
    d_dn = np.linalg.norm(nose - ch)
    pitch = float((d_up - d_dn) / max(d_up + d_dn, 1e-6))

    scale = eye_dist / max(math.cos(yaw), 0.55)
    xs, ys = p[:, 0], p[:, 1]
    return FaceGeometry(center, fh.copy(), ch.copy(), eye_dist, scale, roll, yaw, pitch,
                        (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())))


def geometry_from_pose_head(eye_l: np.ndarray, eye_r: np.ndarray, nose: np.ndarray,
                            ear_l: np.ndarray | None, ear_r: np.ndarray | None) -> FaceGeometry:
    """Coarse face geometry from body-pose head landmarks (fallback when the
    face mesh is lost, e.g. strong head rotation or motion blur)."""
    left, right = (eye_l, eye_r) if eye_l[0] <= eye_r[0] else (eye_r, eye_l)
    center = (left + right) / 2
    v = right - left
    eye_dist = float(max(np.linalg.norm(v), 1e-4))
    roll = math.atan2(v[1], v[0])
    yaw = 0.0
    if ear_l is not None and ear_r is not None:
        a, b = (ear_l, ear_r) if ear_l[0] <= ear_r[0] else (ear_r, ear_l)
        d_l, d_r = np.linalg.norm(nose - a), np.linalg.norm(nose - b)
        yaw = math.asin(float(np.clip((d_l - d_r) / max(d_l + d_r, 1e-6), -0.95, 0.95)))
    scale = eye_dist / max(math.cos(yaw), 0.55)
    # Forehead ≈ 1.1 eye distances above the eye line, perpendicular to it.
    up = np.array([math.sin(roll), -math.cos(roll)])
    forehead = center + up * 1.1 * scale
    chin = center - up * 1.9 * scale
    half = 1.3 * scale
    return FaceGeometry(center, forehead, chin, eye_dist, scale, roll, yaw, 0.0,
                        (center[0] - half, forehead[1], center[0] + half, chin[1]))


class FaceStatus(Enum):
    TRACKED = "tracked"          # face mesh measured this frame
    PREDICTED = "predicted"      # short gap: extrapolate and fade
    POSE_FALLBACK = "pose"       # mesh lost, head estimated from body pose
    LOST = "lost"


class MediaPipeFaceBackend:
    def __init__(self, cfg: FaceConfig, model_path: Path) -> None:
        if not model_path.is_file():
            raise FileNotFoundError(
                f"Face model not found at {model_path}. Run: python scripts/download_models.py")
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python import vision as mpv

        self._mp = mp
        options = mpv.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=mpv.RunningMode.VIDEO,
            num_faces=cfg.num_faces,
            min_face_detection_confidence=cfg.min_detection_confidence,
            min_face_presence_confidence=cfg.min_presence_confidence,
            min_tracking_confidence=cfg.min_tracking_confidence,
        )
        self._landmarker = mpv.FaceLandmarker.create_from_options(options)
        self._last_ts_ms = -1
        log.info("FaceLandmarker ready (num_faces=%d)", cfg.num_faces)

    def detect(self, rgb: np.ndarray, ts_ms: int) -> list[FaceDetection]:
        ts_ms = max(ts_ms, self._last_ts_ms + 1)
        self._last_ts_ms = ts_ms
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(image, ts_ms)
        faces = []
        for lms in result.face_landmarks:
            arr = np.array([[lm.x, lm.y, lm.z] for lm in lms], dtype=np.float64)
            arr[:, 0] = 1.0 - arr[:, 0]      # mirror to display space
            faces.append(FaceDetection(arr))
        return faces

    def close(self) -> None:
        self._landmarker.close()


class PlayerFace:
    """Smoothed face state bound to one player identity."""

    def __init__(self, cfg: FaceConfig) -> None:
        self.cfg = cfg
        self.f_center = OneEuroFilter(cfg.filter_min_cutoff, cfg.filter_beta)
        self.f_forehead = OneEuroFilter(cfg.filter_min_cutoff, cfg.filter_beta)
        self.f_chin = OneEuroFilter(cfg.filter_min_cutoff, cfg.filter_beta)
        self.f_scale = OneEuroFilter(cfg.filter_min_cutoff * 0.7, cfg.filter_beta * 2)
        self.f_roll = AngleFilter(cfg.angle_min_cutoff, cfg.angle_beta)
        self.f_yaw = AngleFilter(cfg.angle_min_cutoff, cfg.angle_beta)
        self.status = FaceStatus.LOST
        self.center: np.ndarray | None = None
        self.forehead: np.ndarray | None = None
        self.chin: np.ndarray | None = None
        self.scale = 0.0
        self.roll = 0.0
        self.yaw = 0.0
        self.pitch = 0.0
        self.alpha = 0.0             # effect opacity (fades while predicted)
        self.last_seen = -math.inf
        self._velocity = np.zeros(2)
        self.raw: FaceGeometry | None = None
        # Metrics (per update): smoothed-vs-measured anchor error in face scales,
        # and frame-to-frame displacement of the smoothed centre.
        self.alignment_error = math.nan
        self.step_displacement = math.nan
        self.detections = 0
        self.updates = 0

    def _filter(self, g: FaceGeometry, t: float) -> None:
        prev = None if self.center is None else self.center.copy()
        self.center = self.f_center(g.center, t)
        self.forehead = self.f_forehead(g.forehead, t)
        self.chin = self.f_chin(g.chin, t)
        self.scale = float(self.f_scale(np.array([g.scale]), t)[0])
        self.roll = float(self.f_roll(np.array([g.roll]), t)[0])
        self.yaw = float(self.f_yaw(np.array([g.yaw]), t)[0])
        self.pitch = g.pitch
        v = self.f_center.velocity
        self._velocity = v if v is not None else np.zeros(2)
        self.step_displacement = (float(np.linalg.norm(self.center - prev) / max(self.scale, 1e-4))
                                  if prev is not None else math.nan)
        self.alignment_error = float(np.linalg.norm(self.forehead - g.forehead) / max(g.scale, 1e-4))

    def _reset_filters(self) -> None:
        for f in (self.f_center, self.f_forehead, self.f_chin, self.f_scale, self.f_roll, self.f_yaw):
            f.reset()

    def update(self, geom: FaceGeometry | None, t: float,
               pose_geom: FaceGeometry | None = None) -> None:
        self.updates += 1
        self.raw = geom
        if geom is not None:
            if self.status in (FaceStatus.LOST,):
                self._reset_filters()
            self.detections += 1
            self._filter(geom, t)
            self.status = FaceStatus.TRACKED
            self.last_seen = t
            self.alpha = min(1.0, self.alpha + 0.34)
            return
        self.alignment_error = math.nan
        gap = t - self.last_seen
        if gap <= self.cfg.hold_s and self.center is not None:
            # Short dropout: constant-velocity extrapolation, fading out.
            dt = 1 / 30
            shift = self._velocity * dt * 0.5
            self.center = self.center + shift
            if self.forehead is not None:
                self.forehead = self.forehead + shift
            if self.chin is not None:
                self.chin = self.chin + shift
            self._velocity *= 0.7
            self.status = FaceStatus.PREDICTED
            self.alpha = max(0.35, 1.0 - gap / self.cfg.hold_s)
            self.step_displacement = math.nan
            return
        if pose_geom is not None:
            if self.status == FaceStatus.LOST:
                self._reset_filters()
            self._filter(pose_geom, t)
            self.alignment_error = math.nan    # no mesh ground reference
            self.status = FaceStatus.POSE_FALLBACK
            self.alpha = 0.75
            return
        self.status = FaceStatus.LOST
        self.alpha = 0.0
        self.center = None
        self.step_displacement = math.nan

    @property
    def visible(self) -> bool:
        return self.status != FaceStatus.LOST and self.center is not None
