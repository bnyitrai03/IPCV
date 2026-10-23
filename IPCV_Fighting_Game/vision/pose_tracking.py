"""Task 2 — multi-person body pose estimation and per-player motion state.

Backend: MediaPipe Tasks ``PoseLandmarker`` with ``num_poses=2``. Unlike the
legacy ``mp.solutions.pose`` (strictly single-person, and removed from
recent MediaPipe releases), the Tasks landmarker runs a person detector and
then a landmark model per detected person, so it genuinely returns up to
``num_poses`` skeletons per frame. Use ``scripts/verify_multipose.py`` to
confirm on your own machine how many people it detects.

Each skeleton has 33 landmarks with normalised image coordinates (x, y),
a relative depth z, a visibility score, and metric "world" coordinates
(metres, origin between the hips). World coordinates are a monocular
*estimate*: useful for arm-extension ratios, not for true 3D measurement.

Downstream code works on :class:`PlayerPose`, which smooths the landmarks
(One Euro), handles missing landmarks and estimates velocities.
"""
from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from config import PoseConfig
from vision.smoothing import LandmarkSmoother

log = logging.getLogger(__name__)

# MediaPipe pose landmark indices (anatomical left/right of the person).
NOSE = 0
L_EYE, R_EYE = 2, 5
L_EAR, R_EAR = 7, 8
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28
NUM_LANDMARKS = 33

# Joints the game depends on (task 2: shoulders, elbows, wrists, hips, head).
KEY_JOINTS = {
    "head": NOSE, "l_shoulder": L_SHOULDER, "r_shoulder": R_SHOULDER,
    "l_elbow": L_ELBOW, "r_elbow": R_ELBOW, "l_wrist": L_WRIST, "r_wrist": R_WRIST,
    "l_hip": L_HIP, "r_hip": R_HIP,
}

SKELETON_EDGES = [
    (L_SHOULDER, R_SHOULDER), (L_SHOULDER, L_ELBOW), (L_ELBOW, L_WRIST),
    (R_SHOULDER, R_ELBOW), (R_ELBOW, R_WRIST), (L_SHOULDER, L_HIP),
    (R_SHOULDER, R_HIP), (L_HIP, R_HIP), (L_HIP, L_KNEE), (L_KNEE, L_ANKLE),
    (R_HIP, R_KNEE), (R_KNEE, R_ANKLE), (NOSE, L_EYE), (NOSE, R_EYE),
    (L_EYE, L_EAR), (R_EYE, R_EAR),
]


@dataclass
class PoseDetection:
    """One raw skeleton from the detector, already mirrored to display space."""

    image: np.ndarray            # (33, 4): x, y (normalised), z, visibility
    world: np.ndarray | None     # (33, 3) metres, or None

    @property
    def visibility(self) -> np.ndarray:
        return self.image[:, 3]

    def center(self, vis_thr: float = 0.3) -> np.ndarray | None:
        """Torso centre (mean of visible shoulders/hips) in normalised coords."""
        idx = [L_SHOULDER, R_SHOULDER, L_HIP, R_HIP]
        pts = [self.image[i, :2] for i in idx if self.image[i, 3] >= vis_thr]
        if not pts:
            pts = [self.image[i, :2] for i in range(NUM_LANDMARKS) if self.image[i, 3] >= vis_thr]
        return np.mean(pts, axis=0) if pts else None

    def shoulder_width(self, aspect: float) -> float:
        """Shoulder distance in normalised-x units (aspect = width / height)."""
        ls, rs = self.image[L_SHOULDER], self.image[R_SHOULDER]
        d = np.hypot((ls[0] - rs[0]), (ls[1] - rs[1]) / aspect)
        return float(max(d, 0.02))

    def bbox(self, vis_thr: float = 0.3) -> tuple[float, float, float, float] | None:
        vis = self.image[:, 3] >= vis_thr
        if not vis.any():
            return None
        xs, ys = self.image[vis, 0], self.image[vis, 1]
        return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())


def mirror_pose(image_lms: np.ndarray, world: np.ndarray | None) -> tuple[np.ndarray, np.ndarray | None]:
    """Mirror x so detections match the mirrored display image.

    Inference runs on the *unmirrored* frame, so MediaPipe's left/right labels
    remain the person's true anatomical sides; only the x coordinates flip.
    """
    img = image_lms.copy()
    img[:, 0] = 1.0 - img[:, 0]
    w = None
    if world is not None:
        w = world.copy()
        w[:, 0] = -w[:, 0]
    return img, w


class MediaPipePoseBackend:
    """Thin wrapper around the MediaPipe Tasks PoseLandmarker (VIDEO mode)."""

    def __init__(self, cfg: PoseConfig, model_path: Path) -> None:
        if not model_path.is_file():
            raise FileNotFoundError(
                f"Pose model not found at {model_path}. Run: python scripts/download_models.py")
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python import vision as mpv

        self._mp = mp
        options = mpv.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=mpv.RunningMode.VIDEO,
            num_poses=cfg.num_poses,
            min_pose_detection_confidence=cfg.min_detection_confidence,
            min_pose_presence_confidence=cfg.min_presence_confidence,
            min_tracking_confidence=cfg.min_tracking_confidence,
            output_segmentation_masks=False,
        )
        self._landmarker = mpv.PoseLandmarker.create_from_options(options)
        self._last_ts_ms = -1
        log.info("PoseLandmarker ready (num_poses=%d, model=%s)", cfg.num_poses, model_path.name)

    def detect(self, rgb: np.ndarray, ts_ms: int) -> list[PoseDetection]:
        ts_ms = max(ts_ms, self._last_ts_ms + 1)   # VIDEO mode needs strictly increasing ts
        self._last_ts_ms = ts_ms
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(image, ts_ms)
        out: list[PoseDetection] = []
        for i, lms in enumerate(result.pose_landmarks):
            arr = np.array([[lm.x, lm.y, lm.z,
                             lm.visibility if lm.visibility is not None else 1.0] for lm in lms],
                           dtype=np.float64)
            world = None
            if result.pose_world_landmarks and i < len(result.pose_world_landmarks):
                world = np.array([[lm.x, lm.y, lm.z] for lm in result.pose_world_landmarks[i]],
                                 dtype=np.float64)
            img, w = mirror_pose(arr, world)
            out.append(PoseDetection(img, w))
        return out

    def close(self) -> None:
        self._landmarker.close()


class PlayerPose:
    """Smoothed pose + motion signals for one tracked player.

    Coordinates are kept in a *pixel-isotropic* normalised space: x in [0,1]
    of the frame width and y scaled by height/width, so distances along x and
    y are comparable (needed for angles and arm lengths).
    """

    def __init__(self, cfg: PoseConfig, aspect: float) -> None:
        self.cfg = cfg
        self.aspect = aspect  # width / height
        self.img = LandmarkSmoother(NUM_LANDMARKS, 2, cfg.landmark_visibility_threshold,
                                    cfg.landmark_hold_s, cfg.filter_min_cutoff, cfg.filter_beta,
                                    cfg.filter_d_cutoff)
        # World landmarks (metres) are noisier in z -> slightly stronger smoothing.
        self.world = LandmarkSmoother(NUM_LANDMARKS, 3, cfg.landmark_visibility_threshold,
                                      cfg.landmark_hold_s, cfg.filter_min_cutoff * 0.8,
                                      cfg.filter_beta, cfg.filter_d_cutoff)
        self.has_world = False
        self._sw_hist: deque[float] = deque(maxlen=15)
        self.shoulder_width = 0.15
        self.last_update_ts = 0.0
        self.raw_visibility = np.zeros(NUM_LANDMARKS)
        self.raw_points = np.full((NUM_LANDMARKS, 2), np.nan)

    def update(self, det: PoseDetection | None, t: float) -> None:
        self.last_update_ts = t
        if det is None:
            self.img.mark_all_missing(t)
            self.world.mark_all_missing(t)
            self.raw_visibility[:] = 0.0
            return
        pts = det.image[:, :2].copy()
        pts[:, 1] /= self.aspect
        vis = det.image[:, 3]
        self.raw_points = pts
        self.raw_visibility = vis.copy()
        self.img.update(pts, vis, t)
        if det.world is not None:
            self.has_world = True
            self.world.update(det.world, vis, t)
        if vis[L_SHOULDER] >= self.cfg.landmark_visibility_threshold and \
                vis[R_SHOULDER] >= self.cfg.landmark_visibility_threshold:
            self._sw_hist.append(float(np.linalg.norm(pts[L_SHOULDER] - pts[R_SHOULDER])))
            # Median over a short window: robust to the shoulder line foreshortening
            # briefly while the player twists to punch.
            self.shoulder_width = max(0.03, float(np.median(self._sw_hist)))

    # ---- accessors (isotropic normalised coords) ----------------------------
    def point(self, idx: int) -> np.ndarray | None:
        return self.img.value[idx].copy() if self.img.valid[idx] else None

    def velocity(self, idx: int) -> np.ndarray:
        return self.img.velocity[idx].copy()

    def world_point(self, idx: int) -> np.ndarray | None:
        if not self.has_world or not self.world.valid[idx]:
            return None
        return self.world.value[idx].copy()

    def to_image(self, p: np.ndarray) -> tuple[float, float]:
        """Isotropic coords -> normalised image coords (x, y in [0, 1])."""
        return float(p[0]), float(p[1] * self.aspect)

    def confidence(self) -> float:
        """Mean raw visibility of the key joints (0..1)."""
        return float(np.mean([self.raw_visibility[i] for i in KEY_JOINTS.values()]))

    def missing_key_joints(self) -> list[str]:
        return [name for name, i in KEY_JOINTS.items() if not self.img.valid[i]]

    def hip_center(self) -> np.ndarray | None:
        l, r = self.point(L_HIP), self.point(R_HIP)
        if l is not None and r is not None:
            return (l + r) / 2
        ls, rs = self.point(L_SHOULDER), self.point(R_SHOULDER)
        if ls is not None and rs is not None:
            # Hips out of frame (players close to the camera): fall back to shoulders.
            return (ls + rs) / 2
        return None
