"""Task 3 — multi-player detection and persistent identity tracking.

Identity strategy
-----------------
1. **Registration** (once, during calibration): when the required number of
   people is stably visible, the person standing on the left becomes Player 1
   and the person on the right becomes Player 2. This is the *only* time
   left/right ordering is used.
2. **Per-frame association** (Hungarian-style optimal assignment, brute force
   over the tiny 2xN problem) with a cost combining
     * motion: distance to the constant-velocity *predicted* position,
       normalised by shoulder width,
     * appearance: Bhattacharyya distance between HSV colour histograms of the
       torso (clothing), learnt at registration and updated slowly,
     * size: log-ratio of shoulder widths (distance from camera).
   While players overlap (crossing), the appearance weight is boosted because
   position becomes ambiguous; appearance models are *frozen* during overlap
   so they cannot be contaminated by the other player.
3. **Occlusion / loss**: an unmatched track coasts on its prediction (LOST)
   for ``lost_timeout_s`` and can be re-acquired by motion+appearance. After
   that it becomes ABSENT and can only be re-acquired by appearance
   (re-identification), or by elimination when it is the only missing player.
4. **Face association**: each face detection is attached to the body whose
   pose-nose is closest (gated), so AR effects inherit the body identity and
   cannot jump between players.

Everything downstream (controls, health, effects, scores) is keyed by the
track id (0 = Player 1, 1 = Player 2), never by detector output order.
"""
from __future__ import annotations

import itertools
import logging
import math
from dataclasses import dataclass, field
from enum import Enum

import cv2
import numpy as np

from config import Config
from vision.face_tracking import (FaceDetection, FaceGeometry, PlayerFace,
                                  compute_face_geometry, geometry_from_pose_head)
from vision.gesture_recognition import GestureEvent, GestureRecognizer
from vision.pose_tracking import (L_EAR, L_EYE, L_HIP, L_SHOULDER, NOSE, R_EAR, R_EYE, R_HIP,
                                  R_SHOULDER, PlayerPose, PoseDetection)

log = logging.getLogger(__name__)

INF = 1e9


class TrackState(Enum):
    UNREGISTERED = "unregistered"
    ACTIVE = "active"
    LOST = "lost"          # temporarily missing, coasting on prediction
    ABSENT = "absent"      # gone for longer, waiting for re-entry


@dataclass
class DetectionInfo:
    det: PoseDetection
    center: np.ndarray            # isotropic normalised coords
    size: float                   # shoulder width, isotropic units
    hist: np.ndarray | None
    n_key_visible: int


@dataclass
class TrackerEvent:
    ts: float
    player: int
    kind: str
    details: dict = field(default_factory=dict)


class PlayerTrack:
    def __init__(self, pid: int, cfg: Config, aspect: float) -> None:
        self.id = pid
        self.cfg = cfg
        self.state = TrackState.UNREGISTERED
        self.pose = PlayerPose(cfg.pose, aspect)
        self.face = PlayerFace(cfg.face)
        self.gestures = GestureRecognizer(pid, cfg.gestures)
        self.center = np.zeros(2)
        self.velocity = np.zeros(2)
        self.size = 0.15
        self.hist: np.ndarray | None = None
        self.last_seen = -math.inf
        self.state_since = 0.0
        self.matched_this_frame = False
        self.bbox: tuple[float, float, float, float] | None = None   # normalised image coords

    @property
    def registered(self) -> bool:
        return self.state != TrackState.UNREGISTERED

    def predicted_center(self, t: float) -> np.ndarray:
        dt = min(max(t - self.last_seen, 0.0), 0.5)
        return self.center + self.velocity * dt


def torso_histogram(image_bgr: np.ndarray, det: PoseDetection, vis_thr: float = 0.4) -> np.ndarray | None:
    """HSV hue/saturation histogram of the torso region (clothing colour)."""
    h, w = image_bgr.shape[:2]
    lm = det.image
    pts = []
    for idx in (L_SHOULDER, R_SHOULDER, R_HIP, L_HIP):
        if lm[idx, 3] >= vis_thr:
            pts.append((lm[idx, 0] * w, lm[idx, 1] * h))
    if len(pts) < 2 or lm[L_SHOULDER, 3] < vis_thr or lm[R_SHOULDER, 3] < vis_thr:
        return None
    if len(pts) < 4:
        # Hips not visible: use a box below the shoulders.
        (x1, y1), (x2, y2) = (lm[L_SHOULDER, 0] * w, lm[L_SHOULDER, 1] * h), \
                             (lm[R_SHOULDER, 0] * w, lm[R_SHOULDER, 1] * h)
        sw = math.hypot(x2 - x1, y2 - y1)
        pts = [(x1, y1), (x2, y2), (x2, y2 + 1.2 * sw), (x1, y1 + 1.2 * sw)]
    poly = np.array(pts, dtype=np.float32)
    # Shrink the polygon towards its centre to avoid background at the edges.
    c = poly.mean(axis=0)
    poly = (c + (poly - c) * 0.75).astype(np.int32)
    # Work on the bounding rectangle only (much cheaper than the full frame).
    x, y, bw, bh = cv2.boundingRect(poly)
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + bw, w), min(y + bh, h)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    roi = image_bgr[y0:y1, x0:x1]
    mask = np.zeros(roi.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, poly - np.array([x0, y0], dtype=np.int32), 255)
    if cv2.countNonZero(mask) < 50:
        return None
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], mask, [16, 8], [0, 180, 0, 256])
    cv2.normalize(hist, hist, alpha=1.0, norm_type=cv2.NORM_L1)
    return hist


def hist_distance(a: np.ndarray | None, b: np.ndarray | None) -> float:
    """Bhattacharyya distance in [0, 1]; 0.5 (uninformative) if unavailable."""
    if a is None or b is None:
        return 0.5
    return float(cv2.compareHist(a.astype(np.float32), b.astype(np.float32),
                                 cv2.HISTCMP_BHATTACHARYYA))


class PlayerTracker:
    def __init__(self, cfg: Config, aspect: float, num_players: int = 2) -> None:
        self.cfg = cfg
        self.tcfg = cfg.tracking
        self.aspect = aspect
        self.tracks = [PlayerTrack(i, cfg, aspect) for i in range(num_players)]
        self._registration_count = 0
        self.events: list[TrackerEvent] = []
        self.overlap = False

    # --------------------------------------------------------------- helpers
    def _emit(self, ts: float, player: int, kind: str, **details) -> None:
        self.events.append(TrackerEvent(ts, player, kind, details))
        who = f"P{player + 1}" if player >= 0 else "both"
        log.info("[tracker] %s %s %s", who, kind, details or "")

    def _set_state(self, track: PlayerTrack, state: TrackState, t: float) -> None:
        track.state = state
        track.state_since = t

    def _describe(self, det: PoseDetection, image: np.ndarray | None) -> DetectionInfo | None:
        vis_thr = self.cfg.pose.landmark_visibility_threshold
        key = [NOSE, L_SHOULDER, R_SHOULDER, L_HIP, R_HIP]
        n_key = int(sum(det.image[i, 3] >= vis_thr for i in key))
        c = det.center(vis_thr=0.3)
        if c is None or n_key < 2:
            return None
        center = np.array([c[0], c[1] / self.aspect])
        size = det.shoulder_width(self.aspect)
        hist = torso_histogram(image, det) if image is not None else None
        return DetectionInfo(det, center, size, hist, n_key)

    def reset_registration(self) -> None:
        for t in self.tracks:
            t.state = TrackState.UNREGISTERED
            t.hist = None
            t.gestures.reset()
            t.gestures.neutral_x = None
        self._registration_count = 0

    def recalibrate(self) -> None:
        for t in self.tracks:
            if t.state == TrackState.ACTIVE:
                t.gestures.calibrate(t.pose)

    @property
    def all_registered(self) -> bool:
        return all(t.registered for t in self.tracks)

    # ----------------------------------------------------------- association
    def _pair_cost(self, track: PlayerTrack, info: DetectionInfo, t: float) -> float:
        tc = self.tcfg
        ref = max(track.size, info.size, 0.03)
        pos = float(np.linalg.norm(track.predicted_center(t) - info.center) / ref)
        app = hist_distance(track.hist, info.hist)
        size = abs(math.log(max(info.size, 1e-3) / max(track.size, 1e-3)))
        w_app = tc.w_appearance * (tc.overlap_appearance_boost if self.overlap else 1.0)
        if track.state == TrackState.ACTIVE:
            if pos > tc.position_gate:
                return INF
            return tc.w_position * pos + w_app * app + tc.w_size * size
        if track.state == TrackState.LOST:
            if pos > tc.position_gate * 2.0 and app > tc.reentry_appearance_threshold:
                return INF
            return 0.5 * tc.w_position * min(pos, 4.0) + w_app * app + tc.w_size * size
        if track.state == TrackState.ABSENT:
            if track.hist is None or info.hist is None or app > tc.reentry_appearance_threshold:
                return INF
            return w_app * app + 0.5
        return INF

    def _best_assignment(self, tracks: list[PlayerTrack], infos: list[DetectionInfo],
                         t: float) -> dict[int, int]:
        """Optimal assignment track->detection (brute force; n is tiny).

        Unassigned tracks pay a miss cost equal to the gate.
        """
        miss = self.tcfg.w_position * self.tcfg.position_gate + 1.0
        costs = [[self._pair_cost(tr, inf, t) for inf in infos] for tr in tracks]
        n_t, n_d = len(tracks), len(infos)
        options = list(range(n_d)) + [None]
        results: list[tuple[float, dict[int, int]]] = []
        for combo in itertools.product(options, repeat=n_t):
            used = [c for c in combo if c is not None]
            if len(used) != len(set(used)):
                continue
            total, mapping, ok = 0.0, {}, True
            for ti, di in enumerate(combo):
                if di is None:
                    total += miss
                else:
                    c = costs[ti][di]
                    if c >= INF:
                        ok = False
                        break
                    total += c
                    mapping[ti] = di
            if ok:
                results.append((total, mapping))
        if not results:
            return {}
        results.sort(key=lambda r: r[0])
        return results[0][1]

    # ------------------------------------------------------------------ update
    def update(self, detections: list[PoseDetection], faces: list[FaceDetection] | None,
               image_bgr: np.ndarray | None, capture_ts: float, now: float) -> list[GestureEvent]:
        """Process one vision frame. ``faces`` is None when the face model was skipped."""
        t = capture_ts
        infos = [i for i in (self._describe(d, image_bgr) for d in detections) if i is not None]
        for tr in self.tracks:
            tr.matched_this_frame = False

        # Crossing / overlap detection between registered, visible players.
        regs = [tr for tr in self.tracks if tr.state in (TrackState.ACTIVE, TrackState.LOST)]
        was_overlap = self.overlap
        self.overlap = False
        if len(regs) == 2:
            a, b = regs
            d = np.linalg.norm(a.predicted_center(t) - b.predicted_center(t))
            self.overlap = d / max((a.size + b.size) / 2, 0.03) < self.tcfg.overlap_distance
        if self.overlap != was_overlap:
            self._emit(t, -1, "overlap_start" if self.overlap else "overlap_end")

        # Associate registered tracks.
        candidates = [tr for tr in self.tracks if tr.registered]
        mapping = self._best_assignment(candidates, infos, t) if candidates else {}
        used = set()
        for ti, di in mapping.items():
            tr, info = candidates[ti], infos[di]
            used.add(di)
            self._apply_match(tr, info, t)

        # Unmatched registered tracks -> LOST -> ABSENT.
        for tr in candidates:
            if tr.matched_this_frame:
                continue
            tr.pose.update(None, t)
            tr.bbox = None
            if tr.state == TrackState.ACTIVE:
                self._set_state(tr, TrackState.LOST, t)
                tr.gestures.reset()
                self._emit(t, tr.id, "lost")
            elif tr.state == TrackState.LOST:
                tr.center = tr.predicted_center(t)
                tr.velocity *= self.tcfg.velocity_decay
                tr.last_seen = t  # prediction is now the reference
                if t - tr.state_since > self.tcfg.lost_timeout_s:
                    self._set_state(tr, TrackState.ABSENT, t)
                    self._emit(t, tr.id, "absent")

        leftovers = [i for i in range(len(infos)) if i not in used]

        # Re-entry by elimination: exactly one missing registered player and one
        # unexplained detection whose appearance is not clearly someone else.
        missing = [tr for tr in self.tracks if tr.state in (TrackState.LOST, TrackState.ABSENT)]
        if len(missing) == 1 and len(leftovers) == 1 and self.all_registered:
            tr, info = missing[0], infos[leftovers[0]]
            if hist_distance(tr.hist, info.hist) < self.tcfg.reentry_appearance_threshold * 1.3:
                self._apply_match(tr, info, t, by_elimination=True)
                leftovers = []

        # Registration of unregistered identities.
        self._register(infos, leftovers, t)

        # Faces -> bodies.
        if faces is not None:
            self._associate_faces(faces, t)
        else:
            for tr in self.tracks:
                if tr.registered and tr.face.status.value == "pose":
                    tr.face.update(None, t, self._pose_face(tr))

        # Gestures.
        events: list[GestureEvent] = []
        for tr in self.tracks:
            if tr.state == TrackState.ACTIVE:
                events.extend(tr.gestures.update(tr.pose, capture_ts, now))
            else:
                tr.gestures.move = 0.0
                tr.gestures.block_active = False
        return events

    def _apply_match(self, tr: PlayerTrack, info: DetectionInfo, t: float,
                     by_elimination: bool = False) -> None:
        prev_state = tr.state
        dt = t - tr.last_seen
        if prev_state == TrackState.ACTIVE and 0 < dt < 0.5:
            inst_v = (info.center - tr.center) / dt
            tr.velocity = 0.6 * tr.velocity + 0.4 * inst_v
        else:
            tr.velocity = np.zeros(2)
        tr.center = info.center
        tr.size = 0.8 * tr.size + 0.2 * info.size if prev_state == TrackState.ACTIVE else info.size
        tr.pose.update(info.det, t)
        tr.bbox = info.det.bbox()
        tr.last_seen = t
        tr.matched_this_frame = True
        # Appearance update only when unambiguous.
        if info.hist is not None and not self.overlap and info.n_key_visible >= 4:
            if tr.hist is None:
                tr.hist = info.hist.copy()
            else:
                r = self.tcfg.appearance_update_rate
                tr.hist = (1 - r) * tr.hist + r * info.hist
        if prev_state != TrackState.ACTIVE:
            gone = t - tr.state_since
            self._set_state(tr, TrackState.ACTIVE, t)
            if prev_state == TrackState.LOST:
                self._emit(t, tr.id, "recovered", gap_s=round(gone, 3))
            elif prev_state == TrackState.ABSENT:
                self._emit(t, tr.id, "reentry", absent_s=round(gone, 3),
                           by_elimination=by_elimination)

    def _register(self, infos: list[DetectionInfo], leftovers: list[int], t: float) -> None:
        unregistered = [tr for tr in self.tracks if not tr.registered]
        if not unregistered:
            return
        good = [i for i in leftovers if infos[i].n_key_visible >= 3]
        if not good:
            self._registration_count = 0
            return
        self._registration_count += 1
        anyone_registered = any(tr.registered for tr in self.tracks)
        if not anyone_registered and len(good) < len(unregistered):
            # Only one person in view: wait a bit longer in case a second player is
            # stepping in (so both get registered left->right), then register the
            # single player as P1. A second player who arrives later becomes P2.
            if self._registration_count < 2 * self.tcfg.registration_frames:
                return
        elif self._registration_count < self.tcfg.registration_frames:
            return
        # Left-to-right ordering is used ONLY here, at registration time.
        good.sort(key=lambda i: infos[i].center[0])
        if any(tr.registered for tr in self.tracks):
            # One player already known: the new person takes the free slot.
            pairs = [(unregistered[0], infos[good[0]])]
        elif len(good) >= len(unregistered):
            pairs = list(zip(unregistered, (infos[i] for i in good)))
        else:
            pairs = [(unregistered[0], infos[good[0]])]   # single player so far -> P1
        for tr, info in pairs:
            self._apply_match(tr, info, t)
            tr.hist = info.hist.copy() if info.hist is not None else None
            tr.gestures.calibrate(tr.pose)
            self._emit(t, tr.id, "registered", x=round(float(info.center[0]), 3))
        self._registration_count = 0

    def _pose_face(self, tr: PlayerTrack) -> FaceGeometry | None:
        p = tr.pose
        le, re, nose = p.point(L_EYE), p.point(R_EYE), p.point(NOSE)
        if le is None or re is None or nose is None:
            return None
        return geometry_from_pose_head(le, re, nose, p.point(L_EAR), p.point(R_EAR))

    def _associate_faces(self, faces: list[FaceDetection], t: float) -> None:
        geoms = [compute_face_geometry(f.landmarks, self.aspect) for f in faces]
        # Only bodies matched in THIS frame may claim a face: a LOST body's stale
        # nose position could otherwise grab the other player's face during a crossing.
        tracks = [tr for tr in self.tracks if tr.state == TrackState.ACTIVE and tr.matched_this_frame]
        anchors = []
        for tr in tracks:
            nose = tr.pose.point(NOSE)
            if nose is None and tr.pose.point(L_EYE) is not None and tr.pose.point(R_EYE) is not None:
                nose = (tr.pose.point(L_EYE) + tr.pose.point(R_EYE)) / 2
            anchors.append(nose)
        best: dict[int, int] = {}
        best_cost = INF
        options = list(range(len(geoms))) + [None]
        for combo in itertools.product(options, repeat=len(tracks)):
            used = [c for c in combo if c is not None]
            if len(used) != len(set(used)):
                continue
            total, mapping, ok = 0.0, {}, True
            for ti, gi in enumerate(combo):
                if gi is None:
                    total += self.cfg.face.association_gate + 0.01
                    continue
                if anchors[ti] is None:
                    ok = False
                    break
                d = float(np.linalg.norm(geoms[gi].center - anchors[ti]) / max(tracks[ti].size, 0.03))
                if d > self.cfg.face.association_gate:
                    ok = False
                    break
                total += d
                mapping[ti] = gi
            if ok and total < best_cost:
                best_cost, best = total, mapping
        for ti, tr in enumerate(tracks):
            gi = best.get(ti)
            tr.face.update(geoms[gi] if gi is not None else None, t, self._pose_face(tr))
        for tr in self.tracks:
            if tr not in tracks and tr.registered:
                tr.face.update(None, t, None)   # short hold/fade, then LOST
