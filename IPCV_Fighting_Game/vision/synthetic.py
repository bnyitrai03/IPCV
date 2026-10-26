"""Synthetic two-person scenario generator (test harness, NOT a CV result).

Produces MediaPipe-shaped pose/face detections plus a rendered image for a
scripted 32 s scenario: idle, hooks, uppercuts, forward jabs (must NOT fire), block, special,
side-stepping, the two players **crossing**, and one player leaving and
re-entering the frame. Detector output order is shuffled every frame, as a
real detector gives no ordering guarantees.

It lets the full tracking -> identity -> gesture -> face-AR -> game pipeline
be exercised automatically (unit tests, ``--vision synthetic`` demo mode)
without a webcam. Real-world performance must be measured with the camera.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from vision.face_tracking import CHEEK_A, CHEEK_B, CHIN, EYE_A, EYE_B, FOREHEAD, NOSE_TIP, FaceDetection
from vision.pose_tracking import (L_ANKLE, L_EAR, L_ELBOW, L_EYE, L_HIP, L_KNEE, L_SHOULDER,
                                  L_WRIST, NOSE, NUM_LANDMARKS, R_ANKLE, R_EAR, R_ELBOW, R_EYE,
                                  R_HIP, R_KNEE, R_SHOULDER, R_WRIST, SKELETON_EDGES, PoseDetection)

SCENARIO_LENGTH = 32.0
SW_METRES = 0.38


@dataclass
class PersonSpec:
    color_bgr: tuple[int, int, int]
    base_x: float


def _ease(p: float) -> float:
    p = max(0.0, min(1.0, p))
    return p * p * (3 - 2 * p)


def _punch_progress(t: float, start: float) -> float:
    dt = t - start
    if dt < 0 or dt > 0.5:
        return 0.0
    if dt < 0.12:
        return _ease(dt / 0.12)
    if dt < 0.27:
        return 1.0
    return 1.0 - _ease((dt - 0.27) / 0.2)


# Scripted gesture times (seconds within the loop), per person index.
# "jab_*" = straight punches at the camera. They are NOT a game action any more,
# so they act as negative examples: they must not produce any event.
SCRIPT = {
    0: {"jab_right": [4.0, 9.5], "jab_left": [5.0], "block": [(7.0, 8.2)],
        "special": [(11.0, 12.0)], "hook_right": [2.0], "uppercut_left": [3.0]},
    1: {"jab_left": [4.6, 10.0], "jab_right": [6.0], "block": [(8.6, 9.4)],
        "special": [(12.5, 13.5)], "uppercut_right": [2.4], "hook_left": [3.3]},
}
EXPECTED_EVENTS = {
    0: {"special": 1, "block_start": 1,
        "hook_right": 1, "uppercut_left": 1},
    1: {"special": 1, "block_start": 1,
        "uppercut_right": 1, "hook_left": 1},
}


class SyntheticScenario:
    def __init__(self, width: int = 640, height: int = 360, noise: float = 0.0015,
                 seed: int = 0) -> None:
        self.w, self.h = width, height
        self.aspect = width / height
        self.noise = noise
        self.rng = np.random.default_rng(seed)
        self.people = [PersonSpec((40, 40, 210), 0.30), PersonSpec((200, 120, 30), 0.70)]
        self.u = 0.13   # shoulder width, normalised-x units

    # ------------------------------------------------------------- scenario
    def person_x(self, idx: int, t: float) -> float:
        x = self.people[idx].base_x
        # Side-step 14-16.5 s.
        if 14.0 <= t < 16.5:
            x += (0.08 if idx == 0 else -0.06) * math.sin((t - 14.0) / 2.5 * math.pi)
        # Crossing 18-22 s, stay swapped, cross back 24-28 s.
        a, b = self.people[0].base_x, self.people[1].base_x
        other = b if idx == 0 else a
        if 18.0 <= t < 22.0:
            x = x + (other - x) * _ease((t - 18.0) / 4.0)
        elif 22.0 <= t < 24.0:
            x = other
        elif 24.0 <= t < 28.0:
            x = other + (self.people[idx].base_x - other) * _ease((t - 24.0) / 4.0)
        return x

    def visible(self, idx: int, t: float) -> bool:
        # Person B leaves the frame 29.0-30.6 s (disappearance and re-entry).
        return not (idx == 1 and 29.0 <= t < 30.6)

    # ----------------------------------------------------------- skeleton
    def _skeleton(self, idx: int, t: float) -> tuple[np.ndarray, np.ndarray]:
        """Returns (image landmarks (33,4) normalised, world (33,3) metres)."""
        u = self.u
        cx = self.person_x(idx, t)
        neck_y = 0.33  # isotropic y of the neck (normalised-x units * aspect handled below)
        iso = np.zeros((NUM_LANDMARKS, 2))
        z = np.zeros(NUM_LANDMARKS)
        # Torso and legs (body units -> isotropic offsets).
        def put(i, bx, by):
            iso[i] = (cx + bx * u, neck_y / self.aspect + by * u)
        breathe = 0.01 * math.sin(t * 2.0 + idx)
        put(L_SHOULDER, -0.5, breathe); put(R_SHOULDER, 0.5, breathe)
        put(L_HIP, -0.33, 1.55); put(R_HIP, 0.33, 1.55)
        put(L_KNEE, -0.35, 2.35); put(R_KNEE, 0.35, 2.35)
        put(L_ANKLE, -0.35, 3.1); put(R_ANKLE, 0.35, 3.1)
        yaw = 0.25 * math.sin(t * 0.9 + idx) if 25.0 > t > 16.5 else 0.0
        put(NOSE, 0.12 * yaw, -0.62)
        put(L_EYE, -0.085 + 0.08 * yaw, -0.70); put(R_EYE, 0.085 + 0.08 * yaw, -0.70)
        put(L_EAR, -0.2, -0.66); put(R_EAR, 0.2, -0.66)
        for i, (bx, by) in ((1, (-0.06, -0.70)), (3, (-0.11, -0.70)), (4, (0.06, -0.70)),
                            (6, (0.11, -0.70)), (9, (-0.05, -0.52)), (10, (0.05, -0.52))):
            put(i, bx + 0.08 * yaw, by)

        script = SCRIPT[idx]
        blocking = any(a <= t < b for a, b in script.get("block", []))
        special = any(a <= t < b for a, b in script.get("special", []))
        for side, s_i, e_i, w_i, sgn in (("left", L_SHOULDER, L_ELBOW, L_WRIST, -1),
                                         ("right", R_SHOULDER, R_ELBOW, R_WRIST, 1)):
            punch = max((_punch_progress(t, s) for s in script.get(f"jab_{side}", [])), default=0.0)
            hook = max((_punch_progress(t, s) for s in script.get(f"hook_{side}", [])), default=0.0)
            upper = max((_punch_progress(t, s) for s in script.get(f"uppercut_{side}", [])),
                        default=0.0)
            s = iso[s_i]
            if special:
                wrist = s + np.array([sgn * 0.1, -1.75]) * u
                elbow = s + np.array([sgn * 0.15, -0.85]) * u
            elif blocking:
                wrist = iso[NOSE] + np.array([sgn * 0.18, 0.05]) * u
                elbow = s + np.array([sgn * 0.05, 0.75]) * u
            else:
                rest_w = s + np.array([-sgn * 0.2, 0.55]) * u
                rest_e = s + np.array([sgn * 0.1, 0.75]) * u
                punch_w = s + np.array([-sgn * 0.05, 0.05]) * u   # towards camera: tiny 2D change
                punch_e = s + np.array([sgn * 0.02, 0.15]) * u
                wrist = rest_w + (punch_w - rest_w) * punch
                elbow = rest_e + (punch_e - rest_e) * punch
                z[w_i] = -1.55 * punch * SW_METRES
                z[e_i] = -0.75 * punch * SW_METRES
                if hook > 0:       # elbow bent ~90 deg, fist swings across at head height
                    hook_w = s + np.array([-sgn * 1.2, -0.1]) * u
                    hook_e = s + np.array([sgn * 0.2, 0.0]) * u
                    wrist = rest_w + (hook_w - rest_w) * hook
                    elbow = rest_e + (hook_e - rest_e) * hook
                    z[e_i] = -0.75 * hook * SW_METRES      # elbow out in front
                    z[w_i] = -0.85 * hook * SW_METRES
                if upper > 0:      # fist driven up and forward to head height
                    up_w = s + np.array([-sgn * 0.35, -1.1]) * u
                    up_e = s + np.array([sgn * 0.05, -0.45]) * u
                    z[w_i] = -0.6 * upper * SW_METRES
                    wrist = rest_w + (up_w - rest_w) * upper
                    elbow = rest_e + (up_e - rest_e) * upper
            iso[w_i], iso[e_i] = wrist, elbow

        img = np.zeros((NUM_LANDMARKS, 4))
        img[:, 0] = iso[:, 0]
        img[:, 1] = iso[:, 1] * self.aspect
        img[:, 2] = z
        img[:, 3] = 0.95
        img[:, :2] += self.rng.normal(0, self.noise, (NUM_LANDMARKS, 2))
        img[img[:, 1] > 0.98, 3] = 0.1            # below the frame -> low visibility
        # World coordinates: metres, origin at hip centre, y down.
        hip = (iso[L_HIP] + iso[R_HIP]) / 2
        world = np.zeros((NUM_LANDMARKS, 3))
        world[:, :2] = (iso - hip) / u * SW_METRES
        world[:, 2] = z
        world += self.rng.normal(0, 0.004, world.shape)
        return img, world

    def _face(self, idx: int, t: float, pose_img: np.ndarray) -> FaceDetection:
        u = self.u
        eyes = (pose_img[L_EYE, :2] + pose_img[R_EYE, :2]) / 2
        c = np.array([eyes[0], eyes[1] / self.aspect])
        yaw = 0.25 * math.sin(t * 0.9 + idx) if 25.0 > t > 16.5 else 0.0
        roll = 0.15 * math.sin(t * 0.7 + idx * 2) if 25.0 > t > 16.5 else 0.0
        R = np.array([[math.cos(roll), -math.sin(roll)], [math.sin(roll), math.cos(roll)]])
        pts = np.zeros((478, 2))
        ang = np.linspace(0, 2 * math.pi, 478, endpoint=False)
        pts[:, 0] = 0.17 * np.cos(ang)
        pts[:, 1] = 0.08 + 0.3 * np.sin(ang)
        ex = 0.085 * math.cos(yaw)
        def setp(i, x, y):
            pts[i] = (x, y)
        setp(EYE_A[0], -ex - 0.045, 0); setp(EYE_A[1], -ex + 0.045, 0)
        setp(EYE_B[0], ex - 0.045, 0); setp(EYE_B[1], ex + 0.045, 0)
        setp(NOSE_TIP, 0.17 * math.sin(yaw), 0.12)
        setp(FOREHEAD, 0.05 * math.sin(yaw), -0.24)
        setp(CHIN, 0.03 * math.sin(yaw), 0.42)
        setp(CHEEK_A, -0.17 * math.cos(yaw) + 0.03 * math.sin(yaw), 0.08)
        setp(CHEEK_B, 0.17 * math.cos(yaw) + 0.03 * math.sin(yaw), 0.08)
        world = (pts @ R.T) * u + c
        out = np.zeros((478, 3))
        out[:, 0] = world[:, 0]
        out[:, 1] = world[:, 1] * self.aspect
        out[:, :2] += self.rng.normal(0, self.noise * 0.5, (478, 2))
        return FaceDetection(out)

    # ------------------------------------------------------------ frame
    def frame(self, t: float) -> tuple[np.ndarray, list[PoseDetection], list[FaceDetection]]:
        t = t % SCENARIO_LENGTH
        image = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        grad = np.linspace(70, 110, self.h, dtype=np.uint8)[:, None]
        image[:] = np.repeat(grad, self.w, axis=1)[:, :, None]
        poses, faces = [], []
        for idx, spec in enumerate(self.people):
            if not self.visible(idx, t):
                continue
            img, world = self._skeleton(idx, t)
            self._draw_person(image, img, spec.color_bgr)
            poses.append(PoseDetection(img, world))
            faces.append(self._face(idx, t, img))
        order = self.rng.permutation(len(poses))
        poses = [poses[i] for i in order]
        faces = [faces[i] for i in self.rng.permutation(len(faces))]
        return image, poses, faces

    def _draw_person(self, image: np.ndarray, lm: np.ndarray, color: tuple[int, int, int]) -> None:
        w, h = self.w, self.h
        P = lambda i: (int(lm[i, 0] * w), int(lm[i, 1] * h))
        torso = np.array([P(L_SHOULDER), P(R_SHOULDER), P(R_HIP), P(L_HIP)], np.int32)
        cv2.fillConvexPoly(image, torso, color)
        for a, b in SKELETON_EDGES:
            if a in (L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_HIP, R_HIP, L_KNEE, R_KNEE):
                cv2.line(image, P(a), P(b), tuple(int(c * 0.8) for c in color), 9)
        r = int(0.17 * self.u * w * 1.6)
        cv2.circle(image, P(NOSE), r, (150, 180, 220), -1)
