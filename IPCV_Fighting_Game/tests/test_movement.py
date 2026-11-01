"""Lean-based movement: leaning the upper body walks, standing position does not."""
import numpy as np

from config import Config
from vision.gesture_recognition import GestureRecognizer
from vision.pose_tracking import (L_ELBOW, L_HIP, L_SHOULDER, L_WRIST, NOSE, R_ELBOW, R_HIP,
                                  R_SHOULDER, R_WRIST, PlayerPose, PoseDetection)

ASPECT = 16 / 9
SW = 0.13
SW_M = 0.38
FPS = 30.0


def body(x=0.4, lean=0.0):
    """Upright person centred at image x. ``lean`` (sw, + = towards image right)
    bends the upper body sideways around the hips: the shoulders move that way
    off the hip line (and tilt a little), the head goes with them. The hips stay
    put. Raising a guard arm only lifts one shoulder; it must not count as a lean."""
    img = np.zeros((33, 4))
    img[:, 3] = 0.95
    sy = 0.4

    def put(i, px, py):
        img[i, 0], img[i, 1] = px, py * ASPECT

    tilt = 0.8 * lean * SW                       # right shoulder lower when leaning right
    shift = lean * SW                            # the upper body moves off the hips
    put(L_HIP, x - 0.33 * SW, sy + 1.55 * SW)
    put(R_HIP, x + 0.33 * SW, sy + 1.55 * SW)
    put(L_SHOULDER, x - 0.5 * SW + shift, sy - tilt / 2)
    put(R_SHOULDER, x + 0.5 * SW + shift, sy + tilt / 2)
    put(NOSE, x + shift, sy - 0.62 * SW)
    for s_i, e_i, w_i, side in ((L_SHOULDER, L_ELBOW, L_WRIST, -1), (R_SHOULDER, R_ELBOW, R_WRIST, 1)):
        sx, syy = img[s_i, 0], img[s_i, 1] / ASPECT
        put(e_i, sx + side * 0.1 * SW, syy + 0.75 * SW)
        put(w_i, sx - side * 0.2 * SW, syy + 0.55 * SW)
    return PoseDetection(img, None)


def run(frames, cfg=None):
    cfg = cfg or Config()
    pose = PlayerPose(cfg.pose, ASPECT)
    rec = GestureRecognizer(0, cfg.gestures)
    moves = []
    for k, det in enumerate(frames):
        t = k / FPS
        pose.update(det, t)
        if k == 5:
            assert rec.calibrate(pose)
        rec.update(pose, t, t)
        moves.append(rec.move)
    return moves


def test_leaning_walks_in_that_direction():
    moves = run([body()] * 10 + [body(lean=-0.9)] * 40)
    assert moves[-1] < -0.8                      # lean left -> walk left (full speed)
    moves = run([body()] * 10 + [body(lean=0.9)] * 40)
    assert moves[-1] > 0.8


def test_hips_out_of_view_still_works():
    frames = []
    for lean in [0.0] * 10 + [0.9] * 40:
        det = body(lean=lean)
        det.image[[L_HIP, R_HIP], 3] = 0.1          # hips not visible
        frames.append(det)
    assert run(frames)[-1] > 0.8


def test_standing_anywhere_upright_does_not_walk():
    # Registered in the middle, then standing far to one side but upright:
    # the old position-based mode would walk forever here.
    moves = run([body(x=0.5)] * 10 + [body(x=0.2)] * 40 + [body(x=0.8)] * 40)
    assert max(abs(m) for m in moves) == 0.0


def test_small_sway_is_ignored_and_returning_upright_stops():
    moves = run([body()] * 10 + [body(lean=0.2)] * 20)
    assert moves[-1] == 0.0
    moves = run([body()] * 10 + [body(lean=0.9)] * 20 + [body()] * 30)
    assert moves[-1] == 0.0


def test_crooked_neutral_posture_is_calibrated_away():
    # Someone standing slightly tilted at registration does not drift.
    moves = run([body(lean=0.15)] * 40)
    assert moves[-1] == 0.0


def test_moderate_sway_is_ignored_but_a_clear_lean_walks():
    moves = run([body()] * 10 + [body(lean=0.15)] * 30)
    assert max(abs(m) for m in moves) == 0.0
    moves = run([body()] * 10 + [body(lean=0.4)] * 40)
    assert moves[-1] > 0.3


def test_one_jittery_frame_does_not_start_walking():
    frames = [body()] * 10
    for _ in range(5):
        frames += [body(lean=0.9)] + [body()] * 6
    moves = run(frames)
    assert max(abs(m) for m in moves) == 0.0


def test_guard_shoulder_and_turned_head_do_not_walk():
    # A raised guard arm lifts one shoulder and a turned head moves the nose.
    # Neither moves the hips, so the torso is still upright: no walking.
    frames = []
    for _ in range(40):
        det = body()
        det.image[L_SHOULDER, 1] -= 0.3 * SW * ASPECT
        det.image[NOSE, 0] += 0.3 * SW
        frames.append(det)
    assert max(abs(m) for m in run(frames)) == 0.0


def _world_shoulders(shift_sw=0.0):
    world = np.zeros((33, 3))
    world[L_SHOULDER] = ((-0.5 + shift_sw) * SW_M, -1.5 * SW_M, 0.0)
    world[R_SHOULDER] = ((0.5 + shift_sw) * SW_M, -1.5 * SW_M, 0.0)
    return world


def test_hips_out_of_view_uses_world_lean_not_shoulder_tilt():
    def frame(shift_sw):
        det = body()
        det.image[[L_HIP, R_HIP], 3] = 0.1                 # hips out of view
        det.image[L_SHOULDER, 1] -= 0.4 * SW * ASPECT      # guard tilts the 2D line
        det.world = _world_shoulders(shift_sw)
        return det
    assert max(abs(m) for m in run([frame(0.0)] * 40)) == 0.0   # upright in 3D
    assert run([frame(0.6)] * 40)[-1] > 0.8                     # leaning in 3D
