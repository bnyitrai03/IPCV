"""Lean-based movement: leaning the upper body walks, standing position does not."""
import numpy as np

from config import Config
from vision.gesture_recognition import GestureRecognizer
from vision.pose_tracking import (L_ELBOW, L_HIP, L_SHOULDER, L_WRIST, NOSE, R_ELBOW, R_HIP,
                                  R_SHOULDER, R_WRIST, PlayerPose, PoseDetection)

ASPECT = 16 / 9
SW = 0.13
FPS = 30.0


def body(x=0.4, lean=0.0):
    """Upright person centred at image x. ``lean`` (sw, + = towards image right)
    bends the upper body sideways: the shoulder on that side drops (shoulder line
    tilts) and the head moves a little that way. The hips are NOT used (often out
    of the picture when standing close to a webcam)."""
    img = np.zeros((33, 4))
    img[:, 3] = 0.95
    sy = 0.4

    def put(i, px, py):
        img[i, 0], img[i, 1] = px, py * ASPECT

    tilt = 0.8 * lean * SW                       # right shoulder lower when leaning right
    put(L_HIP, x - 0.33 * SW, sy + 1.55 * SW)
    put(R_HIP, x + 0.33 * SW, sy + 1.55 * SW)
    put(L_SHOULDER, x - 0.5 * SW, sy - tilt / 2)
    put(R_SHOULDER, x + 0.5 * SW, sy + tilt / 2)
    put(NOSE, x + 0.2 * lean * SW, sy - 0.62 * SW)
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


def test_position_mode_still_available():
    cfg = Config()
    cfg.gestures.move_mode = "position"
    moves = run([body(x=0.4)] * 10 + [body(x=0.6)] * 30, cfg)
    assert moves[-1] > 0.8


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
