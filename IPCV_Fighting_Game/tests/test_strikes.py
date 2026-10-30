"""Attack recognition: hook (elbow ~90 deg + horizontal sweep), uppercut (up and
forward to head height), and motions that must NOT fire (straight punches,
straight-arm swings, slow moves, guard, two-arm raise, holding, pulling back)."""
import numpy as np

from config import Config
from vision.gesture_recognition import GestureRecognizer
from vision.pose_tracking import (L_ELBOW, L_HIP, L_SHOULDER, L_WRIST, NOSE, R_ELBOW, R_HIP,
                                  R_SHOULDER, R_WRIST, PlayerPose, PoseDetection)

ASPECT = 16 / 9
SW = 0.13            # shoulder width in normalised-x units
SW_M = 0.38          # shoulder width in metres (world coords)
FPS = 30.0
REST = {"left": (0.2, 0.55), "right": (-0.2, 0.55)}   # wrist offsets from own shoulder (sw)
GUARD = {"left": (0.35, -0.55), "right": (-0.35, -0.55)}


def make_pose(wrists, depth=None, elbows=None):
    """wrists: side -> (dx, dy) offset from that shoulder in sw (image x right, y down).
    depth: side -> wrist sw towards the camera. elbows: side -> (dx, dy, depth) offset of
    the elbow from the shoulder; default = halfway shoulder-wrist (a straight arm)."""
    depth, elbows = depth or {}, elbows or {}
    img = np.zeros((33, 4))
    img[:, 3] = 0.95
    lsx, sy = 0.4, 0.4

    def put(i, x, y):
        img[i, 0], img[i, 1] = x, y * ASPECT

    put(L_SHOULDER, lsx, sy)
    put(R_SHOULDER, lsx + SW, sy)
    put(NOSE, lsx + SW / 2, sy - 0.62 * SW)
    put(L_HIP, lsx + 0.17 * SW, sy + 1.55 * SW)
    put(R_HIP, lsx + 0.83 * SW, sy + 1.55 * SW)
    world = np.zeros((33, 3))
    world[R_SHOULDER, 0] = SW_M
    for side, w_i, e_i, s_i in (("left", L_WRIST, L_ELBOW, L_SHOULDER),
                                ("right", R_WRIST, R_ELBOW, R_SHOULDER)):
        dx, dy = wrists.get(side, REST[side])
        dz = depth.get(side, 0.0)
        ex, ey, ez = elbows.get(side, (dx / 2, dy / 2, dz / 2))
        sx = img[s_i, 0]
        put(w_i, sx + dx * SW, sy + dy * SW)
        put(e_i, sx + ex * SW, sy + ey * SW)
        world[w_i] = (world[s_i, 0] + dx * SW_M, dy * SW_M, -dz * SW_M)
        world[e_i] = (world[s_i, 0] + ex * SW_M, ey * SW_M, -ez * SW_M)
    return PoseDetection(img, world)


def run(frames):
    """frames: list of (wrists, depth, elbows). Returns gesture names (block_end dropped)."""
    cfg = Config()
    pose = PlayerPose(cfg.pose, ASPECT)
    rec = GestureRecognizer(0, cfg.gestures)
    out = []
    for k, (w, d, e) in enumerate(frames):
        t = k / FPS
        pose.update(make_pose(w, d, e), t)
        out += [ev.gesture for ev in rec.update(pose, t, t) if ev.gesture != "block_end"]
    return out


def still(n, w=None, d=None, e=None):
    return [(w or {}, d or {}, e or {})] * n


def move(side, a, b, n, other=None, depth=(0.0, 0.0), elbow=None):
    """Move one wrist from offset a to b over n frames. elbow(f) -> elbow offset."""
    frames = []
    for k in range(1, n + 1):
        f = k / n
        p = (a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f)
        w = {side: p}
        if other:
            w.update(other)
        e = {side: elbow(f)} if elbow else {}
        frames.append((w, {side: depth[0] + (depth[1] - depth[0]) * f}, e))
    return frames


def bent_elbow(side):
    """Elbow out to the side at shoulder height and forward: ~90 deg arm (hook)."""
    out = -1 if side == "left" else 1          # image direction away from the body
    return lambda f: (out * 0.2, 0.0, 0.7)


def hook_frames(side, start, end, n=4, other=None):
    return move(side, start, end, n, other=other, depth=(0.8, 0.8), elbow=bent_elbow(side))


# ----------------------------------------------------------------- hooks
def test_hook_with_bent_elbow():
    seq = still(10) + hook_frames("left", REST["left"], (1.2, -0.1)) \
        + still(10, {"left": (1.2, -0.1)}, {"left": 0.8}, {"left": bent_elbow("left")(1)})
    assert run(seq) == ["hook_left"]
    seq = still(10) + hook_frames("right", REST["right"], (-1.2, -0.1)) \
        + still(10, {"right": (-1.2, -0.1)}, {"right": 0.8}, {"right": bent_elbow("right")(1)})
    assert run(seq) == ["hook_right"]


def test_hook_from_guard_with_windup():
    frames = still(15, GUARD)
    frames += move("left", GUARD["left"], (-0.45, -0.5), 5, other={"right": GUARD["right"]},
                   elbow=bent_elbow("left"), depth=(0.6, 0.8))
    frames += hook_frames("left", (-0.45, -0.5), (0.9, -0.6), other={"right": GUARD["right"]})
    frames += still(10, {"left": (0.9, -0.6), "right": GUARD["right"]})
    out = [g for g in run(frames) if g != "block_start"]
    assert out == ["hook_left"], out


def test_straight_arm_swing_is_not_a_hook():
    # Same horizontal sweep, but the arm is straight (elbow on the shoulder-wrist line).
    seq = still(10, {"left": (-1.4, 0.0)}) + move("left", (-1.4, 0.0), (0.9, -0.2), 4) \
        + still(10, {"left": (0.9, -0.2)})
    assert run(seq) == []


# ------------------------------------------------------------- uppercuts
def test_uppercut_up_and_forward():
    end = (-0.35, -1.1)
    seq = still(10) + move("right", REST["right"], end, 4, depth=(0.0, 0.6)) \
        + still(15, {"right": end}, {"right": 0.6})
    assert run(seq) == ["uppercut_right"]


def test_uppercut_with_other_hand_in_guard():
    low = (-0.2, 0.4)
    frames = still(15, GUARD) + still(5, {"left": GUARD["left"], "right": low})
    frames += move("right", low, (-0.3, -1.1), 4, other={"left": GUARD["left"]})
    frames += still(15, {"left": GUARD["left"], "right": (-0.3, -1.1)})
    out = [g for g in run(frames) if g != "block_start"]
    assert out == ["uppercut_right"], out


def test_raising_one_hand_to_the_face_is_not_an_uppercut():
    # Hand comes up next to the face (guard height) but does not reach out.
    seq = still(10) + move("right", REST["right"], (-0.35, -0.55), 4) \
        + still(10, {"right": (-0.35, -0.55)})
    assert run(seq) == []


def test_raising_both_arms_is_special_not_uppercut():
    up = {"left": (0.1, -1.8), "right": (-0.1, -1.8)}
    frames = still(10)
    for k in range(1, 5):          # left arm slightly ahead of the right arm
        f, g = k / 4, max(0, (k - 1) / 4)
        frames.append(({"left": (0.2 - 0.1 * f, 0.55 - 2.35 * f),
                        "right": (-0.2 + 0.1 * g, 0.55 - 2.35 * g)}, {}, {}))
    frames += still(25, up)
    out = run(frames)
    assert "special" in out and not any(g.startswith("uppercut") for g in out), out


# ------------------------------------------------------ must NOT fire
def test_straight_punches_do_nothing():
    # Sideways, diagonal and at-the-camera straight punches are not game actions.
    for seq in (still(10) + move("left", REST["left"], (-1.6, 0.0), 4) + still(10, {"left": (-1.6, 0.0)}),
                still(10) + move("right", REST["right"], (1.4, -0.6), 4) + still(10, {"right": (1.4, -0.6)}),
                still(10) + move("right", REST["right"], (-0.05, 0.05), 4, depth=(0.0, 1.5))
                + still(10, {"right": (-0.05, 0.05)}, {"right": 1.5})):
        assert run(seq) == []


def test_slow_and_downward_motions_do_not_fire():
    slow = move("left", REST["left"], (1.2, -0.1), 60, depth=(0.8, 0.8), elbow=bent_elbow("left"))
    assert run(still(10) + slow) == []                                   # 2 s "hook"
    seq = still(10, GUARD) + move("left", GUARD["left"], (0.2, 0.9), 3, other={"right": GUARD["right"]})
    assert [g for g in run(seq) if g != "block_start"] == []


def test_raising_guard_and_holding_it_only_blocks():
    frames = still(10)
    for k in range(1, 5):
        f = k / 4
        frames.append(({"left": (0.2 + 0.15 * f, 0.55 - 1.1 * f),
                        "right": (-0.2 - 0.15 * f, 0.55 - 1.1 * f)}, {}, {}))
    assert run(frames + still(120, GUARD)) == ["block_start"]


def test_holding_does_not_repeat_and_pulling_back_rearms():
    end = (1.2, -0.1)
    held = still(40, {"left": end}, {"left": 0.8}, {"left": bent_elbow("left")(1)})
    seq = (still(10) + hook_frames("left", REST["left"], end) + held
           + move("left", end, REST["left"], 6) + still(15)
           + hook_frames("left", REST["left"], end) + still(5, {"left": end}))
    assert run(seq) == ["hook_left", "hook_left"]


def test_pulling_back_after_uppercut_does_not_fire():
    top = (-0.35, -1.1)
    seq = (still(10) + move("right", REST["right"], top, 4) + still(8, {"right": top})
           + move("right", top, REST["right"], 4) + still(15))
    assert run(seq) == ["uppercut_right"]


def test_compact_uppercut_to_chin_with_bent_arm():
    # From chest height to chin height, elbow bent, fist slightly forward.
    start, end = (-0.1, 0.35), (-0.45, -0.3)
    seq = still(10, {"right": start}) + move("right", start, end, 4, depth=(0.2, 0.6),
                                             elbow=lambda f: (0.1, 0.1, 0.3)) \
        + still(10, {"right": end}, {"right": 0.6}, {"right": (0.1, 0.1, 0.3)})
    assert run(seq) == ["uppercut_right"]


def test_uppercut_with_other_hand_in_high_guard():
    high_guard = (0.35, -0.85)                 # other fist at eye level
    low = (-0.1, 0.35)
    frames = still(15, {"left": high_guard, "right": low})
    frames += move("right", low, (-0.4, -0.4), 4, other={"left": high_guard}, depth=(0.2, 0.6))
    frames += still(15, {"left": high_guard, "right": (-0.4, -0.4)}, {"right": 0.6})
    out = [g for g in run(frames) if g != "block_start"]
    assert out == ["uppercut_right"], out


def test_hook_still_detected_when_fast_fist_is_blurry():
    """During a fast punch MediaPipe's wrist/elbow visibility drops (motion blur)."""
    cfg = Config()
    pose = PlayerPose(cfg.pose, ASPECT)
    rec = GestureRecognizer(0, cfg.gestures)
    frames = still(10) + hook_frames("left", REST["left"], (1.2, -0.1)) \
        + still(10, {"left": (1.2, -0.1)}, {"left": 0.8}, {"left": bent_elbow("left")(1)})
    out = []
    for k, (w, d, e) in enumerate(frames):
        det = make_pose(w, d, e)
        if 10 <= k < 14:                          # the swing frames: blurry arm
            det.image[[L_WRIST, L_ELBOW], 3] = 0.35
        pose.update(det, k / FPS)
        out += [ev.gesture for ev in rec.update(pose, k / FPS, k / FPS)]
    assert out == ["hook_left"]


def test_optional_hook_swung_away_from_the_body():
    # hook_direction="both" also accepts an OUTWARD swing (off by default: the
    # wind-up of a normal hook would then fire as a second hook).
    cfg = Config()
    cfg.gestures.hook_direction = "both"          # optional mode (default: inward only)
    pose = PlayerPose(cfg.pose, ASPECT)
    rec = GestureRecognizer(0, cfg.gestures)
    frames = still(15, GUARD)
    frames += hook_frames("left", GUARD["left"], (-1.0, -0.5), other={"right": GUARD["right"]})
    frames += still(10, {"left": (-1.0, -0.5), "right": GUARD["right"]})
    out = []
    for k, (w, d, e) in enumerate(frames):
        pose.update(make_pose(w, d, e), k / FPS)
        out += [ev.gesture for ev in rec.update(pose, k / FPS, k / FPS) if ev.gesture.startswith("hook")]
    assert out == ["hook_left"], out


def test_inward_only_mode_ignores_outward_swing():
    cfg = Config()
    cfg.gestures.hook_direction = "inward"
    pose = PlayerPose(cfg.pose, ASPECT)
    rec = GestureRecognizer(0, cfg.gestures)
    frames = still(15, GUARD) + hook_frames("left", GUARD["left"], (-1.0, -0.5),
                                            other={"right": GUARD["right"]}) \
        + still(10, {"left": (-1.0, -0.5), "right": GUARD["right"]})
    out = []
    for k, (w, d, e) in enumerate(frames):
        pose.update(make_pose(w, d, e), k / FPS)
        out += [ev.gesture for ev in rec.update(pose, k / FPS, k / FPS)
                if ev.gesture.startswith("hook")]
    assert out == []
