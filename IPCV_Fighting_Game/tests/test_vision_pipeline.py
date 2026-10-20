"""Tests for smoothing, gesture recognition and identity tracking.

These use the synthetic scenario (scripted landmarks), so they verify the
*logic* (state machines, association, re-identification), not the accuracy
of MediaPipe on real video. Live accuracy must be measured with the
evaluation tools and a real webcam.
"""
from collections import Counter

import numpy as np

from config import Config
from vision.face_tracking import FaceStatus, compute_face_geometry
from vision.gesture_recognition import GestureRecognizer
from vision.player_tracking import PlayerTracker, TrackState
from vision.pose_tracking import L_SHOULDER, L_WRIST, NOSE, R_SHOULDER, R_WRIST, PlayerPose, PoseDetection
from vision.smoothing import AngleFilter, LandmarkSmoother, OneEuroFilter
from vision.synthetic import EXPECTED_EVENTS, SCENARIO_LENGTH, SyntheticScenario

FPS = 30.0


def run_scenario(duration: float = SCENARIO_LENGTH, seed: int = 0):
    cfg = Config()
    scen = SyntheticScenario(seed=seed)
    tracker = PlayerTracker(cfg, scen.aspect)
    log = []
    events = []
    n = int(duration * FPS)
    for k in range(n):
        t = k / FPS
        image, poses, faces = scen.frame(t)
        evs = tracker.update(poses, faces, image, t, t)
        events.extend(evs)
        log.append((t, [tr.center.copy() for tr in tracker.tracks],
                    [tr.state for tr in tracker.tracks]))
    return cfg, scen, tracker, log, events


def test_one_euro_reduces_jitter_but_follows_steps():
    rng = np.random.default_rng(1)
    f = OneEuroFilter(min_cutoff=1.0, beta=5.0)
    noisy = 0.5 + rng.normal(0, 0.01, 300)
    out = np.array([f(np.array([v]), i / 30)[0] for i, v in enumerate(noisy)])
    assert out[30:].std() < noisy[30:].std() * 0.5
    f.reset()
    step = [f(np.array([0.0 if i < 30 else 1.0]), i / 30)[0] for i in range(60)]
    assert step[-1] > 0.95


def test_angle_filter_wraps():
    f = AngleFilter(1.0, 0.0)
    f(np.array([3.1]), 0.0)
    out = f(np.array([-3.1]), 1 / 30)[0]
    assert abs(abs(out) - 3.1) < 0.2 or out > 3.0   # stays near +-pi, never jumps to 0


def test_landmark_smoother_holds_then_invalidates():
    sm = LandmarkSmoother(2, 2, vis_threshold=0.5, hold_s=0.2, min_cutoff=1.0, beta=0.0)
    sm.update(np.array([[0.1, 0.1], [0.2, 0.2]]), np.array([1.0, 1.0]), 0.0)
    sm.update(np.zeros((2, 2)), np.array([1.0, 0.0]), 0.1)
    assert sm.valid[1] and np.allclose(sm.value[1], [0.2, 0.2])   # held
    sm.update(np.zeros((2, 2)), np.array([1.0, 0.0]), 0.5)
    assert not sm.valid[1] and np.isnan(sm.value[1]).all()         # expired
    sm.update(np.array([[0, 0], [0.5, 0.5]]), np.array([1.0, 1.0]), 0.6)
    assert sm.valid[1] and sm.recoveries == 1


def _pose_with(wrist_offsets, t_world_z=0.0):
    """Static pose; wrist offsets in shoulder widths relative to each shoulder."""
    img = np.full((33, 4), 0.0)
    img[:, 3] = 0.95
    sw = 0.13
    aspect = 16 / 9
    def put(i, x, y):
        img[i, 0], img[i, 1] = x, y * aspect
    put(L_SHOULDER, 0.4, 0.4); put(R_SHOULDER, 0.4 + sw, 0.4)
    put(NOSE, 0.4 + sw / 2, 0.4 - 0.6 * sw)
    for idx, s_idx in ((L_WRIST, L_SHOULDER), (R_WRIST, R_SHOULDER)):
        dx, dy = wrist_offsets[idx]
        put(idx, img[s_idx, 0] + dx * sw, img[s_idx, 1] / aspect + dy * sw)
    for i in (13, 14):   # elbows halfway
        img[i, :2] = (img[i + 2, :2] + img[i - 2, :2]) / 2
    for i in (23, 24):
        put(i, img[i - 12, 0], img[i - 12, 1] / aspect + 1.5 * sw)
    return PoseDetection(img, None)


def test_punch_fires_once_while_arm_stays_extended():
    cfg = Config()
    pose = PlayerPose(cfg.pose, 16 / 9)
    rec = GestureRecognizer(0, cfg.gestures)
    rest = {L_WRIST: (0.2, 0.55), R_WRIST: (-0.2, 0.55)}
    ext = {L_WRIST: (-1.6, 0.0), R_WRIST: (-0.2, 0.55)}   # left arm out sideways
    events = []
    t = 0.0
    for k in range(10):
        pose.update(_pose_with(rest), t); events += rec.update(pose, t, t); t += 1 / 30
    for k in range(60):   # extend quickly, then HOLD for 2 s
        frac = min(1.0, (k + 1) / 4)
        cur = {L_WRIST: tuple(np.add(rest[L_WRIST], frac * (np.subtract(ext[L_WRIST], rest[L_WRIST])))),
               R_WRIST: rest[R_WRIST]}
        pose.update(_pose_with(cur), t); events += rec.update(pose, t, t); t += 1 / 30
    punches = [e for e in events if e.gesture.startswith("punch")]
    assert [e.gesture for e in punches] == ["punch_left"], punches
    # Retract and punch again -> second event.
    for k in range(15):
        pose.update(_pose_with(rest), t); events += rec.update(pose, t, t); t += 1 / 30
    for k in range(10):
        frac = min(1.0, (k + 1) / 4)
        cur = {L_WRIST: tuple(np.add(rest[L_WRIST], frac * (np.subtract(ext[L_WRIST], rest[L_WRIST])))),
               R_WRIST: rest[R_WRIST]}
        pose.update(_pose_with(cur), t); events += rec.update(pose, t, t); t += 1 / 30
    assert len([e for e in events if e.gesture == "punch_left"]) == 2


def test_slow_reach_does_not_fire():
    cfg = Config()
    pose = PlayerPose(cfg.pose, 16 / 9)
    rec = GestureRecognizer(0, cfg.gestures)
    rest = np.array([0.2, 0.55]); ext = np.array([-1.6, 0.0])
    events, t = [], 0.0
    for k in range(90):  # 3 seconds to extend
        cur = {L_WRIST: tuple(rest + (ext - rest) * min(1, k / 90)), R_WRIST: (-0.2, 0.55)}
        pose.update(_pose_with(cur), t); events += rec.update(pose, t, t); t += 1 / 30
    assert not [e for e in events if e.gesture.startswith("punch")]


def test_scenario_registers_and_preserves_identity_through_crossing():
    cfg, scen, tracker, log, _ = run_scenario()
    errors = 0
    checked = 0
    for t, centers, states in log:
        for pid in (0, 1):
            if states[pid] != TrackState.ACTIVE:
                continue
            truth = scen.person_x(pid, t % SCENARIO_LENGTH)
            other = scen.person_x(1 - pid, t % SCENARIO_LENGTH)
            if abs(truth - other) < 0.05:
                continue   # fully overlapping: identity not observable from position
            checked += 1
            if abs(centers[pid][0] - truth) > abs(centers[pid][0] - other):
                errors += 1
    assert checked > 1000
    assert errors == 0, f"{errors} identity errors out of {checked}"


def test_scenario_reentry_and_events():
    cfg, scen, tracker, log, events = run_scenario()
    kinds = [(e.player, e.kind) for e in tracker.events]
    assert (0, "registered") in kinds and (1, "registered") in kinds
    assert (1, "lost") in kinds
    assert any(k == "reentry" or k == "recovered" for p, k in kinds if p == 1)
    counts = Counter((e.player_id, e.gesture) for e in events)
    for pid, expected in EXPECTED_EVENTS.items():
        for gesture, n in expected.items():
            assert counts[(pid, gesture)] == n, (pid, gesture, counts)
    # No spurious punches anywhere else in the scenario (incl. crossing, side-steps).
    total_punches = sum(v for (p, g), v in counts.items() if g.startswith("punch"))
    assert total_punches == sum(v for e in EXPECTED_EVENTS.values()
                                for g, v in e.items() if g.startswith("punch"))


def test_faces_follow_body_identity():
    cfg, scen, tracker, log, _ = run_scenario(duration=24.0)
    for tr in tracker.tracks:
        assert tr.face.status == FaceStatus.TRACKED
        nose = tr.pose.point(NOSE)
        assert np.linalg.norm(tr.face.center - nose) / tr.size < 0.5


def test_face_geometry_roll_sign():
    scen = SyntheticScenario(noise=0.0)
    _, _, faces = scen.frame(18.0)
    g = compute_face_geometry(faces[0].landmarks, scen.aspect)
    assert g.scale > 0 and abs(g.roll) < 0.5


def test_face_effects_never_switch_players_during_crossing():
    """Every face assigned to P1/P2 must belong to that person (synthetic ground truth)."""
    cfg = Config()
    scen = SyntheticScenario(seed=3)
    tracker = PlayerTracker(cfg, scen.aspect)
    wrong = checked = 0
    for k in range(int(SCENARIO_LENGTH * FPS)):
        t = k / FPS
        image, poses, faces = scen.frame(t)
        tracker.update(poses, faces, image, t, t)
        for tr in tracker.tracks:
            if tr.face.status != FaceStatus.TRACKED or tr.face.raw is None:
                continue
            truth = scen.person_x(tr.id, t)
            other = scen.person_x(1 - tr.id, t)
            if abs(truth - other) < 0.05:
                continue
            checked += 1
            fx = tr.face.raw.center[0]
            if abs(fx - truth) > abs(fx - other):
                wrong += 1
    assert checked > 1000 and wrong == 0, (wrong, checked)


def test_gesture_trial_runner_scores_prompts():
    import time as _t
    from evaluation.gesture_metrics import GestureTrialRunner
    from vision.gesture_recognition import GestureEvent
    runner = GestureTrialRunner(None, players=(0,), repeats=1, seed=0)
    now = _t.perf_counter() + 3.1
    runner.update(now)
    tr = runner.trial
    assert tr is not None
    if tr.gesture != "idle":
        runner.on_gesture(GestureEvent(0, tr.gesture, now + 0.5, now + 0.4, now + 0.55))
    runner.update(now + tr.duration + 0.01)
    assert runner.results and runner.results[0]["correct"] == 1
