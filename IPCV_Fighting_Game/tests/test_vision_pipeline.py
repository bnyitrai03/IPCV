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
from vision.player_tracking import PlayerTracker, TrackState
from vision.pose_tracking import NOSE
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
    expected = {(pid, gesture) for pid, gestures in EXPECTED_EVENTS.items()
                for gesture in gestures}
    allowed = expected | {(pid, "block_end") for pid in EXPECTED_EVENTS}
    assert set(counts) <= allowed, set(counts) - allowed
    for pid, gestures in EXPECTED_EVENTS.items():
        for gesture, n in gestures.items():
            assert counts[(pid, gesture)] == n, (pid, gesture, counts)


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


def test_single_player_registers_alone_and_second_joins_later():
    """One person alone must become P1; a second person arriving later becomes P2."""
    cfg = Config()
    scen = SyntheticScenario(seed=5)
    tracker = PlayerTracker(cfg, scen.aspect)
    for k in range(60):                       # 2 s with only person A visible
        t = k / FPS
        image, poses, faces = scen.frame(t)
        a_only = [p for p in poses if p.center()[0] < 0.5]
        tracker.update(a_only, faces[:1], image, t, t)
    assert tracker.tracks[0].state == TrackState.ACTIVE
    assert not tracker.tracks[1].registered
    for k in range(60, 120):                  # B steps in
        t = k / FPS
        image, poses, faces = scen.frame(t)
        tracker.update(poses, faces, image, t, t)
    assert tracker.tracks[1].state == TrackState.ACTIVE
    assert tracker.tracks[0].center[0] < tracker.tracks[1].center[0]
