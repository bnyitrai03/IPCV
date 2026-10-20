"""Procedural skeletal animation for the fighters.

Poses are dictionaries of joint -> (x, y) in a local frame: origin on the
floor under the hips, +x towards the opponent, -y upwards. Each fighter
state maps to keyframe interpolation driven by the logic state timers, and
an :class:`Animator` blends between successive target poses so state
changes never pop.
"""
from __future__ import annotations

import math

from game.fighter import PUNCH, SPECIAL_CHARGE, SPECIAL_RECOVERY, Fighter, FighterState

Pose = dict[str, tuple[float, float]]

JOINTS = ("hip", "chest", "neck", "head", "sh_f", "el_f", "wr_f", "sh_b", "el_b", "wr_b",
          "kn_f", "an_f", "kn_b", "an_b")

STANCE: Pose = {
    "hip": (0, -108), "chest": (4, -150), "neck": (8, -180), "head": (12, -202),
    "sh_f": (16, -172), "el_f": (42, -142), "wr_f": (54, -174),
    "sh_b": (-4, -174), "el_b": (18, -136), "wr_b": (32, -166),
    "kn_f": (30, -56), "an_f": (42, 0), "kn_b": (-22, -56), "an_b": (-40, 0),
}


def _mod(base: Pose, **changes: tuple[float, float]) -> Pose:
    p = dict(base)
    p.update(changes)
    return p


JAB = _mod(STANCE, chest=(10, -150), neck=(16, -179), head=(20, -201), sh_f=(26, -172),
           el_f=(78, -170), wr_f=(116, -170), wr_b=(30, -170))
CROSS = _mod(STANCE, hip=(4, -106), chest=(14, -150), neck=(22, -178), head=(26, -199),
             sh_b=(18, -172), el_b=(76, -168), wr_b=(120, -168), sh_f=(22, -170),
             el_f=(30, -140), wr_f=(36, -170), an_b=(-30, 0))
BLOCK = _mod(STANCE, hip=(-4, -102), chest=(-2, -146), neck=(2, -175), head=(4, -196),
             el_f=(36, -150), wr_f=(38, -204), el_b=(24, -146), wr_b=(26, -198),
             kn_f=(30, -52), kn_b=(-22, -52))
SPECIAL_CHARGE_POSE = _mod(STANCE, hip=(-6, -100), chest=(-10, -142), neck=(-8, -172),
                           head=(-6, -194), el_f=(-6, -126), wr_f=(-26, -122),
                           el_b=(-14, -124), wr_b=(-30, -118), kn_f=(34, -48), kn_b=(-26, -48))
SPECIAL_RELEASE = _mod(STANCE, hip=(6, -104), chest=(16, -146), neck=(24, -174),
                       head=(28, -196), sh_f=(26, -168), el_f=(68, -156), wr_f=(108, -154),
                       sh_b=(10, -170), el_b=(60, -150), wr_b=(102, -146), an_b=(-56, 0),
                       kn_f=(44, -50))
HIT = _mod(STANCE, hip=(-8, -104), chest=(-20, -146), neck=(-30, -172), head=(-38, -190),
           el_f=(10, -140), wr_f=(30, -158), el_b=(-30, -136), wr_b=(-10, -150))
KO_LYING = {
    "hip": (-60, -16), "chest": (-104, -20), "neck": (-136, -20), "head": (-160, -22),
    "sh_f": (-128, -26), "el_f": (-110, -46), "wr_f": (-92, -24),
    "sh_b": (-128, -16), "el_b": (-150, -8), "wr_b": (-176, -8),
    "kn_f": (-18, -40), "an_f": (14, -6), "kn_b": (-20, -14), "an_b": (24, -4),
}
VICTORY = _mod(STANCE, hip=(0, -112), chest=(2, -156), neck=(4, -186), head=(6, -208),
               sh_f=(12, -178), el_f=(24, -214), wr_f=(20, -250),
               el_b=(-24, -142), wr_b=(-10, -118), kn_f=(18, -58), an_f=(24, 0),
               kn_b=(-16, -58), an_b=(-26, 0))


def lerp_pose(a: Pose, b: Pose, t: float) -> Pose:
    t = max(0.0, min(1.0, t))
    return {j: (a[j][0] + (b[j][0] - a[j][0]) * t, a[j][1] + (b[j][1] - a[j][1]) * t) for j in JOINTS}


def ease_out(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return 1 - (1 - t) ** 3


def _breathe(p: Pose, t: float, amp: float = 2.5, speed: float = 4.2) -> Pose:
    dy = math.sin(t * speed) * amp
    upper = {"chest", "neck", "head", "sh_f", "el_f", "wr_f", "sh_b", "el_b", "wr_b"}
    return {j: (x, y + (dy if j in upper else dy * 0.4 if j in ("hip", "kn_f", "kn_b") else 0))
            for j, (x, y) in p.items()}


def target_pose(f: Fighter, clock: float) -> Pose:
    """Keyframe pose for the fighter's current logic state."""
    st, t = f.state, f.state_time
    if st == FighterState.WALK:
        phase = f.total_time * 9.0
        s = math.sin(phase)
        p = _breathe(STANCE, clock, 1.5)
        dirn = f.walk_dir * f.facing
        p["an_f"] = (STANCE["an_f"][0] + 16 * s * dirn, -max(0.0, 10 * s))
        p["an_b"] = (STANCE["an_b"][0] - 16 * s * dirn, -max(0.0, -10 * s))
        p["kn_f"] = (STANCE["kn_f"][0] + 8 * s * dirn, STANCE["kn_f"][1] - abs(6 * s))
        p["kn_b"] = (STANCE["kn_b"][0] - 8 * s * dirn, STANCE["kn_b"][1] - abs(6 * s))
        return p
    if st == FighterState.PUNCH and f.attack:
        key = JAB if f.attack.side == "left" else CROSS
        if t < PUNCH.startup:
            k = ease_out(t / PUNCH.startup)
        elif t < PUNCH.startup + PUNCH.active:
            k = 1.0
        else:
            k = 1.0 - ease_out((t - PUNCH.startup - PUNCH.active) / PUNCH.recovery)
        return lerp_pose(STANCE, key, k)
    if st in (FighterState.BLOCK, FighterState.BLOCKSTUN):
        p = _breathe(BLOCK, clock, 1.2)
        if st == FighterState.BLOCKSTUN:
            p = {j: (x - 6, y) for j, (x, y) in p.items()}
        return p
    if st == FighterState.SPECIAL:
        if t < SPECIAL_CHARGE:
            shake = math.sin(t * 90) * 1.5
            p = lerp_pose(STANCE, SPECIAL_CHARGE_POSE, ease_out(t / SPECIAL_CHARGE))
            return {j: (x + shake, y) for j, (x, y) in p.items()}
        k = (t - SPECIAL_CHARGE) / SPECIAL_RECOVERY
        return lerp_pose(SPECIAL_RELEASE, STANCE, max(0.0, (k - 0.5) * 2))
    if st == FighterState.HITSTUN:
        k = math.sin(min(1.0, t / max(f.stun_duration, 0.01)) * math.pi)
        return lerp_pose(STANCE, HIT, 0.4 + 0.6 * k)
    if st == FighterState.KO:
        if t < 0.15:
            return HIT
        return lerp_pose(HIT, KO_LYING, ease_out((t - 0.15) / 0.55))
    if st == FighterState.VICTORY:
        p = lerp_pose(STANCE, VICTORY, ease_out(t / 0.4))
        return _breathe(p, clock, 2.0, 6.0)
    return _breathe(STANCE, clock)


class Animator:
    """Smoothly blends towards the target pose (critically-damped style)."""

    def __init__(self) -> None:
        self.pose: Pose = dict(STANCE)

    def update(self, f: Fighter, clock: float, dt: float) -> Pose:
        target = target_pose(f, clock)
        fast = f.state in (FighterState.PUNCH, FighterState.SPECIAL, FighterState.HITSTUN)
        rate = 40.0 if fast else 16.0
        k = 1.0 - math.exp(-rate * dt)
        self.pose = lerp_pose(self.pose, target, k)
        return self.pose
