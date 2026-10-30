"""Central configuration for the IPCV fighting game.

All tunable parameters live here as dataclasses. Any value can be
overridden without editing code by placing a ``config.json`` next to this
file (or passing ``--config path.json``), e.g.::

    {"camera": {"index": 1, "width": 1280, "height": 720},
     "gestures": {"hook_min_dx": 0.4}}

Paths are always resolved relative to the project root, never absolute.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "assets" / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

log = logging.getLogger(__name__)


@dataclass
class CameraConfig:
    index: int = 0
    width: int = 1280
    height: int = 720
    fps: int = 30
    # "auto" picks AVFoundation on macOS, DirectShow/MSMF on Windows, V4L2 on Linux.
    backend: str = "auto"
    # Mirror the image so that moving left in real life moves left on screen.
    mirror: bool = True
    # Frames are downscaled to this width before inference (keeps CV fast).
    inference_width: int = 640


@dataclass
class PoseConfig:
    model_file: str = "pose_landmarker_full.task"   # in assets/models/
    num_poses: int = 2
    min_detection_confidence: float = 0.5
    min_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    # A landmark below this visibility is treated as missing.
    landmark_visibility_threshold: float = 0.5
    # How long a missing landmark keeps its last filtered value (seconds).
    landmark_hold_s: float = 0.25
    # One Euro filter parameters (normalised image coordinates).
    filter_min_cutoff: float = 1.5
    filter_beta: float = 8.0
    filter_d_cutoff: float = 1.0


@dataclass
class FaceConfig:
    model_file: str = "face_landmarker.task"
    num_faces: int = 2
    min_detection_confidence: float = 0.5
    min_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    # Run the face model every N vision frames (1 = every frame).
    run_every_n_frames: int = 1
    # Max face-to-body association distance, in shoulder widths.
    association_gate: float = 0.8
    # Keep drawing a predicted (fading) effect for this long after loss.
    hold_s: float = 0.5
    filter_min_cutoff: float = 1.0
    filter_beta: float = 4.0
    angle_min_cutoff: float = 0.8
    angle_beta: float = 1.0


@dataclass
class TrackingConfig:
    # Association cost weights.
    w_position: float = 1.0
    w_appearance: float = 1.5
    w_size: float = 0.5
    # Gate on predicted-position distance (in shoulder widths) for active tracks.
    position_gate: float = 2.5
    # Track coasts on prediction for this long, then becomes ABSENT.
    lost_timeout_s: float = 1.2
    # Max Bhattacharyya distance for re-identifying a returning player.
    reentry_appearance_threshold: float = 0.55
    # Appearance model exponential update rate (only when players are apart).
    appearance_update_rate: float = 0.03
    # Players closer than this (shoulder widths) are "overlapping" -> crossing mode.
    overlap_distance: float = 1.2
    # During overlap, appearance weight is multiplied by this.
    overlap_appearance_boost: float = 2.5
    # Frames with both players visible needed to register identities.
    registration_frames: int = 20
    velocity_decay: float = 0.85


@dataclass
class GestureConfig:
    # All distances in shoulder widths (sw), speeds in sw per second.
    # --- shared attack motion ---
    strike_onset_speed: float = 1.5      # fist speed that starts a motion window
    attack_min_speed: float = 2.2        # average fist speed over the attack
    attack_cooldown_s: float = 0.30      # per arm, after an attack
    rearm_dist: float = 0.6              # fist back this close to where the attack started...
    rearm_reach: float = 0.9             # ...or arm pulled in (wrist-shoulder < this)
    # --- hook: elbow bent ~90 deg, fast horizontal sweep towards/across the body ---
    hook_direction: str = "inward"       # "inward" = across the body; "both" also away from it
    hook_min_dx: float = 0.4             # horizontal travel from the start of the sweep, sw
    hook_dominance: float = 0.9          # horizontal travel must exceed |dy| * this
    hook_window_s: float = 0.6           # how far back to look for the wind-up point
    hook_min_y: float = -0.7             # sw relative to the nose (- = above): the sweep
    hook_max_y: float = 0.8              # must end around head height
    hook_max_start_y: float = 1.3        # and may start from chest height
    hook_max_duration_s: float = 0.4     # wind-up point -> now must be this fast
    max_fist_speed: float = 25.0         # sw/s; faster single-frame jumps are tracking glitches
    max_sample_gap_s: float = 0.1        # an attack must be tracked without longer gaps
    raw_min_visibility: float = 0.3      # wrist/elbow confidence (fast fists are blurry)
    frame_margin: float = 0.005          # wrist/elbow closer to the image border = not visible
    arm_gap_tolerance_s: float = 0.2     # a blurred/lost wrist this long keeps the motion
    hook_min_elbow_deg: float = 20.0     # elbow angle range (180 = straight arm); the
    hook_max_elbow_deg: float = 145.0    # 3D OR the 2D image angle must be in range
    elbow2d_min_segment: float = 0.4     # sw; shorter (foreshortened) arm segments -> no 2D angle
    # --- uppercut: one fist up and forward, ending at head height ---
    uppercut_min_rise: float = 0.6
    uppercut_end_below_nose: float = 0.45   # fist must end at least at chin height
    uppercut_min_reach: float = 0.7      # fist away from the shoulder (bent arm is fine)
    uppercut_confirm_s: float = 0.12     # wait to make sure it is not a two-arm raise
    uppercut_max_duration_s: float = 0.4 # lowest point -> now must be this fast
    uppercut_min_peak_speed: float = 3.0 # peak fist speed during the rise (sw/s)
    uppercut_max_inward_speed: float = 2.5   # sw/s; fist sweeping inward = hook, not uppercut
    uppercut_max_side: float = 0.6       # sw; end fist at most this far beside its shoulder
    uppercut_dominance: float = 1.3      # rise must exceed sideways travel * this
    # Block: wrists within this many shoulder widths of the nose.
    block_enter_dist: float = 0.85
    block_exit_dist: float = 1.05
    block_confirm_s: float = 0.08
    # Special: wrists above the nose by this many shoulder widths.
    special_raise: float = 0.45
    special_hold_s: float = 0.35
    special_cooldown_s: float = 2.5
    # Movement. "lean": lean the upper body left/right (shoulders vs hips), so
    # players can stand anywhere; "position": step away from the registered spot.
    move_mode: str = "lean"
    lean_dead_zone: float = 0.22          # shoulder tilt + head offset (sw) that is ignored
    lean_release_ratio: float = 0.7       # keep walking until the lean drops below dead*this
    lean_full_scale: float = 0.45         # sw for full walking speed
    move_freeze_after_attack_s: float = 0.35   # torso twist of a punch is not a lean
    move_dead_zone: float = 0.30          # (position mode) step size ignored, sw
    move_full_scale: float = 1.10         # (position mode) step for full speed, sw
    move_smoothing: float = 0.18          # EMA factor per frame


@dataclass
class GameConfig:
    window_width: int = 1280
    window_height: int = 760
    arena_height: int = 560
    target_fps: int = 60
    logic_hz: int = 60
    round_time_s: int = 60
    rounds_to_win: int = 2
    max_health: float = 100.0
    max_energy: float = 100.0
    energy_regen_per_s: float = 6.0
    energy_per_hit: float = 8.0
    special_cost: float = 50.0
    walk_speed: float = 260.0            # px/s
    special_damage: float = 18.0
    block_damage_factor: float = 0.15    # chip damage while blocking
    input_buffer_s: float = 0.15
    fullscreen: bool = False
    sound: bool = True


@dataclass
class Config:
    camera: CameraConfig = field(default_factory=CameraConfig)
    pose: PoseConfig = field(default_factory=PoseConfig)
    face: FaceConfig = field(default_factory=FaceConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    gestures: GestureConfig = field(default_factory=GestureConfig)
    game: GameConfig = field(default_factory=GameConfig)

    def model_path(self, filename: str) -> Path:
        return MODELS_DIR / filename


def _apply_overrides(obj: Any, data: dict) -> None:
    valid = {f.name for f in fields(obj)}
    for key, value in data.items():
        if key not in valid:
            log.warning("Unknown config key ignored: %s", key)
            continue
        current = getattr(obj, key)
        if is_dataclass(current) and isinstance(value, dict):
            _apply_overrides(current, value)
        else:
            setattr(obj, key, type(current)(value) if current is not None else value)


def load_config(path: str | Path | None = None) -> Config:
    """Return defaults, overridden by a JSON file if present."""
    cfg = Config()
    candidate = Path(path) if path else PROJECT_ROOT / "config.json"
    if candidate.is_file():
        with candidate.open("r", encoding="utf-8") as fh:
            _apply_overrides(cfg, json.load(fh))
        log.info("Loaded config overrides from %s", candidate)
    elif path:
        log.warning("Config file %s not found, using defaults", candidate)
    return cfg
