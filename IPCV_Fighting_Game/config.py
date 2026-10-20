"""Central configuration for the IPCV fighting game.

All tunable parameters live here as dataclasses. Any value can be
overridden without editing code by placing a ``config.json`` next to this
file (or passing ``--config path.json``), e.g.::

    {"camera": {"index": 1, "width": 1280, "height": 720},
     "gestures": {"punch_extend_ratio": 1.3}}

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
    # Punch: arm reach (wrist-shoulder distance / shoulder width).
    punch_extend_ratio: float = 1.35
    punch_retract_ratio: float = 1.05
    punch_min_speed: float = 2.5         # shoulder widths per second
    punch_cooldown_s: float = 0.30
    # Block: wrists within this many shoulder widths of the nose.
    block_enter_dist: float = 0.85
    block_exit_dist: float = 1.05
    block_confirm_s: float = 0.08
    # Special: wrists above the nose by this many shoulder widths.
    special_raise: float = 0.45
    special_hold_s: float = 0.35
    special_cooldown_s: float = 2.5
    # Movement: hip offset from calibrated neutral, in shoulder widths.
    move_dead_zone: float = 0.30
    move_full_scale: float = 1.10
    move_smoothing: float = 0.25          # EMA factor per frame


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
    punch_damage: float = 6.0
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
