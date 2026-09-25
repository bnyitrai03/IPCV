# settings.py
import json, os
from dataclasses import dataclass, asdict

BASE_DIR    = os.path.dirname(__file__)
IMAGES_DIR  = os.path.join(BASE_DIR, "images")
SOUNDS_DIR  = os.path.join(BASE_DIR, "sounds")
SETTINGS_JSON = os.path.join(BASE_DIR, "settings.json")

# Constants
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
FPS = 60

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (50, 150, 255)
DARK_BLUE = (20, 60, 120)
GREEN = (50, 200, 50)
RED = (255, 50, 50)
GRAY = (100, 100, 100)
YELLOW = (255, 255, 0)
BROWN = (101, 67, 33)
LIGHT_GREEN = (34, 139, 34)
UI_BG  = (18, 44, 90)
BTN_BG = (236, 236, 236)
BTN_OUT= (25, 25, 25)
ACCENT = (40, 140, 255)

# Game settings
CANOE_WIDTH = 55  # Adjusted width
CANOE_HEIGHT = 100
CANOE_SPEED = 5
UPSTREAM_SPEED = 3  # How fast canoe moves up when rowing
DOWNSTREAM_DRIFT = 1.5  # How fast canoe drifts down when not rowing
CANOE_MAX_UP_Y = SCREEN_HEIGHT // 2  # Highest point canoe can reach (middle of screen)
RIVER_WIDTH = 400
OBSTACLE_WIDTH = 80
OBSTACLE_HEIGHT = 80
OBSTACLE_SPEED = 3
BASE_SPAWN_INTERVAL = 2.0  # seconds (base spawn interval)
BASE_OBSTACLE_SPEED = 3.0  # base obstacle speed
GOAL_DISTANCE = 5000  # Distance to travel to win

@dataclass
class UserSettings:
    music_vol: int = 60      # 0–100
    sfx_vol:   int = 80      # 0–100
    players:   int = 1       # 1..2 (limited to 2 players)
    difficulty: str = "normal"  # easy|hard
    player_names: list | None = None  # ["Alice","Bob"]

    def __post_init__(self):
        if self.player_names is None:
            self.player_names = ["", ""]

def load_settings() -> "UserSettings":
    try:
        if os.path.exists(SETTINGS_JSON):
            with open(SETTINGS_JSON, "r") as f:
                d = json.load(f)
            return UserSettings(**d)
    except Exception:
        pass
    return UserSettings()

def save_settings(s: "UserSettings"):
    with open(SETTINGS_JSON, "w") as f:
        json.dump(asdict(s), f, indent=2)

def difficulty_factors(name: str):
    """Returns (spawn_factor, speed_factor) where:
    - spawn_factor: multiplier for spawn interval (lower = more obstacles)
    - speed_factor: kept at 1.0 since we want difficulty via obstacles, not speed
    """
    name = (name or "normal").lower()
    if name == "easy":   return (1.5, 1.0)  # fewer obstacles, same speed
    if name == "hard":   return (0.5, 1.0)  # more obstacles, same speed
    return (1.0, 1.0)    # normal