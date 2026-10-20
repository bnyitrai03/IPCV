"""Fighter model: position, health, energy and the action state machine.

The state machine is time based (seconds), so it behaves identically
regardless of rendering frame rate. Pure Python, no pygame dependency.

States::

    IDLE/WALK --punch--> PUNCH (startup -> active -> recovery) --> IDLE
    IDLE/WALK --block held--> BLOCK --released--> IDLE
    IDLE/WALK --special (energy)--> SPECIAL (charge -> release -> recovery)
    any --hit--> HITSTUN --> IDLE          any --blocked hit--> BLOCKSTUN
    any --health 0--> KO                   winner --> VICTORY
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from enum import Enum, auto

from config import GameConfig
from game.collision import Box
from game.controls import PlayerInput


class FighterState(Enum):
    IDLE = auto()
    WALK = auto()
    PUNCH = auto()
    BLOCK = auto()
    SPECIAL = auto()
    HITSTUN = auto()
    BLOCKSTUN = auto()
    KO = auto()
    VICTORY = auto()


@dataclass(frozen=True)
class AttackSpec:
    name: str
    startup: float
    active: float
    recovery: float
    damage: float
    reach: float          # px in front of the body edge
    height: float         # hitbox height
    y_offset: float       # hitbox centre above the ground
    hitstun: float
    blockstun: float
    knockback: float      # px/s initial push


PUNCH = AttackSpec("punch", startup=0.07, active=0.09, recovery=0.16, damage=6.0,
                   reach=95.0, height=46.0, y_offset=160.0,
                   hitstun=0.28, blockstun=0.16, knockback=260.0)
SPECIAL_CHARGE = 0.32
SPECIAL_RECOVERY = 0.38

BODY_HALF_WIDTH = 38.0
BODY_HEIGHT = 220.0
PUSH_GAP = 84.0

_attack_ids = itertools.count(1)


@dataclass
class ActiveAttack:
    spec: AttackSpec
    side: str             # "left" / "right" (which hand)
    attack_id: int = field(default_factory=lambda: next(_attack_ids))
    has_hit: bool = False
    source_ts: float | None = None


class Fighter:
    def __init__(self, player_id: int, name: str, x: float, facing: int,
                 cfg: GameConfig) -> None:
        self.player_id = player_id
        self.name = name
        self.cfg = cfg
        self.spawn_x = x
        self.ground_y = 0.0     # set by GameState (arena floor)
        self.reset_for_round(x, facing)

    # ------------------------------------------------------------------ setup
    def reset_for_round(self, x: float, facing: int) -> None:
        self.x = x
        self.facing = facing
        self.vx_knock = 0.0
        self.health = self.cfg.max_health
        self.energy = self.cfg.max_energy * 0.3
        self.state = FighterState.IDLE
        self.state_time = 0.0
        self.stun_duration = 0.0
        self.attack: ActiveAttack | None = None
        self.special_fired = False
        self.pending_special: bool = False
        self.buffer: list[tuple[str, float, float | None]] = []  # (action, age, src_ts)
        self.walk_dir = 0.0
        self.combo_count = 0
        self.combo_timer = 0.0
        self.last_hit_time = 0.0
        self.total_time = 0.0

    # --------------------------------------------------------------- queries
    @property
    def is_alive(self) -> bool:
        return self.health > 0

    @property
    def can_act(self) -> bool:
        return self.state in (FighterState.IDLE, FighterState.WALK, FighterState.BLOCK)

    @property
    def is_blocking(self) -> bool:
        return self.state in (FighterState.BLOCK, FighterState.BLOCKSTUN)

    def hurtbox(self) -> Box:
        return Box(self.x - BODY_HALF_WIDTH, self.ground_y - BODY_HEIGHT,
                   self.x + BODY_HALF_WIDTH, self.ground_y)

    def attack_phase(self) -> str | None:
        """'startup' / 'active' / 'recovery' while punching, else None."""
        if self.state != FighterState.PUNCH or self.attack is None:
            return None
        spec, t = self.attack.spec, self.state_time
        if t < spec.startup:
            return "startup"
        if t < spec.startup + spec.active:
            return "active"
        return "recovery"

    def hitbox(self) -> Box | None:
        if self.attack_phase() != "active" or self.attack is None or self.attack.has_hit:
            return None
        spec = self.attack.spec
        near = self.x + self.facing * (BODY_HALF_WIDTH - 10)
        far = near + self.facing * spec.reach
        cy = self.ground_y - spec.y_offset
        left, right = sorted((near, far))
        return Box(left, cy - spec.height / 2, right, cy + spec.height / 2)

    # --------------------------------------------------------------- control
    def _set_state(self, state: FighterState) -> None:
        self.state = state
        self.state_time = 0.0

    def apply_input(self, inp: PlayerInput) -> list[str]:
        """Queue one-shot actions and update held actions. Returns started actions."""
        if inp.punch_left:
            self.buffer.append(("punch_left", 0.0, inp.source_ts.get("punch_left")))
        if inp.punch_right:
            self.buffer.append(("punch_right", 0.0, inp.source_ts.get("punch_right")))
        if inp.special:
            self.buffer.append(("special", 0.0, inp.source_ts.get("special")))
        self.walk_dir = inp.move
        started: list[str] = []

        if self.can_act:
            # Attacks take priority over blocking/walking.
            while self.buffer:
                action, _age, src_ts = self.buffer.pop(0)
                if action == "special":
                    if self.energy >= self.cfg.special_cost:
                        self.energy -= self.cfg.special_cost
                        self._set_state(FighterState.SPECIAL)
                        self.special_fired = False
                        started.append("special")
                        break
                    started.append("special_denied")
                    continue
                side = "left" if action == "punch_left" else "right"
                self.attack = ActiveAttack(PUNCH, side, source_ts=src_ts)
                self._set_state(FighterState.PUNCH)
                started.append(action)
                break
            if self.can_act:
                if inp.block:
                    if self.state != FighterState.BLOCK:
                        self._set_state(FighterState.BLOCK)
                elif abs(inp.move) > 0.05:
                    if self.state != FighterState.WALK:
                        self._set_state(FighterState.WALK)
                elif self.state != FighterState.IDLE:
                    self._set_state(FighterState.IDLE)
        return started

    # ---------------------------------------------------------------- update
    def update(self, dt: float, arena_left: float, arena_right: float) -> None:
        self.total_time += dt
        self.state_time += dt
        # Expire buffered inputs.
        self.buffer = [(a, age + dt, ts) for a, age, ts in self.buffer
                       if age + dt <= self.cfg.input_buffer_s]
        # Energy regen.
        if self.state not in (FighterState.KO,):
            self.energy = min(self.cfg.max_energy,
                              self.energy + self.cfg.energy_regen_per_s * dt)
        # Combo window.
        if self.combo_timer > 0:
            self.combo_timer -= dt
            if self.combo_timer <= 0:
                self.combo_count = 0

        if self.state == FighterState.WALK:
            # Walking backwards is slower, as in classic fighters.
            speed = self.cfg.walk_speed
            if self.walk_dir * self.facing < 0:
                speed *= 0.75
            self.x += self.walk_dir * speed * dt
        elif self.state == FighterState.PUNCH and self.attack:
            spec = self.attack.spec
            if self.state_time >= spec.startup + spec.active + spec.recovery:
                self.attack = None
                self._set_state(FighterState.IDLE)
        elif self.state == FighterState.SPECIAL:
            if not self.special_fired and self.state_time >= SPECIAL_CHARGE:
                self.special_fired = True
                self.pending_special = True       # GameState spawns the projectile
            if self.state_time >= SPECIAL_CHARGE + SPECIAL_RECOVERY:
                self._set_state(FighterState.IDLE)
        elif self.state in (FighterState.HITSTUN, FighterState.BLOCKSTUN):
            if self.state_time >= self.stun_duration:
                self._set_state(FighterState.BLOCK if self.state == FighterState.BLOCKSTUN
                                else FighterState.IDLE)

        # Knockback slide with friction.
        if abs(self.vx_knock) > 1.0:
            self.x += self.vx_knock * dt
            self.vx_knock *= max(0.0, 1.0 - 8.0 * dt)
        else:
            self.vx_knock = 0.0

        self.x = max(arena_left + BODY_HALF_WIDTH, min(arena_right - BODY_HALF_WIDTH, self.x))

    # ---------------------------------------------------------------- damage
    def receive_hit(self, damage: float, hitstun: float, blockstun: float,
                    knockback_dir: int, knockback: float) -> tuple[bool, float]:
        """Apply a hit. Returns (was_blocked, damage_dealt)."""
        if self.state in (FighterState.KO, FighterState.VICTORY):
            return False, 0.0
        blocked = self.is_blocking
        if blocked:
            dealt = damage * self.cfg.block_damage_factor
            self.stun_duration = blockstun
            self._set_state(FighterState.BLOCKSTUN)
            self.vx_knock = knockback_dir * knockback * 0.6
        else:
            dealt = damage
            self.stun_duration = hitstun
            self.attack = None
            self._set_state(FighterState.HITSTUN)
            self.vx_knock = knockback_dir * knockback
        self.health = max(0.0, self.health - dealt)
        if self.health <= 0:
            self._set_state(FighterState.KO)
            self.vx_knock = knockback_dir * knockback * 1.5
        return blocked, dealt

    def set_victory(self) -> None:
        if self.state != FighterState.KO:
            self.attack = None
            self._set_state(FighterState.VICTORY)

    def face_towards(self, other_x: float) -> None:
        if self.state in (FighterState.IDLE, FighterState.WALK, FighterState.BLOCK):
            self.facing = 1 if other_x >= self.x else -1
