"""Hit resolution, damage calculation and projectiles (special attack)."""
from __future__ import annotations

from dataclasses import dataclass, field

from config import GameConfig
from game.collision import Box
from game.fighter import Fighter, FighterState

COUNTER_HIT_MULT = 1.25      # hitting an opponent during their punch startup
COMBO_WINDOW_S = 0.9
COMBO_SCALING = 0.9          # each extra combo hit deals 10 % less (min 50 %)


@dataclass
class GameEvent:
    kind: str                 # hit, block, special_fire, clash, ko, ...
    data: dict = field(default_factory=dict)


@dataclass
class Projectile:
    owner: int
    x: float
    y: float
    direction: int
    speed: float = 640.0
    radius: float = 30.0
    damage: float = 18.0
    alive: bool = True
    age: float = 0.0

    def box(self) -> Box:
        return Box.from_center(self.x, self.y, self.radius * 2, self.radius * 1.6)

    def update(self, dt: float, left: float, right: float) -> None:
        self.age += dt
        self.x += self.direction * self.speed * dt
        if self.x < left - 60 or self.x > right + 60:
            self.alive = False


def compute_damage(base: float, attacker: Fighter, defender: Fighter) -> tuple[float, bool]:
    """Return (damage, counter_hit). Combo scaling uses the attacker's combo count."""
    counter = defender.attack_phase() == "startup"
    dmg = base * (COUNTER_HIT_MULT if counter else 1.0)
    scale = max(0.5, COMBO_SCALING ** attacker.combo_count)
    return dmg * scale, counter


def _register_hit(attacker: Fighter, defender: Fighter, base: float, hitstun: float,
                  blockstun: float, knockback: float, point: tuple[float, float],
                  cfg: GameConfig, attack_name: str, source_ts: float | None) -> GameEvent:
    dmg, counter = compute_damage(base, attacker, defender)
    direction = 1 if defender.x >= attacker.x else -1
    blocked, dealt = defender.receive_hit(dmg, hitstun, blockstun, direction, knockback)
    if not blocked:
        attacker.combo_count = attacker.combo_count + 1 if attacker.combo_timer > 0 else 1
        attacker.combo_timer = COMBO_WINDOW_S
        attacker.energy = min(cfg.max_energy, attacker.energy + cfg.energy_per_hit)
        defender.energy = min(cfg.max_energy, defender.energy + cfg.energy_per_hit * 0.5)
    return GameEvent("block" if blocked else "hit", {
        "attacker": attacker.player_id, "defender": defender.player_id,
        "damage": dealt, "x": point[0], "y": point[1], "attack": attack_name,
        "counter": counter and not blocked, "combo": attacker.combo_count,
        "ko": defender.state == FighterState.KO, "source_ts": source_ts,
    })


def resolve_melee(fighters: list[Fighter], cfg: GameConfig) -> list[GameEvent]:
    """Check punch hitboxes against opponents' hurtboxes (simultaneous hits allowed)."""
    events: list[GameEvent] = []
    hits = []
    for attacker in fighters:
        hb = attacker.hitbox()
        if hb is None or attacker.attack is None:
            continue
        for defender in fighters:
            if defender is attacker or defender.state == FighterState.KO:
                continue
            if hb.intersects(defender.hurtbox()):
                hits.append((attacker, attacker.attack, defender, hb))
    # Apply after detection so simultaneous punches trade instead of one cancelling the other.
    for attacker, attack, defender, hb in hits:
        attack.has_hit = True
        spec = attack.spec
        point = hb.intersection_center(defender.hurtbox())
        events.append(_register_hit(attacker, defender, spec.damage, spec.hitstun,
                                    spec.blockstun, spec.knockback, point, cfg,
                                    f"{spec.name}_{attack.side}", attack.source_ts))
    return events


def resolve_projectiles(projectiles: list[Projectile], fighters: list[Fighter],
                        cfg: GameConfig) -> list[GameEvent]:
    events: list[GameEvent] = []
    # Projectile clash.
    for i, a in enumerate(projectiles):
        for b in projectiles[i + 1:]:
            if a.alive and b.alive and a.owner != b.owner and a.box().intersects(b.box()):
                a.alive = b.alive = False
                events.append(GameEvent("clash", {"x": (a.x + b.x) / 2, "y": (a.y + b.y) / 2}))
    for p in projectiles:
        if not p.alive:
            continue
        attacker = fighters[p.owner]
        for defender in fighters:
            if defender.player_id == p.owner or defender.state == FighterState.KO:
                continue
            if p.box().intersects(defender.hurtbox()):
                p.alive = False
                point = p.box().intersection_center(defender.hurtbox())
                events.append(_register_hit(attacker, defender, p.damage, 0.55, 0.3, 420.0,
                                            point, cfg, "special", None))
    return events
