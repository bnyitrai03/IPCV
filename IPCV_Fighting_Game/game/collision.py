"""Axis-aligned box collision primitives used for hit/hurt/push boxes."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Box:
    left: float
    top: float
    right: float
    bottom: float

    @staticmethod
    def from_center(cx: float, cy: float, w: float, h: float) -> "Box":
        return Box(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)

    @property
    def width(self) -> float:
        return self.right - self.left

    @property
    def height(self) -> float:
        return self.bottom - self.top

    @property
    def center(self) -> tuple[float, float]:
        return ((self.left + self.right) / 2, (self.top + self.bottom) / 2)

    def intersects(self, other: "Box") -> bool:
        return (self.left < other.right and other.left < self.right
                and self.top < other.bottom and other.top < self.bottom)

    def intersection_center(self, other: "Box") -> tuple[float, float]:
        """Centre of the overlap region (where impact effects are spawned)."""
        left, right = max(self.left, other.left), min(self.right, other.right)
        top, bottom = max(self.top, other.top), min(self.bottom, other.bottom)
        return ((left + right) / 2, (top + bottom) / 2)


def separate_pushboxes(x1: float, x2: float, min_gap: float) -> tuple[float, float]:
    """Push two fighters apart symmetrically so their centres are >= min_gap."""
    gap = abs(x2 - x1)
    if gap >= min_gap:
        return x1, x2
    push = (min_gap - gap) / 2
    if x1 <= x2:
        return x1 - push, x2 + push
    return x1 + push, x2 - push
