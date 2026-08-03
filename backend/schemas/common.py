"""Shared geometric primitives.

Everything downstream of floor-plan calibration works in **metres**, with the
floor plane as XY and Z up. Pixel-space values only ever appear inside the
ingest layer and are suffixed `_px` so the two never get confused.
"""

from __future__ import annotations

import math

from pydantic import BaseModel, Field


class Vec2(BaseModel):
    """A point or vector on the floor plane."""

    x: float
    y: float

    def as_tuple(self) -> tuple[float, float]:
        return (self.x, self.y)

    def __add__(self, other: Vec2) -> Vec2:
        return Vec2(x=self.x + other.x, y=self.y + other.y)

    def __sub__(self, other: Vec2) -> Vec2:
        return Vec2(x=self.x - other.x, y=self.y - other.y)

    def scaled(self, k: float) -> Vec2:
        return Vec2(x=self.x * k, y=self.y * k)

    def length(self) -> float:
        return math.hypot(self.x, self.y)

    def distance_to(self, other: Vec2) -> float:
        return math.hypot(self.x - other.x, self.y - other.y)


class Vec3(BaseModel):
    """A point or vector in world space. Z is up; Z=0 is the floor."""

    x: float
    y: float
    z: float = 0.0

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)

    @classmethod
    def from_vec2(cls, v: Vec2, z: float = 0.0) -> Vec3:
        return cls(x=v.x, y=v.y, z=z)


class Size3(BaseModel):
    """Real-world footprint and height of an object, in metres.

    `width` runs along the object's local X (its "front" face), `depth` along
    local Y, `height` along world Z. Rotation is applied about Z only, which is
    true for essentially all furniture.
    """

    width: float = Field(gt=0)
    depth: float = Field(gt=0)
    height: float = Field(gt=0)

    def footprint_area(self) -> float:
        return self.width * self.depth


def polygon_area(points: list[Vec2]) -> float:
    """Shoelace area. Always positive, so winding order doesn't matter."""
    if len(points) < 3:
        return 0.0
    total = 0.0
    for i, p in enumerate(points):
        q = points[(i + 1) % len(points)]
        total += p.x * q.y - q.x * p.y
    return abs(total) / 2.0


def polygon_centroid(points: list[Vec2]) -> Vec2:
    """Area-weighted centroid. Falls back to the mean for degenerate polygons."""
    area = polygon_area(points)
    if area < 1e-9:
        n = max(len(points), 1)
        return Vec2(
            x=sum(p.x for p in points) / n,
            y=sum(p.y for p in points) / n,
        )
    cx = cy = signed = 0.0
    for i, p in enumerate(points):
        q = points[(i + 1) % len(points)]
        cross = p.x * q.y - q.x * p.y
        cx += (p.x + q.x) * cross
        cy += (p.y + q.y) * cross
        signed += cross
    signed /= 2.0
    return Vec2(x=cx / (6.0 * signed), y=cy / (6.0 * signed))


def point_in_polygon(point: Vec2, polygon: list[Vec2]) -> bool:
    """Standard ray-casting test."""
    inside = False
    n = len(polygon)
    for i in range(n):
        a, b = polygon[i], polygon[(i + 1) % n]
        if (a.y > point.y) != (b.y > point.y):
            t = (point.y - a.y) / (b.y - a.y)
            if point.x < a.x + t * (b.x - a.x):
                inside = not inside
    return inside


def polygon_bounds(points: list[Vec2]) -> tuple[Vec2, Vec2]:
    """Axis-aligned bounding box as (min_corner, max_corner)."""
    xs = [p.x for p in points]
    ys = [p.y for p in points]
    return Vec2(x=min(xs), y=min(ys)), Vec2(x=max(xs), y=max(ys))
