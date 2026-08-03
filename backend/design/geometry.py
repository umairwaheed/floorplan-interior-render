"""Geometry helpers for the placement solver.

Everything here is exact rather than sampled. Overlap in particular is returned
as an **area**, not a boolean: the solver needs a continuous cost so it can tell
"barely touching" from "completely on top of each other" and descend the
gradient between them. A boolean predicate gives it a flat landscape with a
cliff, which simulated annealing cannot navigate.

Convex-only, which is safe here — every furniture footprint is a rotated
rectangle, and room polygons are clipped edge-by-edge rather than assumed
convex (see `polygon_contains_polygon`).
"""

from __future__ import annotations

import math

from ..schemas.common import Vec2, point_in_polygon, polygon_area


def rect_corners(centre: Vec2, width: float, depth: float, rotation_deg: float) -> list[Vec2]:
    """Corners of a rotated rectangle, counter-clockwise from the back-left.

    `width` runs along the object's local X, `depth` along local Y. At rotation
    0 the object's front faces +Y.
    """
    rad = math.radians(rotation_deg)
    cos_r, sin_r = math.cos(rad), math.sin(rad)
    hw, hd = width / 2.0, depth / 2.0
    return [
        Vec2(x=centre.x + dx * cos_r - dy * sin_r, y=centre.y + dx * sin_r + dy * cos_r)
        for dx, dy in ((-hw, -hd), (hw, -hd), (hw, hd), (-hw, hd))
    ]


def facing_direction(rotation_deg: float) -> Vec2:
    """Unit vector the object faces. At rotation 0 this is +Y."""
    rad = math.radians(rotation_deg)
    return Vec2(x=-math.sin(rad), y=math.cos(rad))


def rotation_to_face(direction: Vec2) -> float:
    """Rotation, in degrees, that makes an object face `direction`.

    Inverse of `facing_direction`. Used to turn a wall's inward normal into the
    rotation that puts an object's back against that wall.
    """
    return math.degrees(math.atan2(-direction.x, direction.y)) % 360.0


def _clip_against_edge(polygon: list[Vec2], a: Vec2, b: Vec2) -> list[Vec2]:
    """Sutherland–Hodgman: keep the part of `polygon` left of the directed edge a→b."""

    def side(p: Vec2) -> float:
        return (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x)

    if not polygon:
        return []

    output: list[Vec2] = []
    for i, current in enumerate(polygon):
        previous = polygon[i - 1]
        current_in = side(current) >= 0
        previous_in = side(previous) >= 0

        if current_in != previous_in:
            # The edge crosses the clip line — add the intersection point.
            d1, d2 = side(previous), side(current)
            denominator = d1 - d2
            if abs(denominator) > 1e-12:
                t = d1 / denominator
                output.append(
                    Vec2(
                        x=previous.x + t * (current.x - previous.x),
                        y=previous.y + t * (current.y - previous.y),
                    )
                )
        if current_in:
            output.append(current)
    return output


def _ensure_ccw(polygon: list[Vec2]) -> list[Vec2]:
    """Sutherland–Hodgman assumes a consistent winding; force counter-clockwise."""
    signed = 0.0
    for i, p in enumerate(polygon):
        q = polygon[(i + 1) % len(polygon)]
        signed += p.x * q.y - q.x * p.y
    return polygon if signed >= 0 else list(reversed(polygon))


def convex_intersection_area(a: list[Vec2], b: list[Vec2]) -> float:
    """Exact overlap area of two convex polygons. Zero when disjoint."""
    if len(a) < 3 or len(b) < 3:
        return 0.0

    clip = _ensure_ccw(b)
    subject = _ensure_ccw(a)
    for i in range(len(clip)):
        subject = _clip_against_edge(subject, clip[i], clip[(i + 1) % len(clip)])
        if not subject:
            return 0.0
    return polygon_area(subject)


def rect_overlap_area(
    centre_a: Vec2,
    size_a: tuple[float, float],
    rot_a: float,
    centre_b: Vec2,
    size_b: tuple[float, float],
    rot_b: float,
) -> float:
    """Overlap area of two oriented rectangles.

    Cheap axis-aligned rejection first — most pairs in a room don't overlap, and
    the polygon clip is ~50× more expensive than a bounding-circle test.
    """
    reach_a = math.hypot(size_a[0], size_a[1]) / 2.0
    reach_b = math.hypot(size_b[0], size_b[1]) / 2.0
    if centre_a.distance_to(centre_b) > reach_a + reach_b:
        return 0.0

    return convex_intersection_area(
        rect_corners(centre_a, size_a[0], size_a[1], rot_a),
        rect_corners(centre_b, size_b[0], size_b[1], rot_b),
    )


def polygon_contains_polygon(outer: list[Vec2], inner: list[Vec2]) -> bool:
    """True when every vertex of `inner` lies inside `outer`.

    Sufficient here because `inner` is always a rectangle and `outer` a room:
    a rectangle whose corners are all inside a simple polygon can only escape
    through a concave notch narrower than the rectangle, which would mean the
    room has a feature smaller than a piece of furniture.
    """
    return all(point_in_polygon(vertex, outer) for vertex in inner)


def outside_area(room_polygon: list[Vec2], corners: list[Vec2]) -> float:
    """How much of a footprint sticks out of the room. Zero when fully inside.

    Continuous, so the solver can slide an object back inside rather than
    having to teleport it.
    """
    footprint = polygon_area(corners)
    if footprint <= 0:
        return 0.0
    return max(0.0, footprint - convex_intersection_area(corners, room_polygon))


def point_to_segment_distance(point: Vec2, a: Vec2, b: Vec2) -> float:
    """Shortest distance from a point to a line segment."""
    dx, dy = b.x - a.x, b.y - a.y
    length_sq = dx * dx + dy * dy
    if length_sq < 1e-12:
        return point.distance_to(a)
    t = max(0.0, min(1.0, ((point.x - a.x) * dx + (point.y - a.y) * dy) / length_sq))
    return math.hypot(point.x - (a.x + t * dx), point.y - (a.y + t * dy))


def polygon_edges(polygon: list[Vec2]) -> list[tuple[Vec2, Vec2]]:
    """Consecutive edge pairs, closing the loop."""
    return [(polygon[i], polygon[(i + 1) % len(polygon)]) for i in range(len(polygon))]


def inward_normal(a: Vec2, b: Vec2, polygon: list[Vec2]) -> Vec2:
    """Unit normal of edge a→b pointing into the polygon.

    Derived by testing a probe point rather than assuming a winding order —
    room polygons come from a vision model and their winding is not guaranteed.
    """
    dx, dy = b.x - a.x, b.y - a.y
    length = math.hypot(dx, dy) or 1.0
    normal = Vec2(x=-dy / length, y=dx / length)

    midpoint = Vec2(x=(a.x + b.x) / 2.0, y=(a.y + b.y) / 2.0)
    probe = Vec2(x=midpoint.x + normal.x * 0.05, y=midpoint.y + normal.y * 0.05)
    if point_in_polygon(probe, polygon):
        return normal
    return Vec2(x=-normal.x, y=-normal.y)


def segment_length(a: Vec2, b: Vec2) -> float:
    return a.distance_to(b)
