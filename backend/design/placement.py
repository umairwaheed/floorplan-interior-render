"""Deterministic placement solver.

The LLM chooses *what* belongs in a room; this decides *where*. That split is
deliberate — placement is arithmetic over constraints (clearances, door swings,
non-overlap, real product dimensions), and a language model asked for
coordinates is being asked to do the one thing it is worst at. Given the same
inputs and seed this produces byte-identical output, which is what makes the
scene graph content-hashable and regeneration reproducible.

**Two tiers, mirroring how rooms actually work.** Anchors (bed, sofa, dining
table) are searched globally over candidate positions. Dependents (nightstands,
dining chairs, rugs, pendants) are *derived* from their anchor once it lands —
a nightstand's position is a consequence of the bed's, not an independent
decision. This collapses the search space by orders of magnitude and produces
better rooms than optimizing everything jointly, because the hierarchy encodes
real design relationships that a flat cost function would have to rediscover.
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass, field

from ..config import Settings, get_settings
from ..schemas.common import Size3, Vec2, Vec3, polygon_area, polygon_centroid
from ..schemas.floorplan import Opening, OpeningType, Room
from ..schemas.product import Product, ProductCategory
from ..schemas.scene import ObjectRole, PlacedObject
from .geometry import (
    inward_normal,
    outside_area,
    point_to_segment_distance,
    polygon_edges,
    rect_corners,
    rect_overlap_area,
    rotation_to_face,
    segment_length,
)
from .slots import FurnitureSlot, Placement

logger = logging.getLogger(__name__)

# --- cost weights ----------------------------------------------------------
# Overlap dominates by two orders of magnitude: a room where objects intersect
# is not a worse design, it is a physically impossible one. Everything else is
# a preference and trades off against the others.
W_OVERLAP = 500.0
W_OUTSIDE = 800.0
W_DOOR_BLOCK = 300.0
W_WINDOW_BLOCK = 60.0
W_CIRCULATION = 40.0
W_WALL_GAP = 15.0
W_CLUSTER = 6.0

#: An object above this height is resting on something else (a lamp on a
#: nightstand), not competing for floor space.
FLOOR_Z_TOLERANCE = 0.1


@dataclass
class SlotFill:
    """A slot bound to the real product that will fill it."""

    slot: FurnitureSlot
    product: Product
    #: One entry per unit — quantity 2 means two placed objects from one slot.
    index: int = 0

    @property
    def size(self) -> Size3:
        return self.product.dimensions.to_size3()

    @property
    def instance_id(self) -> str:
        return f"{self.slot.slot_id}#{self.index}"


@dataclass
class Candidate:
    """A possible position and rotation for one object."""

    position: Vec2
    rotation_deg: float
    wall_index: int | None = None
    #: Distance from the object's back face to the wall it's placed against.
    wall_gap: float = 0.0


@dataclass
class RoomContext:
    """Everything the solver needs to know about one room."""

    room: Room
    openings: list[Opening]
    settings: Settings

    polygon: list[Vec2] = field(default_factory=list)
    edges: list[tuple[Vec2, Vec2, Vec2, float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.polygon = self.room.polygon_m
        self.edges = [
            (a, b, inward_normal(a, b, self.polygon), segment_length(a, b))
            for a, b in polygon_edges(self.polygon)
        ]

    @property
    def centroid(self) -> Vec2:
        return polygon_centroid(self.polygon)

    @property
    def area(self) -> float:
        return polygon_area(self.polygon)

    def door_keepouts(self) -> list[tuple[Vec2, float]]:
        """(centre, radius) circles that furniture must stay clear of."""
        return [
            (opening.centre_m, opening.keepout_radius_m())
            for opening in self.openings
            if opening.opening_type != OpeningType.WINDOW
        ]

    def window_spans(self) -> list[tuple[Vec2, Vec2]]:
        return [(o.start_m, o.end_m) for o in self.openings if o.opening_type == OpeningType.WINDOW]


# --- candidate generation --------------------------------------------------


def _wall_candidates(context: RoomContext, size: Size3, step: float = 0.25) -> list[Candidate]:
    """Positions with the object's back flat against each wall."""
    candidates: list[Candidate] = []
    clearance = context.settings.wall_clearance_m

    for index, (a, b, normal, length) in enumerate(context.edges):
        if length < size.width:
            continue  # object is wider than this wall

        direction = Vec2(x=(b.x - a.x) / length, y=(b.y - a.y) / length)
        half_width = size.width / 2.0
        offset = size.depth / 2.0 + clearance
        rotation = rotation_to_face(normal)

        travel = half_width
        while travel <= length - half_width + 1e-9:
            anchor = Vec2(x=a.x + direction.x * travel, y=a.y + direction.y * travel)
            candidates.append(
                Candidate(
                    position=Vec2(x=anchor.x + normal.x * offset, y=anchor.y + normal.y * offset),
                    rotation_deg=rotation,
                    wall_index=index,
                    wall_gap=clearance,
                )
            )
            travel += step

    return candidates


def _centre_candidates(context: RoomContext, size: Size3, steps: int = 7) -> list[Candidate]:
    """A grid over the room's open floor, at two orientations."""
    from ..schemas.common import polygon_bounds

    low, high = polygon_bounds(context.polygon)
    candidates: list[Candidate] = []

    for i in range(1, steps):
        for j in range(1, steps):
            position = Vec2(
                x=low.x + (high.x - low.x) * i / steps,
                y=low.y + (high.y - low.y) * j / steps,
            )
            for rotation in (0.0, 90.0):
                width, depth = (
                    (size.width, size.depth) if rotation == 0.0 else (size.depth, size.width)
                )
                corners = rect_corners(position, width, depth, 0.0)
                if outside_area(context.polygon, corners) < 1e-6:
                    candidates.append(Candidate(position=position, rotation_deg=rotation))
    return candidates


def _corner_candidates(context: RoomContext, size: Size3) -> list[Candidate]:
    """Tucked into each corner, clear of both walls that meet there.

    The offset is built from the two walls' inward normals rather than from a
    single scalar along the diagonal. A diagonal inset of `max(w, d)/2` looks
    right but leaves less than half that along each axis, so the footprint
    still pokes through a wall — and every corner candidate gets rejected,
    silently collapsing the object to the room centre. That is how a floor lamp
    ends up standing in the middle of a studio.
    """
    candidates: list[Candidate] = []
    clearance = context.settings.wall_clearance_m
    centroid = context.centroid
    edges = context.edges

    for index in range(len(context.polygon)):
        vertex = context.polygon[index]
        # The two edges meeting at this vertex, with their inward normals.
        _, _, normal_out, _ = edges[index - 1]
        _, _, normal_in, _ = edges[index]

        # Half-*diagonal*, not half-width: a rotated rectangle reaches
        # hypot(w, d)/2 from its centre in the worst direction. Using max(w, d)/2
        # passes for thin objects by luck and fails for square ones, which is
        # why every corner candidate for a square plant was being rejected.
        reach = math.hypot(size.width, size.depth) / 2.0 + clearance
        position = Vec2(
            x=vertex.x + (normal_out.x + normal_in.x) * reach,
            y=vertex.y + (normal_out.y + normal_in.y) * reach,
        )

        toward_centre = Vec2(x=centroid.x - position.x, y=centroid.y - position.y)
        rotation = rotation_to_face(toward_centre) if toward_centre.length() > 1e-6 else 0.0

        corners = rect_corners(position, size.width, size.depth, rotation)
        if outside_area(context.polygon, corners) < 1e-6:
            candidates.append(Candidate(position=position, rotation_deg=rotation))

    return candidates


def generate_candidates(context: RoomContext, fill: SlotFill) -> list[Candidate]:
    """Candidate placements for one anchor object."""
    size = fill.size
    placement = fill.slot.placement

    if placement == Placement.WALL_BACK:
        candidates = _wall_candidates(context, size)
    elif placement == Placement.CENTRE:
        candidates = _centre_candidates(context, size)
    elif placement == Placement.CORNER:
        candidates = _corner_candidates(context, size)
    else:
        candidates = _centre_candidates(context, size)

    if not candidates:
        # Nothing legal was found — fall back to the room centre so the object
        # is still placed and the overlap cost can push it somewhere sensible.
        candidates = [Candidate(position=context.centroid, rotation_deg=0.0)]
    return candidates


# --- cost ------------------------------------------------------------------


def _footprint(position: Vec2, size: Size3, rotation: float) -> list[Vec2]:
    return rect_corners(position, size.width, size.depth, rotation)


def placement_cost(
    context: RoomContext,
    fills: list[SlotFill],
    chosen: list[Candidate],
) -> float:
    """Total penalty for one complete arrangement. Lower is better."""
    total = 0.0
    footprints = [
        _footprint(c.position, f.size, c.rotation_deg) for f, c in zip(fills, chosen, strict=True)
    ]

    # Pairwise overlap — the hard constraint.
    for i in range(len(fills)):
        for j in range(i + 1, len(fills)):
            overlap = rect_overlap_area(
                chosen[i].position,
                (fills[i].size.width, fills[i].size.depth),
                chosen[i].rotation_deg,
                chosen[j].position,
                (fills[j].size.width, fills[j].size.depth),
                chosen[j].rotation_deg,
            )
            if overlap > 1e-6:
                total += W_OVERLAP * overlap

    door_keepouts = context.door_keepouts()
    windows = context.window_spans()

    for fill, candidate, corners in zip(fills, chosen, footprints, strict=True):
        size = fill.size

        # Outside the room at all.
        total += W_OUTSIDE * outside_area(context.polygon, corners)

        # Door swing arcs must stay clear, or the door cannot open.
        for centre, radius in door_keepouts:
            distance = min(
                point_to_segment_distance(centre, corners[k], corners[(k + 1) % 4])
                for k in range(4)
            )
            if distance < radius:
                total += W_DOOR_BLOCK * (radius - distance) ** 2

        # Tall objects in front of a window block the light; low ones don't.
        if size.height > 1.0:
            for start, end in windows:
                distance = min(point_to_segment_distance(corner, start, end) for corner in corners)
                if distance < 0.4:
                    total += W_WINDOW_BLOCK * (0.4 - distance) ** 2

        # Circulation in front of seating and sleeping — you have to be able to
        # walk up to a sofa and get out of a bed.
        if fill.slot.role in (ObjectRole.PRIMARY_SEATING, ObjectRole.SLEEPING):
            required = context.settings.min_circulation_m
            front = _front_clearance(context, candidate, size)
            if front < required:
                total += W_CIRCULATION * (required - front) ** 2

        # Wall-backed objects should sit flush, not float.
        if fill.slot.placement == Placement.WALL_BACK and candidate.wall_index is None:
            total += W_WALL_GAP

    # Discourage everything piling into one corner of a large room.
    if len(chosen) > 2:
        centre = Vec2(
            x=sum(c.position.x for c in chosen) / len(chosen),
            y=sum(c.position.y for c in chosen) / len(chosen),
        )
        spread = sum(c.position.distance_to(centre) for c in chosen) / len(chosen)
        ideal = math.sqrt(context.area) / 3.0
        if spread < ideal:
            total += W_CLUSTER * (ideal - spread) ** 2

    return total


def _front_clearance(context: RoomContext, candidate: Candidate, size: Size3) -> float:
    """Free distance directly in front of an object before it meets a wall."""
    from .geometry import facing_direction

    direction = facing_direction(candidate.rotation_deg)
    origin = Vec2(
        x=candidate.position.x + direction.x * size.depth / 2.0,
        y=candidate.position.y + direction.y * size.depth / 2.0,
    )
    best = float("inf")
    for a, b, _, _ in context.edges:
        distance = point_to_segment_distance(origin, a, b)
        best = min(best, distance)
    return best if best != float("inf") else 0.0


# --- search ----------------------------------------------------------------


def solve_anchors(
    context: RoomContext,
    fills: list[SlotFill],
    seed: int,
) -> list[Candidate]:
    """Simulated annealing over candidate assignments.

    Restarts because the landscape is multi-modal — a room has several
    genuinely different good layouts, and a single descent gets stuck in
    whichever basin it started in.
    """
    if not fills:
        return []

    settings = context.settings
    candidate_sets = [generate_candidates(context, fill) for fill in fills]

    best_state: list[int] | None = None
    best_cost = float("inf")

    for restart in range(settings.solver_restarts):
        rng = random.Random(seed + restart * 7919)
        state = [rng.randrange(len(options)) for options in candidate_sets]
        current = placement_cost(
            context, fills, [candidate_sets[i][state[i]] for i in range(len(fills))]
        )

        temperature = max(current, 1.0)
        iterations = settings.solver_iterations
        for _ in range(iterations):
            # Geometric cooling: broad exploration early, refinement late.
            temperature = max(0.01, temperature * 0.999)

            index = rng.randrange(len(fills))
            options = candidate_sets[index]
            if len(options) < 2:
                continue

            previous = state[index]
            state[index] = rng.randrange(len(options))
            if state[index] == previous:
                continue

            proposed = placement_cost(
                context, fills, [candidate_sets[i][state[i]] for i in range(len(fills))]
            )
            delta = proposed - current

            if delta <= 0 or rng.random() < math.exp(-delta / temperature):
                current = proposed
            else:
                state[index] = previous

            if current < 1e-9:
                break

        if current < best_cost:
            best_cost = current
            best_state = list(state)

    assert best_state is not None
    logger.debug("room %s solved with cost %.3f", context.room.id, best_cost)
    return [candidate_sets[i][best_state[i]] for i in range(len(fills))]


# --- derived placement -----------------------------------------------------


#: Placements that are meaningless without something to hang off.
_REQUIRES_ANCHOR = {
    Placement.ADJACENT,
    Placement.SURROUNDING,
    Placement.UNDER,
    Placement.ON_TOP,
}


def clamp_into_room(context: RoomContext, position: Vec2, size: Size3, rotation: float) -> Vec2:
    """Slide a footprint back inside the room if it pokes out.

    Derived placements are computed from an anchor without consulting the room
    boundary, so a nightstand beside a bed that sits near a wall can end up
    half outside. The anchor's relationship is worth preserving, so this moves
    the object the minimum distance toward the centroid rather than
    re-deriving it somewhere else.
    """
    corners = rect_corners(position, size.width, size.depth, rotation)
    if outside_area(context.polygon, corners) < 1e-6:
        return position

    centroid = context.centroid
    toward_centre = Vec2(x=centroid.x - position.x, y=centroid.y - position.y)
    distance = toward_centre.length()
    if distance < 1e-9:
        return position
    step = Vec2(x=toward_centre.x / distance, y=toward_centre.y / distance)

    # Walk inward in small increments and keep the first fully-inside spot.
    travelled = 0.0
    while travelled <= distance:
        travelled += 0.05
        candidate = Vec2(x=position.x + step.x * travelled, y=position.y + step.y * travelled)
        if (
            outside_area(context.polygon, rect_corners(candidate, size.width, size.depth, rotation))
            < 1e-6
        ):
            return candidate

    return centroid  # room is smaller than the object; the cost model reports it


def door_intrusion(context: RoomContext, position: Vec2, size: Size3, rotation: float) -> float:
    """How far a footprint reaches into a door's swing arc. Zero when clear."""
    corners = rect_corners(position, size.width, size.depth, rotation)
    worst = 0.0
    for centre, radius in context.door_keepouts():
        distance = min(
            point_to_segment_distance(centre, corners[i], corners[(i + 1) % 4]) for i in range(4)
        )
        worst = max(worst, radius - distance)
    return worst


def separate_from_doors(
    context: RoomContext, position: Vec2, size: Size3, rotation: float, max_steps: int = 20
) -> Vec2:
    """Push a footprint out of any door swing it landed in.

    Derived objects never pass through the annealing search, so the door-swing
    term in `placement_cost` never sees them. Without this a nightstand derived
    from a bed near a doorway sits squarely in the door's arc — the render
    looks fine and the door cannot open.
    """
    keepouts = context.door_keepouts()
    if not keepouts:
        return position

    current = position
    for _ in range(max_steps):
        corners = rect_corners(current, size.width, size.depth, rotation)

        worst_centre: Vec2 | None = None
        worst_intrusion = 0.0
        for centre, radius in keepouts:
            distance = min(
                point_to_segment_distance(centre, corners[i], corners[(i + 1) % 4])
                for i in range(4)
            )
            if radius - distance > worst_intrusion:
                worst_centre, worst_intrusion = centre, radius - distance

        if worst_centre is None:
            return current

        away = Vec2(x=current.x - worst_centre.x, y=current.y - worst_centre.y)
        length = away.length()
        if length < 1e-9:
            away, length = Vec2(x=1.0, y=0.0), 1.0

        stepped = Vec2(x=current.x + away.x / length * 0.08, y=current.y + away.y / length * 0.08)
        if (
            outside_area(context.polygon, rect_corners(stepped, size.width, size.depth, rotation))
            > 1e-6
        ):
            return current  # would leave the room; caller decides whether to drop
        current = stepped

    return current


def separate_from_placed(
    context: RoomContext,
    position: Vec2,
    size: Size3,
    rotation: float,
    placed: list[PlacedObject],
    max_steps: int = 24,
) -> Vec2:
    """Nudge a derived object out of anything it landed on top of.

    Derived positions come from an anchor relationship and never went through
    the annealing search, so they don't participate in its overlap cost. Left
    alone, a nightstand clamped away from a wall can end up inside the
    wardrobe. This pushes it along the axis of deepest penetration in small
    steps, re-checking the room boundary each time so escaping one object
    can't push it through a wall.
    """
    floor_objects = [
        obj
        for obj in placed
        if obj.position_m.z < FLOOR_Z_TOLERANCE and obj.category != ProductCategory.RUG
    ]
    if not floor_objects:
        return position

    current = position
    for _ in range(max_steps):
        worst: PlacedObject | None = None
        worst_overlap = 1e-6
        for obj in floor_objects:
            overlap = rect_overlap_area(
                current,
                (size.width, size.depth),
                rotation,
                Vec2(x=obj.position_m.x, y=obj.position_m.y),
                (obj.size_m.width, obj.size_m.depth),
                obj.rotation_deg,
            )
            if overlap > worst_overlap:
                worst, worst_overlap = obj, overlap

        if worst is None:
            return current

        away = Vec2(x=current.x - worst.position_m.x, y=current.y - worst.position_m.y)
        length = away.length()
        if length < 1e-9:
            away, length = Vec2(x=1.0, y=0.0), 1.0

        stepped = Vec2(x=current.x + away.x / length * 0.06, y=current.y + away.y / length * 0.06)
        if (
            outside_area(context.polygon, rect_corners(stepped, size.width, size.depth, rotation))
            > 1e-6
        ):
            return current  # cornered — leave it and let the cost model report it
        current = stepped

    return current


def derive_placement(
    context: RoomContext,
    fill: SlotFill,
    anchor: PlacedObject | None,
    unit_index: int,
) -> tuple[Vec3, float]:
    """Position a dependent object from its anchor. Fully deterministic."""
    from .geometry import facing_direction

    size = fill.size
    placement = fill.slot.placement
    ceiling = context.room.ceiling_height_m

    # Ceiling, wall-mount and window placements are positioned by the *room*,
    # not by another object — they must still resolve when no anchor exists.
    # Only the genuinely relative placements need one.
    if anchor is None and placement in _REQUIRES_ANCHOR:
        return Vec3.from_vec2(context.centroid, 0.0), 0.0

    if anchor is not None:
        anchor_pos = Vec2(x=anchor.position_m.x, y=anchor.position_m.y)
        front = facing_direction(anchor.rotation_deg)
    else:
        anchor_pos = context.centroid
        front = Vec2(x=0.0, y=1.0)
    right = Vec2(x=front.y, y=-front.x)  # anchor's right-hand side

    if placement == Placement.ADJACENT:
        # Alternate sides so a pair of nightstands brackets the bed.
        side = 1 if unit_index % 2 == 0 else -1
        if fill.slot.category == ProductCategory.COFFEE_TABLE:
            # In front of the sofa, at conversational distance.
            gap = 0.45 + anchor.size_m.depth / 2.0 + size.depth / 2.0
            position = Vec2(x=anchor_pos.x + front.x * gap, y=anchor_pos.y + front.y * gap)
            return Vec3.from_vec2(position, 0.0), anchor.rotation_deg
        offset = anchor.size_m.width / 2.0 + size.width / 2.0 + 0.08
        position = Vec2(
            x=anchor_pos.x + right.x * offset * side,
            y=anchor_pos.y + right.y * offset * side,
        )
        return Vec3.from_vec2(position, 0.0), anchor.rotation_deg

    if placement == Placement.SURROUNDING:
        # Chairs arrayed around a table, each turned to face it.
        count = max(fill.slot.quantity, 1)
        angle = (2.0 * math.pi * unit_index) / count
        radius = max(anchor.size_m.width, anchor.size_m.depth) / 2.0 + size.depth / 2.0 + 0.10
        position = Vec2(
            x=anchor_pos.x + math.cos(angle) * radius,
            y=anchor_pos.y + math.sin(angle) * radius,
        )
        toward_table = Vec2(x=anchor_pos.x - position.x, y=anchor_pos.y - position.y)
        return Vec3.from_vec2(position, 0.0), rotation_to_face(toward_table)

    if placement == Placement.UNDER:
        # A rug sits under the front edge of a sofa, not centred on it.
        shift = anchor.size_m.depth * 0.35
        position = Vec2(x=anchor_pos.x + front.x * shift, y=anchor_pos.y + front.y * shift)
        return Vec3.from_vec2(position, 0.0), anchor.rotation_deg

    if placement == Placement.ON_TOP:
        return (
            Vec3(x=anchor_pos.x, y=anchor_pos.y, z=anchor.top_z),
            anchor.rotation_deg,
        )

    if placement == Placement.CEILING:
        return Vec3(x=anchor_pos.x, y=anchor_pos.y, z=ceiling - size.height), 0.0

    if placement == Placement.WALL_MOUNT:
        # Centred on the longest wall, at eye height.
        longest = max(context.edges, key=lambda e: e[3])
        a, b, normal, _ = longest
        midpoint = Vec2(x=(a.x + b.x) / 2.0, y=(a.y + b.y) / 2.0)
        position = Vec2(x=midpoint.x + normal.x * 0.03, y=midpoint.y + normal.y * 0.03)
        return Vec3(x=position.x, y=position.y, z=1.45), rotation_to_face(normal)

    if placement == Placement.WINDOW:
        windows = context.window_spans()
        if windows:
            start, end = windows[unit_index % len(windows)]

            centre = Vec2(x=(start.x + end.x) / 2.0, y=(start.y + end.y) / 2.0)
            span = Vec2(x=end.x - start.x, y=end.y - start.y)
            # Offset left/right so a pair of curtains flanks the opening.
            length = span.length() or 1.0
            side = 1 if unit_index % 2 == 0 else -1
            position = Vec2(
                x=centre.x + span.x / length * (length / 2.0 + 0.15) * side,
                y=centre.y + span.y / length * (length / 2.0 + 0.15) * side,
            )
            return Vec3(x=position.x, y=position.y, z=0.0), 0.0

    return Vec3.from_vec2(context.centroid, 0.0), 0.0


# --- entry point -----------------------------------------------------------


@dataclass
class PlacementResult:
    """What the solver managed to place, and what it had to give up on."""

    objects: list[PlacedObject] = field(default_factory=list)
    dropped: list[str] = field(default_factory=list)


#: Overlap above this is a real collision rather than a rounding artefact.
MAX_TOLERATED_OVERLAP_M2 = 0.02


def solve_room(
    room: Room,
    openings: list[Opening],
    fills: list[SlotFill],
    seed: int,
    settings: Settings | None = None,
) -> PlacementResult:
    """Place every object in one room. Deterministic for a given seed."""
    settings = settings or get_settings()
    context = RoomContext(room=room, openings=openings, settings=settings)

    anchors = [f for f in fills if not f.slot.placement.is_derived]
    dependents = [f for f in fills if f.slot.placement.is_derived]

    chosen = solve_anchors(context, anchors, seed)

    placed: list[PlacedObject] = []
    dropped: list[str] = []
    by_slot: dict[str, PlacedObject] = {}
    # Priority travels with the object so the final overlap pass knows which of
    # a colliding pair matters less.
    priority: dict[str, int] = {f.instance_id: f.slot.priority for f in fills}

    for fill, candidate in zip(anchors, chosen, strict=True):
        obj = _to_placed(
            fill, Vec3.from_vec2(candidate.position, 0.0), candidate.rotation_deg, seed
        )
        placed.append(obj)
        by_slot.setdefault(fill.slot.slot_id, obj)

    # Dependents in priority order, so a chain (bed → nightstand → lamp)
    # resolves with each link already positioned.
    for fill in sorted(dependents, key=lambda f: f.slot.priority):
        # Curtains with no window, and anything else bound to a feature the
        # room doesn't have, are meaningless — drop rather than dump at the
        # room centre, which is what produced piles of overlapping curtains.
        if fill.slot.placement == Placement.WINDOW and not context.window_spans():
            dropped.append(
                f"{fill.instance_id} ({fill.slot.category.value}): {room.name} has no window"
            )
            continue

        anchor = by_slot.get(fill.slot.anchor_slot_id or "")
        position, rotation = derive_placement(context, fill, anchor, fill.index)

        # Floor-standing derived objects must stay in the room; ceiling and
        # wall-mounted ones are already bound to a surface by construction.
        if fill.slot.placement in {Placement.ADJACENT, Placement.SURROUNDING, Placement.UNDER}:
            planar = clamp_into_room(context, Vec2(x=position.x, y=position.y), fill.size, rotation)
            # A rug is *supposed* to sit under the furniture it anchors to, so
            # it is exempt from both separation and the collision check.
            # Everything else has to claim its own patch of floor.
            if fill.slot.placement != Placement.UNDER:
                planar = separate_from_doors(context, planar, fill.size, rotation)
                planar = separate_from_placed(context, planar, fill.size, rotation, placed)

                # If separation found no clear floor, the object genuinely has
                # nowhere to go — a second nightstand wedged between the bed
                # and the wall, say. Dropping it and saying so beats rendering
                # two objects in the same cubic metre.
                residual = _worst_overlap(planar, fill.size, rotation, placed)
                if residual > MAX_TOLERATED_OVERLAP_M2:
                    dropped.append(
                        f"{fill.instance_id} ({fill.slot.category.value}): no clear floor space "
                        f"in {room.name} (would overlap by {residual:.2f} m²)"
                    )
                    continue

                intrusion = door_intrusion(context, planar, fill.size, rotation)
                if intrusion > 0.25:
                    dropped.append(
                        f"{fill.instance_id} ({fill.slot.category.value}): would block a "
                        f"doorway in {room.name} and has nowhere else to go"
                    )
                    continue

            position = Vec3(x=planar.x, y=planar.y, z=position.z)

        obj = _to_placed(fill, position, rotation, seed)
        placed.append(obj)
        by_slot.setdefault(fill.slot.slot_id, obj)

    survivors, evicted = enforce_no_overlap(placed, priority, room.name)
    return PlacementResult(objects=survivors, dropped=dropped + evicted)


def enforce_no_overlap(
    objects: list[PlacedObject], priority: dict[str, int], room_name: str
) -> tuple[list[PlacedObject], list[str]]:
    """Guarantee no two floor objects intersect, dropping the loser of each clash.

    Annealing minimizes overlap but does not eliminate it — in a tight room
    there may be no arrangement without a collision, and two objects assigned
    the same corner can survive the search. Rather than emit a scene the
    renderer would faithfully draw as interpenetrating furniture, the
    lower-priority object is removed and reported. A missing plant is a much
    smaller failure than a plant growing through a floor lamp.

    Rugs and anything resting on a surface are exempt: they are *meant* to
    share floor space with what they sit under or on.
    """
    survivors = list(objects)
    evicted: list[str] = []

    while True:
        floor_objects = [
            obj
            for obj in survivors
            if obj.position_m.z < FLOOR_Z_TOLERANCE and obj.category != ProductCategory.RUG
        ]

        worst_pair: tuple[PlacedObject, PlacedObject] | None = None
        worst_overlap = MAX_TOLERATED_OVERLAP_M2
        for i, a in enumerate(floor_objects):
            for b in floor_objects[i + 1 :]:
                overlap = rect_overlap_area(
                    Vec2(x=a.position_m.x, y=a.position_m.y),
                    (a.size_m.width, a.size_m.depth),
                    a.rotation_deg,
                    Vec2(x=b.position_m.x, y=b.position_m.y),
                    (b.size_m.width, b.size_m.depth),
                    b.rotation_deg,
                )
                if overlap > worst_overlap:
                    worst_pair, worst_overlap = (a, b), overlap

        if worst_pair is None:
            return survivors, evicted

        a, b = worst_pair
        # Higher priority number means less important; ties break on the larger
        # object surviving, since it is usually the one the room is built around.
        loser = max(
            (a, b),
            key=lambda o: (
                priority.get(o.instance_id, 50),
                -o.size_m.footprint_area(),
                o.instance_id,
            ),
        )
        survivors = [obj for obj in survivors if obj.instance_id != loser.instance_id]
        # Anything resting on the evicted object goes with it.
        orphans = [
            o
            for o in survivors
            if o.position_m.z > FLOOR_Z_TOLERANCE
            and o.instance_id.rsplit("#", 1)[0] != loser.instance_id.rsplit("#", 1)[0]
            and _rests_on(o, loser)
        ]
        for orphan in orphans:
            survivors = [obj for obj in survivors if obj.instance_id != orphan.instance_id]
            evicted.append(
                f"{orphan.instance_id} ({orphan.category.value}): removed with the "
                f"{loser.category.value} it stood on"
            )
        evicted.append(
            f"{loser.instance_id} ({loser.category.value}): overlapped the "
            f"{(b if loser is a else a).category.value} in {room_name} by {worst_overlap:.2f} m²"
        )


def _rests_on(candidate: PlacedObject, support: PlacedObject) -> bool:
    """Whether `candidate` sits on `support`'s top surface."""
    if abs(candidate.position_m.z - support.top_z) > 0.05:
        return False
    return (
        rect_overlap_area(
            Vec2(x=candidate.position_m.x, y=candidate.position_m.y),
            (candidate.size_m.width, candidate.size_m.depth),
            candidate.rotation_deg,
            Vec2(x=support.position_m.x, y=support.position_m.y),
            (support.size_m.width, support.size_m.depth),
            support.rotation_deg,
        )
        > 1e-6
    )


def _worst_overlap(
    position: Vec2, size: Size3, rotation: float, placed: list[PlacedObject]
) -> float:
    """Largest overlap between a proposed footprint and anything already placed."""
    return max(
        (
            rect_overlap_area(
                position,
                (size.width, size.depth),
                rotation,
                Vec2(x=obj.position_m.x, y=obj.position_m.y),
                (obj.size_m.width, obj.size_m.depth),
                obj.rotation_deg,
            )
            for obj in placed
            if obj.position_m.z < FLOOR_Z_TOLERANCE and obj.category != ProductCategory.RUG
        ),
        default=0.0,
    )


def _to_placed(fill: SlotFill, position: Vec3, rotation: float, seed: int) -> PlacedObject:
    product = fill.product
    return PlacedObject(
        instance_id=fill.instance_id,
        product_id=product.id,
        room_id=fill.slot.room_id,
        role=fill.slot.role,
        category=product.category,
        position_m=position,
        rotation_deg=round(rotation % 360.0, 2),
        size_m=fill.size,
        color=product.primary_color or "natural",
        material=product.materials[0] if product.materials else None,
        display_name=product.name,
        # Derived from the scene seed and the instance's stable identity, so a
        # regeneration reproduces it and an unrelated edit elsewhere cannot
        # shift this object's appearance.
        seed=(seed * 31 + hash(fill.instance_id)) % (2**31),
    )
