"""Camera rig — deterministic viewpoints, stored in the scene graph.

Cameras are *scene state*, not render parameters. That distinction is what
makes "regenerate, same scene" mean something: if viewpoints were chosen at
render time, two runs of the same scene would frame it differently and every
downstream comparison — the consistency judge, the user's before/after — would
be measuring camera drift rather than scene drift.

Placement follows how a room is actually photographed: stand in a corner, back
to the wall, and look at the furniture. That maximizes how much of the room
lands in frame and puts the primary furniture group in the middle of it.
"""

from __future__ import annotations

import logging
import math

from ..config import Settings, get_settings
from ..schemas.common import Vec2, Vec3, point_in_polygon, polygon_bounds, polygon_centroid
from ..schemas.floorplan import Room
from ..schemas.scene import Camera, PlacedObject

logger = logging.getLogger(__name__)

#: Sampling resolution for the coverage estimate, per axis.
COVERAGE_SAMPLES = 24


def _look_target(room: Room, objects: list[PlacedObject]) -> Vec2:
    """What the camera should aim at.

    The centroid of the furniture weighted by footprint, so a room reads as
    "sofa and coffee table, framed" rather than "an empty corner, centred".
    Falls back to the room centroid in an unfurnished room.
    """
    floor_objects = [
        obj for obj in objects if obj.position_m.z < 0.6 and obj.size_m.footprint_area() > 0.1
    ]
    if not floor_objects:
        return polygon_centroid(room.polygon_m)

    total_weight = sum(obj.size_m.footprint_area() for obj in floor_objects)
    if total_weight <= 0:
        return polygon_centroid(room.polygon_m)

    return Vec2(
        x=sum(o.position_m.x * o.size_m.footprint_area() for o in floor_objects) / total_weight,
        y=sum(o.position_m.y * o.size_m.footprint_area() for o in floor_objects) / total_weight,
    )


#: A camera needs standing room. Eyes closer than this to a piece of furniture
#: end up inside it, which fills the frame with one object's back face.
#: Tried in order — a small, densely furnished bedroom has almost no floor left
#: at the widest setting, and one cramped viewpoint beats none at all.
PERSONAL_SPACE_STEPS = (0.45, 0.25, 0.10, 0.0)


def _is_standable(eye: Vec2, objects: list[PlacedObject], margin: float) -> bool:
    """Whether a person could actually stand here.

    Corner positions are the best viewpoints and are also exactly where
    wardrobes, plants and floor lamps get placed. Without this check the
    camera ends up inside a cabinet, and the render is a close-up of its side
    panel — geometrically valid and completely useless.
    """
    from ..design.geometry import rect_corners

    for obj in objects:
        if obj.position_m.z > 1.2:
            continue  # a pendant overhead is not in the way
        corners = rect_corners(
            Vec2(x=obj.position_m.x, y=obj.position_m.y),
            obj.size_m.width + margin * 2,
            obj.size_m.depth + margin * 2,
            obj.rotation_deg,
        )
        if point_in_polygon(eye, corners):
            return False
    return True


def _candidate_eyes(room: Room, inset: float) -> list[tuple[Vec2, str]]:
    """Standing positions: room corners first, then wall midpoints.

    Corners see the most of a room. Wall midpoints are the fallback for rooms
    whose corners are too tight to stand in, and give a second, genuinely
    different angle in rooms that need more than four viewpoints.
    """
    polygon = room.polygon_m
    centroid = polygon_centroid(polygon)
    candidates: list[tuple[Vec2, str]] = []

    def anchors() -> list[tuple[Vec2, str]]:
        points: list[tuple[Vec2, str]] = [
            (vertex, f"from corner {i + 1}") for i, vertex in enumerate(polygon)
        ]
        for i in range(len(polygon)):
            a, b = polygon[i], polygon[(i + 1) % len(polygon)]
            points.append(
                (
                    Vec2(x=(a.x + b.x) / 2.0, y=(a.y + b.y) / 2.0),
                    f"from the {_wall_label(i)} wall",
                )
            )
        return points

    for anchor, label in anchors():
        toward_centre = Vec2(x=centroid.x - anchor.x, y=centroid.y - anchor.y)
        length = toward_centre.length() or 1.0
        direction = Vec2(x=toward_centre.x / length, y=toward_centre.y / length)

        # Step further in when the first position is unusable, so a corner
        # occupied by a wardrobe still yields a viewpoint looking over it
        # rather than being discarded outright.
        for step in (0.0, 0.45, 0.9, 1.4):
            distance = inset + step
            if distance >= length:
                break
            eye = Vec2(x=anchor.x + direction.x * distance, y=anchor.y + direction.y * distance)
            if point_in_polygon(eye, polygon):
                candidates.append((eye, label))

    return candidates


def _wall_label(index: int) -> str:
    return ["south", "east", "north", "west"][index % 4]


def estimate_coverage(
    room: Room,
    eye: Vec2,
    target: Vec2,
    fov_deg: float,
    aspect: float,
) -> float:
    """Fraction of the room's floor that falls inside the camera frustum.

    Sampled rather than solved: the exact answer is a polygon-frustum clip in
    3D, and a grid estimate is accurate enough to *rank* viewpoints, which is
    all this is used for.
    """
    polygon = room.polygon_m
    low, high = polygon_bounds(polygon)

    forward = Vec2(x=target.x - eye.x, y=target.y - eye.y)
    length = forward.length()
    if length < 1e-6:
        return 0.0
    forward = Vec2(x=forward.x / length, y=forward.y / length)
    right = Vec2(x=forward.y, y=-forward.x)

    # Floor coverage is bounded by the horizontal fan; the vertical FOV only
    # decides how much wall and ceiling comes with it.
    half_h = math.radians(fov_deg) / 2.0

    inside = 0
    total = 0
    for i in range(COVERAGE_SAMPLES):
        for j in range(COVERAGE_SAMPLES):
            point = Vec2(
                x=low.x + (high.x - low.x) * (i + 0.5) / COVERAGE_SAMPLES,
                y=low.y + (high.y - low.y) * (j + 0.5) / COVERAGE_SAMPLES,
            )
            if not point_in_polygon(point, polygon):
                continue
            total += 1

            offset = Vec2(x=point.x - eye.x, y=point.y - eye.y)
            depth = offset.x * forward.x + offset.y * forward.y
            if depth <= 0.1:
                continue  # behind the camera
            lateral = abs(offset.x * right.x + offset.y * right.y)
            if lateral / depth <= math.tan(half_h):
                inside += 1

    return inside / total if total else 0.0


#: Resolution used to score candidate viewpoints. Small enough that trying a
#: dozen is cheap, large enough that occlusion is measured accurately.
SCORING_WIDTH = 160
SCORING_HEIGHT = 120


def _score_viewpoint(
    room: Room,
    objects: list[PlacedObject],
    eye: Vec2,
    target: Vec3,
    settings: Settings,
    floor_coverage: float,
) -> float:
    """Rank a viewpoint by rendering it, not by approximating it.

    A 2-D floor-coverage estimate cannot see occlusion: a camera with a
    wardrobe half a metre in front of it scores identically to one with a clear
    view of the same floor. Rasterizing the candidate at thumbnail resolution
    measures what actually reaches the frame, which is the thing being ranked.

    Three signals, in order of weight:

    * **how many objects are visible** — the render should show the room's
      furniture, which is the entire point of the design;
    * **whether one object dominates** — a frame that is 60% wardrobe is a
      photograph of a wardrobe, not of a room;
    * **floor coverage** — the cheap 2-D estimate, as a tiebreaker.
    """
    from .raster import rasterize

    if not objects:
        return floor_coverage

    probe = Camera(
        id="probe",
        room_id=room.id,
        position_m=Vec3(x=eye.x, y=eye.y, z=settings.camera_height_m),
        look_at_m=target,
        fov_deg=settings.camera_fov_deg,
    )
    buffers = rasterize(room, objects, probe, SCORING_WIDTH, SCORING_HEIGHT)

    total_pixels = SCORING_WIDTH * SCORING_HEIGHT
    visible = buffers.visible_instances(min_pixels=8)
    visible_fraction = len(visible) / len(objects)

    dominance = max(visible.values(), default=0) / total_pixels
    # Tolerate up to a third of the frame; penalize hard beyond that.
    crowding_penalty = max(0.0, dominance - 0.33) * 2.0

    return round(
        0.55 * visible_fraction
        + 0.25 * floor_coverage
        - crowding_penalty
        + 0.20 * min(sum(visible.values()) / total_pixels, 0.5) * 2,
        4,
    )


def place_cameras(
    room: Room,
    objects: list[PlacedObject],
    count: int,
    settings: Settings | None = None,
) -> list[Camera]:
    """Choose `count` viewpoints for one room, best coverage first.

    Deterministic: the same room and the same objects always yield the same
    cameras, in the same order.
    """
    settings = settings or get_settings()
    target_2d = _look_target(room, objects)
    aspect = settings.render_width / settings.render_height

    eye_height = min(settings.camera_height_m, room.ceiling_height_m - 0.3)
    # Aim slightly below eye level so the floor and furniture dominate the
    # frame rather than blank ceiling.
    target = Vec3(x=target_2d.x, y=target_2d.y, z=eye_height * 0.62)

    candidates = _candidate_eyes(room, settings.camera_corner_inset_m)

    # Relax the standing-room requirement until viewpoints appear. A small
    # densely furnished bedroom has almost no floor left at the widest margin,
    # and a cramped viewpoint is far better than the centroid fallback — which
    # may well be inside the bed.
    scored: list[tuple[float, Vec2, str]] = []
    for margin in PERSONAL_SPACE_STEPS:
        scored = []
        seen_labels: set[str] = set()
        for eye, label in candidates:
            if eye.distance_to(target_2d) < 1.2:
                continue  # too close to frame anything but one object
            if not _is_standable(eye, objects, margin):
                continue
            if label in seen_labels:
                continue  # the first workable step-back per anchor is enough
            seen_labels.add(label)

            floor_coverage = estimate_coverage(
                room, eye, target_2d, settings.camera_fov_deg, aspect
            )
            scored.append(
                (
                    _score_viewpoint(room, objects, eye, target, settings, floor_coverage),
                    eye,
                    label,
                )
            )

        if len(scored) >= count:
            break
        if scored and margin <= PERSONAL_SPACE_STEPS[-2]:
            break  # something is better than nothing; stop relaxing

    if margin < PERSONAL_SPACE_STEPS[0] and scored:
        logger.info(
            "room %s: relaxed camera standing room to %.2f m to find %d viewpoint(s)",
            room.id,
            margin,
            len(scored),
        )

    scored.sort(key=lambda item: (-item[0], item[2]))

    cameras: list[Camera] = []
    for score, eye, label in scored:
        if len(cameras) >= count:
            break
        if score < settings.min_camera_score and cameras:
            break  # keep at least one camera even in an awkward room

        # Skip viewpoints too close to one already chosen — two cameras a
        # handspring apart produce near-identical images, which is the opposite
        # of what "different viewpoints" is asking for.
        if any(eye.distance_to(Vec2(x=c.position_m.x, y=c.position_m.y)) < 1.0 for c in cameras):
            continue

        cameras.append(
            Camera(
                id=f"{room.id}-cam{len(cameras) + 1}",
                room_id=room.id,
                position_m=Vec3(x=eye.x, y=eye.y, z=eye_height),
                look_at_m=target,
                fov_deg=settings.camera_fov_deg,
                label=label,
                coverage_pct=round(score, 3),
            )
        )

    if not cameras:
        # Degenerate room (very small, or a bad polygon): fall back to a single
        # camera at the centroid so the pipeline still produces something.
        centroid = polygon_centroid(room.polygon_m)
        logger.warning("room %s: no viable camera position, using centroid", room.id)
        cameras.append(
            Camera(
                id=f"{room.id}-cam1",
                room_id=room.id,
                position_m=Vec3(x=centroid.x, y=centroid.y, z=eye_height),
                look_at_m=target,
                fov_deg=settings.camera_fov_deg,
                label="room centre",
                coverage_pct=0.0,
            )
        )

    return cameras
