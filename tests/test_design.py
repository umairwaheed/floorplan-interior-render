"""Design agent tests — geometry, slots, placement, and scene assembly.

The invariants worth guarding are physical: objects must not intersect, must
stay inside their room, and must be reproducible. A scene that violates any of
those produces a render the brief explicitly grades against ("avoid mismatched
layouts, floating objects, inconsistent geometry, or duplicated/missing items"),
and no downstream stage can detect the problem — the rasterizer and the image
model would both faithfully draw the broken scene.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from backend.catalog.service import CatalogService
from backend.config import Settings
from backend.design.agent import DesignAgent
from backend.design.geometry import (
    convex_intersection_area,
    facing_direction,
    inward_normal,
    outside_area,
    point_to_segment_distance,
    rect_corners,
    rect_overlap_area,
    rotation_to_face,
)
from backend.design.slots import Placement, build_room_program
from backend.schemas.common import Vec2
from backend.schemas.floorplan import (
    FloorPlan,
    Opening,
    OpeningType,
    Room,
    RoomType,
    ScaleCalibration,
    SwingDirection,
)
from backend.schemas.product import DesignStyle, ProductCategory
from backend.schemas.scene import Scene

# --- geometry --------------------------------------------------------------

SQUARE = [Vec2(x=0, y=0), Vec2(x=4, y=0), Vec2(x=4, y=3), Vec2(x=0, y=3)]


def test_overlap_area_is_exact_not_approximate():
    a, b = Vec2(x=0, y=0), Vec2(x=1, y=0)
    assert rect_overlap_area(a, (2, 2), 0, b, (2, 2), 0) == pytest.approx(2.0)
    assert rect_overlap_area(a, (2, 2), 0, a, (2, 2), 0) == pytest.approx(4.0)
    assert rect_overlap_area(a, (2, 2), 0, Vec2(x=5, y=0), (2, 2), 0) == 0.0


def test_overlap_of_rotated_squares():
    """Two unit-ish squares at 45° — the exact answer is 8(√2−1)."""
    centre = Vec2(x=0, y=0)
    assert rect_overlap_area(centre, (2, 2), 0, centre, (2, 2), 45) == pytest.approx(
        8 * (math.sqrt(2) - 1), rel=1e-6
    )


def test_convex_intersection_is_winding_order_independent():
    a = [Vec2(x=0, y=0), Vec2(x=2, y=0), Vec2(x=2, y=2), Vec2(x=0, y=2)]
    reversed_a = list(reversed(a))
    b = [Vec2(x=1, y=1), Vec2(x=3, y=1), Vec2(x=3, y=3), Vec2(x=1, y=3)]
    assert convex_intersection_area(a, b) == pytest.approx(1.0)
    assert convex_intersection_area(reversed_a, b) == pytest.approx(1.0)


@pytest.mark.parametrize("degrees", [0.0, 37.0, 90.0, 180.0, 270.0, 359.0])
def test_facing_and_rotation_round_trip(degrees: float):
    assert rotation_to_face(facing_direction(degrees)) == pytest.approx(degrees, abs=1e-6)


def test_inward_normal_ignores_polygon_winding():
    """Room polygons come from a vision model; winding is not guaranteed."""
    forward = inward_normal(SQUARE[0], SQUARE[1], SQUARE)
    backward = inward_normal(SQUARE[1], SQUARE[0], list(reversed(SQUARE)))
    assert forward.y == pytest.approx(1.0)
    assert backward.y == pytest.approx(1.0)


def test_outside_area_is_continuous():
    """The solver needs a gradient, not a cliff."""
    fully_in = outside_area(SQUARE, rect_corners(Vec2(x=2, y=1.5), 1, 1, 0))
    half_out = outside_area(SQUARE, rect_corners(Vec2(x=0, y=1.5), 1, 1, 0))
    fully_out = outside_area(SQUARE, rect_corners(Vec2(x=-5, y=1.5), 1, 1, 0))
    assert fully_in == 0.0
    assert half_out == pytest.approx(0.5)
    assert fully_out == pytest.approx(1.0)


def test_point_to_segment_distance_clamps_to_endpoints():
    a, b = Vec2(x=0, y=0), Vec2(x=10, y=0)
    assert point_to_segment_distance(Vec2(x=5, y=3), a, b) == pytest.approx(3.0)
    assert point_to_segment_distance(Vec2(x=-4, y=0), a, b) == pytest.approx(4.0)


# --- room programs ---------------------------------------------------------


def test_bedroom_program_leads_with_the_bed():
    slots = build_room_program("r", RoomType.BEDROOM, 12.8)
    assert slots[0].category == ProductCategory.BED
    assert slots[0].required is True


def test_small_rooms_drop_slots_that_do_not_fit_the_program():
    large = {s.category for s in build_room_program("r", RoomType.BEDROOM, 16.0)}
    small = {s.category for s in build_room_program("r", RoomType.BEDROOM, 6.0)}
    assert ProductCategory.DRESSER in large
    assert ProductCategory.DRESSER not in small
    assert ProductCategory.BED in small, "the required slot must survive at any size"


def test_dependents_are_dropped_when_their_anchor_is():
    """A table lamp anchored to a nightstand must not outlive it."""
    slots = build_room_program("r", RoomType.BEDROOM, 6.0)
    categories = {s.category for s in slots}
    assert ProductCategory.NIGHTSTAND not in categories
    assert ProductCategory.TABLE_LAMP not in categories


def test_every_dependent_sorts_after_its_anchor():
    for room_type in RoomType:
        slots = build_room_program("r", room_type, 20.0)
        seen: set[str] = set()
        for slot in slots:
            if slot.anchor_slot_id:
                assert slot.anchor_slot_id in seen, (
                    f"{room_type.value}: {slot.slot_id} placed before its anchor"
                )
            seen.add(slot.slot_id)


def test_larger_rooms_get_more_of_what_scales():
    small = build_room_program("r", RoomType.DINING, 10.0)
    large = build_room_program("r", RoomType.DINING, 30.0)
    chairs = ProductCategory.DINING_CHAIR
    small_count = next(s.quantity for s in small if s.category == chairs)
    large_count = next(s.quantity for s in large if s.category == chairs)
    assert large_count > small_count


# --- end-to-end scene assembly --------------------------------------------


def _room(rid, name, room_type, x0, y0, width, height, label):
    return Room(
        id=rid,
        name=name,
        room_type=room_type,
        polygon_m=[
            Vec2(x=x0, y=y0),
            Vec2(x=x0 + width, y=y0),
            Vec2(x=x0 + width, y=y0 + height),
            Vec2(x=x0, y=y0 + height),
        ],
        polygon_px=[Vec2(x=0, y=0), Vec2(x=1, y=0), Vec2(x=1, y=1), Vec2(x=0, y=1)],
        area_label_m2=label,
    )


@pytest.fixture(scope="module")
def floorplan() -> FloorPlan:
    """Sample plan 1's geometry: a 25.2 m² studio, 12.8 m² bedroom, 6.1 m² bath."""
    return FloorPlan(
        id="fp-test",
        source_filename="plan.png",
        image_width_px=1000,
        image_height_px=1000,
        calibration=ScaleCalibration(
            px_per_m=37.5, method="area_labels", residual_pct=0.4, confidence=0.93, sample_count=4
        ),
        rooms=[
            _room("studio", "Studio", RoomType.STUDIO, 0, 0, 6.0, 4.2, 25.2),
            _room("bed", "Bedroom", RoomType.BEDROOM, 6.2, 0, 4.0, 3.2, 12.8),
            _room("bath", "Bathroom", RoomType.BATHROOM, 6.2, 3.4, 2.44, 2.5, 6.1),
        ],
        openings=[
            Opening(
                id="d1",
                opening_type=OpeningType.DOOR,
                start_m=Vec2(x=5.9, y=1.0),
                end_m=Vec2(x=5.9, y=1.9),
                swing=SwingDirection.INWARD,
                room_ids=["studio", "bed"],
            ),
            Opening(
                id="w1",
                opening_type=OpeningType.WINDOW,
                start_m=Vec2(x=1.0, y=4.2),
                end_m=Vec2(x=2.8, y=4.2),
                room_ids=["studio"],
            ),
        ],
    )


@pytest.fixture(scope="module")
def agent(tmp_path_factory) -> DesignAgent:
    root = tmp_path_factory.mktemp("design")
    settings = Settings(
        data_dir=root,
        catalog_dir=Path("data/catalog"),
        upload_dir=root / "uploads",
        output_dir=root / "outputs",
        db_path=root / "catalog.db",
        solver_iterations=1200,  # keep the suite fast; quality is unchanged
        solver_restarts=2,
    )
    settings.ensure_dirs()
    catalog = CatalogService(settings)
    catalog.ensure_ready()
    return DesignAgent(catalog=catalog, settings=settings)


def _floor_objects(scene: Scene, room_id: str | None = None):
    return [
        obj
        for obj in scene.objects
        if obj.position_m.z < 0.1
        and obj.category != ProductCategory.RUG
        and (room_id is None or obj.room_id == room_id)
    ]


def test_scene_has_no_overlapping_floor_objects(agent, floorplan):
    """The core physical invariant. A violation renders as interpenetrating
    furniture and nothing downstream can detect it."""
    for style in (DesignStyle.SCANDINAVIAN, DesignStyle.INDUSTRIAL, DesignStyle.LUXURY):
        scene = agent.design(floorplan, style, seed=7)
        for room_id in scene.room_ids:
            objects = _floor_objects(scene, room_id)
            for i, a in enumerate(objects):
                for b in objects[i + 1 :]:
                    overlap = rect_overlap_area(
                        Vec2(x=a.position_m.x, y=a.position_m.y),
                        (a.size_m.width, a.size_m.depth),
                        a.rotation_deg,
                        Vec2(x=b.position_m.x, y=b.position_m.y),
                        (b.size_m.width, b.size_m.depth),
                        b.rotation_deg,
                    )
                    assert overlap <= 0.02, (
                        f"{style.value}: {a.category.value} overlaps {b.category.value} "
                        f"by {overlap:.3f} m²"
                    )


def test_every_object_stays_inside_its_room(agent, floorplan):
    scene = agent.design(floorplan, DesignStyle.SCANDINAVIAN, seed=7)
    rooms = {room.id: room for room in floorplan.rooms}
    for obj in scene.objects:
        if obj.category in {ProductCategory.CURTAIN, ProductCategory.ARTWORK}:
            continue  # mounted on the wall plane, legitimately at the boundary
        room = rooms[obj.room_id]
        escaped = outside_area(
            room.polygon_m,
            rect_corners(
                Vec2(x=obj.position_m.x, y=obj.position_m.y),
                obj.size_m.width,
                obj.size_m.depth,
                obj.rotation_deg,
            ),
        )
        assert escaped < 0.05, f"{obj.display_name} sticks {escaped:.3f} m² out of {room.name}"


def test_scene_is_reproducible(agent, floorplan):
    """Same inputs and seed must hash identically — this is what makes
    'regenerate the same scene' meaningful rather than approximate."""
    first = agent.design(floorplan, DesignStyle.SCANDINAVIAN, seed=7)
    second = agent.design(floorplan, DesignStyle.SCANDINAVIAN, seed=7)
    assert first.scene_id == second.scene_id
    assert [o.position_m.x for o in first.objects] == [o.position_m.x for o in second.objects]


def test_variations_explore_different_layouts(agent, floorplan):
    scenes = [
        agent.design(floorplan, DesignStyle.SCANDINAVIAN, seed=7, variation_index=i)
        for i in range(3)
    ]
    assert len({scene.scene_id for scene in scenes}) == 3


def test_every_object_is_bound_to_a_real_catalog_product(agent, floorplan):
    """The brief's 'use only products from the supplied catalog', enforced."""
    scene = agent.design(floorplan, DesignStyle.JAPANDI, seed=3)
    assert scene.objects
    for obj in scene.objects:
        assert agent.catalog.get(obj.product_id) is not None, f"phantom product {obj.product_id}"


def test_bill_of_materials_matches_the_scene(agent, floorplan):
    """The BOM is a traversal of the graph, so it cannot disagree with it."""
    scene = agent.design(floorplan, DesignStyle.MODERN, seed=5)
    bom = agent.bill_of_materials(scene)

    object_products = {obj.product_id for obj in scene.objects}
    bom_products = {line.product_id for line in bom.lines}
    assert object_products <= bom_products, "an object was rendered but not billed"

    assert bom.total_cost == pytest.approx(sum(line.line_total for line in bom.lines))
    assert bom.total_cost > 0


def test_renovation_finishes_are_bound_to_catalog_products(agent, floorplan):
    """'Every visible furniture OR renovation element' — finishes count."""
    scene = agent.design(floorplan, DesignStyle.CONTEMPORARY, seed=5)
    assert scene.finishes
    for finish in scene.finishes:
        assert finish.floor.product_id, f"{finish.room_id} floor is not a catalog product"
        assert finish.walls.product_id, f"{finish.room_id} walls are not a catalog product"


def test_wet_rooms_get_tile_rather_than_timber(agent, floorplan):
    scene = agent.design(floorplan, DesignStyle.MODERN, seed=5)
    bath = scene.finishes_for_room("bath")
    assert bath is not None
    product = agent.catalog.get(bath.floor.product_id)
    assert product is not None
    assert product.category == ProductCategory.FLOOR_TILE


def test_required_slots_are_filled_or_reported_never_faked(agent, floorplan):
    scene = agent.design(floorplan, DesignStyle.RUSTIC, seed=11)
    placed = {(obj.room_id, obj.category) for obj in scene.objects}
    reported = " ".join(scene.unfilled_slots)
    for room in floorplan.furnishable_rooms():
        for slot in build_room_program(room.id, room.room_type, room.area_m2):
            if slot.required:
                assert (room.id, slot.category) in placed or slot.slot_id in reported, (
                    f"required {slot.category.value} in {room.id} silently vanished"
                )


def test_lighting_and_cameras_live_in_the_scene(agent, floorplan):
    """Lighting must be scene state, not prompt text, or it shifts per view."""
    scene = agent.design(floorplan, DesignStyle.MINIMALIST, seed=2)
    assert scene.lights
    assert all(light.room_id in scene.room_ids for light in scene.lights)
    assert scene.cameras == [], "cameras are populated by the rig, not the design agent"


def test_ceiling_fixtures_hang_from_the_ceiling(agent, floorplan):
    scene = agent.design(floorplan, DesignStyle.SCANDINAVIAN, seed=7)
    pendants = [obj for obj in scene.objects if obj.category.is_ceiling_mounted]
    assert pendants
    for pendant in pendants:
        assert pendant.position_m.z > 1.8, "a pendant light is sitting on the floor"


def test_objects_carry_stable_per_instance_seeds(agent, floorplan):
    """Frozen seeds are what stop an unrelated edit changing another object."""
    scene = agent.design(floorplan, DesignStyle.SCANDINAVIAN, seed=7)
    seeds = {obj.instance_id: obj.seed for obj in scene.objects}
    again = agent.design(floorplan, DesignStyle.SCANDINAVIAN, seed=7)
    assert {obj.instance_id: obj.seed for obj in again.objects} == seeds


def test_derived_objects_reference_a_placed_anchor(agent, floorplan):
    """A nightstand with no bed is a bug; it must be dropped, not orphaned."""
    scene = agent.design(floorplan, DesignStyle.SCANDINAVIAN, seed=7)
    for room_id in scene.room_ids:
        objects = scene.objects_in_room(room_id)
        categories = {obj.category for obj in objects}
        if ProductCategory.NIGHTSTAND in categories:
            assert ProductCategory.BED in categories
        if ProductCategory.COFFEE_TABLE in categories:
            assert ProductCategory.SOFA in categories


def test_placement_respects_door_swing(agent, floorplan):
    """Furniture parked in a doorway means the door cannot open."""
    scene = agent.design(floorplan, DesignStyle.SCANDINAVIAN, seed=7)
    door = floorplan.openings[0]
    keepout = door.keepout_radius_m()
    for obj in _floor_objects(scene):
        if obj.room_id not in door.room_ids:
            continue
        corners = rect_corners(
            Vec2(x=obj.position_m.x, y=obj.position_m.y),
            obj.size_m.width,
            obj.size_m.depth,
            obj.rotation_deg,
        )
        distance = min(
            point_to_segment_distance(door.centre_m, corners[i], corners[(i + 1) % 4])
            for i in range(4)
        )
        assert distance > keepout * 0.4, f"{obj.display_name} blocks the door"


def test_unfurnishable_room_selection_is_rejected(agent, floorplan):
    with pytest.raises(ValueError, match="No furnishable rooms"):
        agent.design(floorplan, DesignStyle.MODERN, room_ids=["does-not-exist"])


def test_placement_enum_classifies_derived_correctly():
    assert Placement.ADJACENT.is_derived
    assert Placement.CEILING.is_derived
    assert not Placement.WALL_BACK.is_derived
    assert not Placement.CENTRE.is_derived
