"""Scaffold-level checks on the core contracts.

The scene-hash tests matter most: content-addressing is what makes
regeneration reproducible, so it needs to be provably stable.
"""

from __future__ import annotations

import math

from backend.schemas import (
    ColorPalette,
    DesignStyle,
    ObjectRole,
    PlacedObject,
    ProductCategory,
    Scene,
    Size3,
    Vec2,
    Vec3,
    point_in_polygon,
    polygon_area,
    polygon_centroid,
)

SQUARE = [Vec2(x=0, y=0), Vec2(x=4, y=0), Vec2(x=4, y=3), Vec2(x=0, y=3)]


def test_polygon_area_is_winding_independent():
    assert polygon_area(SQUARE) == 12.0
    assert polygon_area(list(reversed(SQUARE))) == 12.0


def test_polygon_centroid():
    c = polygon_centroid(SQUARE)
    assert math.isclose(c.x, 2.0)
    assert math.isclose(c.y, 1.5)


def test_point_in_polygon():
    assert point_in_polygon(Vec2(x=2, y=1.5), SQUARE)
    assert not point_in_polygon(Vec2(x=5, y=1.5), SQUARE)


def _object(instance_id: str = "obj-1", rotation: float = 0.0) -> PlacedObject:
    return PlacedObject(
        instance_id=instance_id,
        product_id="comforter:SOFA-001",
        room_id="room-1",
        role=ObjectRole.PRIMARY_SEATING,
        category=ProductCategory.SOFA,
        position_m=Vec3(x=2.0, y=1.0, z=0.0),
        rotation_deg=rotation,
        size_m=Size3(width=2.2, depth=0.9, height=0.85),
        color="sage green",
        display_name="Nordic 3-Seat Sofa",
        seed=12345,
    )


def _scene(objects: list[PlacedObject]) -> Scene:
    return Scene(
        floorplan_id="fp-1",
        style=DesignStyle.SCANDINAVIAN,
        palette=ColorPalette(
            name="Nordic Light",
            description="Warm whites",
            primary="#F4F1EC",
            secondary="#D9CFC1",
            accent="#8FA99B",
            neutral="#EDEAE4",
        ),
        seed=7,
        room_ids=["room-1"],
        objects=objects,
    )


def test_footprint_corners_unrotated():
    corners = _object().footprint_corners()
    xs = sorted(c.x for c in corners)
    ys = sorted(c.y for c in corners)
    assert math.isclose(xs[0], 2.0 - 1.1)
    assert math.isclose(xs[-1], 2.0 + 1.1)
    assert math.isclose(ys[0], 1.0 - 0.45)
    assert math.isclose(ys[-1], 1.0 + 0.45)


def test_footprint_corners_rotated_90_swaps_extent():
    corners = _object(rotation=90.0).footprint_corners()
    width_extent = max(c.x for c in corners) - min(c.x for c in corners)
    depth_extent = max(c.y for c in corners) - min(c.y for c in corners)
    assert math.isclose(width_extent, 0.9, abs_tol=1e-9)
    assert math.isclose(depth_extent, 2.2, abs_tol=1e-9)


def test_scene_hash_is_deterministic():
    """Same content, separately constructed, must hash identically —
    this is what lets regeneration reproduce a scene exactly."""
    assert _scene([_object()]).content_hash() == _scene([_object()]).content_hash()


def test_scene_hash_changes_when_an_object_moves():
    moved = _object()
    moved.position_m = Vec3(x=2.5, y=1.0, z=0.0)
    assert _scene([_object()]).content_hash() != _scene([moved]).content_hash()


def test_finalize_stamps_hash_and_is_idempotent():
    scene = _scene([_object()]).finalize()
    assert len(scene.scene_id) == 16
    first = scene.scene_id
    assert scene.finalize().scene_id == first, "scene_id must not feed its own hash"
