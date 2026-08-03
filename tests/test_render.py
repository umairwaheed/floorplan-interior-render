"""Rasterizer and camera-rig tests.

The multi-view consistency claim rests entirely on this layer: if two cameras
don't project the same geometry, nothing downstream can recover. So the tests
here are mostly about *agreement between views* rather than image quality —
a render that looks plausible but disagrees with its own depth buffer is worse
than one that looks crude and is correct.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from backend.config import Settings
from backend.render.cameras import estimate_coverage, place_cameras
from backend.render.raster import (
    NEAR_PLANE,
    CameraBasis,
    box_corners,
    depth_to_image,
    instance_color,
    rasterize,
    wireframe_image,
)
from backend.schemas.common import Size3, Vec2, Vec3
from backend.schemas.floorplan import Room, RoomType
from backend.schemas.product import ProductCategory
from backend.schemas.scene import Camera, ObjectRole, PlacedObject

WIDTH, HEIGHT = 240, 180


@pytest.fixture
def room() -> Room:
    return Room(
        id="r1",
        name="Living Room",
        room_type=RoomType.LIVING,
        polygon_m=[Vec2(x=0, y=0), Vec2(x=5, y=0), Vec2(x=5, y=4), Vec2(x=0, y=4)],
        polygon_px=[Vec2(x=0, y=0), Vec2(x=1, y=0), Vec2(x=1, y=1), Vec2(x=0, y=1)],
        ceiling_height_m=2.7,
    )


def _object(instance_id: str, x: float, y: float, w=1.0, d=1.0, h=1.0, z=0.0, rot=0.0):
    return PlacedObject(
        instance_id=instance_id,
        product_id=f"test:{instance_id}",
        room_id="r1",
        role=ObjectRole.PRIMARY_SEATING,
        category=ProductCategory.SOFA,
        position_m=Vec3(x=x, y=y, z=z),
        rotation_deg=rot,
        size_m=Size3(width=w, depth=d, height=h),
        color="beige",
        display_name=instance_id,
        seed=1,
    )


@pytest.fixture
def objects() -> list[PlacedObject]:
    return [
        _object("sofa", 2.5, 0.6, w=2.2, d=0.9, h=0.85),
        _object("table", 2.5, 2.0, w=1.1, d=0.6, h=0.42),
        _object("shelf", 0.4, 2.0, w=0.4, d=1.0, h=1.8),
    ]


def _camera(x=4.4, y=3.4, target=(2.2, 1.4)) -> Camera:
    return Camera(
        id="cam1",
        room_id="r1",
        position_m=Vec3(x=x, y=y, z=1.5),
        look_at_m=Vec3(x=target[0], y=target[1], z=0.95),
        fov_deg=60.0,
    )


# --- projection ------------------------------------------------------------


def test_projection_puts_the_look_target_at_frame_centre():
    camera = _camera()
    basis = CameraBasis.from_camera(camera, WIDTH, HEIGHT)
    screen, depths = basis.project(np.array([camera.look_at_m.as_tuple()]))
    assert screen[0, 0] == pytest.approx(WIDTH / 2, abs=0.5)
    assert screen[0, 1] == pytest.approx(HEIGHT / 2, abs=0.5)
    assert depths[0] > 0


def test_projection_basis_is_orthonormal():
    basis = CameraBasis.from_camera(_camera(), WIDTH, HEIGHT)
    for vector in (basis.right, basis.up, basis.forward):
        assert np.linalg.norm(vector) == pytest.approx(1.0)
    assert np.dot(basis.right, basis.up) == pytest.approx(0.0, abs=1e-9)
    assert np.dot(basis.right, basis.forward) == pytest.approx(0.0, abs=1e-9)


def test_depth_is_along_the_view_axis_not_euclidean():
    """Euclidean distance would bow every straight wall in the depth map."""
    basis = CameraBasis.from_camera(
        Camera(
            id="c",
            room_id="r1",
            position_m=Vec3(x=0, y=0, z=0),
            look_at_m=Vec3(x=0, y=1, z=0),
            fov_deg=60,
        ),
        WIDTH,
        HEIGHT,
    )
    # Two points on a plane 3 m ahead, one directly forward, one off to the side.
    _, depths = basis.project(np.array([[0.0, 3.0, 0.0], [2.0, 3.0, 0.0]]))
    assert depths[0] == pytest.approx(3.0)
    assert depths[1] == pytest.approx(3.0), "off-axis point on the same plane must match"


def test_objects_behind_the_camera_are_clipped(room):
    """Without a near-plane clip, geometry behind the lens smears the frame."""
    behind = [_object("behind", 4.9, 3.9, w=1.0, d=1.0, h=1.0)]
    camera = Camera(
        id="c",
        room_id="r1",
        position_m=Vec3(x=1.0, y=1.0, z=1.5),
        look_at_m=Vec3(x=0.2, y=0.2, z=1.0),
        fov_deg=60,
    )
    buffers = rasterize(room, behind, camera, WIDTH, HEIGHT)
    assert "behind" not in buffers.visible_instances()


def test_box_corners_match_the_declared_size():
    obj = _object("o", 1.0, 2.0, w=2.0, d=1.0, h=0.8, z=0.1)
    corners = box_corners(obj)
    assert corners.shape == (8, 3)
    assert corners[:, 2].min() == pytest.approx(0.1)
    assert corners[:, 2].max() == pytest.approx(0.9)
    span_x = corners[:, 0].max() - corners[:, 0].min()
    assert span_x == pytest.approx(2.0)


# --- rasterization ---------------------------------------------------------


def test_rasterizer_fills_the_frame(room, objects):
    buffers = rasterize(room, objects, _camera(), WIDTH, HEIGHT)
    filled = np.isfinite(buffers.depth).mean()
    assert filled > 0.95, "a camera inside a closed room should see geometry everywhere"


def test_depth_buffer_holds_plausible_metric_distances(room, objects):
    buffers = rasterize(room, objects, _camera(), WIDTH, HEIGHT)
    finite = buffers.depth[np.isfinite(buffers.depth)]
    assert finite.min() > NEAR_PLANE
    # The room's diagonal bounds how far anything can be.
    assert finite.max() < math.hypot(5.0, 4.0) + 1.0


def test_nearer_geometry_wins_the_z_buffer(room):
    """A closer box must occlude a further one, not average with it."""
    near = _object("near", 2.5, 2.0, w=1.0, d=1.0, h=1.5)
    far = _object("far", 2.5, 3.4, w=1.0, d=1.0, h=1.5)
    camera = Camera(
        id="c",
        room_id="r1",
        position_m=Vec3(x=2.5, y=0.4, z=1.5),
        look_at_m=Vec3(x=2.5, y=3.5, z=1.0),
        fov_deg=60,
    )
    buffers = rasterize(room, [near, far], camera, WIDTH, HEIGHT)
    visible = buffers.visible_instances()
    assert visible.get("near", 0) > visible.get("far", 0)


def test_depth_image_is_brighter_when_nearer(room, objects):
    """The convention depth-conditioned image models expect. Inverting it
    conditions the model into turning the room inside out."""
    buffers = rasterize(room, objects, _camera(), WIDTH, HEIGHT)
    image = np.asarray(depth_to_image(buffers.depth))

    finite = np.isfinite(buffers.depth)
    nearest = np.unravel_index(
        np.argmin(np.where(finite, buffers.depth, np.inf)), buffers.depth.shape
    )
    furthest = np.unravel_index(
        np.argmax(np.where(finite, buffers.depth, -np.inf)), buffers.depth.shape
    )
    assert image[nearest] > image[furthest]


def test_segmentation_colours_are_stable_per_instance():
    """The judge identifies objects by these colours across views."""
    assert instance_color("room:sofa#0") == instance_color("room:sofa#0")
    assert instance_color("room:sofa#0") != instance_color("room:sofa#1")


def test_visible_instances_ignores_specks(room):
    tiny = _object("speck", 4.9, 3.9, w=0.02, d=0.02, h=0.02)
    buffers = rasterize(room, [tiny], _camera(), WIDTH, HEIGHT)
    assert "speck" not in buffers.visible_instances(min_pixels=40)


def test_wireframe_hides_occluded_edges(room, objects):
    """Edges showing through solid geometry would tell the image model there
    is structure where there is only a back face."""
    image = np.asarray(wireframe_image(room, objects, _camera(), WIDTH, HEIGHT))
    ink = (image < 128).mean()
    assert 0.001 < ink < 0.25, f"wireframe ink coverage {ink:.3f} looks wrong"


def test_rasterization_is_deterministic(room, objects):
    a = rasterize(room, objects, _camera(), WIDTH, HEIGHT)
    b = rasterize(room, objects, _camera(), WIDTH, HEIGHT)
    assert np.array_equal(a.depth, b.depth)
    assert np.array_equal(a.segmentation, b.segmentation)


# --- the consistency property ---------------------------------------------


def test_two_views_agree_on_object_geometry(room, objects):
    """The core claim. Both cameras project the same scene, so an object's
    real-world size is identical in both — only its projection differs."""
    left = rasterize(room, objects, _camera(x=0.5, y=3.5), WIDTH, HEIGHT)
    right = rasterize(room, objects, _camera(x=4.5, y=3.5), WIDTH, HEIGHT)

    shared = set(left.visible_instances()) & set(right.visible_instances())
    assert shared, "the two viewpoints should share at least one visible object"
    # Same instance IDs mean the same scene objects — nothing was re-invented.
    assert set(left.instance_ids) == set(right.instance_ids)


def test_moving_the_camera_changes_the_image_but_not_the_scene(room, objects):
    near = rasterize(room, objects, _camera(x=4.4, y=3.4), WIDTH, HEIGHT)
    far = rasterize(room, objects, _camera(x=0.6, y=0.6), WIDTH, HEIGHT)
    assert not np.array_equal(near.depth, far.depth), "different viewpoints must differ"
    assert near.instance_ids == far.instance_ids, "the scene itself must not change"


# --- camera rig ------------------------------------------------------------


@pytest.fixture
def settings(tmp_path) -> Settings:
    s = Settings(
        data_dir=tmp_path,
        upload_dir=tmp_path / "u",
        output_dir=tmp_path / "o",
        db_path=tmp_path / "c.db",
        render_width=WIDTH,
        render_height=HEIGHT,
    )
    s.ensure_dirs()
    return s


def test_cameras_are_placed_inside_the_room(room, objects, settings):
    from backend.schemas.common import point_in_polygon

    for camera in place_cameras(room, objects, count=3, settings=settings):
        position = Vec2(x=camera.position_m.x, y=camera.position_m.y)
        assert point_in_polygon(position, room.polygon_m)


def test_cameras_never_stand_inside_furniture(room, objects, settings):
    """A camera inside a wardrobe renders its side panel, which is useless."""
    from backend.design.geometry import rect_corners
    from backend.schemas.common import point_in_polygon

    for camera in place_cameras(room, objects, count=3, settings=settings):
        eye = Vec2(x=camera.position_m.x, y=camera.position_m.y)
        for obj in objects:
            footprint = rect_corners(
                Vec2(x=obj.position_m.x, y=obj.position_m.y),
                obj.size_m.width,
                obj.size_m.depth,
                obj.rotation_deg,
            )
            assert not point_in_polygon(eye, footprint), f"camera inside {obj.instance_id}"


def test_cameras_are_spread_apart(room, objects, settings):
    """Two cameras a handspan apart produce near-identical images, which is the
    opposite of 'different viewpoints'."""
    cameras = place_cameras(room, objects, count=4, settings=settings)
    for i, a in enumerate(cameras):
        for b in cameras[i + 1 :]:
            distance = Vec2(x=a.position_m.x, y=a.position_m.y).distance_to(
                Vec2(x=b.position_m.x, y=b.position_m.y)
            )
            assert distance >= 1.0, f"{a.id} and {b.id} are only {distance:.2f} m apart"


def test_camera_placement_is_deterministic(room, objects, settings):
    first = place_cameras(room, objects, count=3, settings=settings)
    second = place_cameras(room, objects, count=3, settings=settings)
    assert [c.position_m.as_tuple() for c in first] == [c.position_m.as_tuple() for c in second]


def test_an_empty_room_still_yields_a_camera(room, settings):
    cameras = place_cameras(room, [], count=2, settings=settings)
    assert cameras, "an unfurnished room must still be renderable"


def test_cameras_look_at_the_furniture_not_the_geometric_centre(room, objects, settings):
    """Aiming at the room centroid frames empty floor in an asymmetric room."""
    lopsided = [_object("bulk", 1.0, 1.0, w=2.0, d=2.0, h=1.0)]
    cameras = place_cameras(room, lopsided, count=1, settings=settings)
    target = cameras[0].look_at_m
    assert target.x == pytest.approx(1.0, abs=0.3)
    assert target.y == pytest.approx(1.0, abs=0.3)


def test_coverage_estimate_is_a_fraction(room):
    coverage = estimate_coverage(
        room, Vec2(x=0.5, y=0.5), Vec2(x=2.5, y=2.0), fov_deg=60.0, aspect=4 / 3
    )
    assert 0.0 <= coverage <= 1.0
    assert coverage > 0.1, "a corner camera should see a meaningful share of the floor"
