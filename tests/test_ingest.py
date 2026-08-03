"""Ingestion tests — loader, coordinate transform, and plan assembly.

The Y-flip tests matter most. Getting that wrong mirrors every room and
silently reverses door swings, and nothing downstream would catch it: the
scene graph, the solver and the renderer would all agree on a mirrored world.
"""

from __future__ import annotations

import math

import pytest
from PIL import Image

from backend.ingest.loader import (
    MIN_LONG_EDGE_PX,
    LoadedPlan,
    UnsupportedPlanFormat,
    crop_to_region,
    load_plan,
    trim_whitespace,
)
from backend.ingest.service import build_floorplan, px_to_m
from backend.schemas.common import Vec2
from backend.schemas.floorplan import (
    FloorPlanExtraction,
    OpeningExtraction,
    OpeningType,
    RoomExtraction,
    RoomType,
    ScaleCalibration,
    SwingDirection,
    WallExtraction,
)

SCALE = 50.0  # px per metre
IMAGE_H = 1000


def _plan_from(image: Image.Image) -> LoadedPlan:
    return LoadedPlan(
        image=image,
        source_filename="test.png",
        page_number=None,
        page_count=1,
        was_upscaled=False,
        original_size_px=(image.width, image.height),
    )


# --- coordinate transform --------------------------------------------------


def test_px_to_m_flips_y_axis():
    """Image Y grows down; world Y grows up."""
    top = px_to_m(Vec2(x=0, y=0), SCALE, IMAGE_H)
    bottom = px_to_m(Vec2(x=0, y=IMAGE_H), SCALE, IMAGE_H)
    assert top.y == pytest.approx(IMAGE_H / SCALE)
    assert bottom.y == pytest.approx(0.0)
    assert top.y > bottom.y


def test_px_to_m_scales_x_without_flipping():
    assert px_to_m(Vec2(x=500, y=0), SCALE, IMAGE_H).x == pytest.approx(10.0)


def test_px_to_m_preserves_distances():
    """A flip and a uniform scale must not distort lengths."""
    a = px_to_m(Vec2(x=100, y=200), SCALE, IMAGE_H)
    b = px_to_m(Vec2(x=400, y=600), SCALE, IMAGE_H)
    assert a.distance_to(b) == pytest.approx(500 / SCALE)


def test_px_to_m_preserves_winding_order():
    """A Y-flip reverses winding; area must stay positive regardless.

    If this ever returns a negative area the shoelace helper has been changed
    to signed, and every polygon in the system would silently invert.
    """
    from backend.schemas.common import polygon_area

    square_px = [Vec2(x=0, y=0), Vec2(x=100, y=0), Vec2(x=100, y=100), Vec2(x=0, y=100)]
    square_m = [px_to_m(p, SCALE, IMAGE_H) for p in square_px]
    assert polygon_area(square_m) == pytest.approx((100 / SCALE) ** 2)


# --- plan assembly ---------------------------------------------------------


def _extraction() -> FloorPlanExtraction:
    """A 6 m × 4 m room labelled 24 m², drawn at SCALE px/m."""
    return FloorPlanExtraction(
        rooms=[
            RoomExtraction(
                id="room-1",
                name="Living Room",
                room_type=RoomType.LIVING,
                polygon_px=[
                    Vec2(x=0, y=0),
                    Vec2(x=6 * SCALE, y=0),
                    Vec2(x=6 * SCALE, y=4 * SCALE),
                    Vec2(x=0, y=4 * SCALE),
                ],
                area_label_m2=24.0,
            )
        ],
        walls=[
            WallExtraction(
                id="wall-1",
                start_px=Vec2(x=0, y=0),
                end_px=Vec2(x=6 * SCALE, y=0),
                thickness_px=0.2 * SCALE,
                is_exterior=True,
            )
        ],
        openings=[
            OpeningExtraction(
                id="door-1",
                opening_type=OpeningType.DOOR,
                wall_id="wall-1",
                start_px=Vec2(x=100, y=0),
                end_px=Vec2(x=100 + 0.9 * SCALE, y=0),
                swing=SwingDirection.INWARD,
                room_ids=["room-1"],
            )
        ],
    )


def _calibration(px_per_m: float = SCALE) -> ScaleCalibration:
    return ScaleCalibration(
        px_per_m=px_per_m, method="area_labels", residual_pct=0.0, confidence=0.95, sample_count=1
    )


def _build(extraction=None, calibration=None):
    return build_floorplan(
        extraction=extraction or _extraction(),
        calibration=calibration or _calibration(),
        image_width_px=800,
        image_height_px=IMAGE_H,
        source_filename="test.png",
    )


def test_room_area_matches_its_printed_label():
    plan = _build()
    assert plan.rooms[0].area_m2 == pytest.approx(24.0)
    assert plan.rooms[0].area_error_pct() == pytest.approx(0.0, abs=0.01)


def test_wall_length_and_thickness_convert_to_metres():
    wall = _build().walls[0]
    assert wall.length_m == pytest.approx(6.0)
    assert wall.thickness_m == pytest.approx(0.2)


def test_door_width_converts_and_windows_get_a_sill():
    plan = _build()
    assert plan.openings[0].width_m == pytest.approx(0.9)
    assert plan.openings[0].sill_height_m == 0.0

    extraction = _extraction()
    extraction.openings[0].opening_type = OpeningType.WINDOW
    assert _build(extraction).openings[0].sill_height_m == pytest.approx(0.9)


def test_pixel_polygon_is_retained_alongside_metres():
    """The UI overlays extraction onto the source raster, so both must survive."""
    room = _build().rooms[0]
    assert len(room.polygon_px) == len(room.polygon_m)
    assert room.polygon_px[1].x == pytest.approx(6 * SCALE)


def test_thin_walls_are_floored_to_a_plausible_thickness():
    extraction = _extraction()
    extraction.walls[0].thickness_px = 0.0
    assert _build(extraction).walls[0].thickness_m >= 0.05


def test_validation_warnings_are_reported_not_raised():
    """A questionable room should be flagged, not fatal."""
    bad = _calibration(px_per_m=SCALE * 4)  # makes the room measure ~1.5 m²
    plan = _build(calibration=bad)
    assert plan.calibration.warnings
    assert any("labelled" in w or "mis-traced" in w for w in plan.calibration.warnings)


def test_dangling_opening_reference_is_flagged():
    extraction = _extraction()
    extraction.openings[0].wall_id = "does-not-exist"
    plan = _build(extraction)
    assert any("unknown wall" in w for w in plan.calibration.warnings)


def test_furnishable_rooms_exclude_circulation():
    extraction = _extraction()
    extraction.rooms[0].room_type = RoomType.HALL
    plan = _build(extraction)
    assert plan.rooms and not plan.furnishable_rooms()
    assert any("No furnishable rooms" in w for w in plan.calibration.warnings)


# --- loader ----------------------------------------------------------------


def test_rejects_unsupported_formats(tmp_path):
    bad = tmp_path / "plan.gif"
    Image.new("RGB", (100, 100), "white").save(bad)
    with pytest.raises(UnsupportedPlanFormat, match="Unsupported floor plan format"):
        load_plan(bad)


def test_missing_file_raises_filenotfound(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_plan(tmp_path / "nope.png")


def test_small_plans_are_upscaled(tmp_path):
    small = tmp_path / "small.png"
    Image.new("RGB", (400, 300), "white").save(small)
    loaded = load_plan(small)
    assert loaded.was_upscaled
    assert max(loaded.width, loaded.height) >= MIN_LONG_EDGE_PX


def test_large_plans_are_capped(tmp_path):
    big = tmp_path / "big.png"
    Image.new("RGB", (5000, 4000), "white").save(big)
    loaded = load_plan(big, max_edge=2000)
    assert max(loaded.width, loaded.height) == 2000


def test_transparency_is_flattened_onto_white(tmp_path):
    """A transparent export must not become a black rectangle."""
    path = tmp_path / "alpha.png"
    Image.new("RGBA", (1500, 1500), (0, 0, 0, 0)).save(path)
    loaded = load_plan(path)
    assert loaded.image.mode == "RGB"
    assert loaded.image.getpixel((10, 10)) == (255, 255, 255)


def test_base64_encoding_has_no_newlines():
    """The Anthropic image API rejects base64 containing newlines."""
    encoded = _plan_from(Image.new("RGB", (100, 100), "white")).to_base64_png()
    assert "\n" not in encoded


def test_crop_to_region_zooms_the_drawing():
    plan = _plan_from(Image.new("RGB", (2000, 2000), "white"))
    cropped = crop_to_region(plan, (0.25, 0.25, 0.5, 0.5), pad=0.0)
    # A quarter-width crop re-normalizes back up toward the working resolution.
    assert cropped.width > 500


def test_crop_to_region_rejects_an_inverted_box():
    plan = _plan_from(Image.new("RGB", (1000, 1000), "white"))
    with pytest.raises(ValueError, match="Invalid normalized crop box"):
        crop_to_region(plan, (0.8, 0.1, 0.2, 0.9))


def test_trim_whitespace_removes_blank_margins():
    image = Image.new("RGB", (2000, 2000), "white")
    for x in range(900, 1100):
        for y in range(900, 1100):
            image.putpixel((x, y), (0, 0, 0))
    trimmed = trim_whitespace(_plan_from(image))
    assert trimmed.width < 2000


def test_trim_whitespace_leaves_a_blank_page_alone():
    plan = _plan_from(Image.new("RGB", (1500, 1500), "white"))
    assert trim_whitespace(plan).width == 1500


def test_page_count_of_the_assessment_pdf():
    """The supplied brief is a 3-page PDF with plans on pages 2 and 3."""
    from pathlib import Path

    from backend.ingest.loader import page_count

    pdf = Path("AI_Engineer_Technical_Assessment_v1.pdf")
    if not pdf.exists():
        pytest.skip("assessment PDF not present (it is gitignored)")
    assert page_count(pdf) == 3
    plan = load_plan(pdf, page=1)
    assert plan.width > 1000 and plan.height > 1000
    assert math.isclose(plan.image.width / plan.image.height, 2481 / 3508, rel_tol=0.02)
