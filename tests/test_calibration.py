"""Scale calibration tests.

These use the real room areas printed on the two floor plans supplied with the
assessment, so a regression here is a regression against the actual inputs.

Plan 1 (studio apartment): 25.2, 12.8, 11.8, 6.1 m²
Plan 2 (Flat 127, 51.0 m² total): 19.3, 14.9, 5.7, 5.4, 3.9 m²
"""

from __future__ import annotations

import math

import pytest

from backend.ingest.calibrate import (
    CalibrationError,
    calibrate,
    calibrate_from_area_labels,
    calibrate_from_dimension_ticks,
)
from backend.schemas.common import Vec2
from backend.schemas.floorplan import DimensionTick, RoomExtraction, RoomType

# A known scale to synthesise pixel geometry from, so the solver has a
# ground truth to be checked against.
TRUE_PX_PER_M = 37.5


def _rect_room(
    room_id: str,
    width_m: float,
    height_m: float,
    area_label: float | None,
    scale: float = TRUE_PX_PER_M,
) -> RoomExtraction:
    """A rectangular room drawn at `scale`, labelled with `area_label` m²."""
    w_px, h_px = width_m * scale, height_m * scale
    return RoomExtraction(
        id=room_id,
        name=room_id,
        room_type=RoomType.OTHER,
        polygon_px=[
            Vec2(x=0, y=0),
            Vec2(x=w_px, y=0),
            Vec2(x=w_px, y=h_px),
            Vec2(x=0, y=h_px),
        ],
        area_label_m2=area_label,
    )


def _plan_one_rooms(scale: float = TRUE_PX_PER_M) -> list[RoomExtraction]:
    """Sample plan 1, with each room's true area matching its printed label."""
    return [
        _rect_room("studio", 6.0, 4.2, 25.2, scale),
        _rect_room("bedroom", 4.0, 3.2, 12.8, scale),
        _rect_room("balcony", 5.9, 2.0, 11.8, scale),
        _rect_room("bath", 2.44, 2.5, 6.1, scale),
    ]


# --- the core solve --------------------------------------------------------


def test_recovers_the_true_scale_from_area_labels():
    result = calibrate_from_area_labels(_plan_one_rooms())
    assert result is not None
    assert result.method == "area_labels"
    assert result.px_per_m == pytest.approx(TRUE_PX_PER_M, rel=1e-6)
    assert result.residual_pct < 0.01
    assert result.confidence > 0.9
    assert result.sample_count == 4


def test_scale_is_independent_of_the_drawing_resolution():
    """The same plan scanned at a different DPI must still calibrate correctly."""
    for scale in (12.0, 37.5, 150.0):
        result = calibrate_from_area_labels(_plan_one_rooms(scale=scale))
        assert result is not None
        assert result.px_per_m == pytest.approx(scale, rel=1e-6)


def test_single_labelled_room_works_but_is_not_over_trusted():
    """One room can fit perfectly while being completely wrong, so confidence
    must stay below that of a multi-room agreement."""
    single = calibrate_from_area_labels([_rect_room("only", 6.0, 4.2, 25.2)])
    many = calibrate_from_area_labels(_plan_one_rooms())
    assert single is not None and many is not None
    assert single.px_per_m == pytest.approx(TRUE_PX_PER_M, rel=1e-6)
    assert single.confidence < many.confidence


def test_rooms_without_labels_are_ignored_not_treated_as_zero():
    rooms = _plan_one_rooms() + [_rect_room("unlabelled", 3.0, 3.0, None)]
    result = calibrate_from_area_labels(rooms)
    assert result is not None
    assert result.sample_count == 4
    assert result.px_per_m == pytest.approx(TRUE_PX_PER_M, rel=1e-6)


def test_returns_none_when_no_room_carries_a_label():
    assert calibrate_from_area_labels([_rect_room("a", 3.0, 3.0, None)]) is None


# --- the self-check, which is the point ------------------------------------


def test_a_mistraced_polygon_is_dropped_and_reported():
    """One badly extracted room must not drag the global scale off."""
    rooms = _plan_one_rooms()
    # Bedroom traced at roughly half scale — a plausible extraction failure.
    rooms[1] = _rect_room("bedroom", 4.0, 3.2, 12.8, scale=TRUE_PX_PER_M * 0.5)

    result = calibrate_from_area_labels(rooms)
    assert result is not None
    assert result.px_per_m == pytest.approx(TRUE_PX_PER_M, rel=0.02), (
        "the three good rooms should still determine the scale"
    )
    assert any("outlier" in w.lower() for w in result.warnings)


def test_systematically_wrong_extraction_is_flagged_loudly():
    """When geometry and labels disagree across the board, the extraction is
    wrong — and the result must say so rather than return a clean number."""
    rooms = [
        _rect_room("a", 6.0, 4.2, 25.2),
        _rect_room("b", 4.0, 3.2, 40.0),  # label wildly inconsistent
        _rect_room("c", 2.44, 2.5, 30.0),
    ]
    result = calibrate_from_area_labels(rooms)
    assert result is not None
    assert result.residual_pct > 20.0
    assert result.confidence < 0.35
    assert result.warnings


def test_residual_accounts_for_dropped_rooms_rather_than_hiding_them():
    rooms = _plan_one_rooms()
    rooms[1] = _rect_room("bedroom", 4.0, 3.2, 12.8, scale=TRUE_PX_PER_M * 0.5)
    result = calibrate_from_area_labels(rooms)
    assert result is not None
    assert result.residual_pct > 10.0, "the dropped room must still count against the residual"


# --- fallback path ---------------------------------------------------------


def test_dimension_ticks_recover_the_scale():
    """Plan 2 prints linear dimensions in cm (370, 810, 210…)."""
    ticks = [
        DimensionTick(
            start_px=Vec2(x=0, y=0),
            end_px=Vec2(x=3.70 * TRUE_PX_PER_M, y=0),
            value=370,
            unit="cm",
        ),
        DimensionTick(
            start_px=Vec2(x=0, y=0),
            end_px=Vec2(x=0, y=8.10 * TRUE_PX_PER_M),
            value=810,
            unit="cm",
        ),
    ]
    result = calibrate_from_dimension_ticks(ticks)
    assert result is not None
    assert result.method == "dimension_ticks"
    assert result.px_per_m == pytest.approx(TRUE_PX_PER_M, rel=1e-6)


def test_tick_units_are_honoured():
    for unit, value in (("mm", 3700), ("cm", 370), ("m", 3.7)):
        tick = DimensionTick(
            start_px=Vec2(x=0, y=0),
            end_px=Vec2(x=3.70 * TRUE_PX_PER_M, y=0),
            value=value,
            unit=unit,
        )
        result = calibrate_from_dimension_ticks([tick])
        assert result is not None
        assert result.px_per_m == pytest.approx(TRUE_PX_PER_M, rel=1e-6), f"unit {unit}"


def test_disagreeing_ticks_are_flagged():
    ticks = [
        DimensionTick(start_px=Vec2(x=0, y=0), end_px=Vec2(x=100, y=0), value=100, unit="cm"),
        DimensionTick(start_px=Vec2(x=0, y=0), end_px=Vec2(x=160, y=0), value=100, unit="cm"),
    ]
    result = calibrate_from_dimension_ticks(ticks)
    assert result is not None
    assert result.warnings


# --- resolution order ------------------------------------------------------


def test_area_labels_are_preferred_over_ticks():
    ticks = [DimensionTick(start_px=Vec2(x=0, y=0), end_px=Vec2(x=999, y=0), value=100, unit="cm")]
    result = calibrate(_plan_one_rooms(), ticks=ticks)
    assert result.method == "area_labels"


def test_falls_back_to_ticks_when_no_labels_exist():
    ticks = [
        DimensionTick(
            start_px=Vec2(x=0, y=0), end_px=Vec2(x=3.7 * TRUE_PX_PER_M, y=0), value=370, unit="cm"
        )
    ]
    result = calibrate([_rect_room("a", 3.0, 3.0, None)], ticks=ticks)
    assert result.method == "dimension_ticks"
    assert result.px_per_m == pytest.approx(TRUE_PX_PER_M, rel=1e-6)


def test_manual_override_wins_over_everything():
    result = calibrate(_plan_one_rooms(), manual_px_per_m=100.0)
    assert result.method == "manual"
    assert result.px_per_m == 100.0
    assert result.warnings, "an unverified manual scale should say so"


def test_fails_loudly_rather_than_inventing_a_scale():
    with pytest.raises(CalibrationError, match="Could not determine the drawing scale"):
        calibrate([_rect_room("a", 3.0, 3.0, None)], ticks=[])


# --- end-to-end sanity against the real plan figures -----------------------


def test_plan_two_rooms_reconcile_to_their_labels():
    """Re-measuring plan 2's rooms at the calibrated scale must reproduce the
    printed labels.

    Note what this test deliberately does *not* assert. The plan is annotated
    "45.6 + 5.4 = 51.0 m²", but the room labels sum to only 49.2. That 1.8 m²
    gap is wall thickness and thresholds — real drawings quote gross floor area
    while room labels are net internal. Calibration must match the labels it
    was fitted against, not the headline total.
    """
    rooms = [
        _rect_room("living", 5.0, 3.86, 19.3),
        _rect_room("bedroom", 4.1, 3.634, 14.9),
        _rect_room("hall", 2.5, 2.28, 5.7),
        _rect_room("balcony", 3.0, 1.8, 5.4),
        _rect_room("wc", 1.95, 2.0, 3.9),
    ]
    result = calibrate_from_area_labels(rooms)
    assert result is not None
    # Tolerance is loose because the fixture's room dimensions are rounded to
    # produce the printed labels, exactly as a real drawing's would be.
    assert math.isclose(result.px_per_m, TRUE_PX_PER_M, rel_tol=1e-3)

    from backend.schemas.common import polygon_area

    for room in rooms:
        measured = polygon_area(room.polygon_px) / (result.px_per_m**2)
        assert measured == pytest.approx(room.area_label_m2, rel=0.01), room.id

    total = sum(polygon_area(r.polygon_px) / (result.px_per_m**2) for r in rooms)
    assert total == pytest.approx(49.2, abs=0.2)
