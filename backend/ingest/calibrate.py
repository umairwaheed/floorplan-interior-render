"""Pixel → metre calibration.

Every geometric claim the system makes downstream — whether a sofa fits, how
far a camera stands from a wall, how much flooring to buy — inherits its error
from this one number. So it is solved and *checked*, not guessed.

**Primary signal: the printed m² labels.** Both sample floor plans print a room
area inside each room (25.2, 12.8, 6.1, 19.3, 14.9…). Given a room's pixel
polygon we know its area in px²; given the label we know it in m². One unknown,
many observations — a linear least-squares fit.

Why not just ask the vision model for room dimensions? Because this is
arithmetic over a global constant, and it self-checks: fit a single scale
against every room at once and the residual tells you whether the *extraction*
is trustworthy. A model asked for dimensions directly gives you a number with
no way to know if it's wrong.

**Fallback: printed dimension ticks** (the `370`, `810` marks on sample plan 2)
when no area labels are present.

If neither exists, calibration fails loudly with zero confidence rather than
inventing a scale.
"""

from __future__ import annotations

import math
import statistics

from ..schemas.common import polygon_area
from ..schemas.floorplan import (
    DimensionTick,
    RoomExtraction,
    ScaleCalibration,
)

#: A room whose implied scale deviates more than this from the median is
#: treated as a bad extraction and dropped before the final fit.
OUTLIER_TOLERANCE = 0.25

#: Above this mean residual, the geometry disagrees with the printed labels
#: badly enough that the polygons are probably wrong.
RESIDUAL_WARN_PCT = 8.0
RESIDUAL_FAIL_PCT = 20.0


def _implied_scale(room: RoomExtraction) -> float | None:
    """px_per_m implied by a single room, from area_px / area_m² = scale²."""
    if not room.area_label_m2 or room.area_label_m2 <= 0:
        return None
    area_px = polygon_area(room.polygon_px)
    if area_px <= 0:
        return None
    return math.sqrt(area_px / room.area_label_m2)


def calibrate_from_area_labels(rooms: list[RoomExtraction]) -> ScaleCalibration | None:
    """Least-squares fit of one global scale against every labelled room."""
    labelled = [(room, scale) for room in rooms if (scale := _implied_scale(room)) is not None]
    if not labelled:
        return None

    warnings: list[str] = []

    # Drop rooms that disagree wildly with the consensus before fitting. A
    # single mis-traced polygon would otherwise drag the global scale off.
    median_scale = statistics.median(scale for _, scale in labelled)
    kept: list[tuple[RoomExtraction, float]] = []
    dropped_names: list[str] = []
    for room, scale in labelled:
        if abs(scale - median_scale) / median_scale <= OUTLIER_TOLERANCE:
            kept.append((room, scale))
        else:
            dropped_names.append(room.name)

    if dropped_names:
        warnings.append(f"Dropped outlier room(s) from the scale fit: {', '.join(dropped_names)}")
    if not kept:  # everything disagreed; fall back to fitting all of it
        kept = labelled

    # Solve in k = m² per px², which makes the fit linear:
    #   predicted_area_m2 = area_px * k   ⇒   k = Σ(area_px·A) / Σ(area_px²)
    numerator = 0.0
    denominator = 0.0
    for room, _ in kept:
        area_px = polygon_area(room.polygon_px)
        numerator += area_px * (room.area_label_m2 or 0.0)
        denominator += area_px * area_px
    if denominator <= 0:
        return None

    k = numerator / denominator
    if k <= 0:
        return None
    px_per_m = 1.0 / math.sqrt(k)

    # Residual is measured against *every* labelled room, including the ones
    # dropped from the fit — hiding them would defeat the point of the check.
    errors = []
    for room, _ in labelled:
        predicted = polygon_area(room.polygon_px) * k
        errors.append(abs(predicted - (room.area_label_m2 or 0.0)) / (room.area_label_m2 or 1.0))
    residual_pct = (sum(errors) / len(errors)) * 100.0

    if residual_pct > RESIDUAL_FAIL_PCT:
        warnings.append(
            f"Room areas disagree with the printed labels by {residual_pct:.1f}% on average — "
            "the extracted polygons are probably wrong, not the scale."
        )
    elif residual_pct > RESIDUAL_WARN_PCT:
        warnings.append(
            f"Scale fit residual is {residual_pct:.1f}%; treat room dimensions as approximate."
        )

    return ScaleCalibration(
        px_per_m=px_per_m,
        method="area_labels",
        residual_pct=round(residual_pct, 2),
        confidence=_confidence(len(labelled), residual_pct, len(dropped_names)),
        sample_count=len(labelled),
        warnings=warnings,
    )


def calibrate_from_dimension_ticks(ticks: list[DimensionTick]) -> ScaleCalibration | None:
    """Fallback: printed linear dimensions, e.g. the '370' marks on plan 2.

    Uses the median rather than a mean — one misread tick shouldn't move the
    answer, and there are usually few enough ticks that outlier rejection has
    nothing to work with.
    """
    scales: list[float] = []
    for tick in ticks:
        metres = tick.value_in_metres()
        pixels = tick.start_px.distance_to(tick.end_px)
        if metres > 0 and pixels > 0:
            scales.append(pixels / metres)
    if not scales:
        return None

    px_per_m = statistics.median(scales)
    spread = (max(scales) - min(scales)) / px_per_m * 100.0 if len(scales) > 1 and px_per_m else 0.0
    warnings = []
    if spread > 15.0:
        warnings.append(f"Dimension ticks disagree by {spread:.1f}%; scale is approximate.")

    return ScaleCalibration(
        px_per_m=px_per_m,
        method="dimension_ticks",
        residual_pct=round(spread, 2),
        confidence=_confidence(len(scales), spread, 0) * 0.85,  # weaker signal than areas
        sample_count=len(scales),
        warnings=warnings,
    )


def _confidence(sample_count: int, residual_pct: float, dropped: int) -> float:
    """Confidence in [0, 1] from agreement and evidence volume.

    Deliberately conservative: a single labelled room can produce a perfect
    residual while being completely wrong, so sample count is capped separately
    from accuracy rather than multiplied into a flattering number.
    """
    if sample_count <= 0:
        return 0.0
    evidence = min(sample_count / 3.0, 1.0)
    accuracy = max(0.0, 1.0 - residual_pct / RESIDUAL_FAIL_PCT)
    penalty = 0.85**dropped
    return round(min(1.0, 0.35 * evidence + 0.65 * accuracy) * penalty, 3)


def calibrate(
    rooms: list[RoomExtraction],
    ticks: list[DimensionTick] | None = None,
    manual_px_per_m: float | None = None,
) -> ScaleCalibration:
    """Resolve the scale, preferring the strongest available signal.

    Order: explicit override → area labels → dimension ticks → failure.
    """
    if manual_px_per_m and manual_px_per_m > 0:
        return ScaleCalibration(
            px_per_m=manual_px_per_m,
            method="manual",
            residual_pct=0.0,
            confidence=1.0,
            sample_count=0,
            warnings=["Scale supplied manually; no automatic verification performed."],
        )

    from_areas = calibrate_from_area_labels(rooms)
    if from_areas and from_areas.confidence > 0.0:
        return from_areas

    from_ticks = calibrate_from_dimension_ticks(ticks or [])
    if from_ticks:
        if from_areas:
            from_ticks.warnings.append("Area labels were present but did not produce a usable fit.")
        return from_ticks

    raise CalibrationError(
        "Could not determine the drawing scale: the plan has no readable m² labels "
        "and no dimension annotations. Supply --px-per-m to proceed."
    )


class CalibrationError(ValueError):
    """Raised when no scale can be established from the plan."""
