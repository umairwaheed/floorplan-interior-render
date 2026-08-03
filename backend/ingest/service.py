"""Floor plan ingestion pipeline: file → calibrated `FloorPlan`.

Assembles the three stages — load, extract, calibrate — and performs the one
irreversible step between them: converting pixel geometry into metres.

**The Y-flip.** Images address pixels top-down; the world uses Y-up so that a
floor plan read on screen matches the plan seen from above, and so rotations
are counter-clockwise-positive as every geometry convention expects. Getting
this wrong mirrors every room and silently reverses door swings, so it happens
in exactly one function and is covered by tests.
"""

from __future__ import annotations

import logging
import uuid

from ..config import Settings, get_settings
from ..schemas.common import Vec2
from ..schemas.floorplan import (
    FloorPlan,
    FloorPlanExtraction,
    Opening,
    Room,
    ScaleCalibration,
    Wall,
)
from .calibrate import CalibrationError, calibrate
from .extract import ExtractionError, FloorPlanExtractor
from .loader import LoadedPlan, load_plan

logger = logging.getLogger(__name__)

#: Walls thinner than this in the drawing are almost certainly a mis-measure.
MIN_WALL_THICKNESS_M = 0.05
DEFAULT_WALL_THICKNESS_M = 0.15


class IngestionError(RuntimeError):
    """Raised when a floor plan cannot be turned into usable geometry."""


def px_to_m(point: Vec2, px_per_m: float, image_height_px: int) -> Vec2:
    """Convert a pixel point to world metres, flipping the Y axis.

    Image Y grows downward; world Y grows upward. This is the single place that
    conversion happens.
    """
    return Vec2(
        x=point.x / px_per_m,
        y=(image_height_px - point.y) / px_per_m,
    )


def _validate(plan: FloorPlan) -> list[str]:
    """Geometry sanity checks. Returns human-readable warnings.

    These are reported rather than raised: a plan with one questionable room is
    still far more useful than no plan at all, and the design stage can work
    around a flagged room. Silence would be the real failure.
    """
    warnings: list[str] = []

    for room in plan.rooms:
        if room.area_m2 < 1.0:
            warnings.append(
                f"Room '{room.name}' measures only {room.area_m2:.2f} m² — likely mis-traced."
            )
        if len(room.polygon_m) < 3:
            warnings.append(f"Room '{room.name}' has a degenerate polygon.")

        error = room.area_error_pct()
        if error is not None and error > 15.0:
            warnings.append(
                f"Room '{room.name}' measures {room.area_m2:.1f} m² but is labelled "
                f"{room.area_label_m2:.1f} m² ({error:.0f}% off)."
            )

    wall_ids = {wall.id for wall in plan.walls}
    for opening in plan.openings:
        if opening.wall_id and opening.wall_id not in wall_ids:
            warnings.append(f"Opening '{opening.id}' references unknown wall '{opening.wall_id}'.")
        if opening.width_m > 3.0:
            warnings.append(
                f"Opening '{opening.id}' is {opening.width_m:.1f} m wide — check the extraction."
            )

    if not plan.furnishable_rooms():
        warnings.append("No furnishable rooms were detected — nothing can be designed.")

    return warnings


def build_floorplan(
    extraction: FloorPlanExtraction,
    calibration: ScaleCalibration,
    image_width_px: int,
    image_height_px: int,
    source_filename: str,
    floorplan_id: str | None = None,
    ceiling_height_m: float = 2.7,
) -> FloorPlan:
    """Convert a raw extraction into a calibrated, metres-based `FloorPlan`."""
    scale = calibration.px_per_m

    rooms = [
        Room(
            id=room.id,
            name=room.name,
            room_type=room.room_type,
            polygon_m=[px_to_m(p, scale, image_height_px) for p in room.polygon_px],
            polygon_px=room.polygon_px,
            area_label_m2=room.area_label_m2,
            ceiling_height_m=ceiling_height_m,
        )
        for room in extraction.rooms
    ]

    walls = [
        Wall(
            id=wall.id,
            start_m=px_to_m(wall.start_px, scale, image_height_px),
            end_m=px_to_m(wall.end_px, scale, image_height_px),
            thickness_m=max(
                MIN_WALL_THICKNESS_M,
                (wall.thickness_px / scale) if wall.thickness_px else DEFAULT_WALL_THICKNESS_M,
            ),
            is_exterior=wall.is_exterior,
            height_m=ceiling_height_m,
        )
        for wall in extraction.walls
    ]

    openings = [
        Opening(
            id=opening.id,
            opening_type=opening.opening_type,
            wall_id=opening.wall_id,
            start_m=px_to_m(opening.start_px, scale, image_height_px),
            end_m=px_to_m(opening.end_px, scale, image_height_px),
            swing=opening.swing,
            room_ids=opening.room_ids,
            # Windows sit above the floor; doors start at it.
            sill_height_m=0.9 if opening.opening_type.value == "window" else 0.0,
            head_height_m=2.1,
        )
        for opening in extraction.openings
    ]

    plan = FloorPlan(
        id=floorplan_id or f"fp-{uuid.uuid4().hex[:10]}",
        source_filename=source_filename,
        image_width_px=image_width_px,
        image_height_px=image_height_px,
        calibration=calibration,
        rooms=rooms,
        walls=walls,
        openings=openings,
        notes=extraction.notes,
    )

    issues = _validate(plan)
    if issues:
        plan.calibration.warnings.extend(issues)
        for issue in issues:
            logger.warning("floor plan %s: %s", plan.id, issue)

    return plan


class FloorPlanIngestService:
    """The entry point the API and CLI both use."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._extractor: FloorPlanExtractor | None = None

    @property
    def extractor(self) -> FloorPlanExtractor:
        # Constructed lazily so importing the service doesn't require an API key.
        if self._extractor is None:
            self._extractor = FloorPlanExtractor(self.settings)
        return self._extractor

    def ingest(
        self,
        path,
        page: int = 0,
        auto_crop: bool = True,
        manual_px_per_m: float | None = None,
        ceiling_height_m: float | None = None,
    ) -> tuple[FloorPlan, LoadedPlan]:
        """Load, extract, calibrate. Returns the plan and the image it was measured from.

        The image comes back alongside because every `polygon_px` is relative to
        it — after an auto-crop that is *not* the file the user uploaded, and
        pairing them is what keeps the pixel overlay in the UI aligned.
        """
        plan_image = load_plan(
            path,
            page=page,
            max_edge=self.settings.max_plan_edge_px,
            dpi=self.settings.plan_render_dpi,
        )

        try:
            extraction, working_image = self.extractor.extract(plan_image, auto_crop=auto_crop)
        except ExtractionError as exc:
            raise IngestionError(str(exc)) from exc

        if not extraction.rooms:
            raise IngestionError(
                "No rooms were detected in this image. Check that it is a floor plan, "
                "and for a multi-page PDF that the right page was selected."
            )

        try:
            calibration = calibrate(
                rooms=extraction.rooms,
                ticks=extraction.dimension_ticks,
                manual_px_per_m=manual_px_per_m,
            )
        except CalibrationError as exc:
            raise IngestionError(str(exc)) from exc

        floorplan = build_floorplan(
            extraction=extraction,
            calibration=calibration,
            image_width_px=working_image.width,
            image_height_px=working_image.height,
            source_filename=working_image.source_filename,
            ceiling_height_m=ceiling_height_m or self.settings.default_ceiling_height_m,
        )

        logger.info(
            "ingested %s: %d rooms, %.1f m², scale %.1f px/m (%s, %.0f%% confidence)",
            floorplan.source_filename,
            len(floorplan.rooms),
            floorplan.total_area_m2,
            calibration.px_per_m,
            calibration.method,
            calibration.confidence * 100,
        )
        return floorplan, working_image
