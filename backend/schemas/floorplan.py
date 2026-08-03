"""Floor plan schema.

Two coordinate systems live here on purpose:

* `*_px` — raw pixel space, exactly as the vision model saw the image.
* metres — everything else, after `ScaleCalibration` resolves px→m.

The vision model is only ever asked for pixel-space values plus the printed
`m²` labels. It is never asked to estimate real-world dimensions, because the
calibration solve recovers those far more reliably (and self-checks).
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from .common import Vec2, polygon_area, polygon_bounds, polygon_centroid


class RoomType(str, Enum):
    LIVING = "living"
    BEDROOM = "bedroom"
    KITCHEN = "kitchen"
    DINING = "dining"
    BATHROOM = "bathroom"
    WC = "wc"
    HALL = "hall"
    BALCONY = "balcony"
    STUDIO = "studio"  # combined living/kitchen/dining, as in sample plan 1
    OTHER = "other"

    @property
    def is_furnishable(self) -> bool:
        """Rooms we generate interior designs for."""
        return self not in {RoomType.WC, RoomType.HALL}


class OpeningType(str, Enum):
    DOOR = "door"
    WINDOW = "window"
    ARCH = "arch"  # opening with no leaf; blocks nothing but sight lines


class SwingDirection(str, Enum):
    INWARD = "inward"
    OUTWARD = "outward"
    SLIDING = "sliding"
    NONE = "none"


class RoomExtraction(BaseModel):
    """A room exactly as the vision model reports it — pixel space only."""

    id: str
    name: str
    room_type: RoomType
    polygon_px: list[Vec2] = Field(min_length=3)
    area_label_m2: float | None = Field(
        default=None,
        description="The m² figure printed inside the room, if the plan shows one. "
        "This is the primary calibration signal — report it verbatim, do not compute it.",
    )


class WallExtraction(BaseModel):
    id: str
    start_px: Vec2
    end_px: Vec2
    thickness_px: float = Field(default=0.0, ge=0)
    is_exterior: bool = False


class OpeningExtraction(BaseModel):
    id: str
    opening_type: OpeningType
    wall_id: str | None = None
    start_px: Vec2
    end_px: Vec2
    swing: SwingDirection = SwingDirection.NONE
    room_ids: list[str] = Field(default_factory=list)


class FloorPlanExtraction(BaseModel):
    """Raw structured output from the vision model, before calibration."""

    rooms: list[RoomExtraction]
    walls: list[WallExtraction] = Field(default_factory=list)
    openings: list[OpeningExtraction] = Field(default_factory=list)
    dimension_ticks: list[DimensionTick] = Field(
        default_factory=list,
        description="Printed dimension annotations (e.g. '370', '810'), used as a "
        "fallback calibration signal when m² labels are absent.",
    )
    notes: str | None = None


class DimensionTick(BaseModel):
    """A printed linear dimension, e.g. the '370' marks on sample plan 2."""

    start_px: Vec2
    end_px: Vec2
    value: float = Field(gt=0, description="Printed number, in `unit`.")
    unit: str = Field(default="cm", description="cm, mm, or m — as printed.")

    def value_in_metres(self) -> float:
        return {"mm": 0.001, "cm": 0.01, "m": 1.0}.get(self.unit, 0.01) * self.value


class ScaleCalibration(BaseModel):
    """Result of the pixel→metre solve."""

    px_per_m: float = Field(gt=0)
    method: str = Field(description="area_labels | dimension_ticks | manual | default")
    residual_pct: float = Field(
        default=0.0,
        description="Mean |predicted - printed| / printed across rooms. "
        "Above ~8% the extraction is probably wrong, not the scale.",
    )
    confidence: float = Field(ge=0, le=1)
    sample_count: int = 0
    warnings: list[str] = Field(default_factory=list)


class Room(BaseModel):
    """A calibrated room, in metres."""

    id: str
    name: str
    room_type: RoomType
    polygon_m: list[Vec2] = Field(min_length=3)
    polygon_px: list[Vec2] = Field(min_length=3)
    area_label_m2: float | None = None
    ceiling_height_m: float = 2.7

    @property
    def area_m2(self) -> float:
        return polygon_area(self.polygon_m)

    @property
    def centroid_m(self) -> Vec2:
        return polygon_centroid(self.polygon_m)

    @property
    def bounds_m(self) -> tuple[Vec2, Vec2]:
        return polygon_bounds(self.polygon_m)

    def area_error_pct(self) -> float | None:
        """How far the calibrated area drifts from the printed label."""
        if not self.area_label_m2:
            return None
        return abs(self.area_m2 - self.area_label_m2) / self.area_label_m2 * 100.0


class Wall(BaseModel):
    id: str
    start_m: Vec2
    end_m: Vec2
    thickness_m: float = 0.15
    is_exterior: bool = False
    height_m: float = 2.7

    @property
    def length_m(self) -> float:
        return self.start_m.distance_to(self.end_m)

    @property
    def normal(self) -> Vec2:
        """Unit normal, pointing to the wall's left as you walk start→end."""
        d = self.end_m - self.start_m
        length = d.length() or 1.0
        return Vec2(x=-d.y / length, y=d.x / length)


class Opening(BaseModel):
    id: str
    opening_type: OpeningType
    wall_id: str | None = None
    start_m: Vec2
    end_m: Vec2
    swing: SwingDirection = SwingDirection.NONE
    room_ids: list[str] = Field(default_factory=list)

    # Sensible architectural defaults; the plan rarely prints these.
    sill_height_m: float = 0.0
    head_height_m: float = 2.1

    @property
    def width_m(self) -> float:
        return self.start_m.distance_to(self.end_m)

    @property
    def centre_m(self) -> Vec2:
        return Vec2(
            x=(self.start_m.x + self.end_m.x) / 2,
            y=(self.start_m.y + self.end_m.y) / 2,
        )

    def keepout_radius_m(self) -> float:
        """Clear floor space this opening needs in front of it.

        A hinged door needs its full swing arc; a window only needs enough
        space that furniture doesn't visually block it.
        """
        if self.opening_type == OpeningType.WINDOW:
            return 0.3
        if self.swing in (SwingDirection.SLIDING, SwingDirection.NONE):
            return 0.6
        return max(self.width_m, 0.8)


class FloorPlan(BaseModel):
    """A calibrated floor plan. The input contract for the design stage."""

    id: str
    source_filename: str
    image_width_px: int
    image_height_px: int
    calibration: ScaleCalibration
    rooms: list[Room]
    walls: list[Wall] = Field(default_factory=list)
    openings: list[Opening] = Field(default_factory=list)
    notes: str | None = None

    def room(self, room_id: str) -> Room | None:
        return next((r for r in self.rooms if r.id == room_id), None)

    def furnishable_rooms(self) -> list[Room]:
        return [r for r in self.rooms if r.room_type.is_furnishable]

    def openings_for_room(self, room_id: str) -> list[Opening]:
        return [o for o in self.openings if room_id in o.room_ids]

    @property
    def total_area_m2(self) -> float:
        return sum(r.area_m2 for r in self.rooms)


# `FloorPlanExtraction` references `DimensionTick` before its definition.
FloorPlanExtraction.model_rebuild()
