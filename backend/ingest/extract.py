"""Vision extraction of floor plan geometry.

**What the model is and isn't asked for.** It reports *pixel* coordinates and
the numbers *printed on the drawing* — nothing else. It is never asked how many
metres wide a room is, because that is arithmetic over a global constant, and
`calibrate.py` solves it from the printed `m²` labels far more reliably while
self-checking the result. Asking a vision model for real-world dimensions gives
you a number with no way to know whether it's wrong.

**Two passes, because resolution is the binding constraint.** Architectural
drawings usually sit inside a page with title blocks, legends and notes, so the
drawing itself may occupy a fraction of the pixels. Pass 1 locates the drawing
region and reads the room labels; pass 2 re-extracts from a crop of just that
region, where the same walls and dimension ticks are several times larger. A
naive ink-density crop can't do this — it cannot tell a floor plan from a
paragraph of text — so the model does it, which is exactly the kind of
perception it's good at.
"""

from __future__ import annotations

import logging

import anthropic
from pydantic import BaseModel, Field

from ..config import Settings, get_settings
from ..schemas.floorplan import FloorPlanExtraction
from .loader import LoadedPlan, crop_to_region

logger = logging.getLogger(__name__)


class DrawingRegion(BaseModel):
    """Where the actual floor plan sits on the page, in normalized coordinates."""

    x0: float = Field(ge=0, le=1, description="Left edge, as a fraction of image width.")
    y0: float = Field(ge=0, le=1, description="Top edge, as a fraction of image height.")
    x1: float = Field(ge=0, le=1, description="Right edge, as a fraction of image width.")
    y1: float = Field(ge=0, le=1, description="Bottom edge, as a fraction of image height.")
    covers_most_of_page: bool = Field(
        description="True when the drawing already fills the page and cropping would gain nothing."
    )
    notes: str | None = None

    def as_box(self) -> tuple[float, float, float, float]:
        return (self.x0, self.y0, self.x1, self.y1)

    @property
    def area_fraction(self) -> float:
        return max(0.0, self.x1 - self.x0) * max(0.0, self.y1 - self.y0)


LOCATE_PROMPT = """\
This image may contain an architectural floor plan somewhere on the page, \
along with unrelated content: title blocks, legends, north arrows, logos, \
tables, or body text.

Return the bounding box of ONLY the floor plan drawing itself — the walls, \
rooms, doors and windows, together with the dimension annotations and room \
labels that belong to it.

Exclude: paragraphs of prose, evaluation criteria, page headers and footers, \
company logos, separate legend boxes, and standalone north-arrow compasses \
that sit away from the drawing.

Coordinates are fractions of the image size: (0,0) is top-left, (1,1) is \
bottom-right. If the drawing already occupies most of the page, set \
covers_most_of_page to true and return the full extent.
"""

EXTRACT_PROMPT = """\
Extract the geometry of this architectural floor plan.

## Coordinates

Report every coordinate in PIXELS of this image: (0,0) is top-left, x \
increases right, y increases down. Do not normalize. Do not convert to metres \
— a separate calibration step derives real-world scale from the printed area \
labels, so your job is pixel geometry and printed text, nothing more.

## Rooms

For each enclosed room, give a polygon tracing its INTERIOR face (the inside \
surface of the surrounding walls, not the wall centrelines). Use 4 points for \
a rectangular room; add points for L-shapes and alcoves. Trace the room's \
actual outline — do not approximate an L-shaped room as a rectangle.

`area_label_m2` is critical: if the drawing prints an area inside the room \
(for example "25.2 m²", "19.3 m²", "5.4 m²"), transcribe that number EXACTLY \
as printed. Do not compute it, round it, or infer it. If no area is printed \
for a room, leave it null — a wrong guess is far worse than a null, because \
these labels are what the scale calibration is fitted against.

Classify `room_type` from the room's printed name and its fixtures: a room \
with a bed is a bedroom, with a toilet or shower is a bathroom or wc, with \
counters and a sink is a kitchen, with a sofa is living. A single open space \
containing kitchen, dining and seating is `studio`. A narrow circulation \
space is `hall`. An outdoor space beyond the building envelope, often with a \
railing, is `balcony`.

## Walls, doors and windows

Give wall segments as start/end pixel points along the wall centreline, and \
mark whether each is exterior (on the building envelope) or interior.

For each door and window, give the two pixel endpoints of the opening in the \
wall. Doors usually show a quarter-circle swing arc; use the arc to set \
`swing` and note that the arc itself is not part of the opening. Windows are \
usually a thin gap with parallel lines across it. List every room the opening \
connects in `room_ids`.

## Dimension ticks

If the drawing prints linear dimensions (numbers along a measured line with \
end ticks, such as "370", "810", "210"), report each one: the two pixel \
endpoints of the measured span, the printed number, and its unit. These \
drawings almost always print centimetres. Report only true linear dimensions \
— not room areas, not flat or room numbers, not text.

## Accuracy

Be precise about which room is which and which rooms are adjacent. If \
something is genuinely ambiguous or illegible, say so in `notes` rather than \
inventing a value.
"""


class ExtractionError(RuntimeError):
    """Raised when the vision model cannot produce usable geometry."""


class FloorPlanExtractor:
    """Wraps the vision model. The only place image pixels meet an LLM."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        if not self.settings.anthropic_api_key:
            raise ExtractionError(
                "ANTHROPIC_API_KEY is not set — floor plan extraction needs it. "
                "Copy .env.example to .env and add a key."
            )
        self.client = anthropic.Anthropic(api_key=self.settings.anthropic_api_key)

    def _image_block(self, plan: LoadedPlan) -> dict[str, object]:
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": plan.to_base64_png(),
            },
        }

    def _parse(self, plan: LoadedPlan, prompt: str, schema: type[BaseModel], max_tokens: int):
        """One structured-output vision call.

        Streams because floor plan extraction can produce a large JSON payload
        and adaptive thinking runs on top of it — a non-streaming call at this
        `max_tokens` risks an HTTP timeout.
        """
        with self.client.messages.stream(
            model=self.settings.vision_model,
            max_tokens=max_tokens,
            output_config={"effort": "high"},
            output_format=schema,
            messages=[
                {
                    "role": "user",
                    "content": [self._image_block(plan), {"type": "text", "text": prompt}],
                }
            ],
        ) as stream:
            response = stream.get_final_message()

        if response.stop_reason == "refusal":
            raise ExtractionError(
                "The vision model declined to process this image "
                f"({getattr(response.stop_details, 'category', 'unknown')})."
            )
        if response.stop_reason == "max_tokens":
            raise ExtractionError(
                "Extraction hit the token limit before completing — the plan may be unusually "
                "complex. Retry with a higher max_tokens or a cropped region."
            )

        parsed = getattr(response, "parsed_output", None)
        if parsed is None:
            raise ExtractionError("The vision model returned no structured output.")
        return parsed

    def locate_drawing(self, plan: LoadedPlan) -> DrawingRegion:
        """Pass 1 — find the drawing region on the page."""
        return self._parse(plan, LOCATE_PROMPT, DrawingRegion, max_tokens=2000)

    def extract_geometry(self, plan: LoadedPlan) -> FloorPlanExtraction:
        """Pass 2 — extract rooms, walls, openings and dimension ticks."""
        return self._parse(plan, EXTRACT_PROMPT, FloorPlanExtraction, max_tokens=16000)

    def extract(
        self, plan: LoadedPlan, auto_crop: bool = True
    ) -> tuple[FloorPlanExtraction, LoadedPlan]:
        """Full extraction. Returns the geometry and the image it was measured against.

        Returning the (possibly cropped) plan matters: every pixel coordinate in
        the extraction is relative to *that* image, so the caller must keep them
        together or the calibration will be fitted against the wrong raster.
        """
        working = plan

        if auto_crop:
            try:
                region = self.locate_drawing(plan)
            except ExtractionError:
                logger.warning("drawing-region pass failed; extracting from the full page")
                region = None

            if region and not region.covers_most_of_page and 0.02 < region.area_fraction < 0.85:
                logger.info(
                    "cropping to drawing region (%.0f%% of page)", region.area_fraction * 100
                )
                try:
                    working = crop_to_region(plan, region.as_box())
                except ValueError:
                    logger.warning(
                        "model returned an invalid crop box %s; using full page", region.as_box()
                    )

        return self.extract_geometry(working), working
