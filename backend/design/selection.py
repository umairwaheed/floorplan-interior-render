"""Binds furniture slots to real catalog products.

This is the step that makes "use only products from the supplied catalog" true
by construction rather than by instruction. The slot says what role needs
filling; retrieval finds products that physically fit and match the style; the
best match wins. Nothing here can produce a product that isn't in the index, so
the render prompt and the bill of materials cannot disagree with each other.

When no product fits a slot, the slot is reported **unfilled** rather than
filled with something that doesn't fit. A sofa 40 cm too wide for the wall is
not a near-miss — it is a room that cannot be built.
"""

from __future__ import annotations

import logging

from ..catalog.service import CatalogService
from ..schemas.common import Size3, polygon_bounds
from ..schemas.floorplan import Room
from ..schemas.product import DesignStyle, Product, ProductCategory, ProductQuery
from ..schemas.scene import ColorPalette
from .placement import SlotFill
from .slots import FurnitureSlot, Placement
from .styles import StyleProfile

logger = logging.getLogger(__name__)

#: Fraction of a room's longer span that a single object may occupy. Stops a
#: 2.6 m sofa being chosen for a 2.8 m room, which "fits" but leaves no floor.
MAX_SPAN_RATIO = 0.75

#: Walking space that must survive in front of a wall-backed object.
MIN_CIRCULATION_M = 0.7

#: Categories where a tight dimensional cap makes no sense — they're small, or
#: sized by area rather than footprint.
_UNCONSTRAINED = {
    ProductCategory.RUG,
    ProductCategory.CURTAIN,
    ProductCategory.CUSHION,
    ProductCategory.THROW,
    ProductCategory.ARTWORK,
    ProductCategory.MIRROR,
    ProductCategory.PENDANT_LIGHT,
    ProductCategory.CEILING_LIGHT,
    ProductCategory.WALL_LIGHT,
    ProductCategory.TABLE_LAMP,
}


def slot_dimension_limits(room: Room, slot: FurnitureSlot) -> tuple[float | None, float | None]:
    """The largest footprint this slot may occupy in this room."""
    if slot.category in _UNCONSTRAINED:
        return None, None

    low, high = polygon_bounds(room.polygon_m)
    span_x, span_y = high.x - low.x, high.y - low.y
    longer, shorter = max(span_x, span_y), min(span_x, span_y)

    if slot.placement == Placement.WALL_BACK:
        # Width runs along the wall; depth projects into the room. The depth cap
        # is "what's left after circulation", not a flat fraction — a double bed
        # is 2.0 m deep and legitimately eats most of a small bedroom, so a
        # 50%-of-span rule would reject every bed that actually fits.
        return longer * MAX_SPAN_RATIO, max(0.4, shorter - MIN_CIRCULATION_M)
    if slot.placement == Placement.CENTRE:
        return longer * 0.6, shorter * 0.6
    return shorter * 0.5, shorter * 0.5


def build_query(
    room: Room,
    slot: FurnitureSlot,
    profile: StyleProfile,
    palette: ColorPalette,
    budget_per_item: float | None,
) -> ProductQuery:
    """Turn a slot plus the user's style choice into a retrieval query.

    The style profile feeds both the hard filters (its preferred materials and
    colours) and the free-text rerank (its retrieval terms), which is why the
    products that come back actually look like the style the prompt claims.
    """
    max_width, max_depth = slot_dimension_limits(room, slot)

    return ProductQuery(
        text=" ".join(
            [
                slot.category.value.replace("_", " "),
                *profile.retrieval_terms[:4],
                palette.name.lower(),
            ]
        ),
        categories=[slot.category],
        styles=[profile.style],
        colors=profile.preferred_colors[:4],
        materials=profile.preferred_materials[:4],
        max_width_m=max_width,
        max_depth_m=max_depth,
        max_height_m=room.ceiling_height_m - 0.1,
        max_price=budget_per_item,
        limit=8,
    )


class SelectionResult:
    """What retrieval could and couldn't fill."""

    def __init__(self) -> None:
        self.fills: list[SlotFill] = []
        self.unfilled: list[str] = []
        self.total_cost: float = 0.0

    def add(self, slot: FurnitureSlot, product: Product, index: int) -> None:
        self.fills.append(SlotFill(slot=slot, product=product, index=index))
        self.total_cost += product.price

    def miss(self, slot: FurnitureSlot, reason: str) -> None:
        self.unfilled.append(f"{slot.slot_id} ({slot.category.value}): {reason}")


def select_products(
    room: Room,
    slots: list[FurnitureSlot],
    profile: StyleProfile,
    palette: ColorPalette,
    catalog: CatalogService,
    budget: float | None = None,
    seed: int = 0,
) -> SelectionResult:
    """Fill each slot from the catalog. Deterministic for a given seed.

    Variation between design variations comes from walking further down the
    ranked candidate list, not from randomising the query — so every variation
    is still drawn from genuinely good matches rather than from noise.
    """
    result = SelectionResult()
    budget_per_item = (budget / max(len(slots), 1) * 3.0) if budget else None

    for slot in slots:
        query = build_query(room, slot, profile, palette, budget_per_item)
        matches = catalog.search(query)

        if not matches:
            # Retry without the dimensional cap to tell "nothing fits" apart
            # from "nothing in this category at all" — the distinction matters
            # to whoever reads the report.
            relaxed = query.model_copy(update={"max_width_m": None, "max_depth_m": None})
            if catalog.search(relaxed):
                result.miss(slot, "no product small enough for this room")
            else:
                result.miss(slot, "no product in this category")
            continue

        # Deterministic pick, varied per variation seed but always from the
        # top of the ranking.
        offset = (seed // 7) % min(len(matches), 3)
        chosen = matches[offset].product

        for index in range(slot.quantity):
            result.add(slot, chosen, index)

    _drop_orphaned_dependents(result)

    if result.unfilled:
        logger.info("room %s: %d slot(s) unfilled", room.id, len(result.unfilled))

    return result


def _drop_orphaned_dependents(result: SelectionResult) -> None:
    """Remove fills whose anchor slot couldn't be filled.

    `build_room_program` already prunes dependents whose anchor slot doesn't
    exist, but a slot can exist and still find no product. Without this, a
    nightstand whose bed was never selected falls back to the room centroid —
    and a pair of them lands in a heap in the middle of the floor, anchored to
    nothing. Dropping them is correct: the point of a nightstand is its
    relationship to the bed.
    """
    filled = {fill.slot.slot_id for fill in result.fills}
    orphans = [
        fill
        for fill in result.fills
        if fill.slot.placement.is_derived
        and fill.slot.anchor_slot_id
        and fill.slot.anchor_slot_id not in filled
    ]
    if not orphans:
        return

    orphan_ids = {fill.instance_id for fill in orphans}
    result.fills = [fill for fill in result.fills if fill.instance_id not in orphan_ids]
    result.total_cost -= sum(fill.product.price for fill in orphans)

    for slot_id in sorted({fill.slot.slot_id for fill in orphans}):
        anchor = next(f.slot.anchor_slot_id for f in orphans if f.slot.slot_id == slot_id)
        result.unfilled.append(f"{slot_id}: dropped — its anchor ({anchor}) was not filled")


def fits_in_room(room: Room, size: Size3) -> bool:
    """Whether an object could physically fit the room at any rotation."""
    low, high = polygon_bounds(room.polygon_m)
    span_x, span_y = high.x - low.x, high.y - low.y
    return (size.width <= span_x and size.depth <= span_y) or (
        size.depth <= span_x and size.width <= span_y
    )


def pick_finish(
    catalog: CatalogService,
    categories: list[ProductCategory],
    profile: StyleProfile,
    palette: ColorPalette,
    style: DesignStyle,
) -> Product | None:
    """Choose a renovation finish — flooring, paint, tile.

    The brief calls for "every visible furniture **or renovation** element" to
    map to a real product, so finishes go through the same retrieval as
    furniture rather than becoming free text in a prompt.
    """
    matches = catalog.search(
        ProductQuery(
            text=" ".join([*profile.retrieval_terms[:3], *profile.preferred_colors[:2]]),
            categories=categories,
            styles=[style],
            colors=profile.preferred_colors[:4],
            materials=profile.preferred_materials[:4],
            limit=5,
        )
    )
    return matches[0].product if matches else None
