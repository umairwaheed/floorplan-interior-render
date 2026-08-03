"""Furniture slots — what a room needs, before any product is chosen.

A *slot* is a role to fill ("this room needs primary seating"), not a product.
Separating the two is what keeps the design grounded: the model reasons about
what belongs in a room, and catalog retrieval decides which real product fills
each slot. The model never names a product, so it can never invent one.

The programs below are the deterministic baseline. They encode ordinary
interior-design practice — a bedroom needs a bed before it needs a plant — and
mean the whole pipeline runs with no API key at all. When a key is present the
LLM proposer refines the list for the specific room, style and budget; its
output is validated back into these same structures, so nothing downstream can
tell which path produced them.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from ..schemas.floorplan import RoomType
from ..schemas.product import ProductCategory
from ..schemas.scene import ObjectRole


class Placement(str, Enum):
    """How an object relates to the room or to another object.

    This is the hierarchy the solver exploits: anchors are searched globally,
    dependents are derived from their anchor. Real interiors are structured
    this way — a nightstand's position is a consequence of the bed's, not an
    independent decision — and modelling it makes the search dramatically
    smaller and the results better.
    """

    WALL_BACK = "wall_back"  # back flat against a wall (bed, sofa, wardrobe)
    CENTRE = "centre"  # free-standing in open floor (dining table)
    CORNER = "corner"  # tucked into a corner (floor lamp, plant)
    ADJACENT = "adjacent"  # beside an anchor (nightstand, coffee table)
    SURROUNDING = "surrounding"  # arrayed around an anchor (dining chairs)
    UNDER = "under"  # beneath an anchor (rug)
    ON_TOP = "on_top"  # on an anchor's surface (table lamp)
    CEILING = "ceiling"  # suspended (pendant, ceiling light)
    WALL_MOUNT = "wall_mount"  # hung on a wall (artwork, mirror)
    WINDOW = "window"  # at a window (curtains)

    @property
    def is_derived(self) -> bool:
        """True when position follows from an anchor rather than a search."""
        return self in {
            Placement.ADJACENT,
            Placement.SURROUNDING,
            Placement.UNDER,
            Placement.ON_TOP,
            Placement.CEILING,
            Placement.WALL_MOUNT,
            Placement.WINDOW,
        }


class FurnitureSlot(BaseModel):
    """One role to fill in one room."""

    slot_id: str
    room_id: str
    role: ObjectRole
    category: ProductCategory
    placement: Placement

    required: bool = Field(
        default=False,
        description="A room missing a required slot is reported as unfilled, never faked.",
    )
    quantity: int = Field(default=1, ge=1, le=12)
    anchor_slot_id: str | None = Field(
        default=None, description="Set for derived placements; the slot this one hangs off."
    )
    priority: int = Field(
        default=50, description="Lower is placed first. Anchors must precede their dependents."
    )
    min_room_area_m2: float = Field(
        default=0.0, description="Skip this slot in rooms smaller than this."
    )
    description: str | None = None


class SlotSpec(BaseModel):
    """A room-program entry, before it is bound to a specific room."""

    role: ObjectRole
    category: ProductCategory
    placement: Placement
    required: bool = False
    quantity: int = 1
    anchor: str | None = None  # category value of the anchoring slot
    priority: int = 50
    min_room_area_m2: float = 0.0
    #: Add one more of these per N m² above `min_room_area_m2`, capped by `quantity_max`.
    scale_per_m2: float | None = None
    quantity_max: int = 1


R = ObjectRole
C = ProductCategory
P = Placement

# Priorities: anchors 10-30, dependents 40-70, decor 80-99. The solver places in
# this order, so an anchor is always positioned before anything derived from it.
ROOM_PROGRAMS: dict[RoomType, list[SlotSpec]] = {
    RoomType.BEDROOM: [
        SlotSpec(
            role=R.SLEEPING, category=C.BED, placement=P.WALL_BACK, required=True, priority=10
        ),
        SlotSpec(
            role=R.STORAGE,
            category=C.WARDROBE,
            placement=P.WALL_BACK,
            priority=20,
            min_room_area_m2=8.0,
        ),
        SlotSpec(
            role=R.SURFACE,
            category=C.NIGHTSTAND,
            placement=P.ADJACENT,
            anchor="bed",
            quantity=2,
            priority=40,
            min_room_area_m2=7.0,
        ),
        SlotSpec(
            role=R.STORAGE,
            category=C.DRESSER,
            placement=P.WALL_BACK,
            priority=45,
            min_room_area_m2=14.0,
        ),
        SlotSpec(
            role=R.SOFT_FURNISHING, category=C.RUG, placement=P.UNDER, anchor="bed", priority=60
        ),
        SlotSpec(
            role=R.LIGHTING_TASK,
            category=C.TABLE_LAMP,
            placement=P.ON_TOP,
            anchor="nightstand",
            priority=65,
            min_room_area_m2=7.0,
        ),
        SlotSpec(
            role=R.LIGHTING_AMBIENT, category=C.PENDANT_LIGHT, placement=P.CEILING, priority=70
        ),
        SlotSpec(
            role=R.SOFT_FURNISHING, category=C.CURTAIN, placement=P.WINDOW, quantity=2, priority=75
        ),
        SlotSpec(role=R.DECOR, category=C.ARTWORK, placement=P.WALL_MOUNT, priority=85),
        SlotSpec(
            role=R.DECOR, category=C.PLANT, placement=P.CORNER, priority=90, min_room_area_m2=12.0
        ),
    ],
    RoomType.LIVING: [
        SlotSpec(
            role=R.PRIMARY_SEATING,
            category=C.SOFA,
            placement=P.WALL_BACK,
            required=True,
            priority=10,
        ),
        SlotSpec(
            role=R.MEDIA,
            category=C.TV_UNIT,
            placement=P.WALL_BACK,
            priority=20,
            min_room_area_m2=12.0,
        ),
        SlotSpec(
            role=R.SURFACE,
            category=C.COFFEE_TABLE,
            placement=P.ADJACENT,
            anchor="sofa",
            priority=40,
        ),
        SlotSpec(
            role=R.SECONDARY_SEATING,
            category=C.ARMCHAIR,
            placement=P.ADJACENT,
            anchor="sofa",
            priority=45,
            min_room_area_m2=16.0,
            scale_per_m2=10.0,
            quantity_max=2,
        ),
        SlotSpec(
            role=R.SOFT_FURNISHING, category=C.RUG, placement=P.UNDER, anchor="sofa", priority=60
        ),
        SlotSpec(
            role=R.STORAGE,
            category=C.SHELVING,
            placement=P.WALL_BACK,
            priority=50,
            min_room_area_m2=18.0,
        ),
        SlotSpec(role=R.LIGHTING_TASK, category=C.FLOOR_LAMP, placement=P.CORNER, priority=65),
        SlotSpec(
            role=R.LIGHTING_AMBIENT, category=C.PENDANT_LIGHT, placement=P.CEILING, priority=70
        ),
        SlotSpec(
            role=R.SOFT_FURNISHING, category=C.CURTAIN, placement=P.WINDOW, quantity=2, priority=75
        ),
        SlotSpec(
            role=R.SOFT_FURNISHING,
            category=C.CUSHION,
            placement=P.ON_TOP,
            anchor="sofa",
            quantity=3,
            priority=80,
        ),
        SlotSpec(role=R.DECOR, category=C.ARTWORK, placement=P.WALL_MOUNT, priority=85),
        SlotSpec(
            role=R.DECOR, category=C.PLANT, placement=P.CORNER, priority=90, min_room_area_m2=14.0
        ),
    ],
    RoomType.STUDIO: [
        # A single open space doing living, dining and kitchen duty — sample
        # plan 1's 25.2 m² room. The zones must not collide, which is what the
        # circulation and spread terms in the solver's cost are for.
        SlotSpec(
            role=R.PRIMARY_SEATING,
            category=C.SOFA,
            placement=P.WALL_BACK,
            required=True,
            priority=10,
        ),
        SlotSpec(
            role=R.DINING, category=C.DINING_TABLE, placement=P.CENTRE, required=True, priority=15
        ),
        SlotSpec(role=R.MEDIA, category=C.TV_UNIT, placement=P.WALL_BACK, priority=25),
        SlotSpec(
            role=R.FIXTURE,
            category=C.KITCHEN_CABINET,
            placement=P.WALL_BACK,
            quantity=3,
            priority=20,
        ),
        SlotSpec(
            role=R.DINING,
            category=C.DINING_CHAIR,
            placement=P.SURROUNDING,
            anchor="dining_table",
            quantity=4,
            priority=45,
        ),
        SlotSpec(
            role=R.SURFACE,
            category=C.COFFEE_TABLE,
            placement=P.ADJACENT,
            anchor="sofa",
            priority=40,
        ),
        SlotSpec(
            role=R.SOFT_FURNISHING, category=C.RUG, placement=P.UNDER, anchor="sofa", priority=60
        ),
        SlotSpec(
            role=R.LIGHTING_AMBIENT,
            category=C.PENDANT_LIGHT,
            placement=P.CEILING,
            anchor="dining_table",
            priority=70,
        ),
        SlotSpec(role=R.LIGHTING_TASK, category=C.FLOOR_LAMP, placement=P.CORNER, priority=65),
        SlotSpec(
            role=R.SOFT_FURNISHING, category=C.CURTAIN, placement=P.WINDOW, quantity=2, priority=75
        ),
        SlotSpec(role=R.DECOR, category=C.ARTWORK, placement=P.WALL_MOUNT, priority=85),
        SlotSpec(role=R.DECOR, category=C.PLANT, placement=P.CORNER, priority=90),
    ],
    RoomType.DINING: [
        SlotSpec(
            role=R.DINING, category=C.DINING_TABLE, placement=P.CENTRE, required=True, priority=10
        ),
        SlotSpec(
            role=R.DINING,
            category=C.DINING_CHAIR,
            placement=P.SURROUNDING,
            anchor="dining_table",
            quantity=4,
            priority=40,
            scale_per_m2=6.0,
            quantity_max=6,
        ),
        SlotSpec(
            role=R.STORAGE,
            category=C.SIDEBOARD,
            placement=P.WALL_BACK,
            priority=30,
            min_room_area_m2=12.0,
        ),
        SlotSpec(
            role=R.LIGHTING_AMBIENT,
            category=C.PENDANT_LIGHT,
            placement=P.CEILING,
            anchor="dining_table",
            priority=70,
        ),
        SlotSpec(
            role=R.SOFT_FURNISHING,
            category=C.RUG,
            placement=P.UNDER,
            anchor="dining_table",
            priority=60,
        ),
        SlotSpec(role=R.DECOR, category=C.ARTWORK, placement=P.WALL_MOUNT, priority=85),
    ],
    RoomType.KITCHEN: [
        SlotSpec(
            role=R.FIXTURE,
            category=C.KITCHEN_CABINET,
            placement=P.WALL_BACK,
            required=True,
            quantity=4,
            priority=10,
            scale_per_m2=3.0,
            quantity_max=8,
        ),
        SlotSpec(role=R.FIXTURE, category=C.COUNTERTOP, placement=P.WALL_BACK, priority=15),
        SlotSpec(role=R.FIXTURE, category=C.SINK, placement=P.WALL_BACK, priority=20),
        SlotSpec(
            role=R.FIXTURE, category=C.APPLIANCE, placement=P.WALL_BACK, quantity=2, priority=25
        ),
        SlotSpec(
            role=R.LIGHTING_AMBIENT, category=C.CEILING_LIGHT, placement=P.CEILING, priority=70
        ),
    ],
    RoomType.BATHROOM: [
        SlotSpec(
            role=R.FIXTURE, category=C.VANITY, placement=P.WALL_BACK, required=True, priority=10
        ),
        SlotSpec(
            role=R.FIXTURE, category=C.TOILET, placement=P.WALL_BACK, required=True, priority=15
        ),
        SlotSpec(role=R.FIXTURE, category=C.SHOWER, placement=P.CORNER, priority=20),
        SlotSpec(
            role=R.FIXTURE,
            category=C.BATHTUB,
            placement=P.WALL_BACK,
            priority=25,
            min_room_area_m2=6.0,
        ),
        SlotSpec(
            role=R.DECOR, category=C.MIRROR, placement=P.WALL_MOUNT, anchor="vanity", priority=80
        ),
        SlotSpec(
            role=R.LIGHTING_AMBIENT, category=C.CEILING_LIGHT, placement=P.CEILING, priority=70
        ),
    ],
    RoomType.WC: [
        SlotSpec(
            role=R.FIXTURE, category=C.TOILET, placement=P.WALL_BACK, required=True, priority=10
        ),
        SlotSpec(role=R.FIXTURE, category=C.SINK, placement=P.WALL_BACK, priority=20),
        SlotSpec(
            role=R.DECOR, category=C.MIRROR, placement=P.WALL_MOUNT, anchor="sink", priority=80
        ),
        SlotSpec(
            role=R.LIGHTING_AMBIENT, category=C.CEILING_LIGHT, placement=P.CEILING, priority=70
        ),
    ],
    RoomType.BALCONY: [
        SlotSpec(
            role=R.SECONDARY_SEATING,
            category=C.ARMCHAIR,
            placement=P.WALL_BACK,
            quantity=2,
            priority=10,
        ),
        SlotSpec(
            role=R.SURFACE,
            category=C.SIDE_TABLE,
            placement=P.ADJACENT,
            anchor="armchair",
            priority=40,
        ),
        SlotSpec(role=R.DECOR, category=C.PLANT, placement=P.CORNER, quantity=2, priority=85),
    ],
    RoomType.HALL: [
        SlotSpec(
            role=R.STORAGE,
            category=C.CONSOLE,
            placement=P.WALL_BACK,
            priority=10,
            min_room_area_m2=4.0,
        ),
        SlotSpec(
            role=R.DECOR, category=C.MIRROR, placement=P.WALL_MOUNT, anchor="console", priority=80
        ),
        SlotSpec(
            role=R.LIGHTING_AMBIENT, category=C.CEILING_LIGHT, placement=P.CEILING, priority=70
        ),
    ],
    RoomType.OTHER: [
        SlotSpec(role=R.SECONDARY_SEATING, category=C.ARMCHAIR, placement=P.WALL_BACK, priority=10),
        SlotSpec(
            role=R.SURFACE,
            category=C.SIDE_TABLE,
            placement=P.ADJACENT,
            anchor="armchair",
            priority=40,
        ),
        SlotSpec(
            role=R.LIGHTING_AMBIENT, category=C.CEILING_LIGHT, placement=P.CEILING, priority=70
        ),
    ],
}


def _scaled_quantity(spec: SlotSpec, area_m2: float) -> int:
    """Larger rooms get more of the things that scale — chairs, cabinets, seating."""
    quantity = spec.quantity
    if spec.scale_per_m2:
        extra = int(max(0.0, area_m2 - spec.min_room_area_m2) // spec.scale_per_m2)
        quantity = min(spec.quantity_max or spec.quantity, quantity + extra)
    return max(1, quantity)


def build_room_program(room_id: str, room_type: RoomType, area_m2: float) -> list[FurnitureSlot]:
    """The deterministic baseline program for one room.

    Slots whose `min_room_area_m2` exceeds the room are dropped — a 6 m² room
    should not be given a dresser just because bedrooms usually have one.
    """
    specs = ROOM_PROGRAMS.get(room_type, ROOM_PROGRAMS[RoomType.OTHER])
    slots: list[FurnitureSlot] = []

    for spec in specs:
        if area_m2 < spec.min_room_area_m2:
            continue
        slots.append(
            FurnitureSlot(
                slot_id=f"{room_id}:{spec.category.value}",
                room_id=room_id,
                role=spec.role,
                category=spec.category,
                placement=spec.placement,
                required=spec.required,
                quantity=_scaled_quantity(spec, area_m2),
                anchor_slot_id=f"{room_id}:{spec.anchor}" if spec.anchor else None,
                priority=spec.priority,
                min_room_area_m2=spec.min_room_area_m2,
            )
        )

    return resolve_dangling_anchors(slots)


def resolve_dangling_anchors(slots: list[FurnitureSlot]) -> list[FurnitureSlot]:
    """Drop derived slots whose anchor didn't survive, and sort into place order.

    A nightstand with no bed, or a mirror with no vanity, has nothing to hang
    off. Dropping it is correct; leaving it would make the solver invent a
    position with no relationship to anything.
    """
    present = {slot.slot_id for slot in slots}
    kept = [
        slot
        for slot in slots
        if not (
            slot.placement.is_derived and slot.anchor_slot_id and slot.anchor_slot_id not in present
        )
    ]

    # A dependent must never sort ahead of its anchor.
    for slot in kept:
        if slot.anchor_slot_id:
            anchor = next((s for s in kept if s.slot_id == slot.anchor_slot_id), None)
            if anchor and slot.priority <= anchor.priority:
                slot.priority = anchor.priority + 5

    return sorted(kept, key=lambda s: (s.priority, s.slot_id))
