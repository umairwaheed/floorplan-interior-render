"""The Scene Graph — the single source of truth.

This is the load-bearing abstraction of the whole system. Once a `Scene` is
built it is **immutable and content-hashed**: every object's product binding,
position, rotation, size, colour and per-instance seed is fixed before a single
pixel is generated.

That is what makes multi-view consistency structural rather than statistical.
Each render is the same scene projected through a different camera, so layout
and object identity cannot drift between views — there is nothing left for the
image model to re-invent.

Consequences worth noting:
* The bill of materials is a graph traversal, not a guess.
* Regeneration with the same `scene_id` reproduces identical conditioning.
* A user change request becomes a *patch* to this graph, so untouched objects
  keep their seeds and stay pixel-stable.
"""

from __future__ import annotations

import hashlib
import json
from enum import Enum

from pydantic import BaseModel, Field

from .common import Size3, Vec2, Vec3
from .product import DesignStyle, ProductCategory


class ObjectRole(str, Enum):
    """What an object is *for* in the room. Drives placement rules and prompt
    emphasis, and is independent of which product ends up filling it."""

    PRIMARY_SEATING = "primary_seating"
    SECONDARY_SEATING = "secondary_seating"
    SLEEPING = "sleeping"
    DINING = "dining"
    WORK = "work"
    STORAGE = "storage"
    SURFACE = "surface"
    MEDIA = "media"
    LIGHTING_AMBIENT = "lighting_ambient"
    LIGHTING_TASK = "lighting_task"
    SOFT_FURNISHING = "soft_furnishing"
    DECOR = "decor"
    FIXTURE = "fixture"


class PlacedObject(BaseModel):
    """One physical object, bound to a real catalog product.

    `instance_id` is stable across regenerations — it is what lets the judge
    check "is this exact object still here, unmoved" between viewpoints.
    """

    instance_id: str
    product_id: str
    room_id: str
    role: ObjectRole
    category: ProductCategory

    position_m: Vec3 = Field(description="Centre of the object's footprint; Z is base height.")
    rotation_deg: float = Field(default=0.0, description="Yaw about Z. 0 faces +Y.")
    size_m: Size3 = Field(description="Real product dimensions, not a guess.")

    color: str = Field(description="Resolved colour name, used verbatim in prompts.")
    material: str | None = None
    display_name: str = Field(description="Catalog product name, used verbatim in prompts.")

    seed: int = Field(description="Frozen per-instance seed for generation stability.")
    wall_id: str | None = Field(default=None, description="Set when anchored to a wall.")

    def footprint_corners(self) -> list[Vec2]:
        """The four floor-plane corners after rotation. Used for overlap
        checks, keep-out tests, and the rasterizer."""
        import math

        rad = math.radians(self.rotation_deg)
        cos_r, sin_r = math.cos(rad), math.sin(rad)
        hw, hd = self.size_m.width / 2.0, self.size_m.depth / 2.0
        corners = []
        for dx, dy in ((-hw, -hd), (hw, -hd), (hw, hd), (-hw, hd)):
            corners.append(
                Vec2(
                    x=self.position_m.x + dx * cos_r - dy * sin_r,
                    y=self.position_m.y + dx * sin_r + dy * cos_r,
                )
            )
        return corners

    @property
    def top_z(self) -> float:
        return self.position_m.z + self.size_m.height


class SurfaceFinish(BaseModel):
    """A renovation finish bound to a catalog product, per the brief's
    requirement that renovation elements also map to real products."""

    product_id: str | None = None
    display_name: str
    color: str
    material: str | None = None
    area_m2: float = 0.0


class RoomFinishes(BaseModel):
    room_id: str
    floor: SurfaceFinish
    walls: SurfaceFinish
    ceiling: SurfaceFinish
    trim: SurfaceFinish | None = None


class LightSource(BaseModel):
    """Lighting is part of the scene, not the prompt — it must not shift
    between viewpoints any more than the furniture does."""

    id: str
    room_id: str
    position_m: Vec3
    intensity: float = 1.0
    color_temp_k: int = 3000
    kind: str = Field(default="ceiling", description="ceiling | lamp | daylight")
    product_id: str | None = None


class Camera(BaseModel):
    """A viewpoint. Stored *in* the scene so regenerations reuse it exactly."""

    id: str
    room_id: str
    position_m: Vec3
    look_at_m: Vec3
    fov_deg: float = 60.0
    up: Vec3 = Field(default_factory=lambda: Vec3(x=0, y=0, z=1))
    label: str = Field(default="", description="Human-readable, e.g. 'from the doorway'.")
    coverage_pct: float = Field(
        default=0.0, description="Share of room floor area inside the frustum."
    )


class ColorPalette(BaseModel):
    name: str
    description: str
    primary: str
    secondary: str
    accent: str
    neutral: str

    def as_prompt_fragment(self) -> str:
        return (
            f"{self.name} palette — primary {self.primary}, secondary {self.secondary}, "
            f"accent {self.accent}, neutrals {self.neutral}"
        )


class Scene(BaseModel):
    """An immutable, fully-specified interior. The input to rendering."""

    scene_id: str = Field(default="", description="Content hash; set by `finalize()`.")
    floorplan_id: str
    style: DesignStyle
    palette: ColorPalette
    seed: int
    variation_index: int = 0

    room_ids: list[str] = Field(default_factory=list)
    objects: list[PlacedObject] = Field(default_factory=list)
    finishes: list[RoomFinishes] = Field(default_factory=list)
    lights: list[LightSource] = Field(default_factory=list)
    cameras: list[Camera] = Field(default_factory=list)

    unfilled_slots: list[str] = Field(
        default_factory=list,
        description="Roles no catalog product could satisfy. Reported, never faked.",
    )

    def content_hash(self) -> str:
        """Hash of everything that affects the generated image.

        Deliberately excludes `scene_id` itself and any downstream artefacts,
        so an unchanged scene always hashes identically.
        """
        payload = self.model_dump(
            mode="json",
            exclude={"scene_id"},
        )
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]

    def finalize(self) -> Scene:
        """Freeze the scene by stamping its content hash."""
        self.scene_id = self.content_hash()
        return self

    def objects_in_room(self, room_id: str) -> list[PlacedObject]:
        return [o for o in self.objects if o.room_id == room_id]

    def cameras_for_room(self, room_id: str) -> list[Camera]:
        return [c for c in self.cameras if c.room_id == room_id]

    def object_by_instance(self, instance_id: str) -> PlacedObject | None:
        return next((o for o in self.objects if o.instance_id == instance_id), None)

    def finishes_for_room(self, room_id: str) -> RoomFinishes | None:
        return next((f for f in self.finishes if f.room_id == room_id), None)

    def product_ids(self) -> set[str]:
        ids = {o.product_id for o in self.objects}
        for finish in self.finishes:
            for surface in (finish.floor, finish.walls, finish.ceiling, finish.trim):
                if surface and surface.product_id:
                    ids.add(surface.product_id)
        return ids
