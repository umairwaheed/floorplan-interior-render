"""The design agent — floor plan plus a style, out comes a scene graph.

Orchestrates the four stages: decide what each room needs (slots), fill those
needs from the catalog (selection), work out where everything goes (the
solver), and bind surface finishes. The result is a content-hashed `Scene`,
frozen before any pixel is generated.

The division of labour is the whole point. The LLM proposes *what* belongs in a
room — taste, which it is good at. Retrieval decides *which product*, from a
real catalog. Deterministic code decides *where*, because placement is
arithmetic over clearances and dimensions. No stage is asked to do another's
job, and the pipeline runs end-to-end with no API key at all, because the slot
programs have a rule-based baseline.
"""

from __future__ import annotations

import logging

from ..catalog.service import CatalogService, get_catalog_service
from ..config import Settings, get_settings
from ..schemas.common import Vec3
from ..schemas.floorplan import FloorPlan, Room
from ..schemas.product import DesignStyle, ProductCategory
from ..schemas.scene import (
    ColorPalette,
    LightSource,
    PlacedObject,
    RoomFinishes,
    Scene,
    SurfaceFinish,
)
from .placement import solve_room
from .selection import pick_finish, select_products
from .slots import build_room_program
from .styles import StyleProfile, get_style_profile

logger = logging.getLogger(__name__)


class DesignAgent:
    """Builds scene graphs. Stateless — safe to share across requests."""

    def __init__(
        self,
        catalog: CatalogService | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.catalog = catalog or get_catalog_service()

    # -- finishes ----------------------------------------------------------

    def _room_finishes(
        self, room: Room, profile: StyleProfile, palette: ColorPalette, style: DesignStyle
    ) -> RoomFinishes:
        """Bind floor, wall, ceiling and trim to catalog products where possible.

        Wet rooms get tile rather than timber — a bathroom floored in laminate
        is a design error the render would faithfully reproduce.
        """
        wet = room.room_type.value in {"bathroom", "wc", "kitchen"}

        floor_product = pick_finish(
            self.catalog,
            [ProductCategory.FLOOR_TILE] if wet else [ProductCategory.FLOORING],
            profile,
            palette,
            style,
        )
        wall_product = pick_finish(
            self.catalog,
            [ProductCategory.WALL_TILE] if wet else [ProductCategory.WALL_PAINT],
            profile,
            palette,
            style,
        )
        trim_product = pick_finish(self.catalog, [ProductCategory.TRIM], profile, palette, style)

        wall_area = self._wall_area(room)

        return RoomFinishes(
            room_id=room.id,
            floor=SurfaceFinish(
                product_id=floor_product.id if floor_product else None,
                display_name=floor_product.name if floor_product else "engineered oak flooring",
                color=(floor_product.primary_color if floor_product else None) or palette.neutral,
                material=floor_product.materials[0]
                if floor_product and floor_product.materials
                else "oak",
                area_m2=room.area_m2,
            ),
            walls=SurfaceFinish(
                product_id=wall_product.id if wall_product else None,
                display_name=wall_product.name if wall_product else "matt interior wall paint",
                color=(wall_product.primary_color if wall_product else None) or palette.primary,
                material=wall_product.materials[0]
                if wall_product and wall_product.materials
                else "paint",
                area_m2=wall_area,
            ),
            ceiling=SurfaceFinish(
                product_id=wall_product.id if wall_product else None,
                display_name="matt white ceiling",
                color=palette.primary,
                material="paint",
                area_m2=room.area_m2,
            ),
            trim=SurfaceFinish(
                product_id=trim_product.id if trim_product else None,
                display_name=trim_product.name if trim_product else "painted skirting",
                color=(trim_product.primary_color if trim_product else None) or palette.primary,
                material=trim_product.materials[0]
                if trim_product and trim_product.materials
                else "mdf",
                area_m2=0.0,
            ),
        )

    @staticmethod
    def _wall_area(room: Room) -> float:
        """Perimeter × ceiling height — what a paint quantity is derived from."""
        polygon = room.polygon_m
        perimeter = sum(
            polygon[i].distance_to(polygon[(i + 1) % len(polygon)]) for i in range(len(polygon))
        )
        return round(perimeter * room.ceiling_height_m, 2)

    # -- lighting ----------------------------------------------------------

    @staticmethod
    def _lights(room: Room, objects: list[PlacedObject]) -> list[LightSource]:
        """Light sources derived from placed fixtures, plus daylight.

        Lighting lives in the scene graph rather than in the prompt for the
        same reason furniture does: it must not shift between viewpoints.
        """
        lights: list[LightSource] = []

        for obj in objects:
            if obj.category.is_ceiling_mounted or obj.category in {
                ProductCategory.FLOOR_LAMP,
                ProductCategory.TABLE_LAMP,
                ProductCategory.WALL_LIGHT,
            }:
                lights.append(
                    LightSource(
                        id=f"light-{obj.instance_id}",
                        room_id=room.id,
                        position_m=obj.position_m,
                        intensity=1.0 if obj.category.is_ceiling_mounted else 0.55,
                        color_temp_k=2700 if not obj.category.is_ceiling_mounted else 3000,
                        kind="ceiling" if obj.category.is_ceiling_mounted else "lamp",
                        product_id=obj.product_id,
                    )
                )

        # Ambient daylight, so a room with no fixtures is still lit rather than
        # rendering as a black box.
        centroid = room.centroid_m
        lights.append(
            LightSource(
                id=f"daylight-{room.id}",
                room_id=room.id,
                position_m=Vec3(x=centroid.x, y=centroid.y, z=room.ceiling_height_m - 0.4),
                intensity=1.2,
                color_temp_k=5600,
                kind="daylight",
            )
        )
        return lights

    # -- main entry point --------------------------------------------------

    def design(
        self,
        floorplan: FloorPlan,
        style: DesignStyle,
        palette_name: str | None = None,
        room_ids: list[str] | None = None,
        seed: int = 0,
        variation_index: int = 0,
        budget: float | None = None,
    ) -> Scene:
        """Produce one complete, frozen scene graph."""
        profile = get_style_profile(style)
        palette = profile.palette_by_name(palette_name)

        rooms = (
            [r for r in floorplan.rooms if r.id in room_ids]
            if room_ids
            else floorplan.furnishable_rooms()
        )
        if not rooms:
            raise ValueError("No furnishable rooms were selected.")

        # The variation index perturbs the seed so different variations explore
        # genuinely different layouts and product picks, while each remains
        # exactly reproducible.
        effective_seed = seed + variation_index * 104729

        objects: list[PlacedObject] = []
        finishes: list[RoomFinishes] = []
        lights: list[LightSource] = []
        unfilled: list[str] = []

        for room in rooms:
            slots = build_room_program(room.id, room.room_type, room.area_m2)
            selection = select_products(
                room=room,
                slots=slots,
                profile=profile,
                palette=palette,
                catalog=self.catalog,
                budget=budget,
                seed=effective_seed,
            )
            unfilled.extend(selection.unfilled)

            placement = solve_room(
                room=room,
                openings=floorplan.openings_for_room(room.id),
                fills=selection.fills,
                seed=effective_seed,
                settings=self.settings,
            )
            objects.extend(placement.objects)
            unfilled.extend(placement.dropped)
            finishes.append(self._room_finishes(room, profile, palette, style))
            lights.extend(self._lights(room, placement.objects))

            logger.info(
                "room %s (%s, %.1f m²): %d objects, %d unfilled, %d unplaceable",
                room.id,
                room.room_type.value,
                room.area_m2,
                len(placement.objects),
                len(selection.unfilled),
                len(placement.dropped),
            )

        scene = Scene(
            floorplan_id=floorplan.id,
            style=style,
            palette=palette,
            seed=effective_seed,
            variation_index=variation_index,
            room_ids=[room.id for room in rooms],
            objects=objects,
            finishes=finishes,
            lights=lights,
            cameras=[],  # populated by the camera rig in the next stage
            unfilled_slots=unfilled,
        )
        return scene.finalize()

    def bill_of_materials(self, scene: Scene):
        """The BOM for a scene — a traversal, not an inference.

        Because the prompt is built from these same products, the list is
        provably what was rendered.
        """
        object_pairs = [(obj.instance_id, obj.product_id) for obj in scene.objects]

        finish_areas: dict[str, float] = {}
        for room_finish in scene.finishes:
            for surface in (room_finish.floor, room_finish.walls, room_finish.ceiling):
                if surface.product_id and surface.area_m2 > 0:
                    finish_areas[surface.product_id] = (
                        finish_areas.get(surface.product_id, 0.0) + surface.area_m2
                    )

        return self.catalog.build_bom(object_pairs, finish_areas)
