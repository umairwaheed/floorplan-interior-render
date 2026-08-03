"""Normalized product schema.

Every supplier adapter maps onto this one shape, so the design agent never
knows or cares which catalog a product came from. Adding a supplier means
writing an adapter, not touching retrieval, placement, or rendering.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, field_validator

from .common import Size3


class ProductCategory(str, Enum):
    """Deliberately coarse. Placement rules key off these, so the list is a
    closed set — anything a supplier calls something else gets mapped in."""

    # Seating
    SOFA = "sofa"
    ARMCHAIR = "armchair"
    DINING_CHAIR = "dining_chair"
    OFFICE_CHAIR = "office_chair"
    STOOL = "stool"
    BENCH = "bench"

    # Tables & surfaces
    COFFEE_TABLE = "coffee_table"
    DINING_TABLE = "dining_table"
    SIDE_TABLE = "side_table"
    DESK = "desk"
    CONSOLE = "console"

    # Sleeping
    BED = "bed"
    MATTRESS = "mattress"
    NIGHTSTAND = "nightstand"

    # Storage
    WARDROBE = "wardrobe"
    DRESSER = "dresser"
    SHELVING = "shelving"
    TV_UNIT = "tv_unit"
    SIDEBOARD = "sideboard"

    # Kitchen & bath
    KITCHEN_CABINET = "kitchen_cabinet"
    COUNTERTOP = "countertop"
    SINK = "sink"
    TOILET = "toilet"
    BATHTUB = "bathtub"
    SHOWER = "shower"
    VANITY = "vanity"
    APPLIANCE = "appliance"

    # Lighting
    CEILING_LIGHT = "ceiling_light"
    PENDANT_LIGHT = "pendant_light"
    FLOOR_LAMP = "floor_lamp"
    TABLE_LAMP = "table_lamp"
    WALL_LIGHT = "wall_light"

    # Soft furnishing & decor
    RUG = "rug"
    CURTAIN = "curtain"
    CUSHION = "cushion"
    THROW = "throw"
    ARTWORK = "artwork"
    MIRROR = "mirror"
    PLANT = "plant"
    DECOR = "decor"

    # Renovation finishes — these are products too, per the brief
    FLOORING = "flooring"
    WALL_PAINT = "wall_paint"
    WALL_TILE = "wall_tile"
    FLOOR_TILE = "floor_tile"
    WALLPAPER = "wallpaper"
    DOOR = "door"
    WINDOW = "window"
    TRIM = "trim"

    @property
    def is_finish(self) -> bool:
        """Finishes are applied to surfaces, not placed as objects."""
        return self in {
            ProductCategory.FLOORING,
            ProductCategory.WALL_PAINT,
            ProductCategory.WALL_TILE,
            ProductCategory.FLOOR_TILE,
            ProductCategory.WALLPAPER,
            ProductCategory.TRIM,
        }

    @property
    def is_ceiling_mounted(self) -> bool:
        return self in {
            ProductCategory.CEILING_LIGHT,
            ProductCategory.PENDANT_LIGHT,
        }

    @property
    def is_wall_mounted(self) -> bool:
        return self in {
            ProductCategory.ARTWORK,
            ProductCategory.MIRROR,
            ProductCategory.WALL_LIGHT,
            ProductCategory.CURTAIN,
        }


class DesignStyle(str, Enum):
    SCANDINAVIAN = "scandinavian"
    JAPANESE = "japanese"
    JAPANDI = "japandi"
    MODERN = "modern"
    CONTEMPORARY = "contemporary"
    MINIMALIST = "minimalist"
    INDUSTRIAL = "industrial"
    CLASSIC = "classic"
    LUXURY = "luxury"
    RUSTIC = "rustic"
    BOHEMIAN = "bohemian"
    MID_CENTURY = "mid_century"


class Dimensions(BaseModel):
    """Product dimensions as the supplier lists them, in millimetres."""

    width_mm: float = Field(gt=0)
    depth_mm: float = Field(gt=0)
    height_mm: float = Field(gt=0)
    is_estimated: bool = Field(
        default=False,
        description="True when dims were inferred (LLM enrichment or category "
        "median) rather than published. Surfaced in the BOM for honesty.",
    )

    def to_size3(self) -> Size3:
        return Size3(
            width=self.width_mm / 1000.0,
            depth=self.depth_mm / 1000.0,
            height=self.height_mm / 1000.0,
        )

    @property
    def footprint_m2(self) -> float:
        return (self.width_mm / 1000.0) * (self.depth_mm / 1000.0)


class Product(BaseModel):
    id: str = Field(description="Stable, globally unique: '{supplier}:{sku}'.")
    supplier: str
    sku: str
    name: str
    url: str | None = None
    description: str | None = None

    category: ProductCategory
    subcategory: str | None = None

    dimensions: Dimensions
    colors: list[str] = Field(default_factory=list)
    materials: list[str] = Field(default_factory=list)
    style_tags: list[DesignStyle] = Field(default_factory=list)

    price: float = Field(ge=0)
    currency: str = "GEL"
    in_stock: bool = True
    image_urls: list[str] = Field(default_factory=list)

    # Finishes are sold by area/volume, so a room needs a computed quantity.
    coverage_per_unit_m2: float | None = Field(
        default=None,
        description="For flooring/paint/tile: m² covered by one purchased unit.",
    )

    @field_validator("colors", "materials", mode="before")
    @classmethod
    def _normalize_tags(cls, v: object) -> list[str]:
        if v is None:
            return []
        if isinstance(v, str):
            return [v.strip().lower()]
        if isinstance(v, list):
            return [str(x).strip().lower() for x in v if str(x).strip()]
        return []

    @property
    def primary_color(self) -> str | None:
        return self.colors[0] if self.colors else None

    @property
    def primary_image(self) -> str | None:
        """The photo handed to the image model as an identity reference."""
        return self.image_urls[0] if self.image_urls else None

    def search_text(self) -> str:
        """Flattened text for embedding. Style and material carry the signal."""
        parts = [
            self.name,
            self.category.value.replace("_", " "),
            self.subcategory or "",
            " ".join(self.materials),
            " ".join(self.colors),
            " ".join(s.value for s in self.style_tags),
            self.description or "",
        ]
        return " ".join(p for p in parts if p).strip()

    def fits_within(self, width_m: float, depth_m: float, tolerance: float = 0.0) -> bool:
        """Footprint test, allowing a 90° turn."""
        size = self.dimensions.to_size3()
        w, d = width_m + tolerance, depth_m + tolerance
        return (size.width <= w and size.depth <= d) or (size.depth <= w and size.width <= d)


class ProductQuery(BaseModel):
    """Retrieval request. Hard filters are applied in SQL; `text` reranks."""

    text: str | None = None
    categories: list[ProductCategory] = Field(default_factory=list)
    styles: list[DesignStyle] = Field(default_factory=list)
    colors: list[str] = Field(default_factory=list)
    materials: list[str] = Field(default_factory=list)
    suppliers: list[str] = Field(default_factory=list)

    max_width_m: float | None = None
    max_depth_m: float | None = None
    max_height_m: float | None = None

    min_price: float | None = None
    max_price: float | None = None
    in_stock_only: bool = True
    limit: int = Field(default=20, ge=1, le=200)


class ProductMatch(BaseModel):
    product: Product
    score: float
    reason: str | None = None


class BOMLine(BaseModel):
    """One line of the bill of materials returned with every design."""

    product_id: str
    name: str
    supplier: str
    url: str | None = None
    category: ProductCategory
    quantity: float
    unit_price: float
    line_total: float
    currency: str = "GEL"
    instance_ids: list[str] = Field(default_factory=list)
    dimensions_estimated: bool = False


class BillOfMaterials(BaseModel):
    lines: list[BOMLine]
    currency: str = "GEL"

    @property
    def total_cost(self) -> float:
        return round(sum(line.line_total for line in self.lines), 2)

    @property
    def item_count(self) -> int:
        return len(self.lines)
