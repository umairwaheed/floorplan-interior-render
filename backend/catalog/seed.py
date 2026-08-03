"""Generates the seeded demo catalog.

The assessment names Gorgia and Comforter as catalog sources but ships no data
files or API, so this builds a realistic stand-in: ~280 products spread across
both suppliers' actual stated ranges, with plausible dimensions, materials,
colours, styles and GEL pricing.

Two deliberate choices:

* Records are emitted in **raw supplier shape** — dimensions as free text like
  `"2100x900x850 mm"`, prices as strings, colour implied by the title — so the
  import path runs through the real adapters and normalizers instead of
  bypassing them. If the parsers regress, the catalog build breaks.
* Generation is **deterministic** (fixed RNG seed), so the committed catalog is
  reproducible and diffs are meaningful.

Swapping in the real feeds means pointing the importer at them. Nothing
downstream changes.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..schemas.product import DesignStyle as S
from ..schemas.product import ProductCategory as C

SEED = 20260803

COLLECTIONS = [
    "Oslo",
    "Bergen",
    "Kyoto",
    "Nara",
    "Aarhus",
    "Malmo",
    "Verona",
    "Siena",
    "Lisbon",
    "Porto",
    "Vienna",
    "Prague",
    "Tbilisi",
    "Batumi",
    "Kutaisi",
    "Aspen",
    "Nordic",
    "Atlas",
    "Vega",
    "Lumen",
    "Terra",
    "Onyx",
    "Ivory",
    "Cascade",
    "Meridian",
    "Solace",
    "Harbour",
    "Linea",
    "Sable",
    "Marlow",
]


@dataclass
class ItemSpec:
    """A family of products to generate."""

    category: C
    supplier: str
    count: int
    noun: str
    colors: list[str]
    materials: list[str]
    styles: list[S]
    width_mm: tuple[int, int]
    depth_mm: tuple[int, int]
    height_mm: tuple[int, int]
    price_gel: tuple[int, int]
    unit: str = "mm"
    coverage_m2: float | None = None
    descriptors: list[str] = field(default_factory=list)


WOOD = ["oak", "walnut", "ash", "birch", "pine", "teak"]
SOFT = ["linen", "cotton", "wool", "velvet", "boucle"]
NEUTRALS = ["white", "off-white", "beige", "grey", "charcoal", "black", "tan"]
WARM = ["beige", "terracotta", "mustard", "tan", "brown", "olive"]
ALL_STYLES = list(S)

# Style tags are *derived from* a product's material and colour rather than
# picked at random. A pale oak and linen piece genuinely reads Scandinavian,
# Japandi and minimalist all at once; a black steel piece does not. Without
# this correlation, filtering for "Scandinavian" returns an arbitrary twelfth
# of the catalog and style adherence collapses — the tags have to mean
# something for retrieval to be worth anything.
MATERIAL_STYLES: dict[str, list[S]] = {
    "oak": [S.SCANDINAVIAN, S.JAPANDI, S.MINIMALIST, S.MODERN, S.CONTEMPORARY],
    "ash": [S.SCANDINAVIAN, S.JAPANDI, S.MINIMALIST],
    "birch": [S.SCANDINAVIAN, S.MINIMALIST, S.JAPANDI],
    "walnut": [S.MID_CENTURY, S.MODERN, S.CLASSIC, S.LUXURY, S.JAPANDI],
    "teak": [S.MID_CENTURY, S.JAPANESE, S.JAPANDI],
    "pine": [S.RUSTIC, S.SCANDINAVIAN, S.BOHEMIAN],
    "bamboo": [S.JAPANESE, S.JAPANDI, S.BOHEMIAN],
    "rattan": [S.BOHEMIAN, S.JAPANDI, S.CONTEMPORARY],
    "leather": [S.MID_CENTURY, S.INDUSTRIAL, S.CLASSIC, S.MODERN, S.LUXURY],
    "faux leather": [S.MODERN, S.CONTEMPORARY, S.INDUSTRIAL],
    "velvet": [S.LUXURY, S.CLASSIC, S.CONTEMPORARY, S.MID_CENTURY],
    "boucle": [S.CONTEMPORARY, S.MODERN, S.MINIMALIST],
    "linen": [S.SCANDINAVIAN, S.JAPANDI, S.CONTEMPORARY, S.MINIMALIST, S.RUSTIC],
    "cotton": [S.SCANDINAVIAN, S.CONTEMPORARY, S.BOHEMIAN],
    "wool": [S.SCANDINAVIAN, S.RUSTIC, S.CONTEMPORARY],
    "jute": [S.BOHEMIAN, S.RUSTIC, S.JAPANDI],
    "steel": [S.INDUSTRIAL, S.MODERN, S.MINIMALIST, S.CONTEMPORARY],
    "iron": [S.INDUSTRIAL, S.RUSTIC, S.CLASSIC],
    "brass": [S.LUXURY, S.CLASSIC, S.CONTEMPORARY, S.MID_CENTURY],
    "aluminium": [S.INDUSTRIAL, S.MODERN, S.MINIMALIST],
    "marble": [S.LUXURY, S.CONTEMPORARY, S.MODERN, S.CLASSIC],
    "granite": [S.MODERN, S.CONTEMPORARY, S.LUXURY],
    "concrete": [S.INDUSTRIAL, S.MINIMALIST, S.MODERN],
    "stone": [S.RUSTIC, S.INDUSTRIAL, S.JAPANESE],
    "glass": [S.MODERN, S.CONTEMPORARY, S.MINIMALIST, S.LUXURY],
    "ceramic": [S.JAPANDI, S.JAPANESE, S.CONTEMPORARY, S.RUSTIC],
    "terracotta": [S.BOHEMIAN, S.RUSTIC, S.JAPANDI],
    "mdf": [S.MODERN, S.CONTEMPORARY, S.MINIMALIST],
    "veneer": [S.MODERN, S.CONTEMPORARY, S.MID_CENTURY],
    "plastic": [S.MODERN, S.CONTEMPORARY, S.MINIMALIST],
    "solid wood": [S.RUSTIC, S.CLASSIC, S.SCANDINAVIAN],
}

COLOR_STYLES: dict[str, list[S]] = {
    "white": [S.MINIMALIST, S.SCANDINAVIAN, S.MODERN, S.JAPANESE],
    "off-white": [S.SCANDINAVIAN, S.JAPANDI, S.MINIMALIST, S.CONTEMPORARY],
    "beige": [S.SCANDINAVIAN, S.JAPANDI, S.CONTEMPORARY, S.RUSTIC],
    "grey": [S.MODERN, S.MINIMALIST, S.CONTEMPORARY, S.INDUSTRIAL],
    "charcoal": [S.INDUSTRIAL, S.MODERN, S.MINIMALIST, S.JAPANDI],
    "black": [S.INDUSTRIAL, S.MINIMALIST, S.MODERN, S.LUXURY, S.JAPANESE],
    "brown": [S.RUSTIC, S.CLASSIC, S.MID_CENTURY],
    "tan": [S.MID_CENTURY, S.CONTEMPORARY, S.BOHEMIAN],
    "natural": [S.SCANDINAVIAN, S.JAPANDI, S.JAPANESE, S.RUSTIC],
    "oak": [S.SCANDINAVIAN, S.JAPANDI, S.MINIMALIST],
    "walnut": [S.MID_CENTURY, S.CLASSIC, S.MODERN],
    "green": [S.BOHEMIAN, S.JAPANDI, S.CONTEMPORARY, S.LUXURY],
    "olive": [S.RUSTIC, S.BOHEMIAN, S.JAPANDI],
    "blue": [S.CLASSIC, S.CONTEMPORARY, S.MODERN],
    "terracotta": [S.BOHEMIAN, S.RUSTIC],
    "mustard": [S.MID_CENTURY, S.BOHEMIAN],
    "pink": [S.BOHEMIAN, S.CONTEMPORARY],
    "gold": [S.LUXURY, S.CLASSIC],
    "silver": [S.MODERN, S.INDUSTRIAL, S.MINIMALIST],
}


def styles_for(material: str, color: str, allowed: list[S], rng: random.Random) -> list[S]:
    """Derive coherent style tags from what the product is actually made of."""
    allowed_set = set(allowed)
    candidates = [
        s
        for s in dict.fromkeys(MATERIAL_STYLES.get(material, []) + COLOR_STYLES.get(color, []))
        if s in allowed_set
    ]
    if not candidates:
        candidates = list(allowed_set)
    rng.shuffle(candidates)
    upper = min(4, len(candidates))
    return candidates[: rng.randint(min(2, upper), upper)]


# --- Comforter: sofas, tables, beds, wardrobes, office furniture, mattresses,
#     textiles and home accessories -----------------------------------------
COMFORTER_SPECS = [
    ItemSpec(
        C.SOFA,
        "comforter",
        14,
        "3-Seat Sofa",
        NEUTRALS + ["green", "blue"],
        SOFT + ["leather", "faux leather"],
        ALL_STYLES,
        (1900, 2600),
        (850, 1000),
        (750, 900),
        (780, 4900),
        descriptors=["deep-seated", "low-profile", "tight-back", "loose-cushion"],
    ),
    ItemSpec(
        C.SOFA,
        "comforter",
        6,
        "2-Seat Sofa",
        NEUTRALS + ["green"],
        SOFT + ["leather"],
        ALL_STYLES,
        (1450, 1800),
        (820, 950),
        (750, 880),
        (620, 3400),
    ),
    ItemSpec(
        C.ARMCHAIR,
        "comforter",
        10,
        "Armchair",
        NEUTRALS + ["green", "mustard", "blue"],
        SOFT + ["leather", "rattan"],
        ALL_STYLES,
        (700, 950),
        (750, 900),
        (720, 1050),
        (620, 2400),
        descriptors=["swivel", "wing-back", "lounge", "accent"],
    ),
    ItemSpec(
        C.DINING_CHAIR,
        "comforter",
        8,
        "Dining Chair",
        NEUTRALS + ["green"],
        WOOD + ["leather", "linen", "steel"],
        ALL_STYLES,
        (420, 500),
        (480, 560),
        (820, 940),
        (180, 720),
    ),
    ItemSpec(
        C.OFFICE_CHAIR,
        "comforter",
        4,
        "Office Chair",
        ["black", "grey", "white"],
        ["leather", "faux leather", "steel", "plastic"],
        [S.MODERN, S.CONTEMPORARY, S.MINIMALIST],
        (580, 680),
        (580, 680),
        (1000, 1250),
        (420, 1600),
    ),
    ItemSpec(
        C.STOOL,
        "comforter",
        5,
        "Stool",
        NEUTRALS + ["mustard"],
        WOOD + ["velvet", "rattan"],
        ALL_STYLES,
        (350, 450),
        (350, 450),
        (420, 760),
        (140, 620),
    ),
    ItemSpec(
        C.BENCH,
        "comforter",
        3,
        "Bench",
        NEUTRALS,
        WOOD + ["linen"],
        ALL_STYLES,
        (1000, 1500),
        (350, 450),
        (420, 480),
        (320, 980),
    ),
    ItemSpec(
        C.COFFEE_TABLE,
        "comforter",
        8,
        "Coffee Table",
        NEUTRALS + ["oak", "walnut"],
        WOOD + ["glass", "marble", "steel"],
        ALL_STYLES,
        (900, 1300),
        (550, 750),
        (350, 460),
        (180, 2200),
        descriptors=["round", "rectangular", "nested", "oval"],
    ),
    ItemSpec(
        C.DINING_TABLE,
        "comforter",
        8,
        "Dining Table",
        NEUTRALS + ["oak", "walnut"],
        WOOD + ["glass", "marble", "steel"],
        ALL_STYLES,
        (1400, 2200),
        (800, 1000),
        (730, 780),
        (450, 4200),
        descriptors=["extendable", "round", "rectangular", "pedestal"],
    ),
    ItemSpec(
        C.SIDE_TABLE,
        "comforter",
        6,
        "Side Table",
        NEUTRALS + ["oak"],
        WOOD + ["marble", "steel", "glass"],
        ALL_STYLES,
        (380, 550),
        (380, 550),
        (480, 620),
        (160, 880),
    ),
    ItemSpec(
        C.DESK,
        "comforter",
        6,
        "Desk",
        NEUTRALS + ["oak", "walnut"],
        WOOD + ["steel", "glass"],
        ALL_STYLES,
        (1100, 1600),
        (600, 800),
        (730, 760),
        (480, 2400),
    ),
    ItemSpec(
        C.CONSOLE,
        "comforter",
        4,
        "Console Table",
        NEUTRALS + ["oak"],
        WOOD + ["marble", "steel"],
        ALL_STYLES,
        (1000, 1400),
        (300, 400),
        (750, 850),
        (380, 1600),
    ),
    ItemSpec(
        C.BED,
        "comforter",
        10,
        "Bed Frame",
        NEUTRALS + ["oak", "walnut", "green"],
        WOOD + ["linen", "velvet", "faux leather"],
        ALL_STYLES,
        (1450, 1900),
        (2000, 2150),
        (850, 1200),
        (540, 4600),
        descriptors=["upholstered", "slatted", "storage", "platform"],
    ),
    ItemSpec(
        C.MATTRESS,
        "comforter",
        6,
        "Mattress",
        ["white", "off-white"],
        ["cotton", "wool"],
        [S.MODERN, S.CONTEMPORARY],
        (1400, 1800),
        (1900, 2100),
        (200, 320),
        (620, 2800),
        descriptors=["pocket-spring", "memory-foam", "hybrid", "orthopaedic"],
    ),
    ItemSpec(
        C.NIGHTSTAND,
        "comforter",
        6,
        "Nightstand",
        NEUTRALS + ["oak", "walnut"],
        WOOD + ["mdf"],
        ALL_STYLES,
        (400, 550),
        (350, 450),
        (450, 620),
        (140, 780),
    ),
    ItemSpec(
        C.WARDROBE,
        "comforter",
        8,
        "Wardrobe",
        NEUTRALS + ["oak", "walnut"],
        WOOD + ["mdf", "glass"],
        ALL_STYLES,
        (1200, 2400),
        (550, 650),
        (2000, 2400),
        (480, 4800),
        descriptors=["sliding-door", "hinged", "mirrored", "open"],
    ),
    ItemSpec(
        C.DRESSER,
        "comforter",
        5,
        "Dresser",
        NEUTRALS + ["oak", "walnut"],
        WOOD + ["mdf"],
        ALL_STYLES,
        (900, 1400),
        (400, 500),
        (750, 900),
        (420, 2200),
    ),
    ItemSpec(
        C.SHELVING,
        "comforter",
        6,
        "Shelving Unit",
        NEUTRALS + ["oak"],
        WOOD + ["steel", "mdf"],
        ALL_STYLES,
        (700, 1100),
        (300, 400),
        (1500, 2100),
        (280, 1700),
    ),
    ItemSpec(
        C.TV_UNIT,
        "comforter",
        5,
        "TV Unit",
        NEUTRALS + ["oak", "walnut"],
        WOOD + ["mdf", "glass"],
        ALL_STYLES,
        (1300, 1900),
        (350, 450),
        (400, 520),
        (320, 1900),
    ),
    ItemSpec(
        C.SIDEBOARD,
        "comforter",
        4,
        "Sideboard",
        NEUTRALS + ["oak", "walnut"],
        WOOD + ["mdf"],
        ALL_STYLES,
        (1400, 1900),
        (400, 480),
        (750, 850),
        (540, 2600),
    ),
    ItemSpec(
        C.RUG,
        "comforter",
        10,
        "Area Rug",
        NEUTRALS + WARM + ["green", "blue"],
        ["wool", "cotton", "jute", "plastic"],
        ALL_STYLES,
        (1600, 3000),
        (1200, 2000),
        (8, 25),
        (180, 2400),
        descriptors=["hand-woven", "flat-weave", "shaggy", "tufted"],
    ),
    ItemSpec(
        C.CURTAIN,
        "comforter",
        8,
        "Curtain Panel",
        NEUTRALS + ["green", "blue"],
        ["linen", "cotton", "velvet"],
        ALL_STYLES,
        (1300, 1600),
        (30, 60),
        (2400, 2800),
        (90, 780),
        descriptors=["blackout", "sheer", "lined", "eyelet"],
    ),
    ItemSpec(
        C.CUSHION,
        "comforter",
        8,
        "Cushion",
        NEUTRALS + WARM + ["green", "blue", "pink"],
        SOFT + ["jute"],
        ALL_STYLES,
        (400, 550),
        (120, 180),
        (400, 550),
        (35, 220),
    ),
    ItemSpec(
        C.THROW,
        "comforter",
        5,
        "Throw Blanket",
        NEUTRALS + WARM + ["green"],
        ["wool", "cotton", "linen"],
        ALL_STYLES,
        (1200, 1500),
        (30, 60),
        (1600, 1900),
        (75, 420),
    ),
    ItemSpec(
        C.ARTWORK,
        "comforter",
        6,
        "Framed Print",
        NEUTRALS + ["gold"],
        ["glass", "oak", "aluminium"],
        ALL_STYLES,
        (400, 900),
        (25, 45),
        (500, 1200),
        (85, 680),
    ),
    ItemSpec(
        C.MIRROR,
        "comforter",
        5,
        "Wall Mirror",
        NEUTRALS + ["gold", "black"],
        ["glass", "oak", "brass", "steel"],
        ALL_STYLES,
        (450, 900),
        (25, 50),
        (600, 1400),
        (140, 1200),
    ),
    ItemSpec(
        C.PLANT,
        "comforter",
        5,
        "Potted Plant",
        ["green"],
        ["ceramic", "terracotta", "jute"],
        ALL_STYLES,
        (300, 600),
        (300, 600),
        (600, 1800),
        (60, 480),
        descriptors=["fiddle-leaf", "olive", "monstera", "rubber"],
    ),
    ItemSpec(
        C.DECOR,
        "comforter",
        6,
        "Ceramic Vase",
        NEUTRALS + WARM,
        ["ceramic", "glass", "stone"],
        ALL_STYLES,
        (140, 280),
        (140, 280),
        (200, 450),
        (35, 320),
    ),
]

# --- Gorgia: furniture, renovation materials, lighting, flooring, paint,
#     bathroom and kitchen products ------------------------------------------
GORGIA_SPECS = [
    ItemSpec(
        C.PENDANT_LIGHT,
        "gorgia",
        8,
        "Pendant Light",
        ["black", "white", "gold", "brass"],
        ["steel", "brass", "glass", "rattan", "ceramic"],
        ALL_STYLES,
        (250, 550),
        (250, 550),
        (300, 650),
        (120, 1800),
        descriptors=["dome", "globe", "cluster", "linear"],
    ),
    ItemSpec(
        C.CEILING_LIGHT,
        "gorgia",
        6,
        "Ceiling Light",
        ["white", "black", "silver"],
        ["steel", "glass", "aluminium"],
        ALL_STYLES,
        (300, 550),
        (300, 550),
        (100, 220),
        (90, 940),
    ),
    ItemSpec(
        C.FLOOR_LAMP,
        "gorgia",
        6,
        "Floor Lamp",
        ["black", "white", "brass", "grey"],
        ["steel", "brass", "linen", "oak"],
        ALL_STYLES,
        (300, 500),
        (300, 500),
        (1400, 1800),
        (160, 1400),
    ),
    ItemSpec(
        C.TABLE_LAMP,
        "gorgia",
        5,
        "Table Lamp",
        ["white", "black", "brass", "beige"],
        ["ceramic", "brass", "linen", "glass"],
        ALL_STYLES,
        (200, 320),
        (200, 320),
        (380, 560),
        (85, 720),
    ),
    ItemSpec(
        C.WALL_LIGHT,
        "gorgia",
        4,
        "Wall Sconce",
        ["black", "brass", "white"],
        ["steel", "brass", "glass"],
        ALL_STYLES,
        (120, 220),
        (150, 260),
        (220, 400),
        (70, 560),
    ),
    ItemSpec(
        C.FLOORING,
        "gorgia",
        8,
        "Laminate Flooring",
        ["oak", "walnut", "grey", "natural"],
        ["oak", "mdf", "veneer"],
        ALL_STYLES,
        (1200, 1380),
        (190, 240),
        (8, 12),
        (28, 145),
        coverage_m2=2.4,
        descriptors=["wide-plank", "herringbone", "matt-lacquered", "brushed"],
    ),
    ItemSpec(
        C.FLOOR_TILE,
        "gorgia",
        6,
        "Floor Tile",
        ["grey", "beige", "white", "charcoal"],
        ["ceramic", "granite", "stone"],
        ALL_STYLES,
        (600, 800),
        (600, 800),
        (9, 12),
        (32, 165),
        coverage_m2=1.44,
    ),
    ItemSpec(
        C.WALL_TILE,
        "gorgia",
        6,
        "Wall Tile",
        ["white", "beige", "green", "grey"],
        ["ceramic"],
        ALL_STYLES,
        (200, 400),
        (400, 800),
        (8, 10),
        (28, 140),
        coverage_m2=1.2,
    ),
    ItemSpec(
        C.WALLPAPER,
        "gorgia",
        4,
        "Wallpaper Roll",
        NEUTRALS + ["green"],
        ["cotton", "plastic"],
        ALL_STYLES,
        (530, 530),
        (1005, 1005),
        (1, 2),
        (45, 320),
        coverage_m2=5.3,
    ),
    ItemSpec(
        C.WALL_PAINT,
        "gorgia",
        8,
        "Interior Wall Paint",
        NEUTRALS + WARM + ["green", "blue"],
        ["plastic"],
        ALL_STYLES,
        (180, 260),
        (180, 260),
        (250, 380),
        (38, 260),
        coverage_m2=12.0,
        descriptors=["matt", "eggshell", "washable", "low-VOC"],
    ),
    ItemSpec(
        C.TRIM,
        "gorgia",
        3,
        "Skirting Board",
        ["white", "oak", "black"],
        ["mdf", "oak", "plastic"],
        ALL_STYLES,
        (2400, 2700),
        (16, 22),
        (60, 120),
        (12, 65),
        coverage_m2=None,
    ),
    ItemSpec(
        C.KITCHEN_CABINET,
        "gorgia",
        6,
        "Kitchen Base Unit",
        NEUTRALS + ["oak", "green"],
        ["mdf", "oak", "veneer"],
        ALL_STYLES,
        (450, 900),
        (560, 620),
        (820, 880),
        (180, 1400),
    ),
    ItemSpec(
        C.COUNTERTOP,
        "gorgia",
        4,
        "Kitchen Countertop",
        ["grey", "white", "black", "oak"],
        ["marble", "granite", "oak", "concrete"],
        ALL_STYLES,
        (1800, 3000),
        (600, 650),
        (30, 45),
        (240, 2600),
        coverage_m2=None,
    ),
    ItemSpec(
        C.SINK,
        "gorgia",
        4,
        "Kitchen Sink",
        ["silver", "black", "white"],
        ["steel", "ceramic", "granite"],
        ALL_STYLES,
        (500, 800),
        (400, 520),
        (180, 260),
        (140, 1100),
    ),
    ItemSpec(
        C.APPLIANCE,
        "gorgia",
        5,
        "Built-in Appliance",
        ["silver", "black", "white"],
        ["steel", "glass"],
        [S.MODERN, S.CONTEMPORARY, S.MINIMALIST, S.INDUSTRIAL],
        (550, 620),
        (550, 620),
        (450, 1800),
        (620, 4200),
        descriptors=["oven", "induction hob", "dishwasher", "refrigerator"],
    ),
    ItemSpec(
        C.TOILET,
        "gorgia",
        3,
        "Wall-Hung Toilet",
        ["white"],
        ["ceramic"],
        ALL_STYLES,
        (360, 400),
        (520, 620),
        (340, 420),
        (240, 1600),
    ),
    ItemSpec(
        C.BATHTUB,
        "gorgia",
        3,
        "Bathtub",
        ["white"],
        ["ceramic", "plastic", "stone"],
        ALL_STYLES,
        (1500, 1800),
        (700, 800),
        (550, 650),
        (620, 4400),
    ),
    ItemSpec(
        C.SHOWER,
        "gorgia",
        3,
        "Shower Enclosure",
        ["silver", "black"],
        ["glass", "aluminium", "steel"],
        ALL_STYLES,
        (800, 1200),
        (800, 1000),
        (1900, 2000),
        (480, 3200),
    ),
    ItemSpec(
        C.VANITY,
        "gorgia",
        4,
        "Bathroom Vanity",
        NEUTRALS + ["oak"],
        ["mdf", "oak", "ceramic"],
        ALL_STYLES,
        (600, 1200),
        (420, 500),
        (500, 850),
        (320, 2400),
    ),
    ItemSpec(
        C.DOOR,
        "gorgia",
        4,
        "Interior Door",
        ["white", "oak", "walnut", "black"],
        ["mdf", "oak", "veneer", "glass"],
        ALL_STYLES,
        (700, 900),
        (35, 45),
        (2000, 2100),
        (280, 1900),
    ),
    ItemSpec(
        C.WINDOW,
        "gorgia",
        3,
        "Window Unit",
        ["white", "grey", "black"],
        ["plastic", "aluminium", "oak"],
        ALL_STYLES,
        (900, 1600),
        (60, 80),
        (1200, 1600),
        (420, 2800),
    ),
]


def _price(rng: random.Random, low: int, high: int) -> str:
    """Prices are emitted as formatted strings to exercise `parse_price`."""
    value = rng.randint(low, high)
    rounded = value - (value % 5) + 9 if value > 100 else value
    return f"{rounded:,.2f}".replace(",", " ")


def _generate(spec: ItemSpec, rng: random.Random, index_start: int) -> list[dict[str, Any]]:
    records = []
    for i in range(spec.count):
        collection = COLLECTIONS[(index_start + i) % len(COLLECTIONS)]
        color = rng.choice(spec.colors)
        material = rng.choice(spec.materials)
        styles = styles_for(material, color, spec.styles, rng)
        descriptor = rng.choice(spec.descriptors) if spec.descriptors else ""

        w = rng.randint(*spec.width_mm)
        d = rng.randint(*spec.depth_mm)
        h = rng.randint(*spec.height_mm)

        name_bits = [collection, descriptor.title() if descriptor else "", spec.noun]
        name = " ".join(b for b in name_bits if b)

        sku = f"{spec.category.value.upper()[:4]}-{index_start + i:04d}"
        records.append(
            {
                "sku": sku,
                "title": f"{name} — {color.title()} {material.title()}",
                "category": spec.category.value.replace("_", " "),
                "normalized_category": spec.category.value,
                "description": (
                    f"{styles[0].value.replace('_', ' ').title()} {spec.noun.lower()} in "
                    f"{color} {material}. {descriptor.title() + '. ' if descriptor else ''}"
                    f"Part of the {collection} collection."
                ),
                # Free text on purpose — the dimension parser has to earn it.
                "dimensions": f"{w}x{d}x{h} {spec.unit}",
                "color": color,
                "material": material,
                "style": [s.value for s in styles],
                "price": _price(rng, *spec.price_gel),
                "currency": "GEL",
                "in_stock": rng.random() > 0.06,
                "url": f"https://example.{spec.supplier}.ge/p/{sku.lower()}",
                "images": [],
                **({"coverage_m2": spec.coverage_m2} if spec.coverage_m2 else {}),
            }
        )
    return records


def build_raw_catalogs() -> dict[str, list[dict[str, Any]]]:
    """Deterministically generate raw records per supplier."""
    rng = random.Random(SEED)
    out: dict[str, list[dict[str, Any]]] = {"gorgia": [], "comforter": []}
    index = 1
    for spec in COMFORTER_SPECS + GORGIA_SPECS:
        out[spec.supplier].extend(_generate(spec, rng, index))
        index += spec.count
    return out


def write_seed_files(catalog_dir: Path) -> dict[str, Path]:
    """Write one raw JSON feed per supplier. Returns supplier → path."""
    catalog_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for supplier, records in build_raw_catalogs().items():
        path = catalog_dir / f"{supplier}_products.json"
        path.write_text(
            json.dumps({"supplier": supplier, "products": records}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        paths[supplier] = path
    return paths


if __name__ == "__main__":
    from ..config import get_settings

    written = write_seed_files(get_settings().catalog_dir)
    for supplier, path in written.items():
        count = len(json.loads(path.read_text(encoding="utf-8"))["products"])
        print(f"{supplier:12s} {count:4d} products → {path}")
