"""Normalization helpers for supplier data.

Supplier feeds are messy: free-text titles, dimensions buried in strings,
inconsistent units, colour and material only implied by the product name. These
functions do the deterministic, cheap part of cleanup. Anything they can't
resolve is left as `None` so the LLM enrichment pass can fill it — and whatever
still isn't resolvable is flagged as estimated rather than silently invented.

Keyword tables are English plus Georgian, since Gorgia and Comforter are
Georgian retailers and their titles are frequently untranslated.
"""

from __future__ import annotations

import re

from ..schemas.product import DesignStyle, Dimensions, ProductCategory

# --- Category inference ----------------------------------------------------
# Order matters: the first match wins, so specific terms precede general ones.
CATEGORY_KEYWORDS: list[tuple[ProductCategory, tuple[str, ...]]] = [
    (ProductCategory.COFFEE_TABLE, ("coffee table", "სასურველი მაგიდა", "журнальный")),
    (ProductCategory.DINING_TABLE, ("dining table", "სასადილო მაგიდა", "обеденный стол")),
    (ProductCategory.SIDE_TABLE, ("side table", "end table", "nightstand table", "bedside table")),
    (ProductCategory.NIGHTSTAND, ("nightstand", "bedside", "საწოლის მაგიდა", "тумба прикроват")),
    (ProductCategory.DESK, ("desk", "writing table", "საწერი მაგიდა", "письменный стол")),
    (ProductCategory.CONSOLE, ("console table", "console", "კონსოლი")),
    (ProductCategory.DINING_CHAIR, ("dining chair", "სასადილო სკამი", "обеденный стул")),
    (ProductCategory.OFFICE_CHAIR, ("office chair", "task chair", "საოფისე სკამი")),
    (ProductCategory.ARMCHAIR, ("armchair", "accent chair", "lounge chair", "სავარძელი", "кресло")),
    (ProductCategory.STOOL, ("stool", "bar stool", "ottoman", "pouf", "სკამი-ბარი")),
    (ProductCategory.BENCH, ("bench", "სკამი-გრძელი", "скамья")),
    (ProductCategory.SOFA, ("sofa", "couch", "settee", "sectional", "დივანი", "диван")),
    (ProductCategory.MATTRESS, ("mattress", "ლეიბი", "матрас")),
    (ProductCategory.BED, ("bed frame", "bed", "საწოლი", "кровать")),
    (ProductCategory.WARDROBE, ("wardrobe", "closet", "armoire", "კარადა", "шкаф")),
    (ProductCategory.DRESSER, ("dresser", "chest of drawers", "კომოდი", "комод")),
    (ProductCategory.TV_UNIT, ("tv unit", "tv stand", "media unit", "ტელევიზორის")),
    (ProductCategory.SIDEBOARD, ("sideboard", "buffet", "credenza")),
    (ProductCategory.SHELVING, ("shelf", "shelving", "bookcase", "თარო", "стеллаж")),
    (ProductCategory.KITCHEN_CABINET, ("kitchen cabinet", "base unit", "wall unit", "სამზარეულო")),
    (ProductCategory.COUNTERTOP, ("countertop", "worktop", "სამუშაო ზედაპირი")),
    (ProductCategory.VANITY, ("vanity", "washbasin cabinet", "სააბაზანო კარადა")),
    (ProductCategory.BATHTUB, ("bathtub", "bath tub", "აბაზანა", "ванна")),
    (ProductCategory.SHOWER, ("shower", "shower cabin", "საშხაპე", "душев")),
    (ProductCategory.TOILET, ("toilet", "wc pan", "უნიტაზი", "унитаз")),
    (ProductCategory.SINK, ("sink", "basin", "ნიჟარა", "раковина")),
    (ProductCategory.APPLIANCE, ("refrigerator", "fridge", "oven", "hob", "dishwasher", "ტექნიკა")),
    (ProductCategory.PENDANT_LIGHT, ("pendant", "chandelier", "hanging lamp", "ჭაღი", "люстра")),
    (ProductCategory.CEILING_LIGHT, ("ceiling light", "flush mount", "spotlight", "ჭერის")),
    (ProductCategory.FLOOR_LAMP, ("floor lamp", "იატაკის ნათურა", "торшер")),
    (ProductCategory.TABLE_LAMP, ("table lamp", "desk lamp", "მაგიდის ნათურა")),
    (ProductCategory.WALL_LIGHT, ("wall light", "sconce", "კედლის ნათურა", "бра")),
    (ProductCategory.RUG, ("rug", "carpet", "ხალიჩა", "ковер")),
    (ProductCategory.CURTAIN, ("curtain", "drape", "blind", "ფარდა", "штора")),
    (ProductCategory.CUSHION, ("cushion", "pillow", "ბალიში", "подушка")),
    (ProductCategory.THROW, ("throw", "blanket", "საბანი", "плед")),
    (ProductCategory.ARTWORK, ("artwork", "wall art", "painting", "poster", "print", "ნახატი")),
    (ProductCategory.MIRROR, ("mirror", "სარკე", "зеркало")),
    (ProductCategory.PLANT, ("plant", "planter", "მცენარე", "растение")),
    (ProductCategory.FLOORING, ("laminate", "parquet", "vinyl floor", "hardwood floor", "პარკეტი")),
    (ProductCategory.FLOOR_TILE, ("floor tile", "იატაკის ფილა", "напольная плитка")),
    (ProductCategory.WALL_TILE, ("wall tile", "კედლის ფილა", "настенная плитка")),
    (ProductCategory.WALLPAPER, ("wallpaper", "შპალერი", "обои")),
    (ProductCategory.WALL_PAINT, ("paint", "emulsion", "საღებავი", "краска")),
    (ProductCategory.TRIM, ("skirting", "baseboard", "cornice", "moulding", "პლინტუსი")),
    (ProductCategory.DOOR, ("door", "კარი", "дверь")),
    (ProductCategory.WINDOW, ("window", "ფანჯარა", "окно")),
    (ProductCategory.DECOR, ("vase", "candle", "decor", "ornament", "დეკორი")),
]

COLOR_KEYWORDS: dict[str, tuple[str, ...]] = {
    "white": ("white", "თეთრი", "белый"),
    "off-white": ("off-white", "ivory", "cream", "კრემისფერი"),
    "beige": ("beige", "sand", "oatmeal", "ბეჟი", "бежевый"),
    "grey": ("grey", "gray", "ნაცრისფერი", "серый"),
    "charcoal": ("charcoal", "anthracite", "graphite"),
    "black": ("black", "შავი", "черный"),
    "brown": ("brown", "chocolate", "ყავისფერი", "коричневый"),
    "tan": ("tan", "camel", "cognac", "caramel"),
    "natural": ("natural", "raw", "unfinished"),
    "oak": ("oak", "მუხა", "дуб"),
    "walnut": ("walnut", "კაკალი", "орех"),
    "green": ("green", "sage", "olive", "emerald", "მწვანე", "зеленый"),
    "blue": ("blue", "navy", "teal", "ლურჯი", "синий"),
    "terracotta": ("terracotta", "rust", "clay"),
    "mustard": ("mustard", "ochre", "amber"),
    "pink": ("pink", "blush", "ვარდისფერი"),
    "gold": ("gold", "brass", "champagne", "ოქროსფერი"),
    "silver": ("silver", "chrome", "steel grey"),
}

MATERIAL_KEYWORDS: dict[str, tuple[str, ...]] = {
    "oak": ("oak", "მუხა"),
    "walnut": ("walnut", "კაკალი"),
    "ash": ("ash wood", "ashwood"),
    "birch": ("birch", "არყის"),
    "pine": ("pine", "ფიჭვი"),
    "teak": ("teak",),
    "bamboo": ("bamboo", "ბამბუკი"),
    "mdf": ("mdf", "chipboard", "particle board"),
    "solid wood": ("solid wood", "მასივი", "массив"),
    "veneer": ("veneer", "ხის შპონი"),
    "leather": ("leather", "ტყავი", "кожа"),
    "faux leather": ("faux leather", "pu leather", "eco leather"),
    "velvet": ("velvet", "ხავერდი", "бархат"),
    "boucle": ("boucle", "bouclé"),
    "linen": ("linen", "სელი", "лен"),
    "cotton": ("cotton", "ბამბა", "хлопок"),
    "wool": ("wool", "მატყლი", "шерсть"),
    "jute": ("jute", "sisal", "seagrass"),
    "rattan": ("rattan", "cane", "wicker"),
    "glass": ("glass", "მინა", "стекло"),
    "marble": ("marble", "მარმარილო", "мрамор"),
    "granite": ("granite", "გრანიტი"),
    "ceramic": ("ceramic", "porcelain", "კერამიკა"),
    "steel": ("steel", "ფოლადი", "сталь"),
    "iron": ("iron", "wrought iron", "რკინა"),
    "brass": ("brass", "თითბერი", "латунь"),
    "aluminium": ("aluminium", "aluminum", "ალუმინი"),
    "concrete": ("concrete", "ბეტონი", "бетон"),
    "stone": ("stone", "ქვა", "камень"),
    "plastic": ("plastic", "polypropylene", "acrylic"),
}

STYLE_KEYWORDS: dict[DesignStyle, tuple[str, ...]] = {
    DesignStyle.SCANDINAVIAN: ("scandinavian", "nordic", "scandi"),
    DesignStyle.JAPANESE: ("japanese", "zen", "japan"),
    DesignStyle.JAPANDI: ("japandi",),
    DesignStyle.MODERN: ("modern",),
    DesignStyle.CONTEMPORARY: ("contemporary",),
    DesignStyle.MINIMALIST: ("minimalist", "minimal"),
    DesignStyle.INDUSTRIAL: ("industrial", "loft"),
    DesignStyle.CLASSIC: ("classic", "traditional", "provence"),
    DesignStyle.LUXURY: ("luxury", "premium", "deluxe", "prestige"),
    DesignStyle.RUSTIC: ("rustic", "farmhouse", "country"),
    DesignStyle.BOHEMIAN: ("bohemian", "boho", "eclectic"),
    DesignStyle.MID_CENTURY: ("mid century", "mid-century", "retro", "vintage"),
}

# Median real-world dimensions, in mm, used only when nothing else is available.
# Anything filled from here is marked `is_estimated=True` and surfaced in the BOM.
CATEGORY_DEFAULT_DIMS: dict[ProductCategory, tuple[float, float, float]] = {
    ProductCategory.SOFA: (2100, 900, 850),
    ProductCategory.ARMCHAIR: (800, 820, 780),
    ProductCategory.DINING_CHAIR: (460, 520, 880),
    ProductCategory.OFFICE_CHAIR: (620, 620, 1100),
    ProductCategory.STOOL: (400, 400, 450),
    ProductCategory.BENCH: (1200, 400, 450),
    ProductCategory.COFFEE_TABLE: (1100, 600, 420),
    ProductCategory.DINING_TABLE: (1600, 900, 750),
    ProductCategory.SIDE_TABLE: (450, 450, 550),
    ProductCategory.DESK: (1400, 700, 750),
    ProductCategory.CONSOLE: (1200, 350, 800),
    ProductCategory.BED: (1600, 2000, 900),
    ProductCategory.MATTRESS: (1600, 2000, 250),
    ProductCategory.NIGHTSTAND: (450, 400, 520),
    ProductCategory.WARDROBE: (1800, 600, 2200),
    ProductCategory.DRESSER: (1000, 450, 800),
    ProductCategory.SHELVING: (800, 350, 1800),
    ProductCategory.TV_UNIT: (1600, 400, 450),
    ProductCategory.SIDEBOARD: (1600, 450, 800),
    ProductCategory.KITCHEN_CABINET: (600, 600, 850),
    ProductCategory.COUNTERTOP: (2000, 600, 40),
    ProductCategory.SINK: (600, 450, 200),
    ProductCategory.TOILET: (380, 650, 800),
    ProductCategory.BATHTUB: (1700, 750, 600),
    ProductCategory.SHOWER: (900, 900, 2000),
    ProductCategory.VANITY: (800, 460, 850),
    ProductCategory.APPLIANCE: (600, 600, 850),
    ProductCategory.CEILING_LIGHT: (400, 400, 150),
    ProductCategory.PENDANT_LIGHT: (350, 350, 400),
    ProductCategory.FLOOR_LAMP: (400, 400, 1600),
    ProductCategory.TABLE_LAMP: (250, 250, 450),
    ProductCategory.WALL_LIGHT: (150, 200, 300),
    ProductCategory.RUG: (2000, 1400, 15),
    ProductCategory.CURTAIN: (1400, 50, 2600),
    ProductCategory.CUSHION: (450, 150, 450),
    ProductCategory.THROW: (1300, 40, 1700),
    ProductCategory.ARTWORK: (600, 30, 800),
    ProductCategory.MIRROR: (600, 30, 900),
    ProductCategory.PLANT: (450, 450, 1200),
    ProductCategory.DECOR: (200, 200, 300),
    ProductCategory.FLOORING: (1200, 190, 8),
    ProductCategory.FLOOR_TILE: (600, 600, 10),
    ProductCategory.WALL_TILE: (300, 600, 9),
    ProductCategory.WALLPAPER: (530, 1005, 1),
    ProductCategory.WALL_PAINT: (200, 200, 300),
    ProductCategory.TRIM: (2400, 20, 80),
    ProductCategory.DOOR: (800, 40, 2000),
    ProductCategory.WINDOW: (1200, 70, 1400),
}

# "220x90x85", "220 x 90 x 85", "220х90х85" (Cyrillic kha is a common typo-alike)
_DIM_TRIPLE = re.compile(
    r"(\d{1,5}(?:[.,]\d+)?)\s*[x×хX*]\s*(\d{1,5}(?:[.,]\d+)?)\s*[x×хX*]\s*(\d{1,5}(?:[.,]\d+)?)"
)
# "W220 D90 H85" in any order
_DIM_LABELLED = re.compile(
    r"(?:^|[^a-z])(?:w|width|სიგანე)\D{0,3}(\d{1,5})"
    r".*?(?:d|depth|სიღრმე)\D{0,3}(\d{1,5})"
    r".*?(?:h|height|სიმაღლე)\D{0,3}(\d{1,5})",
    re.IGNORECASE | re.DOTALL,
)
_UNIT_MM = re.compile(r"\b(mm|მმ|мм)\b", re.IGNORECASE)
_UNIT_CM = re.compile(r"\b(cm|სმ|см)\b", re.IGNORECASE)
_UNIT_M = re.compile(r"\b(m|მ|м)\b", re.IGNORECASE)


def _to_mm(values: tuple[float, float, float], text: str) -> tuple[float, float, float]:
    """Resolve units. Explicit markers win; otherwise magnitude decides.

    A sofa listed as "220x90x85" is centimetres — no supplier sells a 22 cm sofa.
    Values above 400 with no unit are almost always already millimetres.
    """
    if _UNIT_MM.search(text):
        factor = 1.0
    elif _UNIT_CM.search(text):
        factor = 10.0
    elif _UNIT_M.search(text) and max(values) < 10:
        factor = 1000.0
    else:
        factor = 1.0 if max(values) > 400 else 10.0
    return (values[0] * factor, values[1] * factor, values[2] * factor)


def parse_dimensions(text: str | None) -> Dimensions | None:
    """Pull W×D×H out of free text. Returns None rather than guessing."""
    if not text:
        return None

    match = _DIM_TRIPLE.search(text)
    if match:
        raw = tuple(float(g.replace(",", ".")) for g in match.groups())
    else:
        match = _DIM_LABELLED.search(text)
        if not match:
            return None
        raw = tuple(float(g) for g in match.groups())

    w, d, h = _to_mm(raw, text)  # type: ignore[arg-type]
    if not all(0 < v < 20000 for v in (w, d, h)):
        return None
    return Dimensions(width_mm=w, depth_mm=d, height_mm=h, is_estimated=False)


def default_dimensions(category: ProductCategory) -> Dimensions:
    """Category-median fallback, always flagged as estimated."""
    w, d, h = CATEGORY_DEFAULT_DIMS.get(category, (600, 600, 600))
    return Dimensions(width_mm=w, depth_mm=d, height_mm=h, is_estimated=True)


def infer_category(*texts: str | None) -> ProductCategory | None:
    haystack = " ".join(t.lower() for t in texts if t)
    if not haystack:
        return None
    for category, keywords in CATEGORY_KEYWORDS:
        if any(kw in haystack for kw in keywords):
            return category
    return None


def _match_keywords(haystack: str, table: dict[str, tuple[str, ...]]) -> list[str]:
    return [key for key, keywords in table.items() if any(kw in haystack for kw in keywords)]


def extract_colors(*texts: str | None) -> list[str]:
    haystack = " ".join(t.lower() for t in texts if t)
    return _match_keywords(haystack, COLOR_KEYWORDS)


def extract_materials(*texts: str | None) -> list[str]:
    haystack = " ".join(t.lower() for t in texts if t)
    return _match_keywords(haystack, MATERIAL_KEYWORDS)


def infer_styles(*texts: str | None) -> list[DesignStyle]:
    haystack = " ".join(t.lower() for t in texts if t)
    return [
        style
        for style, keywords in STYLE_KEYWORDS.items()
        if any(kw in haystack for kw in keywords)
    ]


def parse_price(value: object) -> float | None:
    """Tolerant price parse: '1 299,00 ₾', '1299.00 GEL', 1299 all work."""
    if value is None:
        return None
    if isinstance(value, int | float):
        return float(value)
    cleaned = re.sub(r"[^\d.,]", "", str(value))
    if not cleaned:
        return None
    # Treat the last separator as the decimal point, whichever it is.
    if "," in cleaned and "." in cleaned:
        cleaned = (
            cleaned.replace(".", "").replace(",", ".")
            if cleaned.rfind(",") > cleaned.rfind(".")
            else cleaned.replace(",", "")
        )
    elif "," in cleaned:
        parts = cleaned.split(",")
        cleaned = cleaned.replace(",", ".") if len(parts[-1]) == 2 else cleaned.replace(",", "")
    try:
        return float(cleaned)
    except ValueError:
        return None


def slugify(text: str, max_length: int = 48) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:max_length] or "item"
