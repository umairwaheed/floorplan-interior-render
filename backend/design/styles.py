"""Style and palette definitions.

Each style carries three things that propagate in different directions:
* `retrieval_terms` / `preferred_materials` — bias catalog *selection*.
* `prompt_fragment` — goes into the image prompt verbatim.
* `palettes` — colour options offered to the user.

Keeping all three in one place is what makes style adherence consistent between
the products chosen and the pixels generated, rather than the prompt claiming
"Scandinavian" while the retrieved sofa is baroque.
"""

from __future__ import annotations

from pydantic import BaseModel

from ..schemas.product import DesignStyle
from ..schemas.scene import ColorPalette


class StyleProfile(BaseModel):
    style: DesignStyle
    label: str
    description: str
    prompt_fragment: str
    retrieval_terms: list[str]
    preferred_materials: list[str]
    preferred_colors: list[str]
    avoid_terms: list[str] = []
    palettes: list[ColorPalette]

    @property
    def default_palette(self) -> ColorPalette:
        return self.palettes[0]

    def palette_by_name(self, name: str | None) -> ColorPalette:
        if not name:
            return self.default_palette
        for palette in self.palettes:
            if palette.name.lower() == name.lower():
                return palette
        return self.default_palette


def _palette(
    name: str, desc: str, primary: str, secondary: str, accent: str, neutral: str
) -> ColorPalette:
    return ColorPalette(
        name=name,
        description=desc,
        primary=primary,
        secondary=secondary,
        accent=accent,
        neutral=neutral,
    )


STYLE_PROFILES: dict[DesignStyle, StyleProfile] = {
    DesignStyle.SCANDINAVIAN: StyleProfile(
        style=DesignStyle.SCANDINAVIAN,
        label="Scandinavian / Nordic",
        description="Light woods, soft neutrals, uncluttered warmth.",
        prompt_fragment=(
            "Scandinavian Nordic interior: pale oak and ash wood, white and soft grey "
            "surfaces, natural linen and wool textiles, clean uncluttered lines, "
            "abundant soft daylight, low contrast, cosy but restrained"
        ),
        retrieval_terms=["scandinavian", "nordic", "oak", "light wood", "minimal", "linen"],
        preferred_materials=["oak", "ash", "birch", "linen", "wool", "cotton"],
        preferred_colors=["white", "beige", "light grey", "natural", "cream", "sage"],
        avoid_terms=["ornate", "baroque", "gilded", "glossy black"],
        palettes=[
            _palette(
                "Nordic Light",
                "Warm whites with pale oak",
                "#F4F1EC",
                "#D9CFC1",
                "#8FA99B",
                "#EDEAE4",
            ),
            _palette(
                "Soft Sage", "Muted green against birch", "#EFEFE9", "#C7D2C6", "#7C8F7B", "#E4E1DA"
            ),
            _palette(
                "Warm Greige", "Greige with tan leather", "#EDE7DF", "#CBBBA5", "#9A7B56", "#E2DCD2"
            ),
        ],
    ),
    DesignStyle.JAPANDI: StyleProfile(
        style=DesignStyle.JAPANDI,
        label="Japandi",
        description="Japanese restraint meets Nordic warmth.",
        prompt_fragment=(
            "Japandi interior: low-profile furniture, warm oak and dark walnut, "
            "handmade ceramics, paper-diffused light, muted earth tones, negative space "
            "treated as a feature, matte finishes throughout, quiet and grounded"
        ),
        retrieval_terms=["japandi", "japanese", "low profile", "oak", "walnut", "linen"],
        preferred_materials=["oak", "walnut", "bamboo", "linen", "paper", "ceramic"],
        preferred_colors=["beige", "taupe", "charcoal", "off-white", "clay", "black"],
        avoid_terms=["chrome", "neon", "ornate", "plastic"],
        palettes=[
            _palette(
                "Muted Earth",
                "Clay and oatmeal with charcoal",
                "#E8E2D8",
                "#C4B49F",
                "#4A4742",
                "#DDD6CA",
            ),
            _palette(
                "Ink & Paper",
                "Off-white with black accents",
                "#F2EFE9",
                "#CFC8BC",
                "#2B2926",
                "#E5E0D6",
            ),
        ],
    ),
    DesignStyle.MINIMALIST: StyleProfile(
        style=DesignStyle.MINIMALIST,
        label="Minimalist",
        description="Reduced to essentials; form over ornament.",
        prompt_fragment=(
            "Minimalist interior: monochrome palette, flush handleless surfaces, "
            "concealed storage, one or two deliberate objects per surface, crisp shadows, "
            "no visible clutter, architectural calm"
        ),
        retrieval_terms=["minimalist", "handleless", "simple", "clean lines"],
        preferred_materials=["lacquer", "concrete", "glass", "steel", "oak"],
        preferred_colors=["white", "black", "grey", "concrete"],
        avoid_terms=["ornate", "floral", "patterned", "rustic"],
        palettes=[
            _palette(
                "Monochrome", "White, grey, black", "#FFFFFF", "#D6D6D6", "#1C1C1C", "#F0F0F0"
            ),
            _palette(
                "Concrete", "Raw grey with warm wood", "#E6E4E0", "#B3B0AA", "#7A6A55", "#D5D2CC"
            ),
        ],
    ),
    DesignStyle.MODERN: StyleProfile(
        style=DesignStyle.MODERN,
        label="Modern",
        description="Clean geometry, mixed materials, considered contrast.",
        prompt_fragment=(
            "Modern interior: strong horizontal lines, mixed materials of wood, matte "
            "metal and glass, bold but limited colour accents, integrated lighting, "
            "polished and intentional"
        ),
        retrieval_terms=["modern", "contemporary", "geometric", "matte"],
        preferred_materials=["walnut", "steel", "glass", "leather", "velvet"],
        preferred_colors=["charcoal", "white", "navy", "tan", "black"],
        avoid_terms=["rustic", "distressed", "floral"],
        palettes=[
            _palette(
                "Charcoal & Tan",
                "Deep grey with tan leather",
                "#E9E7E3",
                "#4C4C4C",
                "#B08155",
                "#DAD7D1",
            ),
            _palette(
                "Navy Accent", "Warm neutrals with navy", "#F1EFEB", "#CFCAC1", "#2C3E57", "#E3DFD8"
            ),
        ],
    ),
    DesignStyle.CONTEMPORARY: StyleProfile(
        style=DesignStyle.CONTEMPORARY,
        label="Contemporary",
        description="Current, soft-edged, layered neutrals.",
        prompt_fragment=(
            "Contemporary interior: curved silhouettes, layered neutral textiles, "
            "bouclé and brushed brass details, soft diffused lighting, gallery-like walls, "
            "current and magazine-ready"
        ),
        retrieval_terms=["contemporary", "curved", "boucle", "brass"],
        preferred_materials=["boucle", "brass", "marble", "oak", "velvet"],
        preferred_colors=["cream", "sand", "camel", "off-white", "brass"],
        avoid_terms=["distressed", "industrial pipe"],
        palettes=[
            _palette(
                "Sand & Cream", "Warm sand with brass", "#F3EEE7", "#DBCBB6", "#A8763E", "#E8E2D9"
            ),
            _palette(
                "Soft Clay", "Terracotta against cream", "#F2ECE5", "#D2B9A5", "#B5654A", "#E6DED4"
            ),
        ],
    ),
    DesignStyle.INDUSTRIAL: StyleProfile(
        style=DesignStyle.INDUSTRIAL,
        label="Industrial",
        description="Raw materials, exposed structure, utilitarian.",
        prompt_fragment=(
            "Industrial interior: exposed brick and concrete, blackened steel frames, "
            "reclaimed timber, visible ductwork and cable-suspended lighting, aged leather, "
            "raw and utilitarian with warm patina"
        ),
        retrieval_terms=["industrial", "metal", "reclaimed", "steel", "loft"],
        preferred_materials=["steel", "iron", "reclaimed wood", "leather", "concrete", "brick"],
        preferred_colors=["black", "rust", "grey", "brown", "copper"],
        avoid_terms=["pastel", "floral", "gilded"],
        palettes=[
            _palette(
                "Raw Steel",
                "Concrete, black steel, tan leather",
                "#D8D5D0",
                "#4A4744",
                "#8B5E3C",
                "#C6C2BB",
            ),
            _palette(
                "Brick & Iron", "Warm brick with iron", "#D9CFC6", "#8C5A44", "#2E2C2A", "#C4B9AE"
            ),
        ],
    ),
    DesignStyle.JAPANESE: StyleProfile(
        style=DesignStyle.JAPANESE,
        label="Japanese",
        description="Low, natural, deeply calm.",
        prompt_fragment=(
            "Japanese interior: tatami-inspired floor treatment, shoji-style screens, "
            "very low furniture, natural bamboo and cedar, indirect diffused light, "
            "profound emptiness and calm, asymmetric balance"
        ),
        retrieval_terms=["japanese", "low", "bamboo", "natural", "zen"],
        preferred_materials=["bamboo", "cedar", "paper", "cotton", "stone"],
        preferred_colors=["natural", "beige", "black", "moss", "off-white"],
        avoid_terms=["ornate", "chrome", "glossy"],
        palettes=[
            _palette(
                "Tatami", "Straw and cedar with black", "#E9E1CF", "#C2A878", "#332F2A", "#DDD4C0"
            ),
            _palette(
                "Moss Garden",
                "Deep green against paper white",
                "#F0EDE4",
                "#C9C7B4",
                "#4F5D45",
                "#E2DED2",
            ),
        ],
    ),
    DesignStyle.CLASSIC: StyleProfile(
        style=DesignStyle.CLASSIC,
        label="Classic",
        description="Symmetry, mouldings, timeless materials.",
        prompt_fragment=(
            "Classic interior: symmetrical arrangement, panelled walls and crown mouldings, "
            "polished hardwood, upholstered wing chairs, framed artwork, warm layered "
            "lighting from table lamps, timeless and formal"
        ),
        retrieval_terms=["classic", "traditional", "upholstered", "wood", "panelled"],
        preferred_materials=["mahogany", "oak", "velvet", "silk", "brass", "marble"],
        preferred_colors=["cream", "burgundy", "navy", "gold", "walnut"],
        avoid_terms=["industrial", "neon", "plastic"],
        palettes=[
            _palette(
                "Cream & Walnut",
                "Cream with rich walnut",
                "#F0E9DC",
                "#6B4A2F",
                "#9C7B3F",
                "#E2D9C8",
            ),
            _palette(
                "Deep Navy", "Navy panelling with brass", "#EDE7DA", "#22304A", "#B08D57", "#DED6C6"
            ),
        ],
    ),
    DesignStyle.LUXURY: StyleProfile(
        style=DesignStyle.LUXURY,
        label="Luxury",
        description="Rich materials, high contrast, statement pieces.",
        prompt_fragment=(
            "Luxury interior: book-matched marble, polished brass and smoked glass, "
            "deep velvet upholstery, statement chandelier, high-contrast dark joinery, "
            "layered accent lighting, opulent and editorial"
        ),
        retrieval_terms=["luxury", "marble", "velvet", "brass", "premium"],
        preferred_materials=["marble", "velvet", "brass", "gold", "smoked glass", "walnut"],
        preferred_colors=["black", "gold", "emerald", "cream", "champagne"],
        avoid_terms=["flatpack", "plastic", "distressed"],
        palettes=[
            _palette(
                "Champagne Noir", "Black, gold, cream", "#F2EDE3", "#1A1A1A", "#C6A25C", "#DED7C9"
            ),
            _palette(
                "Emerald", "Deep green with brass", "#EFEBE2", "#1F4237", "#C09A4E", "#DCD5C7"
            ),
        ],
    ),
    DesignStyle.RUSTIC: StyleProfile(
        style=DesignStyle.RUSTIC,
        label="Rustic",
        description="Weathered timber, handmade texture, warmth.",
        prompt_fragment=(
            "Rustic interior: weathered solid timber beams and furniture, natural stone, "
            "handwoven textiles, wrought iron fittings, warm firelight tones, "
            "imperfect handmade texture, generous and lived-in"
        ),
        retrieval_terms=["rustic", "solid wood", "reclaimed", "handmade", "farmhouse"],
        preferred_materials=["pine", "oak", "stone", "wrought iron", "wool", "jute"],
        preferred_colors=["brown", "terracotta", "cream", "olive", "rust"],
        avoid_terms=["chrome", "glossy", "minimal"],
        palettes=[
            _palette(
                "Farmhouse", "Cream with aged oak", "#EFE6D6", "#9C7A54", "#7A4A32", "#E0D4BF"
            ),
            _palette(
                "Olive & Clay", "Olive with terracotta", "#EAE2D2", "#7C7F55", "#A85F42", "#DCD2BE"
            ),
        ],
    ),
    DesignStyle.BOHEMIAN: StyleProfile(
        style=DesignStyle.BOHEMIAN,
        label="Bohemian",
        description="Layered pattern, plants, collected eclecticism.",
        prompt_fragment=(
            "Bohemian interior: layered patterned rugs and textiles, rattan and cane "
            "furniture, abundant trailing plants, macramé wall hangings, warm eclectic "
            "colour, collected and personal, soft golden light"
        ),
        retrieval_terms=["bohemian", "rattan", "woven", "patterned", "eclectic"],
        preferred_materials=["rattan", "cane", "jute", "cotton", "macrame", "terracotta"],
        preferred_colors=["terracotta", "mustard", "cream", "green", "rust"],
        avoid_terms=["minimal", "monochrome", "corporate"],
        palettes=[
            _palette(
                "Terracotta", "Warm rust with cream", "#F1E5D5", "#C4763F", "#6B7F5A", "#E3D3BC"
            ),
            _palette(
                "Desert Bloom", "Mustard and sand", "#F2E9DA", "#D9A441", "#9C5B4A", "#E5DAC6"
            ),
        ],
    ),
    DesignStyle.MID_CENTURY: StyleProfile(
        style=DesignStyle.MID_CENTURY,
        label="Mid-Century Modern",
        description="Tapered legs, warm walnut, confident colour.",
        prompt_fragment=(
            "Mid-century modern interior: tapered wooden legs, warm walnut and teak, "
            "organic curved forms, mustard and teal upholstery accents, globe pendant "
            "lighting, graphic textiles, optimistic and warm"
        ),
        retrieval_terms=["mid century", "teak", "walnut", "tapered", "retro"],
        preferred_materials=["walnut", "teak", "leather", "wool", "brass"],
        preferred_colors=["mustard", "teal", "walnut", "olive", "cream"],
        avoid_terms=["ornate", "industrial", "distressed"],
        palettes=[
            _palette(
                "Mustard & Teak",
                "Mustard with warm teak",
                "#F0E8D8",
                "#8A5A32",
                "#D9A441",
                "#E1D6C1",
            ),
            _palette("Teal Accent", "Cream with teal", "#F1ECE1", "#C3B49A", "#2E6B6B", "#E4DCCC"),
        ],
    ),
}


def get_style_profile(style: DesignStyle) -> StyleProfile:
    return STYLE_PROFILES[style]


def list_style_profiles() -> list[StyleProfile]:
    return list(STYLE_PROFILES.values())
