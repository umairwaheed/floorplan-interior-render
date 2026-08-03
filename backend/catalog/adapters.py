"""Supplier adapters.

One adapter per supplier, all producing the same `Product`. This is the seam
the brief asks for — "support multiple suppliers and allow replacing catalogs
with minimal changes" — so retrieval, placement and rendering never learn where
a product came from.

A note on fidelity: the real Gorgia and Comforter feed schemas weren't supplied
with the assessment, so the adapters resolve fields by trying a list of
plausible key names rather than hard-coding one. That is the right shape for an
unknown upstream anyway — when the real feed arrives, you extend the key lists
or override `to_product`, and nothing downstream moves.
"""

from __future__ import annotations

import csv
import json
from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from ..schemas.product import DesignStyle, Product, ProductCategory
from .normalize import (
    default_dimensions,
    extract_colors,
    extract_materials,
    infer_category,
    infer_styles,
    parse_dimensions,
    parse_price,
    slugify,
)


def first_of(raw: dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Return the first present, non-empty value among candidate keys.

    Case- and separator-insensitive, because supplier feeds are inconsistent
    about `product_name` vs `productName` vs `Name`.
    """
    normalized = {k.lower().replace("-", "_").replace(" ", "_"): v for k, v in raw.items()}
    for key in keys:
        value = normalized.get(key.lower().replace("-", "_").replace(" ", "_"))
        if value not in (None, "", [], {}):
            return value
    return default


def as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str):
        parts = [p.strip() for p in value.replace(";", ",").replace("|", ",").split(",")]
        return [p for p in parts if p]
    return [str(value)]


class SupplierAdapter(ABC):
    """Maps one supplier's raw records onto the normalized `Product` schema."""

    #: Stable supplier key; becomes the `{supplier}:{sku}` product ID prefix.
    supplier: str = "unknown"
    #: Currency the supplier quotes in.
    currency: str = "GEL"

    def load(self, source: Path) -> Iterator[dict[str, Any]]:
        """Read raw records. Handles a JSON list, a JSON object with a
        `products`/`items` key, or JSONL."""
        text = source.read_text(encoding="utf-8").strip()
        if not text:
            return
        if text.startswith("["):
            yield from json.loads(text)
        elif text.startswith("{"):
            data = json.loads(text)
            if isinstance(data, dict):
                records = first_of(data, "products", "items", "data", "results")
                yield from (records if isinstance(records, list) else [data])
        else:
            for line in text.splitlines():
                if line.strip():
                    yield json.loads(line)

    @abstractmethod
    def to_product(self, raw: dict[str, Any]) -> Product | None:
        """Convert one raw record. Return None to skip unusable rows."""

    def import_products(self, source: Path) -> list[Product]:
        """Load and convert, skipping records that can't be normalized."""
        products: list[Product] = []
        seen: set[str] = set()
        for raw in self.load(source):
            try:
                product = self.to_product(raw)
            except Exception:  # noqa: BLE001 — one bad row must not kill an import
                continue
            if product and product.id not in seen:
                seen.add(product.id)
                products.append(product)
        return products


class GenericAdapter(SupplierAdapter):
    """Best-effort adapter driving the whole normalization pipeline.

    Both concrete suppliers below subclass this; they differ only in defaults
    and category hints, which is exactly how much supplier-specific code the
    architecture should require.
    """

    supplier = "generic"
    #: Fallback when the title yields no category match.
    default_category: ProductCategory | None = None
    #: Extra style tags applied to everything from this supplier.
    house_styles: tuple[DesignStyle, ...] = ()

    def to_product(self, raw: dict[str, Any]) -> Product | None:
        name = first_of(raw, "name", "title", "product_name", "productName")
        if not name:
            return None
        name = str(name).strip()

        description = first_of(raw, "description", "desc", "details", "summary")
        category_hint = first_of(raw, "category", "category_name", "categories", "type", "section")
        category_text = " ".join(as_list(category_hint))

        category = self._resolve_category(raw, name, description, category_text)
        if category is None:
            return None

        sku = str(
            first_of(raw, "sku", "id", "code", "article", "product_id", default=slugify(name))
        )
        dimensions = self._resolve_dimensions(raw, category, name, description)

        price = parse_price(first_of(raw, "price", "price_gel", "cost", "amount", "sale_price"))
        if price is None:
            return None

        colors = as_list(first_of(raw, "color", "colour", "colors")) or extract_colors(
            name, description, category_text
        )
        materials = as_list(first_of(raw, "material", "materials")) or extract_materials(
            name, description, category_text
        )
        styles = self._resolve_styles(raw, name, description, category_text)

        images = as_list(first_of(raw, "image", "images", "image_url", "image_urls", "photo"))
        stock = first_of(raw, "in_stock", "available", "availability", default=True)

        return Product(
            id=f"{self.supplier}:{sku}",
            supplier=self.supplier,
            sku=sku,
            name=name,
            url=first_of(raw, "url", "link", "product_url", "href"),
            description=str(description) if description else None,
            category=category,
            subcategory=category_text or None,
            dimensions=dimensions,
            colors=colors,
            materials=materials,
            style_tags=styles,
            price=price,
            currency=str(first_of(raw, "currency", default=self.currency)),
            in_stock=stock not in (False, "false", "no", "0", 0, "out_of_stock"),
            image_urls=images,
            coverage_per_unit_m2=first_of(raw, "coverage_m2", "coverage_per_unit_m2"),
        )

    # -- resolution steps, split out so subclasses can override one at a time --

    def _resolve_category(
        self, raw: dict[str, Any], name: str, description: Any, category_text: str
    ) -> ProductCategory | None:
        explicit = first_of(raw, "normalized_category")
        if explicit:
            try:
                return ProductCategory(str(explicit))
            except ValueError:
                pass
        return infer_category(name, category_text, description) or self.default_category

    def _resolve_dimensions(
        self, raw: dict[str, Any], category: ProductCategory, name: str, description: Any
    ):
        width = first_of(raw, "width", "width_mm", "w")
        depth = first_of(raw, "depth", "depth_mm", "d", "length")
        height = first_of(raw, "height", "height_mm", "h")
        if all(v is not None for v in (width, depth, height)):
            parsed = parse_dimensions(
                f"{width}x{depth}x{height} {first_of(raw, 'unit', default='')}"
            )
            if parsed:
                return parsed

        dim_text = first_of(raw, "dimensions", "size", "measurements", "dims")
        return (
            parse_dimensions(str(dim_text) if dim_text else None)
            or parse_dimensions(name)
            or parse_dimensions(str(description) if description else None)
            or default_dimensions(category)
        )

    def _resolve_styles(
        self, raw: dict[str, Any], name: str, description: Any, category_text: str
    ) -> list[DesignStyle]:
        declared: list[DesignStyle] = []
        for tag in as_list(first_of(raw, "style", "styles", "style_tags")):
            try:
                declared.append(DesignStyle(tag.strip().lower().replace(" ", "_")))
            except ValueError:
                continue
        inferred = infer_styles(name, description, category_text)
        merged = declared or inferred
        for style in self.house_styles:
            if style not in merged:
                merged.append(style)
        return merged


class GorgiaAdapter(GenericAdapter):
    """Gorgia — furniture, renovation materials, lighting, flooring, paint,
    bathroom and kitchen products."""

    supplier = "gorgia"
    currency = "GEL"
    default_category = ProductCategory.DECOR


class ComforterAdapter(GenericAdapter):
    """Comforter — sofas, tables, beds, wardrobes, office furniture,
    mattresses, textiles and home accessories."""

    supplier = "comforter"
    currency = "GEL"
    default_category = ProductCategory.DECOR


class CSVAdapter(GenericAdapter):
    """For suppliers who hand over a spreadsheet, which is most of them."""

    supplier = "csv"

    def __init__(self, supplier: str = "csv", currency: str = "GEL") -> None:
        self.supplier = supplier
        self.currency = currency

    def load(self, source: Path) -> Iterator[dict[str, Any]]:
        with source.open(encoding="utf-8-sig", newline="") as handle:
            yield from csv.DictReader(handle)


ADAPTERS: dict[str, type[SupplierAdapter]] = {
    "gorgia": GorgiaAdapter,
    "comforter": ComforterAdapter,
    "generic": GenericAdapter,
}


def get_adapter(supplier: str) -> SupplierAdapter:
    """Resolve a supplier key to an adapter instance.

    Unknown suppliers fall back to the generic adapter rather than failing —
    a new catalog should be importable before anyone writes code for it.
    """
    adapter_cls = ADAPTERS.get(supplier.lower())
    if adapter_cls is None:
        adapter = GenericAdapter()
        adapter.supplier = supplier.lower()
        return adapter
    return adapter_cls()
