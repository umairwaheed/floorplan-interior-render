"""Catalog tests.

The hard-filter tests are the important ones. Retrieval returning a *worse*
match is a quality problem; retrieval returning a product that physically does
not fit the room is a correctness bug that propagates all the way into the
render and the BOM.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.catalog.adapters import GenericAdapter, get_adapter
from backend.catalog.embeddings import HashingEmbedder
from backend.catalog.index import ProductIndex
from backend.catalog.normalize import (
    default_dimensions,
    extract_colors,
    extract_materials,
    infer_category,
    parse_dimensions,
    parse_price,
)
from backend.schemas.product import (
    DesignStyle,
    Dimensions,
    Product,
    ProductCategory,
    ProductQuery,
)

# --- normalization ---------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected_mm"),
    [
        ("2100x900x850 mm", (2100, 900, 850)),
        ("220x90x85 cm", (2200, 900, 850)),
        ("220 x 90 x 85", (2200, 900, 850)),  # no unit, small values → cm
        ("2100x900x850", (2100, 900, 850)),  # no unit, large values → mm
        ("W1600 D900 H750", (1600, 900, 750)),
        ("220х90х85 სმ", (2200, 900, 850)),  # Cyrillic 'х' separator, Georgian unit
    ],
)
def test_parse_dimensions(text: str, expected_mm: tuple[float, float, float]):
    dims = parse_dimensions(text)
    assert dims is not None
    assert (dims.width_mm, dims.depth_mm, dims.height_mm) == expected_mm
    assert dims.is_estimated is False


def test_parse_dimensions_returns_none_rather_than_guessing():
    assert parse_dimensions("comfortable three seat sofa") is None
    assert parse_dimensions(None) is None


def test_default_dimensions_are_flagged_estimated():
    dims = default_dimensions(ProductCategory.SOFA)
    assert dims.is_estimated is True
    assert dims.width_mm > 0


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("1299.00", 1299.0),
        ("1 299,00 ₾", 1299.0),
        ("1,299.00 GEL", 1299.0),
        ("1.299,00", 1299.0),
        (1299, 1299.0),
        ("", None),
    ],
)
def test_parse_price(raw: object, expected: float | None):
    assert parse_price(raw) == expected


def test_infer_category_handles_english_and_georgian():
    assert infer_category("Oslo 3-Seat Sofa") == ProductCategory.SOFA
    assert infer_category("დივანი 3-ადგილიანი") == ProductCategory.SOFA
    assert infer_category("Nordic Coffee Table") == ProductCategory.COFFEE_TABLE
    assert infer_category("completely unrelated text") is None


def test_specific_categories_win_over_general_ones():
    """'Coffee Table' must not fall through to the generic table handling."""
    assert infer_category("Kyoto Coffee Table") == ProductCategory.COFFEE_TABLE
    assert infer_category("Kyoto Dining Table") == ProductCategory.DINING_TABLE


def test_extract_colors_and_materials():
    assert "beige" in extract_colors("Oslo Sofa — Beige Linen")
    assert "linen" in extract_materials("Oslo Sofa — Beige Linen")


# --- adapters --------------------------------------------------------------


def test_adapter_normalizes_raw_supplier_record(tmp_path: Path):
    feed = tmp_path / "acme_products.json"
    feed.write_text(
        json.dumps(
            {
                "products": [
                    {
                        "sku": "SF-1",
                        "title": "Oslo 3-Seat Sofa — Beige Linen",
                        "category": "furniture/sofas",
                        "dimensions": "210x90x85 cm",
                        "price": "1 899,00 ₾",
                        "style": "scandinavian",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    products = get_adapter("acme").import_products(feed)
    assert len(products) == 1
    product = products[0]
    assert product.id == "acme:SF-1"
    assert product.category == ProductCategory.SOFA
    assert product.dimensions.width_mm == 2100
    assert product.price == 1899.0
    assert DesignStyle.SCANDINAVIAN in product.style_tags


def test_adapter_skips_unusable_rows_without_failing_the_import(tmp_path: Path):
    feed = tmp_path / "acme_products.json"
    feed.write_text(
        json.dumps(
            [
                {"sku": "OK-1", "title": "Oslo Sofa", "price": "100"},
                {"sku": "NO-NAME", "price": "100"},  # no title
                {"sku": "NO-PRICE", "title": "Oslo Sofa"},  # no price
                {"sku": "NO-CATEGORY", "title": "zzzz", "price": "100"},
            ]
        ),
        encoding="utf-8",
    )
    products = GenericAdapter().import_products(feed)
    assert [p.sku for p in products] == ["OK-1"]


def test_unknown_supplier_falls_back_to_generic_adapter():
    adapter = get_adapter("brand-new-supplier")
    assert adapter.supplier == "brand-new-supplier"


# --- embeddings ------------------------------------------------------------


def test_hashing_embedder_is_deterministic_and_normalized():
    embedder = HashingEmbedder(dim=128).fit(["oak dining table", "linen sofa"])
    first = embedder.embed(["oak dining table"])
    second = embedder.embed(["oak dining table"])
    assert (first == second).all()
    assert abs(float((first[0] ** 2).sum()) - 1.0) < 1e-5


def test_hashing_embedder_ranks_related_text_higher():
    corpus = ["oak dining table", "linen three seat sofa", "ceramic table lamp"]
    embedder = HashingEmbedder(dim=512).fit(corpus)
    vectors = embedder.embed(corpus)
    query = embedder.embed(["oak table"])[0]
    scores = vectors @ query
    assert scores.argmax() == 0


# --- index & hard filters --------------------------------------------------


def _product(
    pid: str,
    category: ProductCategory,
    width_mm: float,
    depth_mm: float,
    price: float,
    styles: list[DesignStyle] | None = None,
    colors: list[str] | None = None,
    in_stock: bool = True,
) -> Product:
    return Product(
        id=pid,
        supplier="test",
        sku=pid.split(":")[-1],
        name=f"{pid} {category.value}",
        category=category,
        dimensions=Dimensions(width_mm=width_mm, depth_mm=depth_mm, height_mm=800),
        colors=colors or ["beige"],
        materials=["linen"],
        style_tags=styles or [DesignStyle.SCANDINAVIAN],
        price=price,
        in_stock=in_stock,
    )


@pytest.fixture
def index(tmp_path: Path) -> ProductIndex:
    idx = ProductIndex(tmp_path / "test.db", embedder=HashingEmbedder(dim=128))
    idx.rebuild(
        [
            _product("test:small", ProductCategory.SOFA, 1500, 850, 900),
            _product("test:medium", ProductCategory.SOFA, 2000, 900, 1800),
            _product("test:huge", ProductCategory.SOFA, 2600, 1000, 3500),
            _product("test:oos", ProductCategory.SOFA, 1400, 800, 700, in_stock=False),
            _product("test:table", ProductCategory.DINING_TABLE, 1600, 900, 1200),
            _product(
                "test:industrial",
                ProductCategory.SOFA,
                1800,
                880,
                2200,
                styles=[DesignStyle.INDUSTRIAL],
                colors=["charcoal"],
            ),
        ]
    )
    return idx


def test_dimension_filter_excludes_products_that_do_not_fit(index: ProductIndex):
    results = index.search(
        ProductQuery(categories=[ProductCategory.SOFA], max_width_m=1.6, max_depth_m=0.9)
    )
    assert {m.product.id for m in results} == {"test:small"}


def test_dimension_filter_allows_a_90_degree_turn(index: ProductIndex):
    """A 2.0×0.9 m sofa fits a 0.95 m × 2.1 m slot when rotated."""
    results = index.search(
        ProductQuery(categories=[ProductCategory.SOFA], max_width_m=0.95, max_depth_m=2.1)
    )
    assert "test:medium" in {m.product.id for m in results}


def test_category_filter_is_absolute(index: ProductIndex):
    results = index.search(ProductQuery(text="dining table", categories=[ProductCategory.SOFA]))
    assert all(m.product.category == ProductCategory.SOFA for m in results)
    assert results, "a matching category should still return results"


def test_price_ceiling_is_never_exceeded(index: ProductIndex):
    results = index.search(ProductQuery(categories=[ProductCategory.SOFA], max_price=1000))
    assert results
    assert all(m.product.price <= 1000 for m in results)


def test_out_of_stock_excluded_by_default(index: ProductIndex):
    ids = {m.product.id for m in index.search(ProductQuery(categories=[ProductCategory.SOFA]))}
    assert "test:oos" not in ids
    ids_incl = {
        m.product.id
        for m in index.search(ProductQuery(categories=[ProductCategory.SOFA], in_stock_only=False))
    }
    assert "test:oos" in ids_incl


def test_style_preference_ranks_but_does_not_exclude(index: ProductIndex):
    """Style is a soft signal — it should reorder, not filter."""
    results = index.search(
        ProductQuery(categories=[ProductCategory.SOFA], styles=[DesignStyle.INDUSTRIAL])
    )
    assert results[0].product.id == "test:industrial"
    assert len(results) > 1, "non-matching styles should still be available as fallbacks"


def test_search_returns_empty_when_nothing_can_fit(index: ProductIndex):
    assert index.search(ProductQuery(categories=[ProductCategory.SOFA], max_width_m=0.2)) == []


def test_stats_report_supplier_and_category_breakdown(index: ProductIndex):
    stats = index.stats()
    assert stats["total"] == 6
    assert stats["by_supplier"] == {"test": 6}
    assert stats["by_category"][ProductCategory.SOFA.value] == 5
