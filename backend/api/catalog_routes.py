"""Catalog endpoints.

Exposes the search axes the brief asks for — category, dimensions, colour,
material, style and price — as query parameters over the same
`CatalogService` the design agent uses. There is no second code path.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from ..catalog.service import get_catalog_service
from ..design.styles import list_style_profiles
from ..schemas.product import (
    DesignStyle,
    Product,
    ProductCategory,
    ProductMatch,
    ProductQuery,
)

router = APIRouter(tags=["catalog"])


@router.get("/catalog/search", response_model=list[ProductMatch])
def search_catalog(
    text: str | None = Query(None, description="Free-text query, used to rerank."),
    category: list[ProductCategory] = Query(default=[]),
    style: list[DesignStyle] = Query(default=[]),
    color: list[str] = Query(default=[]),
    material: list[str] = Query(default=[]),
    supplier: list[str] = Query(default=[]),
    max_width_m: float | None = Query(None, gt=0),
    max_depth_m: float | None = Query(None, gt=0),
    max_height_m: float | None = Query(None, gt=0),
    min_price: float | None = Query(None, ge=0),
    max_price: float | None = Query(None, ge=0),
    in_stock_only: bool = True,
    limit: int = Query(20, ge=1, le=200),
) -> list[ProductMatch]:
    """Hybrid search: hard structured filters first, then semantic rerank."""
    return get_catalog_service().search(
        ProductQuery(
            text=text,
            categories=category,
            styles=style,
            colors=color,
            materials=material,
            suppliers=supplier,
            max_width_m=max_width_m,
            max_depth_m=max_depth_m,
            max_height_m=max_height_m,
            min_price=min_price,
            max_price=max_price,
            in_stock_only=in_stock_only,
            limit=limit,
        )
    )


@router.get("/catalog/products/{product_id:path}", response_model=Product)
def get_product(product_id: str) -> Product:
    product = get_catalog_service().get(product_id)
    if product is None:
        raise HTTPException(status_code=404, detail=f"Unknown product: {product_id}")
    return product


@router.get("/catalog/stats")
def catalog_stats() -> dict[str, object]:
    return get_catalog_service().stats()


@router.post("/catalog/reindex")
def reindex_catalog() -> dict[str, object]:
    """Re-import every supplier feed and rebuild the index.

    This is the whole "catalogs can be imported and indexed" story — drop a new
    `{supplier}_products.json` in the catalog directory and call this.
    """
    return get_catalog_service().rebuild_index()


@router.get("/styles")
def list_styles() -> list[dict[str, object]]:
    """Styles and their palettes, for the style picker."""
    return [
        {
            "style": profile.style.value,
            "label": profile.label,
            "description": profile.description,
            "palettes": [
                {
                    "name": palette.name,
                    "description": palette.description,
                    "swatches": [
                        palette.primary,
                        palette.secondary,
                        palette.accent,
                        palette.neutral,
                    ],
                }
                for palette in profile.palettes
            ],
        }
        for profile in list_style_profiles()
    ]
