"""Bill-of-materials tests.

The brief requires returning product names/IDs, quantities and estimated total
cost. Because the BOM is derived from the scene graph rather than from the
generated image, these are exactness tests, not approximation tests.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from backend.catalog.service import CatalogService
from backend.config import Settings
from backend.schemas.product import Dimensions, Product, ProductCategory


def _product(pid: str, price: float, coverage: float | None = None, estimated: bool = False):
    return Product(
        id=pid,
        supplier="test",
        sku=pid.split(":")[-1],
        name=f"Product {pid}",
        category=ProductCategory.FLOORING if coverage else ProductCategory.SOFA,
        dimensions=Dimensions(width_mm=1200, depth_mm=200, height_mm=10, is_estimated=estimated),
        price=price,
        coverage_per_unit_m2=coverage,
    )


@pytest.fixture
def service(tmp_path: Path) -> CatalogService:
    settings = Settings(
        data_dir=tmp_path,
        catalog_dir=tmp_path / "catalog",
        upload_dir=tmp_path / "uploads",
        output_dir=tmp_path / "outputs",
        db_path=tmp_path / "catalog.db",
    )
    settings.ensure_dirs()
    svc = CatalogService(settings)
    svc.index.rebuild(
        [
            _product("test:sofa", 1800.0),
            _product("test:chair", 250.0),
            _product("test:floor", 45.0, coverage=2.4),
            _product("test:estimated", 500.0, estimated=True),
        ]
    )
    return svc


def test_repeated_products_collapse_into_one_line_with_quantity(service: CatalogService):
    bom = service.build_bom(
        [
            ("chair-1", "test:chair"),
            ("chair-2", "test:chair"),
            ("chair-3", "test:chair"),
            ("sofa-1", "test:sofa"),
        ]
    )
    lines = {line.product_id: line for line in bom.lines}
    assert lines["test:chair"].quantity == 3
    assert lines["test:chair"].line_total == 750.0
    assert lines["test:chair"].instance_ids == ["chair-1", "chair-2", "chair-3"]
    assert bom.total_cost == 750.0 + 1800.0


def test_finish_quantity_is_computed_from_area_with_wastage(service: CatalogService):
    """25.2 m² (sample plan's studio) at 2.4 m² per pack, +10% offcut allowance
    → ceil(27.72 / 2.4) = 12 packs."""
    bom = service.build_bom([], finish_quantities={"test:floor": 25.2})
    line = bom.lines[0]
    assert line.quantity == 12
    assert line.line_total == pytest.approx(540.0)


def test_estimated_dimensions_are_surfaced_not_hidden(service: CatalogService):
    bom = service.build_bom([("x-1", "test:estimated")])
    assert bom.lines[0].dimensions_estimated is True


def test_unknown_product_is_skipped_rather_than_crashing(service: CatalogService):
    bom = service.build_bom([("a", "test:sofa"), ("b", "test:does-not-exist")])
    assert [line.product_id for line in bom.lines] == ["test:sofa"]


def test_total_cost_matches_sum_of_lines(service: CatalogService):
    bom = service.build_bom(
        [("s", "test:sofa"), ("c1", "test:chair"), ("c2", "test:chair")],
        finish_quantities={"test:floor": 10.0},
    )
    assert bom.total_cost == pytest.approx(sum(line.line_total for line in bom.lines))
    assert bom.item_count == 3


def test_bom_lines_are_ordered_by_cost(service: CatalogService):
    bom = service.build_bom([("s", "test:sofa"), ("c", "test:chair")])
    assert [line.product_id for line in bom.lines] == ["test:sofa", "test:chair"]
