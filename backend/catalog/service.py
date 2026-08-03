"""Catalog service — the façade everything else uses.

The design agent, the API and the CLI all go through this. None of them touch
SQL, adapters, or embeddings directly, so swapping the storage or the embedder
is a change in one file.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

from ..config import Settings, get_settings
from ..schemas.product import (
    BillOfMaterials,
    BOMLine,
    Product,
    ProductMatch,
    ProductQuery,
)
from .adapters import get_adapter
from .embeddings import build_embedder
from .index import ProductIndex

logger = logging.getLogger(__name__)


class CatalogService:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.index = ProductIndex(
            db_path=self.settings.db_path,
            embedder=build_embedder(
                backend=self.settings.embedding_backend,
                api_key=self.settings.gemini_api_key,
                dim=self.settings.embedding_dim,
            ),
        )

    # -- import ------------------------------------------------------------

    def discover_sources(self) -> dict[str, Path]:
        """Find `{supplier}_products.json` feeds in the catalog directory.

        Dropping in `newsupplier_products.json` is enough to add a supplier —
        the generic adapter handles it until someone writes a specific one.
        """
        sources: dict[str, Path] = {}
        for path in sorted(self.settings.catalog_dir.glob("*_products.json")):
            sources[path.stem.replace("_products", "")] = path
        return sources

    def import_all(self) -> list[Product]:
        products: list[Product] = []
        for supplier, path in self.discover_sources().items():
            imported = get_adapter(supplier).import_products(path)
            logger.info("imported %d products from %s (%s)", len(imported), supplier, path.name)
            products.extend(imported)
        return products

    def rebuild_index(self, ensure_seed: bool = True) -> dict[str, object]:
        """Import every discovered feed and rebuild the index from scratch."""
        if ensure_seed and not self.discover_sources():
            from .seed import write_seed_files

            logger.info("no catalog feeds found — writing seeded demo catalog")
            write_seed_files(self.settings.catalog_dir)

        products = self.import_all()
        if not products:
            raise RuntimeError(
                f"No products could be imported from {self.settings.catalog_dir}. "
                "Expected one or more '{supplier}_products.json' files."
            )
        self.index.rebuild(products)
        return self.index.stats()

    def ensure_ready(self) -> None:
        """Build the index on first use so a fresh clone just works."""
        if self.index.count() == 0:
            self.rebuild_index()

    # -- query -------------------------------------------------------------

    def search(self, query: ProductQuery) -> list[ProductMatch]:
        self.ensure_ready()
        return self.index.search(query, candidate_limit=self.settings.retrieval_candidates * 10)

    def best_match(self, query: ProductQuery) -> ProductMatch | None:
        matches = self.search(query)
        return matches[0] if matches else None

    def get(self, product_id: str) -> Product | None:
        return self.index.get(product_id)

    def get_many(self, product_ids: list[str]) -> dict[str, Product]:
        return self.index.get_many(product_ids)

    def stats(self) -> dict[str, object]:
        self.ensure_ready()
        return self.index.stats()

    # -- bill of materials -------------------------------------------------

    def build_bom(
        self,
        object_product_ids: list[tuple[str, str]],
        finish_quantities: dict[str, float] | None = None,
    ) -> BillOfMaterials:
        """Aggregate a scene's products into a costed BOM.

        `object_product_ids` is (instance_id, product_id) pairs — one per placed
        object. `finish_quantities` maps a finish product to the area it covers,
        which is converted into purchasable units via `coverage_per_unit_m2`.

        This is a pure traversal of what the scene actually contains, which is
        why the returned products are guaranteed to be the ones rendered.
        """
        finish_quantities = finish_quantities or {}
        wanted = {pid for _, pid in object_product_ids} | set(finish_quantities)
        products = self.get_many(sorted(wanted))

        grouped: dict[str, list[str]] = {}
        for instance_id, product_id in object_product_ids:
            grouped.setdefault(product_id, []).append(instance_id)

        lines: list[BOMLine] = []
        for product_id in sorted(wanted):
            product = products.get(product_id)
            if product is None:
                logger.warning("BOM references unknown product %s — skipped", product_id)
                continue

            instances = grouped.get(product_id, [])
            if product_id in finish_quantities:
                quantity = self._units_for_area(product, finish_quantities[product_id])
            else:
                quantity = float(len(instances))
            if quantity <= 0:
                continue

            lines.append(
                BOMLine(
                    product_id=product.id,
                    name=product.name,
                    supplier=product.supplier,
                    url=product.url,
                    category=product.category,
                    quantity=quantity,
                    unit_price=product.price,
                    line_total=round(product.price * quantity, 2),
                    currency=product.currency,
                    instance_ids=sorted(instances),
                    dimensions_estimated=product.dimensions.is_estimated,
                )
            )

        lines.sort(key=lambda line: (-line.line_total, line.name))
        return BillOfMaterials(lines=lines, currency=self.settings.default_currency)

    @staticmethod
    def _units_for_area(product: Product, area_m2: float) -> float:
        """How many packs/tins to buy for a given area, rounded up.

        Includes a 10% allowance, which is standard for tile and flooring
        offcuts. Paint is quoted by coverage per tin and gets the same margin.
        """
        import math

        coverage = product.coverage_per_unit_m2 or 0.0
        if coverage <= 0:
            return 1.0
        return float(math.ceil((area_m2 * 1.10) / coverage))


@lru_cache
def get_catalog_service() -> CatalogService:
    return CatalogService()
