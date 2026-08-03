"""Build or rebuild the product index.

    make catalog        # or: python -m backend.catalog.build_index

Writes the seeded demo catalog first if no supplier feeds are present, so a
fresh clone is runnable with no setup.
"""

from __future__ import annotations

import argparse
import logging

from ..config import get_settings
from .service import CatalogService


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the product catalog index.")
    parser.add_argument(
        "--no-seed",
        action="store_true",
        help="Fail instead of generating the demo catalog when no feeds are found.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
    settings = get_settings()
    service = CatalogService(settings)

    sources = service.discover_sources()
    if sources:
        print("Sources:")
        for supplier, path in sources.items():
            print(f"  {supplier:12s} {path}")

    stats = service.rebuild_index(ensure_seed=not args.no_seed)

    print(f"\nIndexed {stats['total']} products → {settings.db_path}")
    print(f"Embedder: {settings.embedding_backend} (dim {settings.embedding_dim})")

    print("\nBy supplier:")
    for supplier, count in sorted(stats["by_supplier"].items()):  # type: ignore[union-attr]
        print(f"  {supplier:12s} {count:4d}")

    print("\nTop categories:")
    for category, count in list(stats["by_category"].items())[:12]:  # type: ignore[union-attr]
        print(f"  {category:20s} {count:4d}")

    estimated = stats["estimated_dimensions"]
    if estimated:
        print(f"\n{estimated} products have estimated dimensions (flagged in the BOM).")


if __name__ == "__main__":
    main()
