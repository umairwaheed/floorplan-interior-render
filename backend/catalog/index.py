"""Product index and hybrid retrieval.

Storage is SQLite with real indexes on the fields that get filtered; embeddings
live in the same rows as blobs and are scored in numpy.

**Why filters run before vectors.** Retrieval here has hard physical
constraints: a 2.4 m sofa cannot go in a 2.0 m slot, a product over budget is
not "somewhat" over budget, and a bathtub is never a substitute for a sofa no
matter how close the embeddings land. So structured predicates run first in
SQL, and the embedding only reranks a candidate set that is already valid.
Inverting that order is the classic way a RAG system returns confident nonsense.

**Why numpy and not a vector extension.** At catalog scale here (hundreds of
products, and low thousands for a real supplier feed) scoring a filtered
candidate set in numpy is faster than a KNN index — and, more importantly, it
composes correctly with the hard filters. Vector extensions want to do
top-k first and struggle to apply arbitrary predicates before the search.
Past roughly 10⁵ products the right move is a real vector store (pgvector,
sqlite-vec); `ProductIndex.search` is the only place that would change.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ..schemas.product import (
    DesignStyle,
    Product,
    ProductCategory,
    ProductMatch,
    ProductQuery,
)
from .embeddings import Embedder, HashingEmbedder

SCHEMA = """
CREATE TABLE IF NOT EXISTS products (
    id                   TEXT PRIMARY KEY,
    supplier             TEXT NOT NULL,
    sku                  TEXT,
    name                 TEXT NOT NULL,
    category             TEXT NOT NULL,
    width_mm             REAL NOT NULL,
    depth_mm             REAL NOT NULL,
    height_mm            REAL NOT NULL,
    dims_estimated       INTEGER NOT NULL DEFAULT 0,
    colors               TEXT NOT NULL DEFAULT '[]',
    materials            TEXT NOT NULL DEFAULT '[]',
    style_tags           TEXT NOT NULL DEFAULT '[]',
    price                REAL NOT NULL,
    currency             TEXT NOT NULL DEFAULT 'GEL',
    in_stock             INTEGER NOT NULL DEFAULT 1,
    coverage_per_unit_m2 REAL,
    search_text          TEXT NOT NULL DEFAULT '',
    payload              TEXT NOT NULL,
    embedding            BLOB
);
CREATE INDEX IF NOT EXISTS idx_products_category ON products(category);
CREATE INDEX IF NOT EXISTS idx_products_supplier ON products(supplier);
CREATE INDEX IF NOT EXISTS idx_products_price    ON products(price);
CREATE INDEX IF NOT EXISTS idx_products_stock    ON products(in_stock);
CREATE INDEX IF NOT EXISTS idx_products_dims     ON products(width_mm, depth_mm, height_mm);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

# Weights for the structured half of the hybrid score. Style is weighted above
# colour and material because it is the user's explicit choice, not a detail.
W_SEMANTIC = 0.40
W_STYLE = 0.25
W_COLOR = 0.20
W_MATERIAL = 0.15


class ProductIndex:
    """Owns the SQLite catalog database and answers `ProductQuery`."""

    def __init__(self, db_path: Path, embedder: Embedder | None = None) -> None:
        self.db_path = db_path
        self.embedder = embedder or HashingEmbedder()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self._restore_embedder_state()

    # -- persistence of embedder state ------------------------------------
    # The IDF weights learned at build time must be reused at query time, or
    # query vectors land in a different space than the indexed ones.

    def _restore_embedder_state(self) -> None:
        row = self.conn.execute("SELECT value FROM meta WHERE key = 'idf'").fetchone()
        if row and isinstance(self.embedder, HashingEmbedder):
            idf = np.frombuffer(bytes.fromhex(row["value"]), dtype=np.float32)
            if idf.size == self.embedder.dim:
                self.embedder.idf = idf.copy()

    def _persist_embedder_state(self) -> None:
        if isinstance(self.embedder, HashingEmbedder):
            self.conn.execute(
                "INSERT OR REPLACE INTO meta(key, value) VALUES ('idf', ?)",
                (self.embedder.idf.astype(np.float32).tobytes().hex(),),
            )

    # -- build -------------------------------------------------------------

    def rebuild(self, products: Sequence[Product]) -> int:
        """Replace the index wholesale. Idempotent, so re-running is safe."""
        texts = [p.search_text() for p in products]
        if isinstance(self.embedder, HashingEmbedder):
            self.embedder.fit(texts)
        vectors = self.embedder.embed(texts) if texts else np.zeros((0, self.embedder.dim))

        self.conn.execute("DELETE FROM products")
        self.conn.executemany(
            """INSERT INTO products (
                   id, supplier, sku, name, category,
                   width_mm, depth_mm, height_mm, dims_estimated,
                   colors, materials, style_tags,
                   price, currency, in_stock, coverage_per_unit_m2,
                   search_text, payload, embedding
               ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            [
                (
                    p.id,
                    p.supplier,
                    p.sku,
                    p.name,
                    p.category.value,
                    p.dimensions.width_mm,
                    p.dimensions.depth_mm,
                    p.dimensions.height_mm,
                    int(p.dimensions.is_estimated),
                    json.dumps(p.colors),
                    json.dumps(p.materials),
                    json.dumps([s.value for s in p.style_tags]),
                    p.price,
                    p.currency,
                    int(p.in_stock),
                    p.coverage_per_unit_m2,
                    text,
                    p.model_dump_json(),
                    vectors[i].astype(np.float32).tobytes(),
                )
                for i, (p, text) in enumerate(zip(products, texts, strict=True))
            ],
        )
        self._persist_embedder_state()
        self.conn.commit()
        return len(products)

    # -- query -------------------------------------------------------------

    def _build_filters(self, query: ProductQuery) -> tuple[str, list[Any]]:
        """Translate the hard constraints into SQL. These are non-negotiable —
        anything failing them is not a worse match, it is not a match."""
        clauses: list[str] = []
        params: list[Any] = []

        if query.categories:
            clauses.append(f"category IN ({','.join('?' * len(query.categories))})")
            params += [c.value for c in query.categories]

        if query.suppliers:
            clauses.append(f"supplier IN ({','.join('?' * len(query.suppliers))})")
            params += [s.lower() for s in query.suppliers]

        if query.in_stock_only:
            clauses.append("in_stock = 1")

        if query.min_price is not None:
            clauses.append("price >= ?")
            params.append(query.min_price)
        if query.max_price is not None:
            clauses.append("price <= ?")
            params.append(query.max_price)

        # Footprint fit, allowing the object to be turned 90°.
        if query.max_width_m is not None and query.max_depth_m is not None:
            w_mm, d_mm = query.max_width_m * 1000, query.max_depth_m * 1000
            clauses.append(
                "((width_mm <= ? AND depth_mm <= ?) OR (depth_mm <= ? AND width_mm <= ?))"
            )
            params += [w_mm, d_mm, w_mm, d_mm]
        elif query.max_width_m is not None:
            clauses.append("MIN(width_mm, depth_mm) <= ?")
            params.append(query.max_width_m * 1000)

        if query.max_height_m is not None:
            clauses.append("height_mm <= ?")
            params.append(query.max_height_m * 1000)

        where = " AND ".join(clauses) if clauses else "1=1"
        return where, params

    def search(self, query: ProductQuery, candidate_limit: int = 400) -> list[ProductMatch]:
        where, params = self._build_filters(query)
        rows = self.conn.execute(
            f"SELECT id, payload, embedding, colors, materials, style_tags "  # noqa: S608
            f"FROM products WHERE {where} LIMIT ?",
            [*params, candidate_limit],
        ).fetchall()
        if not rows:
            return []

        semantic = self._semantic_scores(query.text, rows)
        wanted_styles = {s.value for s in query.styles}
        wanted_colors = {c.lower() for c in query.colors}
        wanted_materials = {m.lower() for m in query.materials}

        matches: list[ProductMatch] = []
        for i, row in enumerate(rows):
            styles = set(json.loads(row["style_tags"]))
            colors = set(json.loads(row["colors"]))
            materials = set(json.loads(row["materials"]))

            style_score = self._overlap(wanted_styles, styles)
            color_score = self._overlap(wanted_colors, colors)
            material_score = self._overlap(wanted_materials, materials)

            score, reasons = self._combine(
                semantic=semantic[i] if semantic is not None else None,
                style=(style_score, bool(wanted_styles)),
                color=(color_score, bool(wanted_colors)),
                material=(material_score, bool(wanted_materials)),
            )
            matches.append(
                ProductMatch(
                    product=Product.model_validate_json(row["payload"]),
                    score=round(float(score), 4),
                    reason=", ".join(reasons) or "passes all hard filters",
                )
            )

        matches.sort(key=lambda m: (-m.score, m.product.id))
        return matches[: query.limit]

    def _semantic_scores(self, text: str | None, rows: Sequence[sqlite3.Row]) -> np.ndarray | None:
        if not text:
            return None
        query_vec = self.embedder.embed([text])[0]
        matrix = np.stack([np.frombuffer(row["embedding"], dtype=np.float32) for row in rows])
        # Both sides are L2-normalized, so the dot product is cosine similarity.
        return np.clip(matrix @ query_vec, 0.0, 1.0)

    @staticmethod
    def _overlap(wanted: set[str], have: set[str]) -> float:
        """Share of the requested attributes the product actually has."""
        if not wanted:
            return 0.0
        return len(wanted & have) / len(wanted)

    @staticmethod
    def _combine(
        semantic: float | None,
        style: tuple[float, bool],
        color: tuple[float, bool],
        material: tuple[float, bool],
    ) -> tuple[float, list[str]]:
        """Weighted sum over only the signals the caller actually asked for.

        Renormalizing by active weight matters: a query with no colour
        preference shouldn't cap every result at 80% of the maximum score.
        """
        parts: list[tuple[float, float]] = []
        reasons: list[str] = []

        if semantic is not None:
            parts.append((W_SEMANTIC, semantic))
            reasons.append(f"text {semantic:.2f}")
        for label, weight, (value, active) in (
            ("style", W_STYLE, style),
            ("color", W_COLOR, color),
            ("material", W_MATERIAL, material),
        ):
            if active:
                parts.append((weight, value))
                reasons.append(f"{label} {value:.2f}")

        total_weight = sum(w for w, _ in parts)
        if total_weight == 0:
            return 0.5, reasons  # nothing to rank on; all survivors are equal
        return sum(w * v for w, v in parts) / total_weight, reasons

    # -- lookups -----------------------------------------------------------

    def get(self, product_id: str) -> Product | None:
        row = self.conn.execute(
            "SELECT payload FROM products WHERE id = ?", (product_id,)
        ).fetchone()
        return Product.model_validate_json(row["payload"]) if row else None

    def get_many(self, product_ids: Sequence[str]) -> dict[str, Product]:
        if not product_ids:
            return {}
        placeholders = ",".join("?" * len(product_ids))
        rows = self.conn.execute(
            f"SELECT id, payload FROM products WHERE id IN ({placeholders})",  # noqa: S608
            list(product_ids),
        ).fetchall()
        return {r["id"]: Product.model_validate_json(r["payload"]) for r in rows}

    def count(self) -> int:
        return int(self.conn.execute("SELECT COUNT(*) AS n FROM products").fetchone()["n"])

    def stats(self) -> dict[str, Any]:
        by_supplier = {
            r["supplier"]: r["n"]
            for r in self.conn.execute(
                "SELECT supplier, COUNT(*) AS n FROM products GROUP BY supplier"
            )
        }
        by_category = {
            r["category"]: r["n"]
            for r in self.conn.execute(
                "SELECT category, COUNT(*) AS n FROM products GROUP BY category ORDER BY n DESC"
            )
        }
        estimated = self.conn.execute(
            "SELECT COUNT(*) AS n FROM products WHERE dims_estimated = 1"
        ).fetchone()["n"]
        return {
            "total": self.count(),
            "by_supplier": by_supplier,
            "by_category": by_category,
            "estimated_dimensions": estimated,
        }

    def categories_present(self) -> set[ProductCategory]:
        rows = self.conn.execute("SELECT DISTINCT category FROM products").fetchall()
        out: set[ProductCategory] = set()
        for row in rows:
            try:
                out.add(ProductCategory(row["category"]))
            except ValueError:
                continue
        return out

    def styles_present(self) -> set[DesignStyle]:
        out: set[DesignStyle] = set()
        for row in self.conn.execute("SELECT style_tags FROM products"):
            for tag in json.loads(row["style_tags"]):
                try:
                    out.add(DesignStyle(tag))
                except ValueError:
                    continue
        return out

    def close(self) -> None:
        self.conn.close()
