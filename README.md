# Floor Plan → Photorealistic Interior Renders

Takes a floor plan (PNG/JPG/PDF), detects the rooms, and generates photorealistic
interior renders from multiple camera angles — furnished exclusively with real
products from a supplier catalog, and returning the exact bill of materials for
every render.

> **Status: in progress.** Phases 0–1 are complete and tested. See
> [Roadmap](#roadmap) for what's built and what isn't.

## The hard part

Every requirement in this problem is straightforward except one:

> Every render must represent the exact same interior scene from different camera
> positions. Objects must not move, disappear, appear, rotate, resize, or change
> identity between viewpoints. Only the camera viewpoint may change.

A pipeline that prompts an image model once per viewpoint **cannot** satisfy this.
Diffusion models have no persistent scene state; every sample re-invents the room.
Prompting harder doesn't fix it — "same room, from the corner" is not a constraint,
it's a wish.

**So the image model never decides what the room contains.**

A content-hashed 3D **scene graph** is the single source of truth. Every object's
position, rotation, real-world size, catalog product binding, colour and
per-instance seed is fixed *before a single pixel is generated*. Each render is
that same scene projected through a different camera, conditioned on rasterized
depth and segmentation buffers.

Layout consistency becomes **structural rather than statistical**. There is
nothing left for the image model to re-invent.

```
floor plan ──▶ [Vision LLM] ──▶ Floor Plan JSON (rooms, walls, openings, metres)
                                          │
catalog ──▶ [Adapters] ──▶ Product Index ─┤
                                          ▼
                          [Design agent: retrieve + place]
                                          │
                                          ▼
                            ★ SCENE GRAPH (immutable, hashed) ★
                                          │
                     ┌────────────────────┼────────────────────┐
                     ▼                    ▼                    ▼
                  camera 1             camera 2             camera N
                     │                    │                    │
             [Rasterizer: depth + segmentation + wireframe]
                     │                    │                    │
             [Image model ← geometry + product photos + view-1 anchor]
                     │                    │                    │
                  render 1             render 2             render N
                     └────────────────────┼────────────────────┘
                                          ▼
                            [VLM consistency judge → retry loop]
                                          ▼
                                renders + BOM (products, qty, cost)
```

Three consequences fall out of this design for free:

- **The BOM is exact.** It's a traversal of the scene graph, not an inference from
  the image. The products listed are provably the products rendered.
- **Regeneration is reproducible.** Same `scene_id` + same seed ⇒ identical
  conditioning ⇒ stable renders.
- **Edits are surgical.** "Make the sofa green" becomes a patch to the graph;
  untouched objects keep their seeds and stay pixel-stable.

## Quick start

```bash
make install      # Python 3.12 venv + dependencies
make catalog      # build the product index (~288 seeded products)
make run          # API + web UI on http://localhost:8000
make test         # 116 tests
```

The catalog, retrieval, BOM, calibration and geometry layers are fully offline
and need no keys. **Floor plan extraction needs `ANTHROPIC_API_KEY`** — it is
the one stage that calls a vision model. Image generation defaults to a mock
backend that passes the geometry buffers through.

```bash
cp .env.example .env    # then fill in keys when you want real renders
```

## Design decisions

**Pixel space and metres never mix.** The vision model is only ever asked for
pixel coordinates plus the `m²` labels printed on the plan — never for
real-world dimensions. Scale is recovered by a least-squares solve over
`polygon_area_px / scale² = area_label_m²` across all rooms, which self-checks:
a residual above ~8% means the *extraction* is wrong, not the scale, and gets
flagged rather than silently trusted.

**Hard filters run before vector search.** Retrieval has physical constraints —
a 2.4 m sofa cannot go in a 2.0 m slot, and a bathtub is never a substitute for
a sofa no matter how close the embeddings land. Structured predicates run first
in SQL; the embedding only reranks an already-valid candidate set. Inverting
that order is the classic way a RAG system returns confident nonsense. Tested in
[`tests/test_catalog.py`](tests/test_catalog.py).

**Style tags are derived from material and colour, not assigned randomly.** Pale
oak and linen genuinely reads Scandinavian, Japandi *and* minimalist; black
steel does not. Without that correlation, filtering for a style returns an
arbitrary slice of the catalog and style adherence collapses.

**The LLM does taste; code does geometry.** The model proposes which furniture
*roles* a room needs. Retrieval fills those roles from the real catalog. A
deterministic constraint solver places them — wall anchoring, circulation
clearance, door-swing keep-outs, no overlap. Asking a language model for
coordinates is asking it to do arithmetic it isn't built for.

**Renovation finishes are products too.** The brief says "every visible furniture
*or renovation* element". Flooring, paint, tile and trim are bound to catalog
products in the scene graph, not left as free-text prompt words.

**Honest reporting over flattering output.** Estimated dimensions are flagged in
the BOM. Consistency scores ship in the API response. Furniture roles that no
catalog product could satisfy are reported as unfilled rather than faked.

## Architecture

```
backend/
  config.py               all tunables — model IDs, solver constants, thresholds
  schemas/                the contracts between every stage
    common.py             Vec2/Vec3/Size3 + polygon geometry
    floorplan.py          extraction (pixel space) → calibrated FloorPlan (metres)
    product.py            Product, 40+ categories, ProductQuery, BOM
    scene.py            ★ the scene graph — content-hashed, immutable
    render.py             conditioning maps, consistency scores, jobs
  catalog/
    normalize.py          dimension/price/category parsing (EN + Georgian)
    adapters.py           SupplierAdapter → Gorgia / Comforter / CSV / Generic
    embeddings.py         hashed TF-IDF (offline) or Gemini embeddings
    index.py              SQLite + hybrid retrieval
    service.py            façade + bill of materials
    seed.py               deterministic demo catalog generator
  ingest/
    loader.py             PNG/JPG/PDF → normalized RGB (300 DPI, high-res)
    extract.py            two-pass vision extraction (locate region → geometry)
    calibrate.py          least-squares px→m solve against printed m² labels
    service.py            pipeline + the single Y-flip into world coordinates
  design/
    styles.py             12 styles × 25 palettes
    slots.py              room programs — what each room type needs
    selection.py          binds slots to real catalog products
    geometry.py           exact OBB intersection, inward normals
    placement.py          two-tier constraint solver (anneal + derive)
    agent.py              orchestration → content-hashed Scene
  api/                    FastAPI routers
frontend/                 vanilla JS — no build step
tests/
```

The frontend is deliberately dumb. Everything is reachable from a CLI on the
same service layer, so the architecture doesn't depend on the UI existing.

## Catalog

The assessment names **Gorgia** (furniture, renovation materials, lighting,
flooring, paint, bathroom and kitchen) and **Comforter** (sofas, tables, beds,
wardrobes, office furniture, mattresses, textiles, accessories) as sources, but
ships no data files or API.

This repo therefore includes a **deterministically generated stand-in catalog**
of 288 products spanning both suppliers' stated ranges. It is emitted in *raw
supplier shape* — dimensions as free text, prices as formatted strings, colour
implied by the title — so the import path exercises the real adapters and
parsers rather than bypassing them. If the parsers regress, the catalog build
breaks.

Adding a supplier means dropping `{supplier}_products.json` into `data/catalog/`
and calling `POST /catalog/reindex`. Writing a dedicated adapter is optional;
the generic one handles unknown feeds.

Search supports every axis the brief asks for:

```bash
curl "localhost:8000/catalog/search?style=japandi&material=oak\
&max_width_m=1.8&max_depth_m=0.5&max_price=2000&limit=5"
```

### Known gap

The seeded catalog has no product photography. Real product photos are the
strongest signal for holding object *identity* stable across viewpoints, so with
a real catalog this pipeline would be meaningfully more consistent than it is
here. The view-1 appearance anchor carries identity in the meantime.

## Roadmap

| Phase | Status |
|---|---|
| 0. Scaffold, schemas, config, scene graph | ✅ done |
| 1. Catalog adapters, index, hybrid retrieval, BOM | ✅ done |
| 2. Floor plan ingestion + m² scale calibration | ✅ done |
| 3. Design agent: selection + placement solver | ✅ done |
| 4. Camera rig + numpy geometry rasterizer | ⬜ next |
| 5. Gemini image backend + multi-view prompting | ⬜ |
| 6. VLM consistency judge + retry loop | ⬜ |
| 7. REST API, SSE progress, web UI, CLI | ⬜ |
| 8. Docs, sample outputs, Docker | ⬜ |

## Stack

Python 3.12, FastAPI, Pydantic v2, SQLite, numpy, PyMuPDF, Pillow. Vanilla JS
frontend, no build step. No GPU required.

## Licence

[MIT](LICENSE)
