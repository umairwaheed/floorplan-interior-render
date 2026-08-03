# Floor Plan → Photorealistic Interior Renders — Build Plan

## 1. The one decision that determines the score

Every requirement in the brief is easy except this block:

> Every render must represent the exact same interior scene from different camera positions.
> Objects must not move, disappear, appear, rotate, resize, or change identity between viewpoints.
> Only the camera viewpoint may change.

A pipeline that prompts an image model N times ("same room, from the corner") **cannot** satisfy this. Diffusion models have no persistent scene state; every sample re-invents the room. This is the failure mode the evaluators are screening for.

**Approach: a structured 3D scene graph is the single source of truth. The image model is only a neural renderer.**

```
floor plan ──▶ [Vision LLM] ──▶ Floor Plan JSON (rooms, walls, doors, windows, metres)
                                          │
catalog ──▶ [Importer] ──▶ Product Index ─┤
                                          ▼
                          [Design Agent: retrieve + place]
                                          │
                                          ▼
                            ★ SCENE GRAPH (immutable, hashed) ★
                                          │
                     ┌────────────────────┼────────────────────┐
                     ▼                    ▼                    ▼
                  camera 1             camera 2             camera N
                     │                    │                    │
             [Geometry rasterizer: depth + segmentation + wireframe]
                     │                    │                    │
             [Image model, conditioned on geometry + product photos + view-1 anchor]
                     │                    │                    │
                  render 1             render 2             render N
                     └────────────────────┼────────────────────┘
                                          ▼
                            [VLM consistency judge → retry loop]
                                          ▼
                                 renders + BOM (products, qty, cost)
```

Layout consistency becomes **structural, not statistical** — the same 3D scene projected through different cameras. Every object's position, rotation, size, and catalog product ID are fixed data before a single pixel is generated. The BOM is trivially exact because the prompt is built *from* the selected products.

---

## 2. Components

### 2.1 Floor plan ingestion (`ingest/`)
- Accept PNG/JPG/JPEG/PDF. PDF → raster via PyMuPDF (300 DPI, first page or page picker).
- Preprocess: deskew, greyscale, upscale small plans.
- **Vision LLM extraction** (Claude Opus 5 or Gemini 3 Pro, structured output) → `FloorPlan` schema:
  ```
  rooms[]: { id, name, type(bedroom|living|kitchen|bath|balcony|hall),
             polygon_px[], area_m2_label, ceiling_height_m }
  walls[]: { a, b, thickness_m, is_exterior }
  openings[]: { type(door|window), wall_id, offset_m, width_m, swing_dir }
  ```
- **Scale calibration**: both sample plans print `m²` labels per room (25.2, 12.8, 6.1, 19.3, 14.9…). Solve pixels→metres by least-squares over `polygon_area_px / scale² = area_m2_label` across all rooms. This is far more robust than asking the model for dimensions, and it self-checks: residual > 8% ⇒ flag low confidence and fall back to the printed dimension ticks (`210`, `370`, `810`…) on plan 2.
- Second pass: ask the VLM to verify its own extraction against the image (adjacency, door positions), repair, then validate with Pydantic + geometry checks (closed polygons, no overlap, doors lie on walls).

### 2.2 Catalog ingestion & retrieval (`catalog/`)
- `SupplierAdapter` interface — `GorgiaAdapter`, `ComforterAdapter`, `CSVAdapter`. Each maps supplier fields to one normalized schema:
  ```
  Product: id, supplier, sku, name, url, category, subcategory,
           dims_mm{w,d,h}, colors[], materials[], style_tags[],
           price, currency, image_urls[], in_stock
  ```
- Messy supplier data (free-text titles, missing dims) is normalized by a cheap LLM enrichment pass: infer `category`, `style_tags`, `dominant_color`, and parse dims out of the title/description. Cached by content hash so it runs once.
- Storage: SQLite + `sqlite-vec` for embeddings. **Hybrid retrieval** — hard SQL filters (category, price band, dims fit the slot, colour family) **then** vector rerank on style/description. Hard constraints must never be soft: a 2.4 m sofa cannot go in a 2.0 m slot regardless of embedding similarity.
- Runs offline from a seeded `catalog.json` (a few hundred products across both suppliers) so the reviewer can run it without network access; live importers are the same code path.

### 2.3 Design agent (`design/`)
Two stages, deliberately split so the LLM does taste and code does geometry.

**Stage A — Selection (LLM):** given room type, area, style, palette, budget → a shopping list of *slots*: `{ role: "bed", required: true, target_dims, color_intent, style_intent }`. Each slot is then filled by catalog retrieval, not by the model hallucinating a product. If nothing fits, the slot is dropped and reported.

**Stage B — Placement (deterministic solver):** per-category placement rules (bed → longest free wall, headboard to wall; sofa → faces focal wall; dining table → open centre; rug → under primary group; pendant → above table). Constraints: no overlap, ≥0.7 m circulation, door-swing keep-outs, window keep-outs, wall clearances. Discrete candidate positions + simulated annealing on a weighted cost. Output is fully deterministic given `(floorplan, selections, seed)`.

**Result: the Scene Graph.**
```
Scene: { scene_id (content hash), seed, style, palette,
         rooms[], objects[]: { instance_id, product_id, room_id,
                               position_m{x,y,z}, rotation_deg, size_m, color },
         materials{ floor, walls, ceiling, trim },  # each bound to a catalog finish
         lighting[], cameras[] }
```

### 2.4 Camera rig (`render/cameras.py`)
Deterministic per room: inset each room corner by 0.6 m, eye height 1.5 m, look at the furniture centroid, 60° horizontal FOV, keep cameras whose frustum covers ≥60% of room area and the primary furniture group. Store them **in the scene graph** — identical cameras across every regeneration.

### 2.5 Geometry rasterizer (`render/raster.py`)
No GPU, no headless GL, no platform pain. Every object is an oriented bounding box with known real-world dimensions; walls/floor/ceiling are quads. A ~250-line numpy pinhole projector + painter's-algorithm z-buffer emits three conditioning maps per camera:
1. **Depth map** — the geometric backbone.
2. **Segmentation map** — colour-coded by object instance; doubles as the region map used to tell the image model *which* product occupies *which* pixels.
3. **Wireframe / line map** — clean edges for structure adherence.

Upgrade path if fidelity demands it: swap in Three.js rendered offscreen via Playwright (Node is already available), or `trimesh` + real product 3D proxies. The rasterizer sits behind an interface so this is a one-file change.

### 2.6 Image generation (`imagegen/`)
Pluggable `ImageBackend` interface — this is deliberate; showing the abstraction scores under "scalability and production readiness".

- **`GeminiImageBackend` (primary):** Gemini 3 Pro Image. Per view, send: the segmentation/depth render, the actual **product photos** of every visible item, and — for views 2..N — **view 1 as the appearance anchor**. Multi-reference editing is this model's strength and it is what keeps object *identity* stable, which depth conditioning alone does not.
- **`ReplicateFluxControlNetBackend` (strict-geometry alternative):** Flux + ControlNet-depth, fixed seed per instance, IP-Adapter for style anchoring. Use when geometry adherence matters more than material realism.

Prompting is assembled programmatically from the scene graph — never free-form. Each view's prompt enumerates the exact objects, their catalog names/colours/materials, and their screen regions, plus a fixed style/palette block and a fixed negative block (no added furniture, no moved objects, no layout changes).

**Consistency levers, in order of impact:** (1) shared geometry maps, (2) view-1 anchor + product photos as references, (3) frozen per-instance seeds, (4) identical style/lighting text block across views, (5) rejection sampling via the judge.

### 2.7 Consistency verification (`verify/`)
A VLM judge scores each render against a rubric and returns structured output:
- **Layout fidelity** — does the render match its own depth/segmentation map?
- **Object identity** — for each expected instance: present / correct product / correct colour? Anything present that isn't in the scene graph?
- **Cross-view consistency** — pairwise against view 1.
- **Style & palette adherence.**

Below threshold → regenerate that single view (bumped seed, tightened prompt, up to K attempts), never the whole set. Scores ship in the API response — being honest about a 0.82 consistency score reads better than silently shipping a mismatch.

### 2.8 Regeneration semantics
- `scene_id` = content hash of the scene graph. Same `scene_id` + same seed ⇒ byte-identical prompts and conditioning ⇒ stable renders.
- "Regenerate, keep the scene" → reuse the scene graph verbatim, vary only the image-model seed.
- "Make the sofa green / add a floor lamp" → the LLM emits a **JSON Patch** against the scene graph. Only patched instances get new product bindings or positions; the solver re-runs locally with everything else pinned. Untouched objects keep their seeds, so the rest of the room does not drift.
- **Variations** = N independent scene graphs from the same floor plan + style (different selections/placements), each internally consistent. This is the correct reading of "multiple design variations" — variation happens at the scene level, not the pixel level.

### 2.9 API & UI (`api/`)
FastAPI, async jobs (in-process worker; Redis/RQ noted as the production swap).
```
POST /floorplans              upload → floorplan_id + extracted rooms
GET  /styles                  available styles + palettes
POST /designs                 {floorplan_id, style, palette, rooms[], variations, views_per_room}
                              → job_id
GET  /designs/{id}            status, scene graph, renders, BOM, consistency scores
POST /designs/{id}/regenerate {preserve_scene: true} | {changes: "..."} → new job
GET  /designs/{id}/bom        products (name, id, url, qty, unit price, total)
GET  /catalog/search          category, style, color, material, dims, price
```
Plus a single-page HTMX/vanilla UI: upload → pick style + palette → progress → gallery with per-render product list and total cost. And a `cli.py` mirroring the same service layer, so the core is provably UI-independent.

---

## 3. Phases

| # | Phase | Output | Est. |
|---|---|---|---|
| 0 | Scaffold: uv, FastAPI, schemas, config, Makefile | runnable skeleton | 0.5 h |
| 1 | Catalog importer + hybrid search + seeded catalog | `/catalog/search` works | 1.5 h |
| 2 | Floor plan → JSON + m² scale calibration + validation | both sample plans parse correctly | 1.5 h |
| 3 | Design agent: selection + placement solver → scene graph | valid, non-overlapping scenes | 2 h |
| 4 | Camera rig + geometry rasterizer (depth/seg/wire) | conditioning maps per view | 1.5 h |
| 5 | Image backend + prompt assembly + multi-view anchoring | **photoreal, consistent renders** | 2 h |
| 6 | Consistency judge + retry loop | scores + auto-repair | 1 h |
| 7 | REST API + CLI + minimal web UI | end-to-end demo | 1.5 h |
| 8 | README, architecture doc, sample outputs, Docker | submission package | 1 h |

**~12 hours** for the full build. The brief says 2–3 hours, which is not achievable for these requirements as written — anything delivered in 3 hours is a per-view prompting loop, which fails the consistency section outright. Suggested resolution: build the full pipeline, and in the README state plainly what was built, what was time-boxed, and where the deliberate trade-offs are.

**Minimum defensible cut (~6 h)** if time is hard-capped: phases 0–5 with one hard-coded style, one room, three viewpoints, a 50-product seeded catalog, CLI only. The scene graph and geometry-conditioned rendering are non-negotiable — they *are* the answer.

---

## 4. Stack — locked

- **Python 3.12.11** at `/opt/homebrew/bin/python3.12`, plain `python3.12 -m venv .venv` (no `uv` on this machine, not worth the extra dependency). Not 3.14 — several CV/ML wheels still lag.
- FastAPI + Pydantic v2, PyMuPDF, Pillow + numpy, SQLite + sqlite-vec, `google-genai`, `anthropic`, `pytest`, `ruff`.
- No GPU, no Node, no headless GL. Runs on this laptop as-is.

**Locked decisions:**

| Choice | Decision |
|---|---|
| Image backend | **Gemini 3 Pro Image** as the shipped `ImageBackend`. Multi-reference is what holds object identity across views. The Flux+ControlNet backend stays a documented interface implementation, not built. |
| Catalog | **Seeded `catalog.json`** — ~250 products across Gorgia + Comforter categories, committed. Runs offline; `GorgiaAdapter`/`ComforterAdapter` are the same code path for live import. |
| Interface | **REST API + minimal single-page web UI in vanilla JS** (no build step) — upload → style/palette → SSE progress → gallery with per-render BOM. Plus `cli.py` on the same service layer. |

**Layout:**
```
backend/    FastAPI — ingest, catalog, design, render, imagegen, verify
frontend/   index.html + app.js + style.css, served static by FastAPI
cli.py      same service layer, no HTTP
```
The frontend stays dumb. Everything graded lives in the backend and is reachable from the CLI, so the architecture doesn't depend on the UI. Job progress streams over **SSE** so renders appear per-view as they finish rather than behind a 90-second spinner.

**Prerequisite:** no `GEMINI_API_KEY` is set in this shell. Phases 0–4 and 6–8 don't need it; phase 5 does. I'll build against a `MockImageBackend` that passes the geometry maps through, so everything is runnable end-to-end before the key lands.

---

## 5. Deliverables mapped to their rubric

| Criterion | Where it's earned |
|---|---|
| AI engineering & architecture | Scene graph as source of truth; pluggable backends/adapters |
| Model selection & workflow | VLM for extraction, LLM for taste, code for geometry, diffusion for pixels — each where it's strongest, documented with rejected alternatives |
| Code quality & docs | Typed schemas, tests on the solver + calibrator, ARCHITECTURE.md |
| Photorealism | Gemini 3 Pro Image / Flux, geometry-conditioned |
| Style adherence | Style/palette blocks propagate into selection filters *and* prompts |
| Scene consistency | One 3D scene, N cameras — structural, plus judge + retry |
| Layout preservation | Rasterized depth/segmentation from calibrated real-world metres |
| Furniture placement / object identity | Frozen instance IDs, product-photo references, view-1 anchor |
| Proportion & geometry accuracy | m²-calibrated scale; real product dimensions drive box sizes |
| Correct catalog usage | Prompts built only from retrieved products; BOM derived from the scene graph |
| Scalability & production readiness | Async jobs, caching, supplier adapters, backend interfaces, cost/latency notes |

---

## 6. Risks

| Risk | Mitigation |
|---|---|
| Image model drifts despite conditioning | Judge + per-view retry; fall back to stricter ControlNet backend; report scores honestly |
| Floor plan extraction wrong on plan 2 (Georgian labels, dense dims) | Scale calibration cross-check + confidence flag + `--rooms` manual override |
| Real catalog data missing dimensions | LLM enrichment + category-median fallback, flagged in the BOM |
| Cost/latency (N views × M variations) | Cache by scene hash; default 3 views × 1 variation; expose knobs |
| No supplier API access | Ship seeded `catalog.json`; importers are the same interface |
