# Architecture

## The problem this design exists to solve

> Every render must represent the exact same interior scene from different
> camera positions. Objects must not move, disappear, appear, rotate, resize,
> or change identity between viewpoints.

A pipeline that prompts an image model once per viewpoint cannot satisfy this.
Diffusion models hold no persistent scene state; each sample re-invents the
room. Prompting harder does not help — "the same room, from the corner" is a
wish, not a constraint.

So the image model is never allowed to decide what the room contains.

## The load-bearing idea

A content-hashed 3-D **scene graph** is the single source of truth. Every
object's position, rotation, real-world size, catalog product binding, colour
and per-instance seed is fixed *before a single pixel is generated*. Each render
is that same scene projected through a different camera.

Consistency becomes **structural rather than statistical**. Three properties
fall out for free:

| Property | Why it holds |
|---|---|
| The bill of materials is exact | It is a traversal of the graph, not an inference from the image |
| Regeneration is reproducible | Same `scene_id` + same seed ⇒ identical conditioning |
| Edits are surgical | A change is a patch; untouched objects keep their seeds |

## Stage map

```
 PNG/JPG/PDF
     │
     ▼
┌─────────────┐  pixel coords + printed m² labels only
│   ingest    │  ─────────────────────────────────────▶ Claude (vision)
│             │  least-squares px→m solve, self-checking
└─────┬───────┘
      │ FloorPlan (metres, Y-up)
      ▼
┌─────────────┐  slots ──▶ hybrid retrieval ──▶ SQLite catalog
│   design    │  placement: simulated annealing + derived dependents
└─────┬───────┘
      │ ★ Scene (content-hashed, immutable)
      ▼
┌─────────────┐  camera rig scored by actual rasterization
│   render    │  pinhole + z-buffer → depth · segmentation · wireframe · preview
└─────┬───────┘
      │ ConditioningMaps per camera
      ▼
┌─────────────┐  view 1 → anchor;  views 2..N ← anchor + own geometry
│  imagegen   │  ─────────────────────────────────────▶ Gemini 3 Pro Image
└─────┬───────┘
      │ Render[]
      ▼
┌─────────────┐  render + segmentation map + anchor
│   verify    │  ─────────────────────────────────────▶ Claude (vision)
└─────┬───────┘  per-view retry, honest scores
      ▼
  renders + BOM + consistency report
```

## Decisions, and what was rejected

### Who decides what

| Decision | Made by | Why not the alternative |
|---|---|---|
| Room geometry | Vision model, pixel space only | Asking it for *metres* gives a number with no way to check it. Pixels plus printed labels lets a least-squares solve recover the scale **and** self-check via the residual. |
| Which furniture roles a room needs | Rule-based programs (LLM proposer designed, unbuilt) | — |
| Which product fills a role | Catalog retrieval | An LLM naming products invents them. Retrieval cannot return a product that is not in the index. |
| Where each object goes | Deterministic solver | Placement is arithmetic over clearances and published dimensions. An LLM asked for coordinates does the one thing it is worst at, and destroys reproducibility. |
| What the room looks like | Image model | This is the only thing it is actually good at. |
| Whether the result is right | A *different* model from a *different* provider | A model grading its own output excuses its own biases. |

### Rejected: a real 3-D renderer

Three.js in headless Chrome, or `trimesh` + `pyrender`, would give nicer
previews. Rejected because the conditioning maps do not need to be pretty — they
need to be *exact and reproducible*, and to run anywhere without a GPU or a
browser. A numpy pinhole rasterizer with a z-buffer is ~400 lines, has no
platform dependencies, and produces byte-identical output across runs. The image
model supplies the realism; this supplies the truth.

### Rejected: vector search first, filters second

Retrieval here has hard physical constraints. A 2.4 m sofa cannot go in a 2.0 m
slot, a product over budget is not "somewhat" over budget, and a bathtub is
never a near-miss for a sofa however close the embeddings land. Structured
predicates run first in SQL; the embedding only reranks an already-valid
candidate set. Inverting that order is the standard way a RAG system returns
confident nonsense.

### Rejected: sqlite-vec

Added, then removed. At a few hundred to a few thousand products, scoring a
SQL-filtered candidate set in numpy is faster *and* composes correctly with hard
filters — vector extensions want to do top-k first. Keeping the dependency for
appearance would have been worse than not having it. Past roughly 10⁵ products
the right move is pgvector or similar; `ProductIndex.search` is the only place
that changes.

### Rejected: optimising all object positions jointly

The solver runs two tiers. Anchors (bed, sofa, dining table) are searched
globally by simulated annealing. Dependents (nightstands, dining chairs, rugs,
pendants) are *derived* from their anchor once it lands. A nightstand's position
is a consequence of the bed's, not an independent variable — modelling that
collapses the search space by orders of magnitude and encodes design
relationships a flat cost function would have to rediscover.

The cost of that choice is a whole class of bug: derived objects bypass the
annealing cost function, so every constraint enforced there silently does not
apply to them. Four were found and fixed individually (escaping the room,
pendants at floor level, curtains piled at the centroid of a windowless room, a
nightstand inside a door's swing arc) before a final pass was added that
guarantees the invariant globally.

### Rejected: retrying a whole scene when one view fails

Regenerating a room to fix one bad view discards acceptable images and mints a
fresh anchor, leaving the survivors inconsistent with it. Retries are per view.
When an *anchor* is re-rolled, its dependent views are regenerated too —
otherwise the reported cross-view score is measured against an image no longer
in the output.

### Rejected: Amazon Nova Canvas (for now)

Nova Canvas takes a segmentation map as a first-class control signal, which fits
this architecture more closely than Gemini, where geometry goes in as reference
images. It was the front-runner when measured layout fidelity was 0.12. Re-framing
the task (below) moved that to 0.72, which removed the urgency. The
`ImageBackend` seam exists for it; the finding is documented in
[`prompt-iteration.md`](prompt-iteration.md).

## The measured finding

The single highest-leverage change in the system was not architectural. Asked to
**generate** a photorealistic room from geometry maps, Gemini treats them as a
mood board — layout fidelity **0.12**, three of six objects missing, two
invented. Asked to **re-render an existing 3-D scene** with a material and
lighting pass only, it treats the same maps as the thing to preserve — **0.72**,
nothing missing. Which conditioning image was attached barely mattered by
comparison (depth-only scored 0.65 under the same framing).

Cross-view consistency with the anchor mechanism: **0.87**.

Full method, numbers and caveats — including that judge variance is ±0.1 and
n=1 per variant — in [`prompt-iteration.md`](prompt-iteration.md).

## Coordinate conventions

One place each, because getting them wrong is silent:

- **Pixel space** (`*_px`) exists only inside `ingest/`. The vision model is
  never asked for metres.
- **World space** is metres, Y-up, Z-up-from-floor. The single Y-flip lives in
  `ingest/service.py::px_to_m`; images address pixels top-down, and mirroring
  the world would silently reverse every door swing.
- **Object local space**: `width` along local X, `depth` along local Y, rotation
  is yaw about Z only. At rotation 0 an object faces +Y.

## Determinism

`Scene.content_hash()` covers everything that affects the generated image and
excludes `scene_id` itself, so finalization is idempotent. Per-instance seeds
use blake2b, **not** Python's `hash()` — string hashing is salted per process,
which made every "frozen" seed differ on every run and the reproducibility claim
hold only inside a single interpreter. Caught by running the real pipeline
twice; now covered by a test that designs the same scene in two subprocesses.

## What is not built

Stated plainly rather than implied:

- **The LLM slot proposer.** Room programs are rule-based. The interface is
  designed for a model to refine them; nothing calls one.
- **Natural-language edits.** `POST /designs/{id}/regenerate` re-renders the same
  scene graph with new seeds. The scene-graph patch path is designed and
  returns `501` rather than silently doing a full re-design.
- **Persistence.** Floor plans and jobs live in process memory. Fine for a demo
  and the CLI; the swap to Redis or a database touches `services/store.py` only.
- **Product photography.** The seeded catalog has none, and real product photos
  are the strongest signal for holding object identity across viewpoints. With a
  real catalog this pipeline would be measurably more consistent than it is here.
