# Prompt iteration: getting Gemini to honour the geometry

Measured, not argued. One scene, one camera, one seed throughout — the only
variable is the conditioning strategy. Scored by the Claude judge against the
segmentation map projected from the frozen scene graph.

## Results

| # | Variant | layout | identity | style | photo | overall | missing | invented |
|---|---------|-------:|---------:|------:|------:|--------:|--------:|---------:|
| 0 | Baseline — depth+seg+wireframe, *generation* framing | **0.12** | 0.30 | 0.78 | 0.80 | 0.36 | 3 | 3 |
| 1 | Edit framing, untextured preview only | **0.70** | 0.85 | 0.87 | 0.80 | 0.79 | 0 | 1 |
| 2 | Edit framing, depth map only | 0.65 | 0.85 | 0.90 | 0.80 | 0.77 | 0 | 2 |
| 3 | Edit framing + screen positions as text | **0.72** | 0.85 | 0.86 | 0.80 | 0.79 | 0 | 1 |
| 4 | 3 + explicit anti-invention clause | 0.62 | 0.82 | 0.88 | 0.85 | 0.75 | 0 | 2 |
| 5 | **Second view, anchored to view 1** | **0.76** | 0.88 | 0.90 | 0.85 | **0.83** | 0 | 2 |

**Cross-view consistency (view 2 vs view 1): 0.87.**

## What actually mattered

**The framing, by a mile.** Layout fidelity went 0.12 → 0.70 by changing what
the model was asked to *do*, not what it was shown. Asked to generate a
photorealistic room from geometry maps, it treats them as a mood board and
designs its own layout — the baseline put the sofa on the wrong wall, dropped
the TV unit, shelving and floor lamp, and invented two wall sconces. Asked to
re-render an existing 3D scene with a material and lighting pass only, it
treats the same maps as the thing to preserve.

**Which image is attached barely mattered.** Depth-only scored 0.65 under the
edit framing, against 0.70 for the untextured preview. Both are far above the
0.12 baseline that had *more* conditioning images. This was the surprise: the
instinct to add more geometry signal was wrong, and the earlier assumption that
Gemini simply cannot be structurally conditioned was also wrong.

**Screen positions as text helped slightly** (0.70 → 0.72) — within judge noise,
but free and principled, so it stayed.

**The anti-invention clause did not work.** Variant 4 still invented two
fixtures and scored *lower* on layout than variant 3. It is retained because it
is cheap and correct in principle, but it should not be described as effective.
Invented objects remain the main open failure mode.

**The anchor mechanism works.** View 2, conditioned on its own geometry plus
view 1 as an appearance reference, scored the highest of any variant and 0.87
on cross-view consistency — the number the whole architecture exists to produce.

## Caveats

- **n = 1 per variant.** Judging the same image twice gave 0.44 and 0.36, so
  judge variance is roughly ±0.1. Differences smaller than that (variants 1 vs
  3) are not real; the 0.12 → 0.70 jump is far outside it.
- **One scene, one room type, one style.** Whether the framing holds for a
  cramped bathroom or a Bohemian palette is untested.
- Total cost of this iteration: 5 image generations and 8 judge calls.

## Still open

Layout fidelity of 0.7 is a large improvement but not fidelity. The judge's
remaining complaints are consistently about *small objects* — a cushion against
the wrong sofa arm, a rug wider than its footprint, a coffee table shifted a few
percent. Large objects land correctly. A backend with true structural
conditioning (Amazon Nova Canvas takes a segmentation map as a real control
signal) would likely close that gap; the `ImageBackend` seam exists for it.
