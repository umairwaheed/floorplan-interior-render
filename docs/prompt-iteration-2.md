# Round 2: why layout fidelity does not generalize

Round 1 tuned one living room to 0.72 layout fidelity. The production four-room
run scored **0.43**, with the judge's complaints concentrated in the small,
sparse rooms — invented taps, cisterns, sconces, radiators. Round 1's own
caveat named this gap: *"whether the framing holds for a cramped bathroom is
untested"*. This round tested it.

**The headline: the prompt hypothesis was refuted, and the real problem turned
out to be somewhere else entirely.**

---

## What was tested

Each variant is a hypothesis about the *cause*, not a reword.

| | Variant | Hypothesis |
|---|---|---|
| V0 | production prompt | control |
| V1 | + `ARCHITECTURE` block | Nothing ever asserts a window's *absence*, so the model supplies what the room type implies. |
| V2 | + fixtures forbidden **by name** | Round 1's generic anti-invention clause failed because "do not add fixtures" is abstract; models respond to nouns. |
| V3 | + exact object count | A sparse frame reads as under-specified; an integer gives a stopping condition. |

Rooms: bathroom 3.9 m² (2 objects), balcony 5.4 m² (4), living room 19.5 m² (7)
— a dense control, because a change that helps sparse rooms and quietly hurts
the ones that already worked is not an improvement.

## The result that looked good

At two seeds per variant, V2 was a clear winner on the bathroom:

| variant | layout | invented |
|---|---:|---:|
| V0 | 0.54 | 2.0 |
| V1 | 0.40 | 1.0 |
| **V2** | **0.72** | **0.5** |
| V3 | 0.57 | 1.0 |

Both seeds scored exactly 0.72. It looked like a +0.18 result that also halved
hallucinations, and it appeared to overturn round 1's finding — the difference
being *named nouns* rather than an abstract instruction.

## The result that was true

Three phrasings of the same hypothesis, on the same two seeds, scored the
bathroom **0.72, 0.64 and 0.54**. That is the entire claimed effect size,
produced by rewording. So the sample size went up:

**Bathroom, n = 6 seeds per arm**

| variant | layout | sd | range | invented |
|---|---:|---:|:--|---:|
| V0 | 0.62 | 0.18 | 0.30–0.82 | 1.2 |
| V2 | 0.59 | 0.17 | 0.35–0.79 | 0.8 |

**Living room, n = 6 seeds per arm**

| variant | layout | sd | range | invented |
|---|---:|---:|:--|---:|
| V0 | 0.57 | 0.14 | 0.32–0.72 | 1.0 |
| V2 | **0.44** | 0.12 | 0.22–0.55 | **2.2** |

The bathroom effect disappeared. In the living room V2 was actively harmful and
*doubled* invented objects — naming a fixture makes the model more likely to
draw it, which is the failure mode round 1 saw and this round confirms at
proper sample size.

**V2 was not shipped.** The n=2 result was a false positive.

## Where the variance actually lives

The obvious next question: is a score a property of the image, or of the judge?
One unchanged image, re-judged six times:

| | layout |
|---|---|
| mean | 0.62 |
| sd | **0.04** |
| range | 0.58–0.68 |

The judge is stable. Round 1 had estimated judge noise at ±0.1 from *two*
calls and used it to dismiss differences; the real figure is about ±0.04.

So the ±0.15–0.18 spread is the **generator**. Same prompt, same camera, same
scene, same conditioning images — only the seed differs — and layout fidelity
ranges from 0.30 to 0.82.

That reframes everything:

- **Single-view scores are draws from a wide distribution.** A reported 0.43 on
  a seven-view run carries a standard error near 0.18/√7 ≈ 0.07.
- **Prompt tuning cannot be evaluated at n < 6.** Round 1's table is n = 1 per
  variant, so every difference in it except the 0.12 → 0.70 framing jump is
  inside noise and should not be read as a ranking.
- **The retry threshold sits inside the noise band.** Re-rolling a view that
  scored 0.72 against a 0.75 threshold mostly resamples the distribution rather
  than fixing anything.

## What was shipped instead

Not a prompt change — a selection change.

If a retry is a fresh draw from a distribution with sd ≈ 0.18, then returning
the *last* draw discards better ones already paid for. The loop did exactly
that: `current = retried`, unconditionally. The README's own logged example
shows the cost — a view went `0.62 → 0.72 → 0.64` and shipped **0.64**.

Two fixes, no extra API calls:

1. **Keep the best-scoring attempt, not the last.** Unverified verdicts never
   win, so running without a judge cannot outrank a measured render.
2. **Write each attempt to its own file.** Attempts shared one filename, so a
   retry overwrote the image an earlier verdict described. Best-of-N is only
   honest if the file that ships is the file that was scored.

`attempts` still reports total generations, not the winner's ordinal — the run
should not look cheaper than it was.

This is strictly ≥ the old behaviour for the same spend, and it is the largest
single lever found in this round, precisely because the generator is the noisy
component.

## Caveats

- Three rooms, one style, one plan. Bathroom and living room at n = 6; balcony
  at n = 2 only, so its numbers are indicative and are not used for any claim.
- The n = 6 arms are not statistically significant at p < 0.05 either. The
  living-room regression (0.57 → 0.44) has p ≈ 0.1. It is reported as "do not
  ship this", which is the decision it supports, not as a proven harm.
- Detecting a genuine 0.05 prompt effect against sd = 0.18 would need roughly
  50 samples per arm. That is the honest cost of prompt tuning here, and it is
  why this round stopped rewording.
- Total cost: 44 image generations, 50 judge calls.

## Still open

Generator variance is the binding constraint. Best-of-N mitigates it; it does
not reduce it. The routes worth trying next, in the order their expected effect
size justifies:

1. **Wider best-of-N on the anchor only.** The anchor sets each room's
   appearance and every later view inherits it, so variance there is the most
   expensive. Cost scales on one view per room rather than all of them.
2. **A stronger structural signal than a prompt.** Every intervention tested
   here is text. The 0.12 → 0.70 jump came from changing the *task*, not the
   wording, which suggests the remaining headroom is also structural — a
   ControlNet-style conditioned backend rather than a better sentence.
3. **Judging with a fixed rubric per room type**, to cut the residual 0.04.
   Cheapest of the three, smallest payoff.
