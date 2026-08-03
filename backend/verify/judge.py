"""Consistency judge — checks the render against the scene it was meant to be.

The judge is deliberately a *different model from a different provider* than
the generator. A model grading its own output is a weak signal: the same biases
that produced a mistake tend to excuse it. Claude judges Gemini's images.

What makes this checkable rather than vibes-based is that the judge is given
ground truth, not just the picture. It sees the segmentation map — which is
projected from the frozen scene graph and therefore *cannot* be wrong — and the
list of objects that are supposed to be in frame. So the question is never "does
this look like a nice room" but "is the sofa where the map says it is, and is it
still the same sofa as in the reference".

`NullJudge` exists because running unverified is a legitimate state, but
pretending to have verified is not. It returns scores flagged `verified=False`,
and `ConsistencyScores.passes()` refuses them.
"""

from __future__ import annotations

import base64
import logging
from abc import ABC, abstractmethod
from pathlib import Path

from ..config import Settings, get_settings
from ..schemas.render import ConsistencyScores, Render
from ..schemas.scene import Scene

logger = logging.getLogger(__name__)


class JudgeError(RuntimeError):
    """Raised when a judge cannot produce a verdict."""


JUDGE_SYSTEM = """\
You are a strict quality inspector for architectural interior renders. You are \
checking whether a generated photograph faithfully represents a specific 3D \
scene, not whether it is attractive.

You will be shown:
1. The generated render.
2. A SEGMENTATION MAP projected from the authoritative 3D scene. Each distinct \
colour is one object, in its true position. This map is ground truth — where it \
disagrees with the render, the render is wrong.
3. Optionally a REFERENCE PHOTOGRAPH: an earlier view of the same room, from a \
different camera position.
4. The list of objects that should be visible.

Be sceptical and specific. A render that looks good but puts the sofa on the \
wrong wall is a failure. Do not award marks for style when geometry is wrong. \
If you cannot tell, say so in `issues` rather than guessing high."""


def _judge_prompt(render: Render, scene: Scene, has_reference: bool) -> str:
    objects = {obj.instance_id: obj for obj in scene.objects}
    expected = [
        objects[instance_id]
        for instance_id in (render.conditioning.visible_instance_ids if render.conditioning else [])
        if instance_id in objects
    ]

    inventory = (
        "\n".join(
            f"- {obj.instance_id}: {obj.display_name} — {obj.color}"
            f"{', ' + obj.material if obj.material else ''}, "
            f"{obj.size_m.width:.2f}×{obj.size_m.depth:.2f}×{obj.size_m.height:.2f} m"
            for obj in expected
        )
        or "- (nothing should be visible in this view)"
    )

    cross_view = (
        """
**cross_view_consistency** — compare against the REFERENCE PHOTOGRAPH. This is \
the same physical room from a different camera position. Objects appearing in \
both must be the same objects: same colour, same material, same shape, same \
finish. Wall and floor finishes must match. Lighting temperature and direction \
must be consistent. Score 1.0 only if someone could believe both photographs \
were taken in the same room minutes apart.
"""
        if has_reference
        else """
**cross_view_consistency** — set this to null. This is the first view of the \
room, so there is no earlier view to compare against.
"""
    )

    return f"""\
Assess this render of "{scene.style.value}" interior in the {scene.palette.name} palette.

Objects that should be visible, per the 3D scene:
{inventory}

Score each dimension from 0.0 to 1.0:

**layout_fidelity** — does the render's geometry match the segmentation map? \
Are objects in the same positions, at the same relative scales, against the \
same walls? Is the room's shape the same?

**object_identity** — is each listed object present, and is it the right kind \
of object in the right colour and material? List the `instance_id` of anything \
missing in `missing_instance_ids`.
{cross_view}
**style_adherence** — does it read as {scene.style.value}, in the stated palette?

**photorealism** — does it look like a photograph rather than a render or an \
illustration?

Also report in `hallucinated_objects` any clearly visible furniture or fixture \
that is NOT in the list above — an invented object is as much a consistency \
failure as a missing one.

Put concrete, specific observations in `issues` — at most five, one sentence \
each. "The sofa is against the left wall but the map places it against the back \
wall" is useful; "layout could be better" is not. Keep them short: the verdict \
matters more than the prose, and long lists have truncated the response."""


class ConsistencyJudge(ABC):
    """Scores one render against the scene it came from."""

    name = "abstract"

    @abstractmethod
    def judge(self, render: Render, scene: Scene, reference_path: Path | None) -> ConsistencyScores:
        """Return scores. Should not raise for ordinary model failures."""


class NullJudge(ConsistencyJudge):
    """Runs when verification is disabled or unavailable.

    Returns neutral placeholders flagged `verified=False`. This is not a
    passing grade and `passes()` will reject it — a system that cannot check
    its own output should say so, not report 0.9 and hope.
    """

    name = "null"

    def judge(self, render: Render, scene: Scene, reference_path: Path | None) -> ConsistencyScores:
        return ConsistencyScores(
            layout_fidelity=0.0,
            object_identity=0.0,
            cross_view_consistency=None,
            style_adherence=0.0,
            photorealism=0.0,
            verified=False,
            issues=["Not verified — no consistency judge was configured."],
        )


class ClaudeJudge(ConsistencyJudge):
    """Claude vision with structured output.

    Independent of the generator by construction: a different provider, a
    different model family, and no access to the generation prompt — only to
    the image, the ground-truth geometry, and the object list.
    """

    name = "claude"

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        if not self.settings.anthropic_api_key:
            raise JudgeError(
                "ANTHROPIC_API_KEY is not set — the consistency judge needs it. "
                "Set ENABLE_JUDGE=false to render without verification."
            )
        import anthropic

        self.client = anthropic.Anthropic(api_key=self.settings.anthropic_api_key)

    @staticmethod
    def _image_block(path: Path, media_type: str = "image/png") -> dict[str, object]:
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": base64.standard_b64encode(path.read_bytes()).decode("ascii"),
            },
        }

    def judge(self, render: Render, scene: Scene, reference_path: Path | None) -> ConsistencyScores:
        if not render.image_path or not Path(render.image_path).exists():
            return ConsistencyScores(
                layout_fidelity=0.0,
                object_identity=0.0,
                style_adherence=0.0,
                photorealism=0.0,
                verified=False,
                issues=["No image to judge — the render failed."],
            )

        content: list[dict[str, object]] = [
            {"type": "text", "text": "GENERATED RENDER:"},
            self._image_block(Path(render.image_path)),
        ]

        if render.conditioning and Path(render.conditioning.segmentation_path).exists():
            content += [
                {
                    "type": "text",
                    "text": "SEGMENTATION MAP (ground truth from the 3D scene):",
                },
                self._image_block(Path(render.conditioning.segmentation_path)),
            ]

        has_reference = reference_path is not None and reference_path.exists()
        if has_reference:
            content += [
                {
                    "type": "text",
                    "text": "REFERENCE PHOTOGRAPH (same room, earlier view):",
                },
                self._image_block(reference_path),  # type: ignore[arg-type]
            ]

        content.append({"type": "text", "text": _judge_prompt(render, scene, has_reference)})

        try:
            response = self.client.messages.parse(
                model=self.settings.judge_model,
                # Not lowballed: adaptive thinking and the verdict share this
                # budget, and at 4000 the JSON was being truncated mid-string —
                # a valid-looking verdict became a parse error and the render
                # was reported unverified. A judge that silently stops judging
                # is worse than no judge, because the score is the product.
                max_tokens=16000,
                system=JUDGE_SYSTEM,
                output_format=ConsistencyScores,
                messages=[{"role": "user", "content": content}],
            )
        except Exception as exc:  # noqa: BLE001 — a judge failure must not fail the render
            logger.warning("judge failed for %s: %s", render.id, exc)
            return ConsistencyScores(
                layout_fidelity=0.0,
                object_identity=0.0,
                style_adherence=0.0,
                photorealism=0.0,
                verified=False,
                issues=[f"Judge unavailable: {exc}"],
            )

        if response.stop_reason == "refusal":
            return ConsistencyScores(
                layout_fidelity=0.0,
                object_identity=0.0,
                style_adherence=0.0,
                photorealism=0.0,
                verified=False,
                issues=["The judge declined to assess this image."],
            )

        scores = response.parsed_output
        if scores is None:
            return ConsistencyScores(
                layout_fidelity=0.0,
                object_identity=0.0,
                style_adherence=0.0,
                photorealism=0.0,
                verified=False,
                issues=["The judge returned no structured verdict."],
            )

        scores.verified = True
        # An anchor cannot be cross-view consistent with anything; drop any
        # value the model volunteered rather than let it inflate the mean.
        if not has_reference:
            scores.cross_view_consistency = None
        return scores


def build_judge(settings: Settings | None = None) -> ConsistencyJudge:
    """Resolve the configured judge, degrading to `NullJudge` rather than failing."""
    settings = settings or get_settings()
    if not settings.enable_judge:
        return NullJudge()
    try:
        return ClaudeJudge(settings)
    except JudgeError as exc:
        logger.warning("%s — renders will be reported as unverified", exc)
        return NullJudge()
