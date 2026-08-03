"""Verification and per-view retry.

Retries are **per view, never per scene**. Regenerating a whole room because
one of three views came out badly would throw away two acceptable images and,
worse, produce a fresh anchor — so the two good views would now be inconsistent
with the new one. The scene graph is fixed; only the failing image is redrawn,
with a bumped seed and the same conditioning.

The anchor is a special case worth being explicit about: it is the reference
every other view of that room was matched against. Re-rolling it silently
invalidates them. So when an anchor fails and is regenerated, the views
downstream of it are regenerated too — otherwise the system would be reporting
consistency scores measured against an image no longer in the output.

Scores ship in the API response either way. A render that scores 0.62 is more
useful reported honestly than hidden behind a retry loop that gave up.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

from ..config import Settings, get_settings
from ..imagegen.service import RenderService
from ..schemas.floorplan import FloorPlan
from ..schemas.render import ConditioningMaps, Render, RenderStatus
from ..schemas.scene import Scene
from .judge import ConsistencyJudge, build_judge

logger = logging.getLogger(__name__)

ProgressHook = Callable[[Render], None]


class VerificationService:
    """Judges renders and re-rolls the ones that fail."""

    def __init__(
        self,
        render_service: RenderService,
        judge: ConsistencyJudge | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.renders = render_service
        self.judge = judge or build_judge(self.settings)

    def verify_and_retry(
        self,
        scene: Scene,
        floorplan: FloorPlan,
        renders: list[Render],
        conditioning: dict[str, ConditioningMaps],
        output_dir: Path | None = None,
        on_progress: ProgressHook | None = None,
    ) -> list[Render]:
        """Score every render, retrying failures view by view.

        Processes a room's anchor first so that later views in that room are
        judged against a *final* reference rather than one that is about to be
        replaced.
        """
        output_dir = output_dir or (self.settings.output_dir / scene.output_key)
        by_room: dict[str, list[Render]] = {}
        for render in renders:
            by_room.setdefault(render.room_id, []).append(render)

        final: list[Render] = []

        for room_id, room_renders in by_room.items():
            room = floorplan.room(room_id)
            room_name = room.name if room else "room"

            anchors = [r for r in room_renders if r.is_anchor]
            followers = [r for r in room_renders if not r.is_anchor]

            settled_anchor: Render | None = None
            for anchor in anchors:
                settled_anchor = self._settle(
                    render=anchor,
                    scene=scene,
                    conditioning=conditioning,
                    output_dir=output_dir,
                    reference_path=None,
                    room_name=room_name,
                    on_progress=on_progress,
                )
                final.append(settled_anchor)

            reference = (
                Path(settled_anchor.image_path)
                if settled_anchor and settled_anchor.image_path
                else None
            )

            # If the anchor was re-rolled, every follower in this room was
            # generated against an image that no longer exists. Judging them
            # against the new anchor would report consistency with a reference
            # they never saw — so they are regenerated first.
            anchor_changed = settled_anchor is not None and settled_anchor.attempts > 1
            if anchor_changed and reference is not None:
                logger.info(
                    "room %s: anchor was re-rolled — regenerating %d dependent view(s)",
                    room_id,
                    len(followers),
                )
                followers = [
                    self._regenerate_against(
                        follower, scene, conditioning, output_dir, reference, room_name
                    )
                    for follower in followers
                ]

            for follower in followers:
                final.append(
                    self._settle(
                        render=follower,
                        scene=scene,
                        conditioning=conditioning,
                        output_dir=output_dir,
                        reference_path=reference,
                        room_name=room_name,
                        on_progress=on_progress,
                    )
                )

        # Preserve the caller's original ordering; the room grouping above is
        # an implementation detail, not something the API should expose.
        order = {render.id: index for index, render in enumerate(renders)}
        final.sort(key=lambda r: order.get(r.id, 1_000))
        return final

    def _regenerate_against(
        self,
        render: Render,
        scene: Scene,
        conditioning: dict[str, ConditioningMaps],
        output_dir: Path,
        reference_path: Path,
        room_name: str,
    ) -> Render:
        """Redraw a follower view against a replaced anchor."""
        maps = conditioning.get(render.camera_id)
        if maps is None:
            return render
        retried = self.renders.render_view(
            scene=scene,
            camera=self._camera(scene, render.camera_id),
            maps=maps,
            output_dir=output_dir,
            anchor_path=reference_path,
            room_name=room_name,
            attempt=render.attempts,
        )
        return retried if retried.status != RenderStatus.FAILED else render

    def _settle(
        self,
        render: Render,
        scene: Scene,
        conditioning: dict[str, ConditioningMaps],
        output_dir: Path,
        reference_path: Path | None,
        room_name: str,
        on_progress: ProgressHook | None,
    ) -> Render:
        """Judge one render, re-rolling until it passes or attempts run out.

        Keeps the best-scoring attempt, not the last one. Measured generator
        spread on a fixed prompt and camera is sd ~0.15-0.18 of layout fidelity
        (n=6, judge sd 0.04 on a fixed image — so the spread is the generator,
        not the measurement). Against that, returning the final sample throws
        away a better one it has already paid for: a real run went
        0.62 -> 0.72 -> 0.64 and kept 0.64.
        """
        current = render
        maps = conditioning.get(render.camera_id)
        best = render
        best_score = -1.0

        def remember(candidate: Render) -> None:
            """Track the leader. Unverified scores never win — an unchecked
            render is not a good render, it is an unmeasured one."""
            nonlocal best, best_score
            scored = candidate.scores
            if scored is not None and scored.verified and scored.overall > best_score:
                best, best_score = candidate, scored.overall

        for attempt in range(self.settings.max_render_attempts):
            current.status = RenderStatus.JUDGING
            scores = self.judge.judge(current, scene, reference_path)
            current.scores = scores

            remember(current)

            if not scores.verified:
                # Nothing was checked, so there is nothing to retry against —
                # re-rolling would burn budget on an unmeasurable difference.
                current.status = RenderStatus.COMPLETED
                break

            if scores.passes(self.settings.consistency_threshold):
                current.status = RenderStatus.COMPLETED
                logger.info(
                    "render %s passed at %.2f (attempt %d)",
                    current.id,
                    scores.overall,
                    attempt + 1,
                )
                break

            is_last = attempt == self.settings.max_render_attempts - 1
            logger.info(
                "render %s scored %.2f%s%s",
                current.id,
                scores.overall,
                f", missing {scores.missing_instance_ids}" if scores.missing_instance_ids else "",
                " — accepting, out of attempts" if is_last else " — retrying",
            )

            if is_last or maps is None:
                # Out of attempts. Fall back to the best attempt seen, with its
                # real score — a 0.62 reported honestly is more useful than a
                # gap, and better than a 0.55 that merely happened to be last.
                current.status = RenderStatus.COMPLETED
                break

            current.status = RenderStatus.RETRYING
            if on_progress is not None:
                on_progress(current)

            retried = self.renders.render_view(
                scene=scene,
                camera=self._camera(scene, current.camera_id),
                maps=maps,
                output_dir=output_dir,
                anchor_path=reference_path,
                room_name=room_name,
                attempt=attempt + 1,
            )
            if retried.status == RenderStatus.FAILED:
                # Generation broke rather than scored badly — keep the earlier
                # image, which at least exists.
                current.error = retried.error
                current.status = RenderStatus.COMPLETED
                break

            retried.scores = None
            current = retried

        if best_score >= 0 and best is not current:
            logger.info(
                "render %s keeping attempt %d at %.2f over final attempt at %.2f",
                best.id,
                best.attempts,
                best.scores.overall if best.scores else 0.0,
                current.scores.overall if current.scores else 0.0,
            )
            best.status = RenderStatus.COMPLETED
            # `attempts` is what the view cost, not which draw won. Reporting
            # the winner's ordinal would under-count generations already paid
            # for and make the run look cheaper than it was.
            best.attempts = max(best.attempts, current.attempts)
            current = best

        if on_progress is not None:
            on_progress(current)
        return current

    @staticmethod
    def _camera(scene: Scene, camera_id: str):
        camera = next((c for c in scene.cameras if c.id == camera_id), None)
        if camera is None:
            raise ValueError(f"Scene {scene.scene_id} has no camera {camera_id}")
        return camera


def summarize(renders: list[Render]) -> dict[str, object]:
    """Scene-level consistency summary for the API response.

    Reports the *worst* view alongside the mean: an average of 0.85 with one
    view at 0.4 is a broken set, and a mean alone would hide that.
    """
    scored = [r.scores for r in renders if r.scores and r.scores.verified]
    if not scored:
        return {
            "verified": False,
            "note": "No renders were verified — no consistency judge ran.",
            "render_count": len(renders),
        }

    overalls = [s.overall for s in scored]
    return {
        "verified": True,
        "render_count": len(renders),
        "verified_count": len(scored),
        "mean_consistency": round(sum(overalls) / len(overalls), 3),
        "worst_consistency": round(min(overalls), 3),
        "mean_layout_fidelity": round(sum(s.layout_fidelity for s in scored) / len(scored), 3),
        "mean_object_identity": round(sum(s.object_identity for s in scored) / len(scored), 3),
        "missing_objects": sorted({i for s in scored for i in s.missing_instance_ids}),
        "hallucinated_objects": sorted({o for s in scored for o in s.hallucinated_objects}),
    }


def mean_consistency(renders: list[Render]) -> float | None:
    scored = [r.scores.overall for r in renders if r.scores and r.scores.verified]
    return round(sum(scored) / len(scored), 3) if scored else None
