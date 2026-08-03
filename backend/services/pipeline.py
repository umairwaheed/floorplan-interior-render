"""End-to-end design pipeline.

The one place the six stages are wired together: design → cameras → rasterize →
render → verify → cost. Everything it calls is a service that also works
standalone, so the CLI and the API run identical code rather than parallel
implementations that drift.

Progress is reported through a callback rather than logged, because the UI
needs per-view updates as they land — a 90-second blank spinner is a worse
answer than a slightly uglier one that shows the anchor view appearing first.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ..catalog.service import CatalogService, get_catalog_service
from ..config import Settings, get_settings
from ..design.agent import DesignAgent
from ..imagegen.service import RenderService
from ..render.service import SceneRenderer
from ..schemas.floorplan import FloorPlan
from ..schemas.render import (
    DesignJob,
    DesignRequest,
    DesignVariation,
    JobStatus,
    Render,
)
from ..schemas.scene import Scene
from ..verify.service import VerificationService, mean_consistency, summarize

logger = logging.getLogger(__name__)


@dataclass
class ProgressEvent:
    """One thing that happened, in a shape the UI can render directly."""

    job_id: str
    status: JobStatus
    progress: float
    detail: str
    render: Render | None = None
    variation_index: int | None = None
    #: Monotonic per job. A stream subscribes before replaying history, so an
    #: event landing in that window would otherwise be delivered twice.
    seq: int = 0

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "seq": self.seq,
            "job_id": self.job_id,
            "status": self.status.value,
            "progress": round(self.progress, 3),
            "detail": self.detail,
        }
        if self.variation_index is not None:
            payload["variation_index"] = self.variation_index
        if self.render is not None:
            payload["render"] = {
                "id": self.render.id,
                "camera_id": self.render.camera_id,
                "room_id": self.render.room_id,
                "status": self.render.status.value,
                "is_anchor": self.render.is_anchor,
                "attempts": self.render.attempts,
                "image_url": _static_url(self.render.image_path),
                "preview_url": _static_url(
                    self.render.conditioning.preview_path if self.render.conditioning else None
                ),
                "scores": self.render.scores.model_dump() if self.render.scores else None,
                "overall": self.render.scores.overall if self.render.scores else None,
                "product_ids": self.render.product_ids,
                "error": self.render.error,
            }
        return payload


ProgressHook = Callable[[ProgressEvent], None]


def _static_url(path: str | None) -> str | None:
    """Map a filesystem path into the URL the static mount serves it at."""
    if not path:
        return None
    resolved = Path(path).resolve()
    output_root = get_settings().output_dir.resolve()
    try:
        return f"/static/outputs/{resolved.relative_to(output_root)}"
    except ValueError:
        return None


class DesignPipeline:
    """Runs a design request to completion."""

    def __init__(
        self,
        catalog: CatalogService | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.catalog = catalog or get_catalog_service()
        self.agent = DesignAgent(catalog=self.catalog, settings=self.settings)
        self.scenes = SceneRenderer(self.settings)
        self.renders = RenderService(settings=self.settings)
        self.verifier = VerificationService(self.renders, settings=self.settings)
        # Scenes are kept so a regeneration reuses the exact graph rather than
        # rebuilding it. Rebuilding would produce the same hash, but reusing
        # makes that a guarantee rather than a coincidence.
        self._scene_cache: dict[str, Scene] = {}

    def run(
        self,
        job: DesignJob,
        floorplan: FloorPlan,
        on_progress: ProgressHook | None = None,
        scenes: list[Scene] | None = None,
    ) -> DesignJob:
        """Execute a job, mutating it in place so the caller can poll it too.

        `scenes` re-photographs an existing design: the graphs are used exactly
        as given rather than re-solved. Passing them is the only way to honour
        "regenerate, same scene" — re-running the solver with a different seed
        moves furniture, however small the change to the request looks.
        """
        request = job.request
        started = time.monotonic()

        def emit(status: JobStatus, progress: float, detail: str, **kwargs) -> None:
            job.status = status
            job.progress = progress
            job.stage_detail = detail
            if on_progress is not None:
                on_progress(ProgressEvent(job.id, status, progress, detail, **kwargs))

        try:
            total = max(request.variations, 1) if scenes is None else len(scenes)
            for index in range(total):
                base = index / total
                span = 1.0 / total

                if scenes is not None:
                    scene = scenes[index]
                    emit(
                        JobStatus.DESIGNING,
                        base + span * 0.05,
                        f"Reusing scene {scene.scene_id} ({len(scene.objects)} objects)",
                        variation_index=index,
                    )
                else:
                    emit(
                        JobStatus.DESIGNING,
                        base + span * 0.05,
                        f"Designing variation {index + 1} of {total}",
                        variation_index=index,
                    )
                    scene = self._design(floorplan, request, index)

                emit(
                    JobStatus.RASTERIZING,
                    base + span * 0.20,
                    f"Placing cameras and projecting geometry ({len(scene.cameras)} views)",
                    variation_index=index,
                )
                output_dir = self.settings.output_dir / scene.output_key
                conditioning = self.scenes.render_scene(scene, floorplan, output_dir)

                # Renders stream out as they finish, so the gallery fills in
                # rather than appearing all at once at the end.
                done = 0
                expected = max(len(scene.cameras), 1)

                def on_render(
                    render: Render, _index=index, _base=base, _span=span, _expected=expected
                ) -> None:
                    nonlocal done
                    done += 1
                    emit(
                        JobStatus.RENDERING,
                        _base + _span * (0.20 + 0.55 * done / _expected),
                        f"Rendered {done} of {_expected} views",
                        render=render,
                        variation_index=_index,
                    )

                emit(
                    JobStatus.RENDERING,
                    base + span * 0.25,
                    f"Generating {expected} view(s)",
                    variation_index=index,
                )
                renders = self.renders.render_scene(
                    scene, floorplan, conditioning, output_dir, on_progress=on_render
                )

                emit(
                    JobStatus.VERIFYING,
                    base + span * 0.80,
                    "Checking consistency against the scene geometry",
                    variation_index=index,
                )
                renders = self.verifier.verify_and_retry(
                    scene,
                    floorplan,
                    renders,
                    conditioning,
                    output_dir,
                    on_progress=lambda r, _i=index: emit(
                        JobStatus.VERIFYING,
                        job.progress,
                        f"Verified {r.camera_id}",
                        render=r,
                        variation_index=_i,
                    ),
                )

                job.variations.append(
                    DesignVariation(
                        scene_id=scene.scene_id,
                        variation_index=index,
                        renders=renders,
                        bom=self.agent.bill_of_materials(scene),
                        mean_consistency=mean_consistency(renders),
                    )
                )
                self._scene_cache[scene.scene_id] = scene

            emit(JobStatus.COMPLETED, 1.0, "Done")
            logger.info(
                "job %s finished in %.1fs: %d variation(s), %d render(s)",
                job.id,
                time.monotonic() - started,
                len(job.variations),
                len(job.all_renders()),
            )

        except Exception as exc:  # noqa: BLE001 — a job failure is data, not a crash
            job.status = JobStatus.FAILED
            job.error = str(exc)
            logger.exception("job %s failed", job.id)
            if on_progress is not None:
                on_progress(ProgressEvent(job.id, JobStatus.FAILED, job.progress, str(exc)))

        return job

    def _design(self, floorplan: FloorPlan, request: DesignRequest, index: int) -> Scene:
        scene = self.agent.design(
            floorplan=floorplan,
            style=request.style,
            palette_name=request.palette_name,
            room_ids=request.room_ids or None,
            seed=request.seed if request.seed is not None else 0,
            variation_index=index,
            budget=request.budget_max,
        )
        return self.scenes.attach_cameras(scene, floorplan, request.views_per_room)

    def scene(self, scene_id: str) -> Scene | None:
        return self._scene_cache.get(scene_id)

    def summary(self, job: DesignJob) -> dict[str, object]:
        """Consistency summary across every render in the job."""
        return summarize(job.all_renders())
