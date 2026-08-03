"""Floor plan upload and design endpoints.

Every capability here is also reachable from `cli.py` through the same service
layer, so the architecture doesn't depend on the UI existing — the brief allows
a REST API, a CLI *or* a web interface, and building the core so all three are
thin over one pipeline is the point.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile
from sse_starlette.sse import EventSourceResponse

from ..config import get_settings
from ..ingest.loader import SUPPORTED_SUFFIXES, UnsupportedPlanFormat, page_count
from ..ingest.service import FloorPlanIngestService, IngestionError
from ..schemas.render import DesignRequest, RegenerateRequest
from ..services.store import FLOORPLAN_STORE, JOB_STORE

logger = logging.getLogger(__name__)
router = APIRouter(tags=["design"])


# --- floor plans -----------------------------------------------------------


@router.post("/floorplans")
async def upload_floorplan(
    file: UploadFile = File(...),
    page: int = Form(0),
    ceiling_height_m: float | None = Form(None),
    px_per_m: float | None = Form(None),
) -> dict[str, object]:
    """Upload a plan, extract its geometry, and calibrate it to metres.

    `px_per_m` is an escape hatch: a plan with no printed m² labels and no
    dimension ticks cannot be calibrated automatically, and guessing a scale
    would silently corrupt every measurement downstream.
    """
    settings = get_settings()
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format '{suffix}'. Supported: "
            f"{', '.join(sorted(SUPPORTED_SUFFIXES))}",
        )

    destination = settings.upload_dir / f"{uuid.uuid4().hex[:10]}{suffix}"
    with destination.open("wb") as handle:
        shutil.copyfileobj(file.file, handle)

    service = FloorPlanIngestService(settings)
    try:
        # Extraction is a blocking vision call; keep it off the event loop.
        floorplan, working = await asyncio.to_thread(
            service.ingest,
            destination,
            page,
            True,
            px_per_m,
            ceiling_height_m,
        )
    except (IngestionError, UnsupportedPlanFormat) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("floor plan ingestion failed")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}") from exc

    stored = FLOORPLAN_STORE.add(floorplan, destination, working)
    return _floorplan_payload(stored)


@router.get("/floorplans")
def list_floorplans() -> list[dict[str, object]]:
    return [_floorplan_payload(plan) for plan in FLOORPLAN_STORE.list()]


@router.get("/floorplans/{floorplan_id}")
def get_floorplan(floorplan_id: str) -> dict[str, object]:
    stored = FLOORPLAN_STORE.get(floorplan_id)
    if stored is None:
        raise HTTPException(status_code=404, detail=f"Unknown floor plan: {floorplan_id}")
    return _floorplan_payload(stored, include_geometry=True)


@router.get("/floorplans/pages")
def count_pages(path: str) -> dict[str, int]:
    """Page count for a multi-page PDF, so the UI can offer a page picker."""
    return {"pages": page_count(Path(path))}


def _floorplan_payload(stored, include_geometry: bool = False) -> dict[str, object]:
    plan = stored.floorplan
    payload: dict[str, object] = {
        "id": plan.id,
        "source_filename": plan.source_filename,
        "uploaded_at": stored.uploaded_at,
        "image_url": f"/static/uploads/{stored.working_image_path.name}",
        "total_area_m2": round(plan.total_area_m2, 2),
        "calibration": {
            "px_per_m": round(plan.calibration.px_per_m, 2),
            "method": plan.calibration.method,
            "residual_pct": plan.calibration.residual_pct,
            "confidence": plan.calibration.confidence,
            # Surfaced, not buried: a low-confidence scale means every
            # dimension downstream is suspect and the user should know.
            "warnings": plan.calibration.warnings,
        },
        "rooms": [
            {
                "id": room.id,
                "name": room.name,
                "type": room.room_type.value,
                "area_m2": round(room.area_m2, 2),
                "area_label_m2": room.area_label_m2,
                "furnishable": room.room_type.is_furnishable,
            }
            for room in plan.rooms
        ],
    }
    if include_geometry:
        payload["geometry"] = plan.model_dump(mode="json")
    return payload


# --- designs ---------------------------------------------------------------


@router.post("/designs")
async def create_design(request: DesignRequest, background: BackgroundTasks) -> dict[str, object]:
    """Queue a design job. Returns immediately; follow `/designs/{id}/stream`."""
    stored = FLOORPLAN_STORE.get(request.floorplan_id)
    if stored is None:
        raise HTTPException(status_code=404, detail=f"Unknown floor plan: {request.floorplan_id}")

    job = JOB_STORE.create(request)
    JOB_STORE.start(job, stored.floorplan)
    return {
        "job_id": job.id,
        "status": job.status.value,
        "stream_url": f"/designs/{job.id}/stream",
    }


@router.get("/designs")
def list_designs() -> list[dict[str, object]]:
    return [
        {
            "id": job.id,
            "floorplan_id": job.floorplan_id,
            "status": job.status.value,
            "progress": job.progress,
            "style": job.request.style.value,
            "created_at": job.created_at,
            "render_count": len(job.all_renders()),
        }
        for job in JOB_STORE.list()
    ]


@router.get("/designs/{job_id}")
def get_design(job_id: str) -> dict[str, object]:
    job = JOB_STORE.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Unknown job: {job_id}")
    return _job_payload(job)


@router.get("/designs/{job_id}/stream")
async def stream_design(job_id: str):
    """Server-sent events for one job.

    Streaming rather than polling because renders take 25 seconds each and
    arrive one at a time — the gallery should fill in as they land rather than
    appear all at once behind a 90-second spinner.
    """
    record = JOB_STORE.record(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Unknown job: {job_id}")

    queue = JOB_STORE.subscribe(job_id)
    if queue is None:
        raise HTTPException(status_code=404, detail=f"Unknown job: {job_id}")

    async def events():
        try:
            # Replay what already happened, so a client that connects late (or
            # reconnects) sees the whole run rather than joining mid-way. The
            # subscription is already open, so an event arriving during the
            # replay lands in the queue too — hence the sequence check below.
            replayed = -1
            for event in list(record.events):
                replayed = max(replayed, event.seq)
                yield {"event": "progress", "data": json.dumps(event.to_dict())}

            if record.done.is_set():
                yield {"event": "complete", "data": json.dumps(_job_payload(record.job))}
                return

            while True:
                event = await queue.get()
                if event is None:
                    break
                if event.seq <= replayed:
                    continue  # already sent during replay
                replayed = event.seq
                yield {"event": "progress", "data": json.dumps(event.to_dict())}

            yield {"event": "complete", "data": json.dumps(_job_payload(record.job))}
        finally:
            JOB_STORE.unsubscribe(job_id, queue)

    return EventSourceResponse(events())


@router.get("/designs/{job_id}/bom")
def get_bom(job_id: str, variation: int = 0) -> dict[str, object]:
    """Products used, quantities and estimated total cost.

    A traversal of the scene graph rather than an inference from the images, so
    the list is provably what was rendered.
    """
    job = JOB_STORE.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Unknown job: {job_id}")
    if variation >= len(job.variations):
        raise HTTPException(status_code=404, detail=f"No variation {variation} in this job")

    entry = job.variations[variation]
    if entry.bom is None:
        raise HTTPException(status_code=409, detail="This variation has no bill of materials yet")

    return {
        "job_id": job.id,
        "scene_id": entry.scene_id,
        "variation_index": entry.variation_index,
        "currency": entry.bom.currency,
        "total_cost": entry.bom.total_cost,
        "item_count": entry.bom.item_count,
        "lines": [line.model_dump(mode="json") for line in entry.bom.lines],
    }


@router.post("/designs/{job_id}/regenerate")
async def regenerate(job_id: str, request: RegenerateRequest) -> dict[str, object]:
    """Re-render an existing design.

    With `preserve_scene` the scene graph is reused verbatim and only the image
    seeds change — which is what "regenerate, same room" has to mean for the
    result to be comparable. Patch-based edits from a natural-language change
    request are not implemented; see the response note rather than a silent
    full re-design.
    """
    job = JOB_STORE.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Unknown job: {job_id}")

    stored = FLOORPLAN_STORE.get(job.floorplan_id)
    if stored is None:
        raise HTTPException(status_code=404, detail="The job's floor plan is no longer available")

    if request.changes and not request.preserve_scene:
        raise HTTPException(
            status_code=501,
            detail=(
                "Natural-language edits are not implemented. The scene-graph patch path "
                "is designed but unbuilt — see README. Use preserve_scene=true to "
                "re-render the same scene, or create a new design to change the brief."
            ),
        )

    new_request = job.request.model_copy(
        update={
            "views_per_room": request.views_per_room or job.request.views_per_room,
            # A different seed with the same scene graph is exactly "same room,
            # new photographs".
            "seed": (job.request.seed or 0) + 1,
        }
    )
    new_job = JOB_STORE.create(new_request)
    JOB_STORE.start(new_job, stored.floorplan)
    return {
        "job_id": new_job.id,
        "status": new_job.status.value,
        "stream_url": f"/designs/{new_job.id}/stream",
        "preserved_scene": request.preserve_scene,
    }


def _job_payload(job) -> dict[str, object]:
    from ..services.pipeline import _static_url

    return {
        "id": job.id,
        "floorplan_id": job.floorplan_id,
        "status": job.status.value,
        "progress": job.progress,
        "detail": job.stage_detail,
        "error": job.error,
        "created_at": job.created_at,
        "completed_at": job.completed_at,
        "style": job.request.style.value,
        "palette": job.request.palette_name,
        "consistency": JOB_STORE.pipeline.summary(job),
        "variations": [
            {
                "scene_id": variation.scene_id,
                "variation_index": variation.variation_index,
                "mean_consistency": variation.mean_consistency,
                "total_cost": variation.bom.total_cost if variation.bom else None,
                "currency": variation.bom.currency if variation.bom else None,
                "renders": [
                    {
                        "id": render.id,
                        "camera_id": render.camera_id,
                        "room_id": render.room_id,
                        "status": render.status.value,
                        "is_anchor": render.is_anchor,
                        "attempts": render.attempts,
                        "image_url": _static_url(render.image_path),
                        "preview_url": _static_url(
                            render.conditioning.preview_path if render.conditioning else None
                        ),
                        "depth_url": _static_url(
                            render.conditioning.depth_path if render.conditioning else None
                        ),
                        "segmentation_url": _static_url(
                            render.conditioning.segmentation_path if render.conditioning else None
                        ),
                        "overall": render.scores.overall if render.scores else None,
                        "verified": render.scores.verified if render.scores else False,
                        "scores": render.scores.model_dump() if render.scores else None,
                        "product_ids": render.product_ids,
                        "error": render.error,
                    }
                    for render in variation.renders
                ],
            }
            for variation in job.variations
        ],
    }
