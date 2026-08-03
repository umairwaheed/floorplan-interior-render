"""In-process stores for floor plans and jobs.

Deliberately the simplest thing that works: dictionaries plus files on disk. It
is honest about being single-process — `JOB_STORE` is fine for a demo and for a
CLI, and would be replaced by Redis or a database the moment a second worker
exists. The interface is narrow enough that the swap touches this file only.

Jobs run on a thread pool rather than in the event loop. Image generation is a
25-second blocking HTTP call; running it inline would stall every other request
including the SSE stream that is meant to report its progress.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from ..config import get_settings
from ..ingest.loader import LoadedPlan
from ..schemas.floorplan import FloorPlan
from ..schemas.render import DesignJob, DesignRequest
from .pipeline import DesignPipeline, ProgressEvent

logger = logging.getLogger(__name__)


@dataclass
class StoredFloorPlan:
    """An uploaded plan and everything derived from it."""

    id: str
    floorplan: FloorPlan
    source_path: Path
    working_image_path: Path
    uploaded_at: str


class FloorPlanStore:
    def __init__(self) -> None:
        self._plans: dict[str, StoredFloorPlan] = {}
        self._lock = threading.Lock()

    def add(self, floorplan: FloorPlan, source: Path, working: LoadedPlan) -> StoredFloorPlan:
        settings = get_settings()
        working_path = settings.upload_dir / f"{floorplan.id}_working.png"
        working.save(working_path)

        stored = StoredFloorPlan(
            id=floorplan.id,
            floorplan=floorplan,
            source_path=source,
            working_image_path=working_path,
            uploaded_at=datetime.now(UTC).isoformat(),
        )
        with self._lock:
            self._plans[floorplan.id] = stored
        return stored

    def get(self, floorplan_id: str) -> StoredFloorPlan | None:
        return self._plans.get(floorplan_id)

    def list(self) -> list[StoredFloorPlan]:
        return sorted(self._plans.values(), key=lambda p: p.uploaded_at, reverse=True)


@dataclass
class JobRecord:
    """A job plus the machinery for streaming its progress."""

    job: DesignJob
    events: list[ProgressEvent] = field(default_factory=list)
    subscribers: list[asyncio.Queue] = field(default_factory=list)
    loop: asyncio.AbstractEventLoop | None = None
    done: threading.Event = field(default_factory=threading.Event)


class JobStore:
    """Runs design jobs on a thread pool and fans progress out to SSE clients."""

    def __init__(self, max_workers: int = 2) -> None:
        self._jobs: dict[str, JobRecord] = {}
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="design")
        self._pipeline = DesignPipeline()

    @property
    def pipeline(self) -> DesignPipeline:
        return self._pipeline

    def create(self, request: DesignRequest) -> DesignJob:
        job = DesignJob(
            id=f"job-{uuid.uuid4().hex[:10]}",
            floorplan_id=request.floorplan_id,
            request=request,
            created_at=datetime.now(UTC).isoformat(),
        )
        with self._lock:
            self._jobs[job.id] = JobRecord(job=job)
        return job

    def get(self, job_id: str) -> DesignJob | None:
        record = self._jobs.get(job_id)
        return record.job if record else None

    def record(self, job_id: str) -> JobRecord | None:
        return self._jobs.get(job_id)

    def list(self) -> list[DesignJob]:
        return sorted(
            (r.job for r in self._jobs.values()), key=lambda j: j.created_at, reverse=True
        )

    def start(self, job: DesignJob, floorplan: FloorPlan) -> None:
        """Queue a job. Returns immediately; progress arrives via `subscribe`."""
        record = self._jobs[job.id]
        try:
            record.loop = asyncio.get_running_loop()
        except RuntimeError:
            record.loop = None  # started from the CLI, no event loop

        self._executor.submit(self._run, record, floorplan)

    def _run(self, record: JobRecord, floorplan: FloorPlan) -> None:
        def on_progress(event: ProgressEvent) -> None:
            # Numbered before publishing, so a replaying stream can tell which
            # events it has already sent.
            event.seq = len(record.events)
            record.events.append(event)
            self._publish(record, event)

        try:
            self._pipeline.run(record.job, floorplan, on_progress=on_progress)
        finally:
            record.job.completed_at = datetime.now(UTC).isoformat()
            record.done.set()
            # Wake any stream still waiting, so it closes rather than hanging
            # until its own timeout.
            self._publish(record, None)

    def _publish(self, record: JobRecord, event: ProgressEvent | None) -> None:
        """Hand an event to every subscriber, from a worker thread.

        Queues belong to the event loop, so the put has to be scheduled onto
        it rather than called directly — doing it directly from this thread is
        the classic way to corrupt asyncio state under load.
        """
        loop = record.loop
        if loop is None:
            return
        for queue in list(record.subscribers):
            # A closed loop just means that client is gone.
            with contextlib.suppress(RuntimeError):
                loop.call_soon_threadsafe(queue.put_nowait, event)

    def subscribe(self, job_id: str) -> asyncio.Queue | None:
        record = self._jobs.get(job_id)
        if record is None:
            return None
        queue: asyncio.Queue = asyncio.Queue()
        record.subscribers.append(queue)
        return queue

    def unsubscribe(self, job_id: str, queue: asyncio.Queue) -> None:
        record = self._jobs.get(job_id)
        if record and queue in record.subscribers:
            record.subscribers.remove(queue)

    def run_sync(
        self,
        job: DesignJob,
        floorplan: FloorPlan,
        on_progress: Callable[[ProgressEvent], None] | None = None,
    ) -> DesignJob:
        """Run in the calling thread. Used by the CLI, which wants to block."""
        return self._pipeline.run(job, floorplan, on_progress=on_progress)


FLOORPLAN_STORE = FloorPlanStore()
JOB_STORE = JobStore()
