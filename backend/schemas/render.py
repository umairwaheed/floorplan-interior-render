"""Render, job, and evaluation schemas."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from .product import BillOfMaterials, DesignStyle


class ConditioningMaps(BaseModel):
    """The geometry buffers rasterized from the scene for one camera.

    These are what force layout agreement between viewpoints — the image model
    is conditioned on projected 3D truth rather than asked to imagine a room.
    """

    depth_path: str
    segmentation_path: str
    wireframe_path: str
    preview_path: str | None = Field(
        default=None, description="Flat-shaded colour preview, for debugging and the UI."
    )
    visible_instance_ids: list[str] = Field(
        default_factory=list,
        description="Objects actually inside this frustum. The prompt only ever "
        "names these, so the model is never told to draw something off-screen.",
    )
    instance_pixel_share: dict[str, float] = Field(
        default_factory=dict,
        description="instance_id → fraction of frame. Drives prompt emphasis and "
        "lets the judge ignore objects too small to assess.",
    )
    instance_screen_boxes: dict[str, tuple[int, int, int, int]] = Field(
        default_factory=dict,
        description="instance_id → (x0, y0, x1, y1) as percentages of the frame. "
        "Stating positions in text as well as pixels measurably improved layout "
        "fidelity — the model grounds on the numbers where it drifts on the image.",
    )


class RenderStatus(str, Enum):
    PENDING = "pending"
    GENERATING = "generating"
    JUDGING = "judging"
    RETRYING = "retrying"
    COMPLETED = "completed"
    FAILED = "failed"


class ConsistencyScores(BaseModel):
    """Structured judgement of one render. Reported honestly, not hidden."""

    layout_fidelity: float = Field(ge=0, le=1)
    object_identity: float = Field(ge=0, le=1)
    cross_view_consistency: float | None = Field(
        default=None,
        ge=0,
        le=1,
        description="None for an anchor view — it has no earlier view to be "
        "consistent with, and scoring it 1.0 would flatter every anchor.",
    )
    style_adherence: float = Field(ge=0, le=1)
    photorealism: float = Field(ge=0, le=1)

    verified: bool = Field(
        default=True,
        description="False when no judge ran. An unverified render carries "
        "placeholder scores that must never be read as a passing grade.",
    )

    missing_instance_ids: list[str] = Field(default_factory=list)
    hallucinated_objects: list[str] = Field(
        default_factory=list, description="Visible items absent from the scene graph."
    )
    issues: list[str] = Field(default_factory=list)

    @property
    def overall(self) -> float:
        """Weighted toward the criteria the brief grades hardest.

        Renormalized when cross-view doesn't apply, so an anchor is scored on
        what it can actually be scored on rather than given a free quarter.
        """
        weights = [
            (0.30, self.layout_fidelity),
            (0.25, self.object_identity),
            (0.10, self.style_adherence),
            (0.10, self.photorealism),
        ]
        if self.cross_view_consistency is not None:
            weights.append((0.25, self.cross_view_consistency))

        total_weight = sum(weight for weight, _ in weights)
        return round(sum(weight * value for weight, value in weights) / total_weight, 3)

    def passes(self, threshold: float) -> bool:
        """A missing object fails outright, however good the rest looks.

        The brief's hard requirement is that objects don't disappear between
        viewpoints, so that is not something a high style score can offset.
        """
        return self.verified and self.overall >= threshold and not self.missing_instance_ids


class Render(BaseModel):
    id: str
    scene_id: str
    camera_id: str
    room_id: str
    status: RenderStatus = RenderStatus.PENDING

    image_path: str | None = None
    conditioning: ConditioningMaps | None = None
    scores: ConsistencyScores | None = None

    attempts: int = 0
    is_anchor: bool = Field(
        default=False,
        description="The first view of a room. Later views receive it as an "
        "appearance reference, which is what holds object identity stable.",
    )
    seed: int = 0
    prompt: str | None = None
    error: str | None = None
    duration_s: float | None = None

    # Per the brief: return the products used in *every* render.
    product_ids: list[str] = Field(default_factory=list)


class JobStatus(str, Enum):
    QUEUED = "queued"
    EXTRACTING = "extracting"
    DESIGNING = "designing"
    RASTERIZING = "rasterizing"
    RENDERING = "rendering"
    VERIFYING = "verifying"
    COMPLETED = "completed"
    FAILED = "failed"


class DesignRequest(BaseModel):
    floorplan_id: str
    style: DesignStyle
    palette_name: str | None = None
    room_ids: list[str] = Field(
        default_factory=list, description="Empty means all furnishable rooms."
    )
    variations: int = Field(default=1, ge=1, le=4)
    views_per_room: int = Field(default=3, ge=1, le=6)
    budget_max: float | None = None
    seed: int | None = Field(default=None, description="Omit for a random scene.")


class RegenerateRequest(BaseModel):
    preserve_scene: bool = Field(
        default=True,
        description="True re-renders the identical scene graph with new image "
        "seeds. False lets `changes` patch the graph.",
    )
    changes: str | None = Field(
        default=None,
        description="Natural-language change request, e.g. 'make the sofa green'. "
        "Applied as a patch so untouched objects stay pixel-stable.",
    )
    views_per_room: int | None = None


class DesignVariation(BaseModel):
    scene_id: str
    variation_index: int
    renders: list[Render] = Field(default_factory=list)
    bom: BillOfMaterials | None = None
    mean_consistency: float | None = None


class DesignJob(BaseModel):
    id: str
    floorplan_id: str
    request: DesignRequest
    status: JobStatus = JobStatus.QUEUED
    progress: float = Field(default=0.0, ge=0, le=1)
    stage_detail: str = ""
    variations: list[DesignVariation] = Field(default_factory=list)
    error: str | None = None
    created_at: str = ""
    completed_at: str = ""

    def all_renders(self) -> list[Render]:
        return [r for v in self.variations for r in v.renders]
