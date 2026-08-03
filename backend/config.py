"""Configuration. Everything tunable lives here, nothing is hard-coded downstream."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Paths -------------------------------------------------------------
    data_dir: Path = PROJECT_ROOT / "data"
    catalog_dir: Path = PROJECT_ROOT / "data" / "catalog"
    upload_dir: Path = PROJECT_ROOT / "data" / "uploads"
    output_dir: Path = PROJECT_ROOT / "data" / "outputs"
    frontend_dir: Path = PROJECT_ROOT / "frontend"
    db_path: Path = PROJECT_ROOT / "data" / "catalog.db"

    # --- Model selection ---------------------------------------------------
    # Vision extraction and judging need reasoning over images; product
    # enrichment is bulk work where a cheap model is the right call.
    vision_model: str = "claude-opus-5"
    enrichment_model: str = "claude-haiku-4-5"
    judge_model: str = "claude-opus-5"

    # Opus 5 supports high-resolution vision (2576px long edge). Floor plans are
    # dense line drawings, so resolution directly determines extraction quality.
    max_plan_edge_px: int = 2200
    plan_render_dpi: int = 300
    image_model: str = "gemini-3-pro-image-preview"

    anthropic_api_key: str | None = None
    gemini_api_key: str | None = None

    # `mock` renders the conditioning maps straight through, so the entire
    # pipeline is runnable and testable without an image-model key.
    image_backend: str = "mock"

    # --- Rendering ---------------------------------------------------------
    render_width: int = 1024
    render_height: int = 768
    conditioning_width: int = 1024
    conditioning_height: int = 768

    camera_height_m: float = 1.5
    camera_corner_inset_m: float = 0.6
    camera_fov_deg: float = 60.0
    min_camera_coverage_pct: float = 0.45

    # --- Placement solver --------------------------------------------------
    min_circulation_m: float = 0.7
    wall_clearance_m: float = 0.05
    solver_iterations: int = 4000
    solver_restarts: int = 3

    # --- Consistency verification -----------------------------------------
    consistency_threshold: float = 0.75
    max_render_attempts: int = 3
    enable_judge: bool = True

    # --- Retrieval ---------------------------------------------------------
    # `hashing` is offline and deterministic; `gemini` gives real semantic
    # recall. Hard structured filters run before either, so the default is
    # sufficient at demo-catalog scale.
    embedding_backend: str = "hashing"
    embedding_dim: int = 256
    retrieval_candidates: int = 60

    default_ceiling_height_m: float = 2.7
    default_currency: str = "GEL"

    def ensure_dirs(self) -> None:
        for path in (self.data_dir, self.catalog_dir, self.upload_dir, self.output_dir):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_dirs()
    return settings
