"""FastAPI application.

The API is a thin transport over the service layer — every capability it
exposes is also reachable from `cli.py`, which keeps the architecture honest
about not depending on the UI.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .api import catalog_routes
from .config import get_settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)

settings = get_settings()

app = FastAPI(
    title="Floor Plan → Interior Render",
    description=(
        "Generates photorealistic, multi-viewpoint interior renders from a floor plan, "
        "furnished exclusively with real catalog products. Multi-view consistency is "
        "enforced structurally by a content-hashed 3D scene graph."
    ),
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(catalog_routes.router)


@app.get("/health", tags=["meta"])
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "image_backend": settings.image_backend,
        "has_anthropic_key": bool(settings.anthropic_api_key),
        "has_gemini_key": bool(settings.gemini_api_key),
    }


# Generated renders and conditioning maps are served straight off disk.
app.mount("/static/outputs", StaticFiles(directory=settings.output_dir), name="outputs")
app.mount("/static/uploads", StaticFiles(directory=settings.upload_dir), name="uploads")

if settings.frontend_dir.exists():
    app.mount("/app", StaticFiles(directory=settings.frontend_dir, html=True), name="frontend")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(settings.frontend_dir / "index.html")
