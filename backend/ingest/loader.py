"""Floor plan intake — PNG, JPG, JPEG, PDF in; a normalized RGB image out.

Resolution is the whole game here. A floor plan is a dense line drawing where
the signal is thin strokes and small printed labels (`25.2 m²`, `370`), so
anything that blurs or downsamples costs extraction accuracy directly. PDFs are
therefore rendered at 300 DPI rather than at their embedded raster size, and
small scans are upscaled rather than left alone.
"""

from __future__ import annotations

import base64
import io
import logging
from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image, ImageOps

logger = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = {".png", ".jpg", ".jpeg", ".pdf"}

#: Below this, small scans are upscaled — the vision model reads fine detail
#: better at size, and upscaling never invents geometry that wasn't there.
MIN_LONG_EDGE_PX = 1400


@dataclass(frozen=True)
class LoadedPlan:
    """A floor plan image ready for the vision model."""

    image: Image.Image
    source_filename: str
    page_number: int | None
    page_count: int
    was_upscaled: bool
    original_size_px: tuple[int, int]

    @property
    def width(self) -> int:
        return self.image.width

    @property
    def height(self) -> int:
        return self.image.height

    def to_base64_png(self) -> str:
        """Encode for the Anthropic vision API (no newlines, per the API contract)."""
        buffer = io.BytesIO()
        self.image.save(buffer, format="PNG", optimize=True)
        return base64.standard_b64encode(buffer.getvalue()).decode("ascii")

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.image.save(path, format="PNG")
        return path


class UnsupportedPlanFormat(ValueError):
    """Raised for a file type we can't read."""


def _normalize(
    image: Image.Image, max_edge: int, source_filename: str, page: int | None, page_count: int
) -> LoadedPlan:
    """Apply EXIF orientation, flatten transparency, and fit within `max_edge`."""
    original_size = (image.width, image.height)

    image = ImageOps.exif_transpose(image)

    # Floor plans are frequently exported with an alpha channel. Compositing
    # onto white keeps thin black linework crisp; converting straight to RGB
    # would render transparent regions black and swallow the drawing.
    if image.mode in ("RGBA", "LA", "P"):
        image = image.convert("RGBA")
        canvas = Image.new("RGBA", image.size, (255, 255, 255, 255))
        image = Image.alpha_composite(canvas, image).convert("RGB")
    elif image.mode != "RGB":
        image = image.convert("RGB")

    was_upscaled = False
    long_edge = max(image.width, image.height)

    if long_edge < MIN_LONG_EDGE_PX:
        factor = MIN_LONG_EDGE_PX / long_edge
        image = image.resize(
            (round(image.width * factor), round(image.height * factor)),
            Image.LANCZOS,
        )
        was_upscaled = True
        logger.info("upscaled small plan %s by %.2fx", source_filename, factor)
    elif long_edge > max_edge:
        factor = max_edge / long_edge
        image = image.resize(
            (round(image.width * factor), round(image.height * factor)),
            Image.LANCZOS,
        )
        logger.info("downscaled plan %s to fit %dpx", source_filename, max_edge)

    return LoadedPlan(
        image=image,
        source_filename=source_filename,
        page_number=page,
        page_count=page_count,
        was_upscaled=was_upscaled,
        original_size_px=original_size,
    )


def _render_pdf_page(path: Path, page_index: int, dpi: int) -> tuple[Image.Image, int]:
    """Rasterize one PDF page at `dpi`.

    Rendering at an explicit DPI rather than the page's native resolution is
    deliberate: vector floor plans have no inherent pixel size, and PyMuPDF's
    default 72 DPI turns a readable drawing into an unreadable one.
    """
    with fitz.open(path) as document:
        page_count = document.page_count
        if page_count == 0:
            raise UnsupportedPlanFormat(f"{path.name} has no pages.")
        if not 0 <= page_index < page_count:
            raise UnsupportedPlanFormat(
                f"{path.name} has {page_count} page(s); page {page_index + 1} was requested."
            )
        page = document.load_page(page_index)
        pixmap = page.get_pixmap(matrix=fitz.Matrix(dpi / 72.0, dpi / 72.0), alpha=False)
        image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
    return image, page_count


def load_plan(
    path: Path,
    page: int = 0,
    max_edge: int = 2200,
    dpi: int = 300,
) -> LoadedPlan:
    """Load a floor plan from disk, normalized and ready for extraction.

    `page` is zero-based and only meaningful for PDFs.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No such floor plan: {path}")

    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise UnsupportedPlanFormat(
            f"Unsupported floor plan format '{suffix}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_SUFFIXES))}"
        )

    if suffix == ".pdf":
        image, page_count = _render_pdf_page(path, page, dpi)
        return _normalize(image, max_edge, path.name, page, page_count)

    with Image.open(path) as opened:
        opened.load()
        return _normalize(opened, max_edge, path.name, None, 1)


def crop_to_region(
    plan: LoadedPlan, box: tuple[float, float, float, float], pad: float = 0.02
) -> LoadedPlan:
    """Crop to a normalized (x0, y0, x1, y1) region, then re-normalize.

    Used with the drawing-region box the vision model reports on its first
    pass. Coordinates are fractions of width/height so the caller never has to
    know the pixel size, and `pad` keeps a margin so edge walls and the
    outermost dimension ticks aren't clipped off.
    """
    x0, y0, x1, y1 = box
    if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0):
        raise ValueError(f"Invalid normalized crop box: {box}")

    x0, y0 = max(0.0, x0 - pad), max(0.0, y0 - pad)
    x1, y1 = min(1.0, x1 + pad), min(1.0, y1 + pad)

    width, height = plan.image.width, plan.image.height
    cropped = plan.image.crop(
        (round(x0 * width), round(y0 * height), round(x1 * width), round(y1 * height))
    )
    # Re-normalizing upscales the crop back toward the working resolution,
    # which is the entire point — the drawing now fills the frame.
    return _normalize(
        cropped,
        max_edge=max(width, height),
        source_filename=plan.source_filename,
        page=plan.page_number,
        page_count=plan.page_count,
    )


def trim_whitespace(plan: LoadedPlan, threshold: int = 247) -> LoadedPlan:
    """Crop uniform near-white margins. Deliberately conservative.

    This only removes blank border, never content — it cannot distinguish a
    drawing from a paragraph of text, so it is not a substitute for
    `crop_to_region`. It exists to stop huge blank margins from eating the
    working resolution before the model ever sees the page.
    """
    import numpy as np

    pixels = np.asarray(plan.image.convert("L"))
    ink_rows = np.where((pixels < threshold).any(axis=1))[0]
    ink_cols = np.where((pixels < threshold).any(axis=0))[0]
    if ink_rows.size == 0 or ink_cols.size == 0:
        return plan  # blank page; nothing to trim

    height, width = pixels.shape
    box = (
        max(0, int(ink_cols[0]) - 8),
        max(0, int(ink_rows[0]) - 8),
        min(width, int(ink_cols[-1]) + 9),
        min(height, int(ink_rows[-1]) + 9),
    )
    if box == (0, 0, width, height):
        return plan

    return _normalize(
        plan.image.crop(box),
        max_edge=max(width, height),
        source_filename=plan.source_filename,
        page=plan.page_number,
        page_count=plan.page_count,
    )


def page_count(path: Path) -> int:
    """Number of pages, so a caller can offer a page picker for multi-page PDFs."""
    path = Path(path)
    if path.suffix.lower() != ".pdf":
        return 1
    with fitz.open(path) as document:
        return document.page_count
