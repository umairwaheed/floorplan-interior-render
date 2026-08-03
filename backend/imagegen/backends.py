"""Image generation backends.

`ImageBackend` is the seam between "what the room is" and "what it looks like".
Everything above it — the scene graph, the conditioning maps, the prompt — is
provider-agnostic, so swapping the generator is one class rather than a
rewrite.

That seam is not hypothetical. The two credible providers condition on geometry
in genuinely different ways:

* **Gemini 3 Pro Image** takes the geometry buffers as *reference images* and
  relies on prompt discipline to honour them. Its strength is multi-image
  reference, which is what carries object identity from the anchor view into
  later views.
* **Amazon Nova Canvas** (Bedrock) has native structural control modes that
  would consume the segmentation buffer as a first-class conditioning signal —
  a closer fit to this architecture — but is weaker at holding one specific
  object's appearance across views.

Gemini is the shipped backend because identity drift between viewpoints is the
failure this whole system exists to prevent, and that is the axis Gemini wins.
`NovaCanvasBackend` would slot in here unchanged.
"""

from __future__ import annotations

import base64
import io
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

from ..config import Settings, get_settings

logger = logging.getLogger(__name__)


class ImageGenerationError(RuntimeError):
    """Raised when a backend cannot produce an image."""


@dataclass
class ReferenceImage:
    """An image handed to the model alongside the prompt.

    `role` is not sent to the API — it exists so the ordering and captioning
    logic stays readable, and so a backend that *does* distinguish conditioning
    types (Nova Canvas) can route them correctly.
    """

    path: Path
    role: str  # depth | segmentation | wireframe | anchor | product
    caption: str = ""


@dataclass
class GenerationRequest:
    """Everything needed to produce one view."""

    prompt: str
    references: list[ReferenceImage] = field(default_factory=list)
    seed: int = 0
    width: int = 1024
    height: int = 768
    negative_prompt: str | None = None


@dataclass
class GenerationResult:
    image: Image.Image
    backend: str
    model: str
    duration_s: float
    seed: int


class ImageBackend(ABC):
    """One method, so a new provider is one class."""

    name: str = "abstract"

    @abstractmethod
    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Produce a single image."""

    @property
    def is_mock(self) -> bool:
        return False


class MockImageBackend(ImageBackend):
    """Renders the conditioning maps into a labelled placeholder.

    Not a stub that returns a blank rectangle. It composites the actual
    geometry the real backend would receive, so the whole pipeline —
    conditioning, prompt assembly, per-view seeds, the judge, the API, the UI —
    is exercised and inspectable without an API key or a cent of spend. When a
    layout bug exists, this surfaces it just as well as a photorealistic render
    would, because the geometry is the part that was wrong.
    """

    name = "mock"

    @property
    def is_mock(self) -> bool:
        return True

    def generate(self, request: GenerationRequest) -> GenerationResult:
        started = time.monotonic()

        preview = next((r for r in request.references if r.role == "preview"), None)
        segmentation = next((r for r in request.references if r.role == "segmentation"), None)
        source = preview or segmentation or (request.references[0] if request.references else None)

        if source and Path(source.path).exists():
            with Image.open(source.path) as opened:
                image = opened.convert("RGB").resize((request.width, request.height), Image.LANCZOS)
            # A soft blur makes it obvious at a glance that this is a
            # placeholder, not a real render — nobody should mistake a mock
            # for output they can evaluate photorealism on.
            image = image.filter(ImageFilter.GaussianBlur(radius=1.2))
        else:
            image = Image.new("RGB", (request.width, request.height), (222, 218, 210))

        self._annotate(image, request)
        return GenerationResult(
            image=image,
            backend=self.name,
            model="mock",
            duration_s=time.monotonic() - started,
            seed=request.seed,
        )

    @staticmethod
    def _annotate(image: Image.Image, request: GenerationRequest) -> None:
        draw = ImageDraw.Draw(image, "RGBA")
        draw.rectangle([(0, 0), (image.width, 34)], fill=(20, 20, 24, 190))
        draw.text(
            (10, 10),
            f"MOCK RENDER — seed {request.seed} — {len(request.references)} conditioning image(s)",
            fill=(255, 255, 255),
        )


class GeminiImageBackend(ImageBackend):
    """Gemini 3 Pro Image.

    The geometry buffers and the anchor view go in as reference images, in a
    fixed order: structure first, then appearance. Ordering is deliberate —
    the model weights earlier images more heavily, and geometry is the
    constraint that must not bend, while appearance is what should be carried
    forward.
    """

    name = "gemini"

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        if not self.settings.gemini_api_key:
            raise ImageGenerationError(
                "GEMINI_API_KEY is not set. Set IMAGE_BACKEND=mock to run the pipeline "
                "without an image model."
            )
        from google import genai

        self.client = genai.Client(api_key=self.settings.gemini_api_key)
        self.model = self.settings.image_model

    def generate(self, request: GenerationRequest) -> GenerationResult:
        from google.genai import types

        started = time.monotonic()
        contents: list[object] = []

        # Structure before appearance; the anchor view last so it reads as
        # "and make it look like this".
        # The untextured preview leads: it reads as "a room to re-render" where the
        # abstract maps read as "a diagram to interpret". Anchor last, so it
        # lands as "and make it look like this".
        order = {
            "preview_primary": 0,
            "segmentation": 1,
            "depth": 2,
            "wireframe": 3,
            "product": 4,
            "anchor": 5,
        }
        for reference in sorted(request.references, key=lambda r: order.get(r.role, 9)):
            if reference.role == "preview":
                continue  # a debug aid, not a conditioning signal
            path = Path(reference.path)
            if not path.exists():
                logger.warning("reference image missing: %s", path)
                continue
            if reference.caption:
                contents.append(reference.caption)
            contents.append(types.Part.from_bytes(data=path.read_bytes(), mime_type="image/png"))

        contents.append(request.prompt)

        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=contents,
                config=types.GenerateContentConfig(
                    response_modalities=["IMAGE"],
                    image_config=types.ImageConfig(
                        aspect_ratio=self._aspect_ratio(request.width, request.height)
                    ),
                ),
            )
        except Exception as exc:  # noqa: BLE001 — surfaced with context to the caller
            raise ImageGenerationError(f"Gemini image generation failed: {exc}") from exc

        image = self._extract_image(response)
        if image.size != (request.width, request.height):
            image = image.resize((request.width, request.height), Image.LANCZOS)

        return GenerationResult(
            image=image,
            backend=self.name,
            model=self.model,
            duration_s=time.monotonic() - started,
            seed=request.seed,
        )

    @staticmethod
    def _aspect_ratio(width: int, height: int) -> str:
        """Nearest supported aspect ratio label."""
        ratio = width / height
        options = {"1:1": 1.0, "4:3": 4 / 3, "3:4": 0.75, "16:9": 16 / 9, "9:16": 9 / 16}
        return min(options, key=lambda label: abs(options[label] - ratio))

    @staticmethod
    def _extract_image(response: object) -> Image.Image:
        """Pull the first inline image out of the response.

        Raises with the model's own text when no image came back — a refusal or
        a safety block returns prose, and surfacing it is far more useful than
        an IndexError.
        """
        candidates = getattr(response, "candidates", None) or []
        text_parts: list[str] = []

        for candidate in candidates:
            content = getattr(candidate, "content", None)
            for part in getattr(content, "parts", None) or []:
                inline = getattr(part, "inline_data", None)
                if inline is not None and getattr(inline, "data", None):
                    data = inline.data
                    if isinstance(data, str):
                        data = base64.b64decode(data)
                    return Image.open(io.BytesIO(data)).convert("RGB")
                if getattr(part, "text", None):
                    text_parts.append(part.text)

        detail = " ".join(text_parts).strip()
        raise ImageGenerationError(
            f"Gemini returned no image{': ' + detail[:400] if detail else '.'}"
        )


def build_backend(settings: Settings | None = None) -> ImageBackend:
    """Resolve the configured backend, falling back to mock rather than failing.

    A missing key should degrade the output, not break the pipeline — the rest
    of the system is still worth exercising and inspecting.
    """
    settings = settings or get_settings()
    choice = (settings.image_backend or "mock").lower()

    if choice == "gemini":
        try:
            return GeminiImageBackend(settings)
        except ImageGenerationError as exc:
            logger.warning("%s — falling back to the mock backend", exc)
            return MockImageBackend()

    if choice != "mock":
        logger.warning("unknown IMAGE_BACKEND %r — using mock", choice)
    return MockImageBackend()
