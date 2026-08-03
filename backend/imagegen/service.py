"""Multi-view render orchestration.

Two mechanisms carry consistency between viewpoints, and they cover different
failure modes:

1. **Shared geometry.** Every view is conditioned on depth and segmentation
   buffers projected from the same frozen scene, so layout cannot drift. This
   is structural — it holds whether or not the model cooperates.
2. **The anchor view.** The first view of a room is generated, then handed to
   every later view of that room as an appearance reference. Geometry
   conditioning fixes *where* the sofa is; only the anchor fixes that it is the
   same sofa, in the same fabric, under the same light.

Ordering matters: views of a room are generated strictly in sequence, because
view 2 cannot start until view 1 exists to reference. Different rooms are
independent.

Per-view seeds are derived from the scene seed and the camera ID, so a
regeneration reproduces the same view and an edit elsewhere in the scene cannot
shift it.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from pathlib import Path

from ..config import Settings, get_settings
from ..schemas.floorplan import FloorPlan
from ..schemas.render import ConditioningMaps, Render, RenderStatus
from ..schemas.scene import Camera, Scene
from .backends import (
    GenerationRequest,
    ImageBackend,
    ImageGenerationError,
    ReferenceImage,
    build_backend,
)
from .prompts import build_view_prompt

logger = logging.getLogger(__name__)

ProgressHook = Callable[[Render], None]


def view_seed(scene_seed: int, camera_id: str, attempt: int = 0) -> int:
    """A stable per-view seed.

    Derived from the scene seed and the camera's identity rather than from a
    counter, so adding a camera to one room cannot renumber another room's
    views and silently change images the user already approved.
    """
    digest = 0
    for char in camera_id:
        digest = (digest * 131 + ord(char)) % (2**31)
    return (scene_seed * 7919 + digest + attempt * 104729) % (2**31)


class RenderService:
    """Generates photorealistic views for a scene."""

    def __init__(
        self,
        backend: ImageBackend | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.backend = backend or build_backend(self.settings)

    def _references(
        self,
        maps: ConditioningMaps,
        anchor_path: Path | None,
    ) -> list[ReferenceImage]:
        """Conditioning images for one view, captioned so their role is explicit.

        The captions are part of the prompt: an uncaptioned image is just
        pixels, and the model has to guess whether a depth map is a
        constraint or a picture of a room it should reproduce.
        """
        # The untextured preview leads. Measured on one scene: leading with it
        # under an edit framing put layout fidelity at 0.70 against 0.12 for the
        # depth/segmentation-led generation framing. It reads as "a room to
        # re-render" where the abstract maps read as "a diagram to interpret".
        references = [
            ReferenceImage(
                path=Path(maps.preview_path or maps.depth_path),
                role="preview_primary",
                caption=(
                    "UNTEXTURED 3D RENDER OF THIS EXACT SHOT — reproduce this composition "
                    "precisely, replacing each placeholder volume with real furniture."
                ),
            ),
            ReferenceImage(
                path=Path(maps.segmentation_path),
                role="segmentation",
                caption=(
                    "SEGMENTATION MAP — each distinct colour is one object, same camera. "
                    "Render exactly one object per coloured region, in that region only."
                ),
            ),
            ReferenceImage(
                path=Path(maps.depth_path),
                role="depth",
                caption="DEPTH MAP — brighter is closer to the camera.",
            ),
        ]

        if anchor_path is not None and anchor_path.exists():
            references.append(
                ReferenceImage(
                    path=anchor_path,
                    role="anchor",
                    caption=(
                        "REFERENCE PHOTOGRAPH — this is the SAME room, already rendered "
                        "from a different camera position. Every object here keeps its "
                        "exact colour, material and finish. Match its lighting, its "
                        "palette, and the identity of every object it shares with this view."
                    ),
                )
            )

        return references

    def render_view(
        self,
        scene: Scene,
        camera: Camera,
        maps: ConditioningMaps,
        output_dir: Path,
        anchor_path: Path | None,
        room_name: str,
        attempt: int = 0,
    ) -> Render:
        """Generate one view. Never raises — failures land on the `Render`."""
        is_anchor = anchor_path is None
        seed = view_seed(scene.seed, camera.id, attempt)

        render = Render(
            id=f"{scene.scene_id}_{camera.id}",
            scene_id=scene.scene_id,
            camera_id=camera.id,
            room_id=camera.room_id,
            status=RenderStatus.GENERATING,
            conditioning=maps,
            is_anchor=is_anchor,
            seed=seed,
            attempts=attempt + 1,
            product_ids=sorted(
                {
                    obj.product_id
                    for obj in scene.objects
                    if obj.instance_id in set(maps.visible_instance_ids)
                }
            ),
        )

        prompt = build_view_prompt(
            scene=scene, camera=camera, maps=maps, is_anchor=is_anchor, room_name=room_name
        )
        render.prompt = prompt

        started = time.monotonic()
        try:
            result = self.backend.generate(
                GenerationRequest(
                    prompt=prompt,
                    references=self._references(maps, anchor_path),
                    seed=seed,
                    width=self.settings.render_width,
                    height=self.settings.render_height,
                )
            )
        except ImageGenerationError as exc:
            render.status = RenderStatus.FAILED
            render.error = str(exc)
            render.duration_s = round(time.monotonic() - started, 2)
            logger.error("render %s failed: %s", render.id, exc)
            return render

        output_dir.mkdir(parents=True, exist_ok=True)
        image_path = output_dir / f"{render.id}.png"
        result.image.save(image_path)

        render.image_path = str(image_path)
        render.status = RenderStatus.COMPLETED
        render.duration_s = round(result.duration_s, 2)
        return render

    def render_scene(
        self,
        scene: Scene,
        floorplan: FloorPlan,
        conditioning: dict[str, ConditioningMaps],
        output_dir: Path | None = None,
        on_progress: ProgressHook | None = None,
    ) -> list[Render]:
        """Render every camera, anchoring each room to its own first view.

        Rooms are independent; views within a room are sequential because a
        later view needs the anchor image to exist on disk.
        """
        output_dir = output_dir or (self.settings.output_dir / scene.scene_id)
        renders: list[Render] = []

        for room_id in scene.room_ids:
            room = floorplan.room(room_id)
            room_name = room.name if room else "room"
            anchor_path: Path | None = None

            for camera in scene.cameras_for_room(room_id):
                maps = conditioning.get(camera.id)
                if maps is None:
                    logger.warning("no conditioning maps for camera %s — skipped", camera.id)
                    continue

                render = self.render_view(
                    scene=scene,
                    camera=camera,
                    maps=maps,
                    output_dir=output_dir,
                    anchor_path=anchor_path,
                    room_name=room_name,
                )
                renders.append(render)

                if on_progress is not None:
                    on_progress(render)

                # The first successful view of a room becomes its anchor. A
                # failed first view must not become one — anchoring later views
                # to a missing file would silently drop the identity signal.
                if anchor_path is None and render.image_path:
                    anchor_path = Path(render.image_path)
                    logger.info("room %s anchored to %s", room_id, camera.id)

        return renders

    def anchor_for_room(self, renders: list[Render], room_id: str) -> Path | None:
        """The anchor image for a room, for re-rendering a single view later."""
        for render in renders:
            if render.room_id == room_id and render.is_anchor and render.image_path:
                return Path(render.image_path)
        return None
