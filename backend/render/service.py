"""Turns a scene graph into per-camera conditioning maps on disk.

This is the handoff to image generation. Everything upstream decided *what the
room is*; everything downstream only decides what it looks like. The buffers
written here are the contract between those halves — and because every camera
projects the same frozen scene, two views cannot disagree about the geometry
they were conditioned on.

`visible_instance_ids` matters as much as the images. The prompt for a view
names only the objects actually in that frame, so the model is never told to
draw something off-screen, and the consistency judge knows exactly which
objects it is entitled to look for.
"""

from __future__ import annotations

import logging
from pathlib import Path

from ..config import Settings, get_settings
from ..schemas.floorplan import FloorPlan
from ..schemas.render import ConditioningMaps
from ..schemas.scene import Camera, Scene
from .cameras import place_cameras
from .raster import depth_to_image, rasterize, to_pil, wireframe_image

logger = logging.getLogger(__name__)


class SceneRenderer:
    """Rasterizes a scene's cameras into conditioning maps."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def attach_cameras(self, scene: Scene, floorplan: FloorPlan, views_per_room: int) -> Scene:
        """Populate the scene's camera list and re-freeze it.

        Cameras belong to the scene, not to a render request, so this mutates
        the graph and re-hashes it. A scene with cameras is a different scene
        from one without — pretending otherwise would let two runs with
        different view counts collide on the same `scene_id`.
        """
        cameras: list[Camera] = []
        for room_id in scene.room_ids:
            room = floorplan.room(room_id)
            if room is None:
                logger.warning("scene references unknown room %s", room_id)
                continue
            cameras.extend(
                place_cameras(
                    room=room,
                    objects=scene.objects_in_room(room_id),
                    count=views_per_room,
                    settings=self.settings,
                )
            )

        scene.cameras = cameras
        return scene.finalize()

    def render_camera(
        self,
        scene: Scene,
        floorplan: FloorPlan,
        camera: Camera,
        output_dir: Path,
    ) -> ConditioningMaps:
        """Rasterize one camera and write its four buffers."""
        room = floorplan.room(camera.room_id)
        if room is None:
            raise ValueError(f"Camera {camera.id} references unknown room {camera.room_id}")

        objects = scene.objects_in_room(camera.room_id)
        width, height = self.settings.conditioning_width, self.settings.conditioning_height

        buffers = rasterize(
            room=room,
            objects=objects,
            camera=camera,
            width=width,
            height=height,
            finishes=scene.finishes_for_room(camera.room_id),
        )

        output_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{scene.output_key}_{camera.id}"

        depth_path = output_dir / f"{stem}_depth.png"
        segmentation_path = output_dir / f"{stem}_segmentation.png"
        wireframe_path = output_dir / f"{stem}_wireframe.png"
        preview_path = output_dir / f"{stem}_preview.png"

        depth_to_image(buffers.depth).save(depth_path)
        to_pil(buffers.segmentation).save(segmentation_path)
        to_pil(buffers.preview).save(preview_path)
        wireframe_image(room, objects, camera, width, height).save(wireframe_path)

        visible = buffers.visible_instances()
        total_pixels = float(width * height)

        return ConditioningMaps(
            depth_path=str(depth_path),
            segmentation_path=str(segmentation_path),
            wireframe_path=str(wireframe_path),
            preview_path=str(preview_path),
            visible_instance_ids=sorted(visible, key=lambda i: -visible[i]),
            instance_pixel_share={
                instance_id: round(count / total_pixels, 5)
                for instance_id, count in visible.items()
            },
            instance_screen_boxes=buffers.screen_boxes(),
        )

    def render_scene(
        self, scene: Scene, floorplan: FloorPlan, output_dir: Path | None = None
    ) -> dict[str, ConditioningMaps]:
        """Rasterize every camera in a scene. Returns camera_id → maps."""
        output_dir = output_dir or (self.settings.output_dir / scene.output_key)
        maps: dict[str, ConditioningMaps] = {}

        for camera in scene.cameras:
            maps[camera.id] = self.render_camera(scene, floorplan, camera, output_dir)
            logger.info(
                "rasterized %s (%s): %d instances visible",
                camera.id,
                camera.label,
                len(maps[camera.id].visible_instance_ids),
            )

        return maps
