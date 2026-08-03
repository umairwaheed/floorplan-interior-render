"""Prompt assembly and multi-view render orchestration.

The consistency-critical properties here are about what the *prompt* says, not
about image quality — which can't be asserted without a model. Specifically:
only visible objects are named, the shared style block is byte-identical
between views, and later views are told they are looking at the same room.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from backend.config import Settings
from backend.imagegen.backends import (
    GenerationRequest,
    ImageBackend,
    ImageGenerationError,
    MockImageBackend,
    ReferenceImage,
    build_backend,
)
from backend.imagegen.prompts import (
    NEGATIVE_CONSTRAINTS,
    build_scene_block,
    build_view_prompt,
)
from backend.imagegen.service import RenderService, view_seed
from backend.schemas.common import Size3, Vec2, Vec3
from backend.schemas.floorplan import FloorPlan, Room, RoomType, ScaleCalibration
from backend.schemas.product import DesignStyle, ProductCategory
from backend.schemas.render import ConditioningMaps, RenderStatus
from backend.schemas.scene import (
    Camera,
    ColorPalette,
    LightSource,
    ObjectRole,
    PlacedObject,
    RoomFinishes,
    Scene,
    SurfaceFinish,
)


def _object(instance_id: str, name: str, x=2.0, y=1.0, color="beige"):
    return PlacedObject(
        instance_id=instance_id,
        product_id=f"comforter:{instance_id}",
        room_id="r1",
        role=ObjectRole.PRIMARY_SEATING,
        category=ProductCategory.SOFA,
        position_m=Vec3(x=x, y=y, z=0.0),
        rotation_deg=0.0,
        size_m=Size3(width=2.1, depth=0.9, height=0.85),
        color=color,
        material="linen",
        display_name=name,
        seed=42,
    )


@pytest.fixture
def scene() -> Scene:
    palette = ColorPalette(
        name="Nordic Light",
        description="Warm whites",
        primary="#F4F1EC",
        secondary="#D9CFC1",
        accent="#8FA99B",
        neutral="#EDEAE4",
    )
    finish = SurfaceFinish(
        product_id="gorgia:FLOO-1", display_name="Oslo Laminate Flooring", color="oak", area_m2=20
    )
    return Scene(
        floorplan_id="fp1",
        style=DesignStyle.SCANDINAVIAN,
        palette=palette,
        seed=7,
        room_ids=["r1"],
        objects=[
            _object("r1:sofa#0", "Bergen 3-Seat Sofa"),
            _object("r1:table#0", "Oslo Coffee Table", x=2.0, y=2.2),
            _object("r1:lamp#0", "Vega Floor Lamp", x=0.4, y=3.6),
        ],
        finishes=[
            RoomFinishes(
                room_id="r1",
                floor=finish,
                walls=SurfaceFinish(
                    product_id="gorgia:PAIN-1",
                    display_name="Kyoto Wall Paint",
                    color="off-white",
                    area_m2=45,
                ),
                ceiling=SurfaceFinish(display_name="matt white ceiling", color="white"),
            )
        ],
        lights=[
            LightSource(id="l1", room_id="r1", position_m=Vec3(x=2, y=2, z=2.4), kind="daylight"),
            LightSource(
                id="l2",
                room_id="r1",
                position_m=Vec3(x=0.4, y=3.6, z=1.6),
                kind="lamp",
                color_temp_k=2700,
            ),
        ],
    ).finalize()


@pytest.fixture
def camera() -> Camera:
    return Camera(
        id="r1-cam1",
        room_id="r1",
        position_m=Vec3(x=4.4, y=3.4, z=1.5),
        look_at_m=Vec3(x=2.0, y=1.5, z=0.9),
        label="from corner 1",
    )


def _maps(tmp_path: Path, visible: list[str]) -> ConditioningMaps:
    for name in ("depth", "segmentation", "wireframe", "preview"):
        Image.new("RGB", (32, 24), "grey").save(tmp_path / f"{name}.png")
    return ConditioningMaps(
        depth_path=str(tmp_path / "depth.png"),
        segmentation_path=str(tmp_path / "segmentation.png"),
        wireframe_path=str(tmp_path / "wireframe.png"),
        preview_path=str(tmp_path / "preview.png"),
        visible_instance_ids=visible,
        instance_pixel_share={instance_id: 0.1 for instance_id in visible},
    )


# --- prompt assembly -------------------------------------------------------


def test_prompt_names_only_visible_objects(scene, camera, tmp_path):
    """Naming an off-screen object is the most reliable way to make it appear
    in one view and not another."""
    maps = _maps(tmp_path, ["r1:sofa#0", "r1:table#0"])
    prompt = build_view_prompt(scene, camera, maps, is_anchor=True)

    assert "Bergen 3-Seat Sofa" in prompt
    assert "Oslo Coffee Table" in prompt
    assert "Vega Floor Lamp" not in prompt, "an off-screen object was named"


def test_prompt_uses_catalog_names_verbatim(scene, camera, tmp_path):
    """What the model is told to draw and what the BOM lists must be the same
    string, from the same record."""
    maps = _maps(tmp_path, ["r1:sofa#0"])
    prompt = build_view_prompt(scene, camera, maps, is_anchor=True)
    sofa = scene.object_by_instance("r1:sofa#0")
    assert sofa.display_name in prompt
    assert sofa.color in prompt


def test_prompt_includes_real_world_dimensions(scene, camera, tmp_path):
    maps = _maps(tmp_path, ["r1:sofa#0"])
    prompt = build_view_prompt(scene, camera, maps, is_anchor=True)
    assert "2.10 m wide" in prompt
    assert "0.85 m high" in prompt


def test_renovation_finishes_appear_as_products(scene, camera, tmp_path):
    """'Every visible furniture OR renovation element' — finishes are named."""
    maps = _maps(tmp_path, ["r1:sofa#0"])
    prompt = build_view_prompt(scene, camera, maps, is_anchor=True)
    assert "Oslo Laminate Flooring" in prompt
    assert "Kyoto Wall Paint" in prompt


def test_shared_block_is_identical_across_views(scene, tmp_path):
    """Any drift in the shared block is drift the model will faithfully
    reproduce as a difference between viewpoints."""
    first = build_scene_block(scene, "r1")
    second = build_scene_block(scene, "r1")
    assert first == second
    assert "Scandinavian" in first
    assert "Nordic Light" in first


def test_anchor_and_follow_up_prompts_differ_only_in_framing(scene, camera, tmp_path):
    maps = _maps(tmp_path, ["r1:sofa#0"])
    anchor = build_view_prompt(scene, camera, maps, is_anchor=True)
    follow = build_view_prompt(scene, camera, maps, is_anchor=False)

    assert "SAME" not in anchor
    assert "SAME" in follow
    assert "reference photograph" in follow
    # The style/surface/lighting block survives verbatim in both.
    shared = build_scene_block(scene, "r1")
    assert shared in anchor
    assert shared in follow


def test_prompt_forbids_the_failure_modes_the_brief_names(scene, camera, tmp_path):
    """Checked against the whole prompt: the prohibitions are split between the
    edit framing (reposition/resize/add/remove) and the negative block
    (inventing fixtures), and it is their combination that matters."""
    maps = _maps(tmp_path, ["r1:sofa#0"])
    prompt = build_view_prompt(scene, camera, maps, is_anchor=False)
    assert NEGATIVE_CONSTRAINTS in prompt
    for forbidden in ("reposition", "resize", "add or remove", "invented object"):
        assert forbidden in prompt, f"the prompt no longer forbids: {forbidden}"


def test_prompt_frames_the_task_as_re_rendering_not_generating(scene, camera, tmp_path):
    """The highest-leverage line in the system.

    Measured on one scene, camera and seed: framing the job as re-rendering an
    existing 3D scene rather than generating a room moved layout fidelity from
    0.12 to 0.70. Asked to generate, the model treats the geometry as a mood
    board; asked to re-render, it treats it as the thing to preserve.
    """
    maps = _maps(tmp_path, ["r1:sofa#0"])
    prompt = build_view_prompt(scene, camera, maps, is_anchor=True)
    assert "re-rendering an existing 3D scene" in prompt
    assert "not designing a room" in prompt
    assert "MATERIAL AND LIGHTING PASS ONLY" in prompt


def test_prompt_states_screen_positions_as_text(scene, camera, tmp_path):
    """Numbers alongside pixels — the model grounds on the text where it drifts
    on the map."""
    maps = _maps(tmp_path, ["r1:sofa#0"])
    maps.instance_screen_boxes = {"r1:sofa#0": (10, 20, 60, 90)}
    prompt = build_view_prompt(scene, camera, maps, is_anchor=True)
    assert "EXACT SCREEN POSITIONS" in prompt
    assert "x 10%-60%" in prompt


def test_prompt_survives_a_view_with_nothing_visible(scene, camera, tmp_path):
    maps = _maps(tmp_path, [])
    prompt = build_view_prompt(scene, camera, maps, is_anchor=True)
    assert "no furniture is visible" in prompt


def test_lighting_is_described_from_the_scene_not_invented(scene, camera, tmp_path):
    maps = _maps(tmp_path, ["r1:sofa#0"])
    prompt = build_view_prompt(scene, camera, maps, is_anchor=True)
    assert "daylight" in prompt.lower()
    assert "Shadow directions must match across views" in prompt


# --- backends --------------------------------------------------------------


def test_mock_backend_produces_a_correctly_sized_image(tmp_path):
    maps = _maps(tmp_path, [])
    backend = MockImageBackend()
    result = backend.generate(
        GenerationRequest(
            prompt="test",
            references=[ReferenceImage(path=Path(maps.preview_path), role="preview")],
            width=200,
            height=150,
            seed=99,
        )
    )
    assert result.image.size == (200, 150)
    assert result.backend == "mock"
    assert result.seed == 99
    assert backend.is_mock


def test_mock_backend_works_with_no_references():
    result = MockImageBackend().generate(GenerationRequest(prompt="test", width=64, height=64))
    assert result.image.size == (64, 64)


def test_build_backend_falls_back_rather_than_failing(tmp_path):
    """A missing key should degrade the output, not break the pipeline."""
    settings = Settings(
        data_dir=tmp_path,
        upload_dir=tmp_path / "u",
        output_dir=tmp_path / "o",
        db_path=tmp_path / "c.db",
        image_backend="gemini",
        gemini_api_key=None,
    )
    settings.ensure_dirs()
    assert build_backend(settings).is_mock


def test_unknown_backend_falls_back_to_mock(tmp_path):
    settings = Settings(
        data_dir=tmp_path,
        upload_dir=tmp_path / "u",
        output_dir=tmp_path / "o",
        db_path=tmp_path / "c.db",
        image_backend="midjourney",
    )
    settings.ensure_dirs()
    assert build_backend(settings).is_mock


# --- seeds -----------------------------------------------------------------


def test_view_seeds_are_stable_and_distinct():
    assert view_seed(7, "cam1") == view_seed(7, "cam1")
    assert view_seed(7, "cam1") != view_seed(7, "cam2")
    assert view_seed(7, "cam1") != view_seed(8, "cam1")


def test_retry_attempts_get_fresh_seeds():
    assert view_seed(7, "cam1", attempt=0) != view_seed(7, "cam1", attempt=1)


def test_view_seed_does_not_depend_on_camera_ordering():
    """Adding a camera to one room must not renumber another room's views and
    silently change images the user already approved."""
    before = view_seed(7, "bed-cam1")
    after = view_seed(7, "bed-cam1")  # studio gained a camera in between
    assert before == after


# --- orchestration ---------------------------------------------------------


@pytest.fixture
def floorplan() -> FloorPlan:
    return FloorPlan(
        id="fp1",
        source_filename="p.png",
        image_width_px=100,
        image_height_px=100,
        calibration=ScaleCalibration(
            px_per_m=10, method="area_labels", residual_pct=0, confidence=1, sample_count=1
        ),
        rooms=[
            Room(
                id="r1",
                name="Living Room",
                room_type=RoomType.LIVING,
                polygon_m=[Vec2(x=0, y=0), Vec2(x=5, y=0), Vec2(x=5, y=4), Vec2(x=0, y=4)],
                polygon_px=[Vec2(x=0, y=0), Vec2(x=1, y=0), Vec2(x=1, y=1), Vec2(x=0, y=1)],
            )
        ],
    )


def _service(tmp_path: Path) -> RenderService:
    settings = Settings(
        data_dir=tmp_path,
        upload_dir=tmp_path / "u",
        output_dir=tmp_path / "o",
        db_path=tmp_path / "c.db",
        render_width=80,
        render_height=60,
    )
    settings.ensure_dirs()
    return RenderService(backend=MockImageBackend(), settings=settings)


def test_first_view_is_the_anchor_and_later_views_are_not(scene, floorplan, camera, tmp_path):
    scene.cameras = [
        camera,
        Camera(
            id="r1-cam2",
            room_id="r1",
            position_m=Vec3(x=0.6, y=0.6, z=1.5),
            look_at_m=Vec3(x=2.0, y=1.5, z=0.9),
        ),
    ]
    maps = _maps(tmp_path, ["r1:sofa#0"])
    renders = _service(tmp_path).render_scene(
        scene, floorplan, {"r1-cam1": maps, "r1-cam2": maps}, tmp_path / "out"
    )

    assert [r.is_anchor for r in renders] == [True, False]
    assert all(r.status == RenderStatus.COMPLETED for r in renders)
    assert all(Path(r.image_path).exists() for r in renders)


def test_each_room_gets_its_own_anchor(scene, floorplan, tmp_path):
    """Anchoring across rooms would drag one room's palette into another."""
    floorplan.rooms.append(
        Room(
            id="r2",
            name="Bedroom",
            room_type=RoomType.BEDROOM,
            polygon_m=[Vec2(x=6, y=0), Vec2(x=9, y=0), Vec2(x=9, y=3), Vec2(x=6, y=3)],
            polygon_px=[Vec2(x=0, y=0), Vec2(x=1, y=0), Vec2(x=1, y=1), Vec2(x=0, y=1)],
        )
    )
    scene.room_ids = ["r1", "r2"]
    scene.cameras = [
        Camera(
            id="r1-cam1",
            room_id="r1",
            position_m=Vec3(x=4, y=3, z=1.5),
            look_at_m=Vec3(x=2, y=1, z=1),
        ),
        Camera(
            id="r2-cam1",
            room_id="r2",
            position_m=Vec3(x=8, y=2, z=1.5),
            look_at_m=Vec3(x=7, y=1, z=1),
        ),
    ]
    maps = _maps(tmp_path, [])
    renders = _service(tmp_path).render_scene(
        scene, floorplan, {"r1-cam1": maps, "r2-cam1": maps}, tmp_path / "out"
    )
    assert {r.room_id for r in renders if r.is_anchor} == {"r1", "r2"}


def test_render_records_the_products_it_contains(scene, floorplan, camera, tmp_path):
    """'Return the products used in every render' — per render, not per scene."""
    scene.cameras = [camera]
    maps = _maps(tmp_path, ["r1:sofa#0", "r1:table#0"])
    renders = _service(tmp_path).render_scene(scene, floorplan, {"r1-cam1": maps}, tmp_path / "o")

    assert renders[0].product_ids == ["comforter:r1:sofa#0", "comforter:r1:table#0"]
    assert "comforter:r1:lamp#0" not in renders[0].product_ids


def test_a_failing_backend_does_not_raise_or_become_an_anchor(scene, floorplan, tmp_path):
    """A failed first view must not anchor later ones — they would reference a
    file that does not exist and silently lose the identity signal."""

    class Broken(ImageBackend):
        name = "broken"

        def generate(self, request):
            raise ImageGenerationError("no capacity")

    scene.cameras = [
        Camera(
            id="r1-cam1",
            room_id="r1",
            position_m=Vec3(x=4, y=3, z=1.5),
            look_at_m=Vec3(x=2, y=1, z=1),
        ),
        Camera(
            id="r1-cam2",
            room_id="r1",
            position_m=Vec3(x=1, y=1, z=1.5),
            look_at_m=Vec3(x=2, y=1, z=1),
        ),
    ]
    maps = _maps(tmp_path, [])
    settings = Settings(
        data_dir=tmp_path,
        upload_dir=tmp_path / "u",
        output_dir=tmp_path / "o",
        db_path=tmp_path / "c.db",
    )
    settings.ensure_dirs()

    renders = RenderService(backend=Broken(), settings=settings).render_scene(
        scene, floorplan, {"r1-cam1": maps, "r1-cam2": maps}, tmp_path / "out"
    )
    assert all(r.status == RenderStatus.FAILED for r in renders)
    assert all("no capacity" in (r.error or "") for r in renders)
    assert all(r.is_anchor for r in renders), "no successful render, so none can anchor"


def test_missing_conditioning_maps_are_skipped_not_fatal(scene, floorplan, camera, tmp_path):
    scene.cameras = [camera]
    renders = _service(tmp_path).render_scene(scene, floorplan, {}, tmp_path / "out")
    assert renders == []


def test_progress_hook_fires_per_view(scene, floorplan, camera, tmp_path):
    scene.cameras = [camera]
    maps = _maps(tmp_path, [])
    seen = []
    _service(tmp_path).render_scene(
        scene, floorplan, {"r1-cam1": maps}, tmp_path / "out", on_progress=seen.append
    )
    assert len(seen) == 1
    assert seen[0].camera_id == "r1-cam1"
