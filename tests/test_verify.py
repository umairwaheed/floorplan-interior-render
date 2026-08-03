"""Consistency judge and retry-loop tests.

The retry loop is where honest reporting is easiest to quietly lose: it is
tempting to keep re-rolling until something scores well, or to report the
attempt rather than the result. These tests pin the opposite behaviour — the
score that ships is the score of the image that ships, and running unverified
is never reported as passing.

A scripted judge drives the loop deterministically; the real judge needs a
model and is exercised only through its scoring semantics.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from backend.config import Settings
from backend.imagegen.backends import ImageBackend, ImageGenerationError, MockImageBackend
from backend.imagegen.service import RenderService
from backend.schemas.common import Size3, Vec2, Vec3
from backend.schemas.floorplan import FloorPlan, Room, RoomType, ScaleCalibration
from backend.schemas.product import DesignStyle, ProductCategory
from backend.schemas.render import ConditioningMaps, ConsistencyScores, RenderStatus
from backend.schemas.scene import (
    Camera,
    ColorPalette,
    ObjectRole,
    PlacedObject,
    Scene,
)
from backend.verify.judge import ConsistencyJudge, NullJudge, build_judge
from backend.verify.service import VerificationService, mean_consistency, summarize


def _scores(value: float, *, cross=None, missing=None, verified=True) -> ConsistencyScores:
    return ConsistencyScores(
        layout_fidelity=value,
        object_identity=value,
        cross_view_consistency=cross,
        style_adherence=value,
        photorealism=value,
        verified=verified,
        missing_instance_ids=missing or [],
    )


# --- scoring semantics -----------------------------------------------------


def test_anchor_is_not_given_a_free_quarter():
    """An anchor has no earlier view to match, so cross-view must not count."""
    anchor = _scores(0.8)
    follower = _scores(0.8, cross=0.8)
    assert anchor.overall == pytest.approx(follower.overall)


def test_a_weak_cross_view_score_drags_the_total_down():
    strong = _scores(0.9, cross=0.9)
    weak = _scores(0.9, cross=0.2)
    assert weak.overall < strong.overall - 0.1


def test_a_missing_object_fails_however_good_the_rest_looks():
    """The brief's hard requirement: objects must not disappear between views.
    No style score can offset that."""
    scores = _scores(0.98, cross=0.98, missing=["r1:sofa#0"])
    assert scores.overall > 0.9
    assert not scores.passes(0.75)


def test_unverified_scores_never_pass():
    assert not _scores(1.0, cross=1.0, verified=False).passes(0.0)


def test_null_judge_reports_itself_as_unverified():
    judge = NullJudge()
    scores = judge.judge(render=None, scene=None, reference_path=None)  # type: ignore[arg-type]
    assert scores.verified is False
    assert not scores.passes(0.0)
    assert "Not verified" in scores.issues[0]


def test_build_judge_degrades_rather_than_failing(tmp_path):
    settings = Settings(
        data_dir=tmp_path,
        upload_dir=tmp_path / "u",
        output_dir=tmp_path / "o",
        db_path=tmp_path / "c.db",
        enable_judge=True,
        anthropic_api_key=None,
    )
    settings.ensure_dirs()
    assert isinstance(build_judge(settings), NullJudge)


def test_disabling_the_judge_yields_the_null_judge(tmp_path):
    settings = Settings(
        data_dir=tmp_path,
        upload_dir=tmp_path / "u",
        output_dir=tmp_path / "o",
        db_path=tmp_path / "c.db",
        enable_judge=False,
    )
    settings.ensure_dirs()
    assert isinstance(build_judge(settings), NullJudge)


# --- fixtures for the loop -------------------------------------------------


@pytest.fixture
def scene() -> Scene:
    return Scene(
        floorplan_id="fp1",
        style=DesignStyle.SCANDINAVIAN,
        palette=ColorPalette(
            name="Nordic Light",
            description="warm whites",
            primary="#F4F1EC",
            secondary="#D9CFC1",
            accent="#8FA99B",
            neutral="#EDEAE4",
        ),
        seed=7,
        room_ids=["r1"],
        objects=[
            PlacedObject(
                instance_id="r1:sofa#0",
                product_id="comforter:sofa",
                room_id="r1",
                role=ObjectRole.PRIMARY_SEATING,
                category=ProductCategory.SOFA,
                position_m=Vec3(x=2, y=1, z=0),
                size_m=Size3(width=2.1, depth=0.9, height=0.85),
                color="beige",
                display_name="Bergen Sofa",
                seed=1,
            )
        ],
        cameras=[
            Camera(
                id="r1-cam1",
                room_id="r1",
                position_m=Vec3(x=4, y=3, z=1.5),
                look_at_m=Vec3(x=2, y=1, z=1),
            ),
            Camera(
                id="r1-cam2",
                room_id="r1",
                position_m=Vec3(x=1, y=3, z=1.5),
                look_at_m=Vec3(x=2, y=1, z=1),
            ),
        ],
    ).finalize()


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


@pytest.fixture
def maps(tmp_path) -> ConditioningMaps:
    for name in ("depth", "segmentation", "wireframe", "preview"):
        Image.new("RGB", (24, 18), "grey").save(tmp_path / f"{name}.png")
    return ConditioningMaps(
        depth_path=str(tmp_path / "depth.png"),
        segmentation_path=str(tmp_path / "segmentation.png"),
        wireframe_path=str(tmp_path / "wireframe.png"),
        preview_path=str(tmp_path / "preview.png"),
        visible_instance_ids=["r1:sofa#0"],
        instance_pixel_share={"r1:sofa#0": 0.2},
    )


class ScriptedJudge(ConsistencyJudge):
    """Returns a prepared verdict per call, so the loop is deterministic."""

    name = "scripted"

    def __init__(self, verdicts: list[ConsistencyScores]) -> None:
        self.verdicts = verdicts
        self.calls = 0

    def judge(self, render, scene, reference_path):
        verdict = self.verdicts[min(self.calls, len(self.verdicts) - 1)]
        self.calls += 1
        return verdict.model_copy(deep=True)


def _settings(tmp_path, attempts=3, threshold=0.75) -> Settings:
    settings = Settings(
        data_dir=tmp_path,
        upload_dir=tmp_path / "u",
        output_dir=tmp_path / "o",
        db_path=tmp_path / "c.db",
        render_width=48,
        render_height=36,
        max_render_attempts=attempts,
        consistency_threshold=threshold,
    )
    settings.ensure_dirs()
    return settings


def _pipeline(tmp_path, verdicts, attempts=3, backend=None):
    settings = _settings(tmp_path, attempts=attempts)
    renders = RenderService(backend=backend or MockImageBackend(), settings=settings)
    return (
        VerificationService(renders, judge=ScriptedJudge(verdicts), settings=settings),
        renders,
        settings,
    )


def _render_all(renders, scene, floorplan, maps, tmp_path):
    return renders.render_scene(
        scene,
        floorplan,
        {"r1-cam1": maps, "r1-cam2": maps},
        tmp_path / "out",
    )


# --- the retry loop --------------------------------------------------------


def test_a_passing_render_is_not_retried(scene, floorplan, maps, tmp_path):
    verifier, renders, _ = _pipeline(tmp_path, [_scores(0.9, cross=0.9)])
    initial = _render_all(renders, scene, floorplan, maps, tmp_path)

    final = verifier.verify_and_retry(
        scene, floorplan, initial, {"r1-cam1": maps, "r1-cam2": maps}, tmp_path / "out"
    )
    assert all(r.attempts == 1 for r in final)
    assert all(r.status == RenderStatus.COMPLETED for r in final)


def test_a_failing_render_is_retried_and_the_better_one_kept(scene, floorplan, maps, tmp_path):
    verifier, renders, _ = _pipeline(tmp_path, [_scores(0.3, cross=0.3), _scores(0.92, cross=0.92)])
    initial = _render_all(renders, scene, floorplan, maps, tmp_path)

    final = verifier.verify_and_retry(
        scene, floorplan, initial, {"r1-cam1": maps, "r1-cam2": maps}, tmp_path / "out"
    )
    anchor = next(r for r in final if r.is_anchor)
    assert anchor.attempts > 1, "a failing render should have been re-rolled"
    assert anchor.scores.overall > 0.75


def test_retries_are_bounded_and_the_real_score_is_reported(scene, floorplan, maps, tmp_path):
    """Out of attempts, keep the image and its true score. A 0.4 reported
    honestly beats a gap or a fabricated pass."""
    verifier, renders, settings = _pipeline(tmp_path, [_scores(0.4, cross=0.4)], attempts=2)
    initial = _render_all(renders, scene, floorplan, maps, tmp_path)

    final = verifier.verify_and_retry(
        scene, floorplan, initial, {"r1-cam1": maps, "r1-cam2": maps}, tmp_path / "out"
    )
    anchor = next(r for r in final if r.is_anchor)
    assert anchor.attempts <= settings.max_render_attempts
    assert anchor.status == RenderStatus.COMPLETED
    assert anchor.scores.overall == pytest.approx(0.4, abs=0.01)
    assert not anchor.scores.passes(settings.consistency_threshold)


def test_retrying_only_touches_the_failing_view(scene, floorplan, maps, tmp_path):
    """Regenerating the whole room would discard acceptable images."""

    class PerCamera(ConsistencyJudge):
        name = "per-camera"

        def judge(self, render, scene, reference_path):
            good = render.camera_id == "r1-cam1"
            return _scores(0.95 if good else 0.2, cross=None if render.is_anchor else 0.2)

    settings = _settings(tmp_path, attempts=2)
    renders = RenderService(backend=MockImageBackend(), settings=settings)
    verifier = VerificationService(renders, judge=PerCamera(), settings=settings)

    initial = _render_all(renders, scene, floorplan, maps, tmp_path)
    final = verifier.verify_and_retry(
        scene, floorplan, initial, {"r1-cam1": maps, "r1-cam2": maps}, tmp_path / "out"
    )

    by_camera = {r.camera_id: r for r in final}
    assert by_camera["r1-cam1"].attempts == 1, "a passing view was needlessly re-rolled"
    assert by_camera["r1-cam2"].attempts > 1


def test_rerolling_an_anchor_regenerates_the_views_that_referenced_it(
    scene, floorplan, maps, tmp_path
):
    """Otherwise the reported cross-view score is measured against an image
    that is no longer in the output."""
    verdicts = [_scores(0.2), _scores(0.95), _scores(0.95, cross=0.95)]
    verifier, renders, _ = _pipeline(tmp_path, verdicts)
    initial = _render_all(renders, scene, floorplan, maps, tmp_path)
    original_follower_seed = next(r for r in initial if not r.is_anchor).seed

    final = verifier.verify_and_retry(
        scene, floorplan, initial, {"r1-cam1": maps, "r1-cam2": maps}, tmp_path / "out"
    )
    anchor = next(r for r in final if r.is_anchor)
    follower = next(r for r in final if not r.is_anchor)

    assert anchor.attempts > 1
    assert follower.seed != original_follower_seed, "the follower was not regenerated"


def test_unverified_renders_are_not_retried(scene, floorplan, maps, tmp_path):
    """With nothing measured, a re-roll spends budget on an unmeasurable
    difference."""
    verifier, renders, _ = _pipeline(tmp_path, [_scores(0.0, verified=False)])
    initial = _render_all(renders, scene, floorplan, maps, tmp_path)

    final = verifier.verify_and_retry(
        scene, floorplan, initial, {"r1-cam1": maps, "r1-cam2": maps}, tmp_path / "out"
    )
    assert all(r.attempts == 1 for r in final)
    assert all(r.scores.verified is False for r in final)


def test_a_broken_generator_mid_retry_keeps_the_earlier_image(scene, floorplan, maps, tmp_path):
    class BreaksOnRetry(ImageBackend):
        name = "flaky"

        def __init__(self) -> None:
            self.calls = 0

        def generate(self, request):
            self.calls += 1
            if self.calls > 2:  # the two initial views succeed; retries fail
                raise ImageGenerationError("out of capacity")
            return MockImageBackend().generate(request)

    verifier, renders, _ = _pipeline(tmp_path, [_scores(0.2, cross=0.2)], backend=BreaksOnRetry())
    initial = _render_all(renders, scene, floorplan, maps, tmp_path)

    final = verifier.verify_and_retry(
        scene, floorplan, initial, {"r1-cam1": maps, "r1-cam2": maps}, tmp_path / "out"
    )
    assert all(r.image_path and Path(r.image_path).exists() for r in final)
    assert any("capacity" in (r.error or "") for r in final)


def test_output_order_is_preserved(scene, floorplan, maps, tmp_path):
    verifier, renders, _ = _pipeline(tmp_path, [_scores(0.9, cross=0.9)])
    initial = _render_all(renders, scene, floorplan, maps, tmp_path)
    final = verifier.verify_and_retry(
        scene, floorplan, initial, {"r1-cam1": maps, "r1-cam2": maps}, tmp_path / "out"
    )
    assert [r.camera_id for r in final] == [r.camera_id for r in initial]


# --- reporting -------------------------------------------------------------


def test_summary_reports_the_worst_view_not_just_the_mean(scene, floorplan, maps, tmp_path):
    """A mean of 0.85 hiding one view at 0.4 is a broken set."""

    class Mixed(ConsistencyJudge):
        name = "mixed"

        def judge(self, render, scene, reference_path):
            return _scores(
                0.95 if render.is_anchor else 0.4, cross=None if render.is_anchor else 0.4
            )

    settings = _settings(tmp_path, attempts=1)
    renders = RenderService(backend=MockImageBackend(), settings=settings)
    verifier = VerificationService(renders, judge=Mixed(), settings=settings)

    initial = _render_all(renders, scene, floorplan, maps, tmp_path)
    final = verifier.verify_and_retry(
        scene, floorplan, initial, {"r1-cam1": maps, "r1-cam2": maps}, tmp_path / "out"
    )

    report = summarize(final)
    assert report["verified"] is True
    assert report["worst_consistency"] < report["mean_consistency"]
    assert report["worst_consistency"] == pytest.approx(0.4, abs=0.01)


def test_summary_says_so_when_nothing_was_verified(scene, floorplan, maps, tmp_path):
    verifier, renders, _ = _pipeline(tmp_path, [_scores(0.0, verified=False)])
    initial = _render_all(renders, scene, floorplan, maps, tmp_path)
    final = verifier.verify_and_retry(
        scene, floorplan, initial, {"r1-cam1": maps, "r1-cam2": maps}, tmp_path / "out"
    )
    report = summarize(final)
    assert report["verified"] is False
    assert "no consistency judge" in report["note"].lower()
    assert mean_consistency(final) is None


def test_summary_surfaces_missing_and_hallucinated_objects(scene, floorplan, maps, tmp_path):
    verdict = _scores(0.8, cross=0.8, missing=["r1:sofa#0"])
    verdict.hallucinated_objects = ["a piano nobody asked for"]
    verifier, renders, _ = _pipeline(tmp_path, [verdict], attempts=1)

    initial = _render_all(renders, scene, floorplan, maps, tmp_path)
    final = verifier.verify_and_retry(
        scene, floorplan, initial, {"r1-cam1": maps, "r1-cam2": maps}, tmp_path / "out"
    )
    report = summarize(final)
    assert report["missing_objects"] == ["r1:sofa#0"]
    assert report["hallucinated_objects"] == ["a piano nobody asked for"]
