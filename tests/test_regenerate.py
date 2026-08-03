"""Regeneration must preserve the scene.

The brief lists "regenerate designs while preserving the same scene unless the
user requests changes" as a user feature, and the consistency requirements
spend eight bullets on objects not moving. So the guarantee under test is not
"the response says preserved_scene: true" — it is that every object keeps its
product, position, rotation and size, and that the *images* still differ.

An earlier implementation satisfied the first sentence and violated the second:
it bumped `DesignRequest.seed`, which feeds the placement solver, and 21 of 32
objects moved. These tests exist so that cannot come back quietly.
"""

from __future__ import annotations

import pytest

from backend.imagegen.service import view_seed
from backend.render.service import SceneRenderer
from backend.schemas.product import DesignStyle
from backend.schemas.scene import Scene

from .test_design import agent, floorplan  # noqa: F401 — pytest fixtures


@pytest.fixture
def scene(agent, floorplan) -> Scene:  # noqa: F811
    designed = agent.design(
        floorplan=floorplan,
        style=DesignStyle.SCANDINAVIAN,
        palette_name=None,
        room_ids=None,
        seed=0,
        variation_index=0,
        budget=None,
    )
    # Cameras are attached by the pipeline, not the agent, and a scene without
    # them has no per-view seeds to compare.
    return SceneRenderer().attach_cameras(designed, floorplan, views_per_room=2)


def _fingerprint(scene: Scene) -> dict[str, tuple]:
    """Everything the consistency requirements say must not change."""
    return {
        obj.instance_id: (
            obj.product_id,
            obj.position_m.x,
            obj.position_m.y,
            obj.position_m.z,
            obj.rotation_deg,
            obj.size_m.width,
            obj.size_m.depth,
            obj.size_m.height,
        )
        for obj in scene.objects
    }


def test_salting_a_scene_changes_nothing_about_the_scene(scene):
    regenerated = scene.model_copy(update={"render_salt": scene.render_salt + 1})

    assert _fingerprint(regenerated) == _fingerprint(scene)
    assert regenerated.content_hash() == scene.content_hash()
    # Same id is the point: "same scene_id" has to keep meaning "same room".
    assert regenerated.scene_id == scene.scene_id


def test_salting_a_scene_changes_every_image_seed(scene):
    regenerated = scene.model_copy(update={"render_salt": scene.render_salt + 1})

    before = [view_seed(scene.seed, c.id, 0, scene.render_salt) for c in scene.cameras]
    after = [view_seed(scene.seed, c.id, 0, regenerated.render_salt) for c in regenerated.cameras]

    assert before and not set(before) & set(after)


def test_salted_output_does_not_overwrite_the_original(scene):
    regenerated = scene.model_copy(update={"render_salt": 1})

    assert scene.output_key == scene.scene_id
    assert regenerated.output_key != scene.output_key


def test_reseeding_the_design_moves_furniture(agent, floorplan):  # noqa: F811
    """Why the salt exists at all — the guard on the original bug.

    If this ever stops holding, the seed no longer drives placement and the
    regeneration fix is protecting against nothing.
    """
    kwargs = dict(
        floorplan=floorplan,
        style=DesignStyle.SCANDINAVIAN,
        palette_name=None,
        room_ids=None,
        variation_index=0,
        budget=None,
    )
    a = agent.design(seed=0, **kwargs)
    b = agent.design(seed=1, **kwargs)

    fa, fb = _fingerprint(a), _fingerprint(b)
    moved = [k for k in fa if k in fb and fa[k] != fb[k]]
    assert moved, "reseeding no longer changes placement — the salt may be redundant"
