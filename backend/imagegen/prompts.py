"""Prompt assembly — built from the scene graph, never written by hand.

Every noun in a render prompt traces back to a `PlacedObject` or a
`SurfaceFinish` in the frozen scene. That is what makes "use only products from
the supplied catalog" enforceable rather than hopeful: there is no code path
that can put a product name into a prompt unless retrieval already bound that
product to a slot.

Two properties follow, and both matter for consistency:

* **Only visible objects are named.** The rasterizer reports which instances
  actually land in this frame, so the model is never asked to draw something
  off-screen — the single most reliable way to make an object appear in one
  view and not another.
* **The style, palette, finishes and negative constraints are byte-identical
  across views of a scene.** Only the per-view object list and camera
  description change. Any variation in the shared block is variation the image
  model will faithfully reproduce as a difference between viewpoints.
"""

from __future__ import annotations

from ..design.styles import get_style_profile
from ..schemas.render import ConditioningMaps
from ..schemas.scene import Camera, PlacedObject, RoomFinishes, Scene

#: Things the model must not do. Phrased as observable outcomes rather than
#: abstractions ("do not add furniture" beats "be consistent"), because the
#: former is checkable and the latter is not.
NEGATIVE_CONSTRAINTS = """\
Do not add, remove, move, resize or restyle any furniture. Do not invent \
objects that are not listed. Do not change the room's shape, the position of \
walls, doors or windows, or the ceiling height. Do not add people, text, \
watermarks, labels, or measurement annotations. Do not produce a floor plan, \
a diagram, a collage, or a split view — this is a single photograph."""

CAMERA_BRIEF = """\
Photograph the room from the camera position implied by the geometry images. \
Use a natural interior-photography perspective at standing eye height, with \
straight vertical lines and no fisheye distortion."""


def _describe_object(obj: PlacedObject, pixel_share: float | None) -> str:
    """One object, as the prompt should name it.

    Uses the catalog product's own name, colour and material — so what the
    model is told to draw and what the bill of materials lists are the same
    string, drawn from the same record.
    """
    size = obj.size_m
    parts = [f"{obj.display_name}"]

    descriptor = " ".join(filter(None, [obj.color, obj.material]))
    if descriptor:
        parts.append(f"({descriptor})")

    parts.append(f"— {size.width:.2f} m wide × {size.depth:.2f} m deep × {size.height:.2f} m high")

    if pixel_share is not None and pixel_share > 0.12:
        parts.append("— prominent in this view")
    elif pixel_share is not None and pixel_share < 0.02:
        parts.append("— small or partly occluded in this view")

    return " ".join(parts)


def _finish_block(finishes: RoomFinishes | None) -> str:
    """Renovation surfaces, named as products.

    The brief asks for "every visible furniture *or renovation* element" to map
    to a real product, so floors and walls are described from their catalog
    binding rather than left to the model's taste.
    """
    if finishes is None:
        return "Floor, walls and ceiling in a neutral, coherent finish."

    lines = [
        f"Floor: {finishes.floor.display_name}"
        + (f", {finishes.floor.color}" if finishes.floor.color else ""),
        f"Walls: {finishes.walls.display_name}"
        + (f", {finishes.walls.color}" if finishes.walls.color else ""),
        f"Ceiling: {finishes.ceiling.display_name}",
    ]
    if finishes.trim:
        lines.append(f"Skirting: {finishes.trim.display_name}")
    return "\n".join(lines)


def _lighting_block(scene: Scene, room_id: str) -> str:
    """Lighting described from the scene's own light sources.

    Lighting lives in the graph for the same reason furniture does: if it were
    left to the prompt, the sun would move between viewpoints.
    """
    lights = [light for light in scene.lights if light.room_id == room_id]
    if not lights:
        return "Soft, even daylight."

    has_daylight = any(light.kind == "daylight" for light in lights)
    fixtures = [light for light in lights if light.kind != "daylight"]

    parts = []
    if has_daylight:
        parts.append("soft natural daylight from the window")
    if fixtures:
        warm = sum(1 for light in fixtures if light.color_temp_k <= 3000)
        parts.append(
            f"{len(fixtures)} lit fixture(s), {'warm' if warm else 'neutral'} tone, "
            "casting consistent shadows"
        )
    return "Lighting: " + ", ".join(parts) + ". Shadow directions must match across views."


def build_scene_block(scene: Scene, room_id: str) -> str:
    """The part of the prompt that is identical for every view of a scene.

    Kept separate and assembled verbatim so it cannot drift between views —
    and so it forms a stable prefix, which is also what prompt caching wants.
    """
    profile = get_style_profile(scene.style)
    return "\n".join(
        [
            "STYLE",
            profile.prompt_fragment + ".",
            f"Colour palette: {scene.palette.as_prompt_fragment()}.",
            "",
            "SURFACES",
            _finish_block(scene.finishes_for_room(room_id)),
            "",
            _lighting_block(scene, room_id),
        ]
    )


def build_view_prompt(
    scene: Scene,
    camera: Camera,
    maps: ConditioningMaps,
    is_anchor: bool,
    room_name: str = "room",
) -> str:
    """The complete prompt for one view.

    `is_anchor` selects between "establish this room" and "the same room again,
    from here" — the second framing is what the appearance-reference image is
    there to satisfy.
    """
    objects = {obj.instance_id: obj for obj in scene.objects_in_room(camera.room_id)}
    visible = [
        objects[instance_id] for instance_id in maps.visible_instance_ids if instance_id in objects
    ]

    inventory = (
        "\n".join(
            f"- {_describe_object(obj, maps.instance_pixel_share.get(obj.instance_id))}"
            for obj in visible
        )
        or "- (no furniture is visible from this angle)"
    )

    header = (
        f"Photorealistic interior photograph of a {room_name}."
        if is_anchor
        else (
            f"Photorealistic interior photograph of the SAME {room_name} shown in the "
            "reference photograph, from a different camera position."
        )
    )

    consistency = (
        "This is the first view; it establishes the room's appearance."
        if is_anchor
        else (
            "CRITICAL — this is the same physical room as the reference photograph, "
            "at the same moment, with nothing rearranged. Every object keeps the exact "
            "same colour, material, shape and finish it has in the reference. Only the "
            "camera has moved. Objects visible in both views must be recognisably the "
            "same objects."
        )
    )

    return "\n".join(
        [
            header,
            "",
            "GEOMETRY",
            "The attached depth map, segmentation map and wireframe define the exact "
            "geometry of this shot. Match them precisely: every surface, wall angle and "
            "object position must align with them. Brighter regions of the depth map are "
            "closer to the camera. Each distinct colour in the segmentation map is one "
            "object; render one object per region and nothing in between them.",
            "",
            build_scene_block(scene, camera.room_id),
            "",
            "OBJECTS IN THIS VIEW",
            "Render exactly these, and only these:",
            inventory,
            "",
            "CAMERA",
            CAMERA_BRIEF,
            "",
            "CONSISTENCY",
            consistency,
            "",
            "DO NOT",
            NEGATIVE_CONSTRAINTS,
        ]
    )


def build_change_request_prompt(scene: Scene, change: str) -> str:
    """Ask the model to turn a user's change request into a scene-graph patch.

    Deliberately narrow: the model proposes *which objects change and how*, and
    the design layer re-runs retrieval and placement for those objects only.
    Letting it rewrite the whole scene would defeat the point of a patch, which
    is that untouched objects keep their seeds and stay pixel-stable.
    """
    inventory = "\n".join(
        f"- {obj.instance_id}: {obj.display_name} ({obj.color}, {obj.category.value}) "
        f"in room {obj.room_id}"
        for obj in scene.objects
    )
    return "\n".join(
        [
            "A user has asked for a change to an existing interior design.",
            "",
            f'Their request: "{change}"',
            "",
            "Current objects in the scene:",
            inventory,
            "",
            f"Current style: {scene.style.value}, palette: {scene.palette.name}.",
            "",
            "Identify the minimal set of changes. Only list objects the user actually "
            "asked to change — everything else must stay exactly as it is, because "
            "untouched objects are reused verbatim and must remain pixel-stable.",
        ]
    )
