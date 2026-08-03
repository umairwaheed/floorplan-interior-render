"""Geometry rasterizer — the scene graph projected through a camera.

This is where multi-view consistency stops being a promise and becomes
measurable. Each camera projects the *same* 3D scene, so the depth and
segmentation buffers for view 1 and view 2 are two views of one geometry rather
than two independent guesses. Conditioned on these, the image model is filling
in surfaces, not deciding where the furniture goes.

No GPU, no OpenGL, no headless browser. A pinhole projection plus a z-buffer
over oriented boxes is a few hundred lines of numpy, runs anywhere, and is
exactly reproducible — which matters more here than photorealism, because the
image model supplies the realism and this supplies the truth.

Four buffers come out of each camera:

* **depth** — the geometric backbone; nearer is brighter, the convention
  depth-conditioned image models expect.
* **segmentation** — a deterministic colour per object instance, so the prompt
  can say which product occupies which region and the judge can check it.
* **wireframe** — clean edges, for structural conditioning.
* **preview** — flat-shaded colour for the UI and for debugging by eye.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageDraw

from ..schemas.common import Vec2, Vec3
from ..schemas.floorplan import Room
from ..schemas.scene import Camera, PlacedObject, RoomFinishes

#: Geometry nearer than this is clipped — it is behind or inside the lens.
NEAR_PLANE = 0.05

#: Reserved segmentation colours for architecture, kept away from object hues.
FLOOR_COLOR = (48, 48, 56)
WALL_COLOR = (96, 96, 104)
CEILING_COLOR = (144, 144, 152)

#: The 12 edges of a box, as index pairs into `box_corners`.
BOX_EDGES = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),  # bottom
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4),  # top
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),  # verticals
)

#: The 6 faces of a box, as index quads, wound consistently.
BOX_FACES = (
    (0, 1, 2, 3),  # bottom
    (7, 6, 5, 4),  # top
    (0, 4, 5, 1),  # side
    (1, 5, 6, 2),
    (2, 6, 7, 3),
    (3, 7, 4, 0),
)


def instance_color(instance_id: str) -> tuple[int, int, int]:
    """A stable, well-separated colour for one object instance.

    Derived from the instance ID so the same object keeps the same segmentation
    colour across every view and every regeneration — which is what lets the
    judge ask "is instance X still where the map says it is" rather than
    guessing from appearance.
    """
    digest = hashlib.blake2b(instance_id.encode(), digest_size=8).digest()
    # Sample hue widely but keep saturation and value high, so colours stay
    # distinguishable and never collide with the greys reserved above.
    hue = int.from_bytes(digest[:2], "big") / 65535.0
    saturation = 0.55 + (digest[2] / 255.0) * 0.4
    value = 0.65 + (digest[3] / 255.0) * 0.35
    return _hsv_to_rgb(hue, saturation, value)


def _hsv_to_rgb(h: float, s: float, v: float) -> tuple[int, int, int]:
    i = int(h * 6.0) % 6
    f = h * 6.0 - int(h * 6.0)
    p, q, t = v * (1 - s), v * (1 - f * s), v * (1 - (1 - f) * s)
    r, g, b = [(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)][i]
    return (int(r * 255), int(g * 255), int(b * 255))


@dataclass
class CameraBasis:
    """A camera resolved into a projection basis."""

    eye: np.ndarray
    right: np.ndarray
    up: np.ndarray
    forward: np.ndarray
    focal_px: float
    width: int
    height: int

    @classmethod
    def from_camera(cls, camera: Camera, width: int, height: int) -> CameraBasis:
        eye = np.array(camera.position_m.as_tuple(), dtype=np.float64)
        target = np.array(camera.look_at_m.as_tuple(), dtype=np.float64)

        forward = target - eye
        norm = np.linalg.norm(forward)
        forward = forward / norm if norm > 1e-9 else np.array([0.0, 1.0, 0.0])

        world_up = np.array([0.0, 0.0, 1.0])
        right = np.cross(forward, world_up)
        if np.linalg.norm(right) < 1e-9:
            # Camera is looking straight up or down; any horizontal right works.
            right = np.array([1.0, 0.0, 0.0])
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)

        focal_px = (width / 2.0) / math.tan(math.radians(camera.fov_deg) / 2.0)
        return cls(eye, right, up, forward, focal_px, width, height)

    def project(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """World points (N,3) → screen points (N,2) and view depths (N,).

        Depth is distance along the view axis, which is what a depth map wants;
        Euclidean distance would bow straight walls.
        """
        offset = points - self.eye
        x_view = offset @ self.right
        y_view = offset @ self.up
        z_view = offset @ self.forward

        safe_z = np.where(np.abs(z_view) < 1e-9, 1e-9, z_view)
        screen_x = self.width / 2.0 + self.focal_px * x_view / safe_z
        screen_y = self.height / 2.0 - self.focal_px * y_view / safe_z
        return np.stack([screen_x, screen_y], axis=1), z_view


def box_corners(obj: PlacedObject) -> np.ndarray:
    """The 8 world-space corners of an object's bounding box."""
    rad = math.radians(obj.rotation_deg)
    cos_r, sin_r = math.cos(rad), math.sin(rad)
    hw, hd = obj.size_m.width / 2.0, obj.size_m.depth / 2.0
    z0, z1 = obj.position_m.z, obj.position_m.z + obj.size_m.height

    corners = []
    for z in (z0, z1):
        for dx, dy in ((-hw, -hd), (hw, -hd), (hw, hd), (-hw, hd)):
            corners.append(
                (
                    obj.position_m.x + dx * cos_r - dy * sin_r,
                    obj.position_m.y + dx * sin_r + dy * cos_r,
                    z,
                )
            )
    return np.array(corners, dtype=np.float64)


@dataclass
class Fragment:
    """One planar face queued for rasterization."""

    world: np.ndarray  # (N,3) polygon vertices
    color: tuple[int, int, int]
    seg_color: tuple[int, int, int]
    instance_id: str | None = None


@dataclass
class RasterBuffers:
    """The output of rasterizing one camera."""

    depth: np.ndarray  # (H,W) float, view-space metres; inf where nothing was hit
    instance: np.ndarray  # (H,W) int32 index into `instance_ids`; -1 for architecture
    segmentation: np.ndarray  # (H,W,3) uint8
    preview: np.ndarray  # (H,W,3) uint8
    instance_ids: list[str] = field(default_factory=list)

    def visible_instances(self, min_pixels: int = 40) -> dict[str, int]:
        """instance_id → pixel count, for objects actually big enough to see."""
        counts: dict[str, int] = {}
        if not self.instance_ids:
            return counts
        flat = self.instance[self.instance >= 0]
        if flat.size == 0:
            return counts
        indices, totals = np.unique(flat, return_counts=True)
        for index, total in zip(indices, totals, strict=True):
            if total >= min_pixels:
                counts[self.instance_ids[int(index)]] = int(total)
        return counts


def _clip_near(world: np.ndarray, depths: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Sutherland–Hodgman against the near plane, in 3D.

    Without this, a wall passing behind the camera projects to wild
    coordinates and smears across the frame — the classic symptom of a
    missing near-plane clip.
    """
    if np.all(depths > NEAR_PLANE):
        return world, depths
    if np.all(depths <= NEAR_PLANE):
        return None

    output_world: list[np.ndarray] = []
    output_depth: list[float] = []
    count = len(world)
    for i in range(count):
        current_w, current_d = world[i], depths[i]
        previous_w, previous_d = world[i - 1], depths[i - 1]

        if (current_d > NEAR_PLANE) != (previous_d > NEAR_PLANE):
            t = (NEAR_PLANE - previous_d) / (current_d - previous_d)
            output_world.append(previous_w + t * (current_w - previous_w))
            output_depth.append(NEAR_PLANE)
        if current_d > NEAR_PLANE:
            output_world.append(current_w)
            output_depth.append(current_d)

    if len(output_world) < 3:
        return None
    return np.array(output_world), np.array(output_depth)


def _fill_triangle(
    depth_buffer: np.ndarray,
    instance_buffer: np.ndarray,
    seg_buffer: np.ndarray,
    preview_buffer: np.ndarray,
    screen: np.ndarray,
    view_depths: np.ndarray,
    seg_color: tuple[int, int, int],
    shade_color: tuple[int, int, int],
    instance_index: int,
) -> None:
    """Z-buffered triangle fill with perspective-correct depth.

    Depth is interpolated as 1/z, which is what varies linearly in screen
    space. Interpolating z directly bends every surface toward the camera and
    would put a visible bow in every wall of the depth map.
    """
    height, width = depth_buffer.shape

    min_x = max(int(np.floor(screen[:, 0].min())), 0)
    max_x = min(int(np.ceil(screen[:, 0].max())), width - 1)
    min_y = max(int(np.floor(screen[:, 1].min())), 0)
    max_y = min(int(np.ceil(screen[:, 1].max())), height - 1)
    if min_x > max_x or min_y > max_y:
        return

    (x0, y0), (x1, y1), (x2, y2) = screen
    area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
    if abs(area) < 1e-9:
        return  # degenerate after projection

    ys, xs = np.mgrid[min_y : max_y + 1, min_x : max_x + 1]
    px, py = xs + 0.5, ys + 0.5

    w0 = ((x1 - px) * (y2 - py) - (x2 - px) * (y1 - py)) / area
    w1 = ((x2 - px) * (y0 - py) - (x0 - px) * (y2 - py)) / area
    w2 = 1.0 - w0 - w1

    inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
    if not inside.any():
        return

    inv_z = w0 / view_depths[0] + w1 / view_depths[1] + w2 / view_depths[2]
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(np.abs(inv_z) > 1e-12, 1.0 / inv_z, np.inf)

    region = (slice(min_y, max_y + 1), slice(min_x, max_x + 1))
    closer = inside & (z > NEAR_PLANE) & (z < depth_buffer[region])
    if not closer.any():
        return

    depth_buffer[region][closer] = z[closer]
    instance_buffer[region][closer] = instance_index
    seg_buffer[region][closer] = seg_color
    preview_buffer[region][closer] = shade_color


def _shade(base: tuple[int, int, int], normal: np.ndarray) -> tuple[int, int, int]:
    """Flat lambert shading, for the human-readable preview only.

    Purely so the preview reads as a room rather than a flat colour field —
    nothing downstream depends on it.
    """
    light = np.array([0.35, -0.45, 0.82])
    light = light / np.linalg.norm(light)
    intensity = 0.55 + 0.45 * max(0.0, float(np.dot(normal, light)))
    return tuple(int(min(255, channel * intensity)) for channel in base)  # type: ignore[return-value]


def _face_normal(vertices: np.ndarray) -> np.ndarray:
    normal = np.cross(vertices[1] - vertices[0], vertices[2] - vertices[0])
    length = np.linalg.norm(normal)
    return normal / length if length > 1e-9 else np.array([0.0, 0.0, 1.0])


def _room_fragments(room: Room, finishes: RoomFinishes | None) -> list[Fragment]:
    """Floor, ceiling and walls as planar quads."""
    polygon = room.polygon_m
    height = room.ceiling_height_m
    fragments: list[Fragment] = []

    floor = np.array([(p.x, p.y, 0.0) for p in polygon])
    ceiling = np.array([(p.x, p.y, height) for p in reversed(polygon)])

    floor_rgb = _finish_rgb(finishes.floor.color if finishes else None, (150, 132, 110))
    wall_rgb = _finish_rgb(finishes.walls.color if finishes else None, (232, 228, 220))
    ceiling_rgb = (245, 245, 242)

    fragments.append(Fragment(world=floor, color=floor_rgb, seg_color=FLOOR_COLOR))
    fragments.append(Fragment(world=ceiling, color=ceiling_rgb, seg_color=CEILING_COLOR))

    for i in range(len(polygon)):
        a, b = polygon[i], polygon[(i + 1) % len(polygon)]
        wall = np.array([(a.x, a.y, 0.0), (b.x, b.y, 0.0), (b.x, b.y, height), (a.x, a.y, height)])
        fragments.append(Fragment(world=wall, color=wall_rgb, seg_color=WALL_COLOR))

    return fragments


def _finish_rgb(color_name: str | None, fallback: tuple[int, int, int]) -> tuple[int, int, int]:
    """Map a finish colour name or hex to something displayable."""
    if not color_name:
        return fallback
    name = color_name.strip().lower()
    if name.startswith("#") and len(name) == 7:
        return (int(name[1:3], 16), int(name[3:5], 16), int(name[5:7], 16))
    return {
        "white": (245, 245, 243),
        "off-white": (240, 236, 228),
        "beige": (225, 212, 190),
        "grey": (178, 178, 178),
        "charcoal": (74, 74, 78),
        "black": (38, 38, 40),
        "brown": (120, 86, 60),
        "tan": (190, 152, 110),
        "natural": (206, 184, 152),
        "oak": (198, 168, 126),
        "walnut": (110, 76, 50),
        "green": (120, 142, 112),
        "olive": (128, 132, 88),
        "blue": (96, 122, 150),
        "terracotta": (188, 108, 78),
        "mustard": (214, 168, 70),
        "pink": (226, 190, 190),
        "gold": (200, 168, 100),
        "silver": (192, 196, 200),
    }.get(name, fallback)


def rasterize(
    room: Room,
    objects: list[PlacedObject],
    camera: Camera,
    width: int,
    height: int,
    finishes: RoomFinishes | None = None,
) -> RasterBuffers:
    """Project one room through one camera into depth, instance and colour buffers."""
    basis = CameraBasis.from_camera(camera, width, height)

    depth_buffer = np.full((height, width), np.inf, dtype=np.float64)
    instance_buffer = np.full((height, width), -1, dtype=np.int32)
    seg_buffer = np.zeros((height, width, 3), dtype=np.uint8)
    preview_buffer = np.zeros((height, width, 3), dtype=np.uint8)

    instance_ids = [obj.instance_id for obj in objects]
    index_of = {instance_id: i for i, instance_id in enumerate(instance_ids)}

    fragments = _room_fragments(room, finishes)
    for obj in objects:
        corners = box_corners(obj)
        base_rgb = _finish_rgb(obj.color, (190, 185, 176))
        seg_rgb = instance_color(obj.instance_id)
        for face in BOX_FACES:
            fragments.append(
                Fragment(
                    world=corners[list(face)],
                    color=base_rgb,
                    seg_color=seg_rgb,
                    instance_id=obj.instance_id,
                )
            )

    for fragment in fragments:
        _, depths = basis.project(fragment.world)
        clipped = _clip_near(fragment.world, depths)
        if clipped is None:
            continue
        world, view_depths = clipped
        screen, _ = basis.project(world)

        shade = _shade(fragment.color, _face_normal(world))
        instance_index = index_of.get(fragment.instance_id or "", -1)

        # Fan-triangulate the (convex) face.
        for i in range(1, len(world) - 1):
            _fill_triangle(
                depth_buffer,
                instance_buffer,
                seg_buffer,
                preview_buffer,
                screen[[0, i, i + 1]],
                view_depths[[0, i, i + 1]],
                fragment.seg_color,
                shade,
                instance_index,
            )

    return RasterBuffers(
        depth=depth_buffer,
        instance=instance_buffer,
        segmentation=seg_buffer,
        preview=preview_buffer,
        instance_ids=instance_ids,
    )


def depth_to_image(depth: np.ndarray) -> Image.Image:
    """Normalize a depth buffer to an 8-bit image, nearer = brighter.

    That polarity is the convention depth-conditioned image models expect;
    inverting it produces a plausible-looking map that conditions the model
    into turning the room inside out.
    """
    finite = np.isfinite(depth)
    if not finite.any():
        return Image.fromarray(np.zeros(depth.shape, dtype=np.uint8), mode="L")

    near, far = depth[finite].min(), depth[finite].max()
    span = max(far - near, 1e-6)

    normalized = np.zeros(depth.shape, dtype=np.float64)
    normalized[finite] = 1.0 - (depth[finite] - near) / span
    return Image.fromarray((normalized * 255).astype(np.uint8), mode="L")


def wireframe_image(
    room: Room,
    objects: list[PlacedObject],
    camera: Camera,
    width: int,
    height: int,
) -> Image.Image:
    """Projected edges on white — clean structural conditioning.

    Drawn with a depth test against the filled buffers so hidden edges don't
    show through, which would tell the image model there is geometry where
    there is only a back face.
    """
    basis = CameraBasis.from_camera(camera, width, height)
    buffers = rasterize(room, objects, camera, width, height)

    image = Image.new("L", (width, height), 255)
    draw = ImageDraw.Draw(image)

    segments: list[tuple[np.ndarray, np.ndarray]] = []

    polygon = room.polygon_m
    ceiling_h = room.ceiling_height_m
    for i in range(len(polygon)):
        a, b = polygon[i], polygon[(i + 1) % len(polygon)]
        for z0, z1 in ((0.0, 0.0), (ceiling_h, ceiling_h)):
            segments.append((np.array([a.x, a.y, z0]), np.array([b.x, b.y, z1])))
        segments.append((np.array([a.x, a.y, 0.0]), np.array([a.x, a.y, ceiling_h])))

    for obj in objects:
        corners = box_corners(obj)
        for start, end in BOX_EDGES:
            segments.append((corners[start], corners[end]))

    for start_world, end_world in segments:
        pair = np.stack([start_world, end_world])
        _, depths = basis.project(pair)
        clipped = _clip_near(pair, depths)
        if clipped is None or len(clipped[0]) < 2:
            continue
        screen, _ = basis.project(clipped[0])
        _draw_visible_segment(draw, screen, clipped[1], buffers.depth)

    return image


def _draw_visible_segment(
    draw: ImageDraw.ImageDraw,
    screen: np.ndarray,
    depths: np.ndarray,
    depth_buffer: np.ndarray,
    samples: int = 48,
    tolerance: float = 0.06,
) -> None:
    """Draw a line only where it isn't hidden behind filled geometry."""
    height, width = depth_buffer.shape
    previous: tuple[float, float] | None = None

    for i in range(samples + 1):
        t = i / samples
        x = screen[0, 0] + (screen[1, 0] - screen[0, 0]) * t
        y = screen[0, 1] + (screen[1, 1] - screen[0, 1]) * t
        # Depth along the segment varies linearly in 1/z, same as the fill.
        inv_z = (1 - t) / depths[0] + t / depths[1]
        z = 1.0 / inv_z if abs(inv_z) > 1e-12 else np.inf

        px, py = int(x), int(y)
        visible = 0 <= px < width and 0 <= py < height and z <= depth_buffer[py, px] + tolerance
        if visible:
            if previous is not None:
                draw.line([previous, (x, y)], fill=0, width=2)
            previous = (x, y)
        else:
            previous = None


def to_pil(buffer: np.ndarray) -> Image.Image:
    return Image.fromarray(buffer, mode="RGB")


def camera_from_vecs(
    camera_id: str, room_id: str, eye: Vec3, target: Vec3, fov_deg: float = 60.0
) -> Camera:
    """Small helper for tests and one-off renders."""
    return Camera(id=camera_id, room_id=room_id, position_m=eye, look_at_m=target, fov_deg=fov_deg)


def screen_position(camera: Camera, point: Vec2, width: int, height: int, z: float = 0.0):
    """Where a world point lands on screen, or None if behind the camera."""
    basis = CameraBasis.from_camera(camera, width, height)
    screen, depths = basis.project(np.array([[point.x, point.y, z]]))
    if depths[0] <= NEAR_PLANE:
        return None
    return float(screen[0, 0]), float(screen[0, 1])
