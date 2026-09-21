"""PNG rendering: a clean technical preview and a wood-textured mockup.

Two very different jobs share this module because they share the geometry
plumbing:

``render_preview``
    A flat top view that shows what the file contains - outer profile, through
    cuts, recesses tinted by depth, engraving.  This is the honest picture,
    used in the contact sheet and for checking a design by eye.

``render_mockup``
    A procedurally wood-grained render of the same geometry, suitable as a
    listing image.  The grain is generated from the design's own slug, so a
    given design always renders identically and a bundle of fifty designs does
    not look like fifty photographs of the same board.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.patheffects as patheffects  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.patches import PathPatch  # noqa: E402
from matplotlib.path import Path as MplPath  # noqa: E402

from .design import Design, Part  # noqa: E402
from .geometry import Point, Ring, ensure_ccw, ensure_cw  # noqa: E402

__all__ = [
    "WOOD_PALETTES",
    "render_preview",
    "render_mockup",
    "compound_path",
]

#: Light and dark grain colours per material keyword, as 0-1 RGB.
WOOD_PALETTES: dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]] = {
    "oak": ((0.847, 0.714, 0.522), (0.612, 0.435, 0.259)),
    "walnut": ((0.545, 0.396, 0.278), (0.286, 0.184, 0.125)),
    "maple": ((0.933, 0.859, 0.718), (0.784, 0.667, 0.502)),
    "cherry": ((0.820, 0.604, 0.443), (0.588, 0.341, 0.224)),
    "birch": ((0.925, 0.855, 0.733), (0.796, 0.686, 0.541)),
    "plywood": ((0.906, 0.831, 0.698), (0.729, 0.616, 0.467)),
    "pine": ((0.949, 0.867, 0.706), (0.780, 0.639, 0.443)),
    "bamboo": ((0.898, 0.812, 0.647), (0.694, 0.573, 0.388)),
    "acrylic": ((0.831, 0.851, 0.867), (0.635, 0.667, 0.698)),
    "mdf": ((0.792, 0.714, 0.616), (0.616, 0.529, 0.435)),
}
_DEFAULT_PALETTE = WOOD_PALETTES["oak"]

#: Preview palette.  The DXF keeps its AutoCAD colour indices, which exist to
#: be unambiguous in CAD rather than to look like anything; a preview that a
#: buyer sees needs colours that were chosen.  Operations stay distinguishable.
PREVIEW_COLORS = {
    "background": "#ffffff",
    "material": "#f4f0e8",
    "outline": "#b04a3f",
    "inside": "#3a6ea5",
    "pocket_shallow": "#e0d3b8",
    "pocket_deep": "#bda478",
    "pocket_edge": "#96794a",
    "engrave": "#7d57a4",
    "drill": "#c87a2c",
    "info": "#8a8a8a",
    "caption": "#4a4a4a",
}
_PREVIEW_BACKGROUND = PREVIEW_COLORS["background"]
_PREVIEW_MATERIAL = PREVIEW_COLORS["material"]
_MOCKUP_BACKGROUND = "#e9e4dc"


# --------------------------------------------------------------------------- #
# geometry to matplotlib
# --------------------------------------------------------------------------- #
def compound_path(exterior: Sequence[Point], holes: Iterable[Sequence[Point]] = ()) -> MplPath:
    """Build a matplotlib path with holes.

    Two details matter.  Rings are closed explicitly, because matplotlib wants
    the closing vertex the rest of this project leaves implicit.  And holes are
    wound *opposite* to the exterior, because Agg fills compound paths by the
    nonzero winding rule: a hole wound the same way as the exterior adds to the
    winding number instead of cancelling it, and fills in solid.

    Args:
        exterior: Outer boundary.
        holes: Interior boundaries.

    Returns:
        A path whose interior is the material, holes excluded.
    """
    vertices: list[Point] = []
    codes: list[int] = []
    outer = ensure_ccw(list(exterior))
    for ring in (outer, *[ensure_cw(list(h)) for h in holes]):
        if len(ring) < 3:
            continue
        vertices.extend(ring)
        vertices.append(ring[0])
        codes.extend([MplPath.MOVETO] + [MplPath.LINETO] * (len(ring) - 1))
        codes.append(MplPath.CLOSEPOLY)
    return MplPath(np.asarray(vertices, dtype=float), codes)


def _figure(
    bbox: tuple[float, float, float, float],
    margin: float,
    px_width: int,
    dpi: int,
    background: str,
) -> tuple[Figure, "matplotlib.axes.Axes"]:
    """Create a figure whose axes exactly cover the design plus a margin."""
    x0, y0, x1, y1 = bbox
    width = (x1 - x0) + 2 * margin
    height = (y1 - y0) + 2 * margin
    fig = Figure(figsize=(px_width / dpi, px_width / dpi * height / width), dpi=dpi)
    fig.patch.set_facecolor(background)
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    ax.set_xlim(x0 - margin, x1 + margin)
    ax.set_ylim(y0 - margin, y1 + margin)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor(background)
    return fig, ax


def _seed_of(text: str) -> int:
    """Derive a stable seed from a string, so renders are reproducible."""
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "big")


# --------------------------------------------------------------------------- #
# clean technical preview
# --------------------------------------------------------------------------- #
def render_preview(
    design: Design,
    path: str | Path,
    px_width: int = 1400,
    dpi: int = 200,
    margin_mm: float | None = None,
    caption: bool = True,
) -> Path:
    """Render a flat top view of a design.

    Cuts, recesses and engraving are drawn in their layer colours, recesses
    are tinted in proportion to depth, and through cuts show the background,
    so the picture says the same thing the DXF does.

    Args:
        design: The design to draw; use :meth:`Design.normalized` first.
        path: Destination PNG path; parent directories are created.
        px_width: Image width in pixels.
        dpi: Dots per inch used to size the figure.
        margin_mm: Blank margin around the design; defaults to 4% of its size.
        caption: Draw the overall dimensions under the design.

    Returns:
        The written path.
    """
    placed = design.normalized()
    bbox = placed.bbox()
    span = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
    margin = span * 0.04 if margin_mm is None else margin_mm
    caption_space = span * 0.07 if caption else 0.0
    frame = (bbox[0], bbox[1] - caption_space, bbox[2], bbox[3])
    fig, ax = _figure(frame, margin, px_width, dpi, _PREVIEW_BACKGROUND)

    # Line width is in points, so it must come from the figure size rather
    # than from millimetres: a millimetre-derived width is hairline on a
    # coaster and a fat stripe on a sheet of plywood.
    line = max(0.6, (px_width / dpi) * 0.22)
    deepest = max(placed.pocket_depths(), default=1.0)
    # Matplotlib sizes text in points, but a label's height is in millimetres
    # and has to scale with the drawing: a 16 mm label passed straight through
    # as 16 points is invisible on a tray and enormous on a sheet of plywood.
    view_width = (frame[2] - frame[0]) + 2 * margin
    points_per_mm = (px_width / dpi * 72.0) / view_width

    for part in placed.placed_parts():
        ax.add_patch(
            PathPatch(
                compound_path(part.outline, part.holes),
                facecolor=_PREVIEW_MATERIAL,
                edgecolor="none",
                zorder=1,
            )
        )
        for pocket in part.pockets:
            ax.add_patch(
                PathPatch(
                    compound_path(pocket.ring, pocket.islands),
                    facecolor=_depth_colour(pocket.depth, deepest),
                    edgecolor=PREVIEW_COLORS["pocket_edge"],
                    linewidth=line,
                    zorder=2,
                )
            )
        for hole in part.holes:
            ax.add_patch(
                PathPatch(
                    compound_path(hole),
                    facecolor="none",
                    edgecolor=PREVIEW_COLORS["inside"],
                    linewidth=line,
                    zorder=4,
                )
            )
        for contour in part.engrave:
            points = np.asarray(
                list(contour.points) + ([contour.points[0]] if contour.closed else []),
                dtype=float,
            )
            ax.plot(
                points[:, 0],
                points[:, 1],
                color=PREVIEW_COLORS["engrave"],
                linewidth=line * 0.9,
                zorder=4,
            )
        for drill in part.drills:
            ax.add_patch(
                PathPatch(
                    compound_path(_circle_ring(drill.center, drill.radius)),
                    facecolor=_PREVIEW_BACKGROUND,
                    edgecolor=PREVIEW_COLORS["drill"],
                    linewidth=line,
                    zorder=5,
                )
            )
        ax.add_patch(
            PathPatch(
                compound_path(part.outline),
                facecolor="none",
                edgecolor=PREVIEW_COLORS["outline"],
                linewidth=line * 1.5,
                zorder=6,
            )
        )
        # Annotation is part of what the file contains, so the preview shows
        # it.  The mockup does not: that is the finished object, which has no
        # INFO layer on it.
        for label in part.labels:
            ax.text(
                label.position[0],
                label.position[1],
                label.text,
                ha={"left": "left", "center": "center", "right": "right"}[label.align],
                va="center" if label.align == "center" else "baseline",
                rotation=label.rotation,
                rotation_mode="anchor",
                fontsize=max(label.height * points_per_mm * 0.8, 1.5),
                color=PREVIEW_COLORS["info"],
                zorder=7,
            )

    if caption:
        width, height = placed.size()
        ax.text(
            (bbox[0] + bbox[2]) / 2.0,
            bbox[1] - caption_space * 0.62,
            f"{width:.0f} x {height:.0f} x {placed.thickness:g} mm   {placed.material}",
            ha="center",
            va="center",
            fontsize=min(max(caption_space * 0.45 * points_per_mm, 5.0), 16.0),
            color=PREVIEW_COLORS["caption"],
        )

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, facecolor=fig.get_facecolor())
    return out


def _depth_colour(depth: float, deepest: float) -> tuple[float, float, float]:
    """Blend the pocket palette so deeper recesses read darker.

    Args:
        depth: This pocket's depth in mm.
        deepest: The deepest pocket in the design, in mm.

    Returns:
        An RGB triple.
    """
    from matplotlib.colors import to_rgb

    shallow = np.asarray(to_rgb(PREVIEW_COLORS["pocket_shallow"]))
    deep = np.asarray(to_rgb(PREVIEW_COLORS["pocket_deep"]))
    ratio = min(1.0, depth / deepest) if deepest > 0 else 1.0
    return tuple(shallow + (deep - shallow) * ratio)


def _circle_ring(centre: Point, radius: float, segments: int = 48) -> Ring:
    """A small polygonal circle, used where a patch is needed."""
    cx, cy = centre
    return [
        (cx + radius * math.cos(2 * math.pi * i / segments),
         cy + radius * math.sin(2 * math.pi * i / segments))
        for i in range(segments)
    ]


# --------------------------------------------------------------------------- #
# wood texture
# --------------------------------------------------------------------------- #
def _upsample(grid: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Smoothly resize a small grid to ``shape`` with bilinear interpolation."""
    height, width = shape
    gy, gx = grid.shape
    ys = np.linspace(0, gy - 1, height)
    xs = np.linspace(0, gx - 1, width)
    y0 = np.clip(np.floor(ys).astype(int), 0, gy - 2)
    x0 = np.clip(np.floor(xs).astype(int), 0, gx - 2)
    fy = (ys - y0)[:, None]
    fx = (xs - x0)[None, :]
    fy = fy * fy * (3 - 2 * fy)  # smoothstep, so octaves do not show seams
    fx = fx * fx * (3 - 2 * fx)
    g00 = grid[np.ix_(y0, x0)]
    g01 = grid[np.ix_(y0, x0 + 1)]
    g10 = grid[np.ix_(y0 + 1, x0)]
    g11 = grid[np.ix_(y0 + 1, x0 + 1)]
    return (g00 * (1 - fx) + g01 * fx) * (1 - fy) + (g10 * (1 - fx) + g11 * fx) * fy


def _fbm(rng: np.random.Generator, shape: tuple[int, int], octaves: int = 5) -> np.ndarray:
    """Fractal noise in ``[-1, 1]``, used to distort the grain."""
    total = np.zeros(shape)
    amplitude = 1.0
    cells = 3
    norm = 0.0
    for _ in range(octaves):
        total += amplitude * _upsample(rng.random((cells + 1, cells + 1)), shape)
        norm += amplitude
        amplitude *= 0.5
        cells *= 2
    return (total / norm) * 2.0 - 1.0


def _palette_for(material: str) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Pick grain colours from a free-text material description."""
    lowered = material.lower()
    for key, palette in WOOD_PALETTES.items():
        if key in lowered:
            return palette
    return _DEFAULT_PALETTE


def _wood_image(
    shape: tuple[int, int], extent: tuple[float, float, float, float], material: str, seed: int
) -> np.ndarray:
    """Generate a wood grain image covering ``extent`` in millimetres.

    Growth rings are sine waves across the grain direction, distorted by
    fractal noise so they wander the way real rings do, with a second noise
    field adding fine streaks along the grain.

    Args:
        shape: ``(rows, cols)`` of the image.
        extent: ``(x0, x1, y0, y1)`` in mm that the image covers.
        material: Material description, used to pick the palette.
        seed: Seed making the result reproducible.

    Returns:
        An ``(rows, cols, 3)`` float RGB array.
    """
    rng = np.random.default_rng(seed)
    light, dark = (np.asarray(c) for c in _palette_for(material))
    x0, x1, y0, y1 = extent
    ys = np.linspace(y0, y1, shape[0])[:, None]
    xs = np.linspace(x0, x1, shape[1])[None, :]

    angle = math.radians(rng.uniform(-6.0, 6.0))
    across = xs * math.sin(angle) + ys * math.cos(angle)
    spacing = rng.uniform(11.0, 26.0)
    wander = _fbm(rng, shape, octaves=5)
    density = _fbm(rng, shape, octaves=3)
    fine = _fbm(rng, shape, octaves=7)

    # Real boards do not have evenly spaced rings, so the spacing itself is
    # modulated, and the rings stay sinusoidal rather than being sharpened into
    # bands - hard banding is what makes procedural wood look like a cartoon.
    phase = across * (2 * math.pi / spacing) * (1.0 + 0.22 * density)
    rings = 0.5 + 0.5 * np.sin(phase + wander * 3.0)
    value = 0.60 * rings + 0.40 * (0.5 + 0.5 * fine)
    value = 0.5 + (value - 0.5) * 0.62  # keep contrast well short of the extremes
    grain = np.clip(0.16 + 0.68 * value, 0.0, 1.0)

    image = light[None, None, :] + grain[:, :, None] * (dark - light)[None, None, :]
    # A gentle diagonal light falloff keeps the panel from looking like a swatch.
    ramp = np.linspace(1.06, 0.92, shape[1])[None, :, None] * np.linspace(
        1.04, 0.94, shape[0]
    )[:, None, None]
    return np.clip(image * ramp, 0.0, 1.0)


# --------------------------------------------------------------------------- #
# listing mockup
# --------------------------------------------------------------------------- #
def render_mockup(
    design: Design,
    path: str | Path,
    px_width: int = 1600,
    dpi: int = 200,
    texture_px: int = 900,
) -> Path:
    """Render a wood-textured view of a design, for use as a listing image.

    Args:
        design: The design to draw.
        path: Destination PNG path; parent directories are created.
        px_width: Image width in pixels.
        dpi: Dots per inch used to size the figure.
        texture_px: Resolution of the generated grain along the longer axis.

    Returns:
        The written path.
    """
    placed = design.normalized()
    bbox = placed.bbox()
    width, height = bbox[2] - bbox[0], bbox[3] - bbox[1]
    span = max(width, height)
    margin = span * 0.10
    fig, ax = _figure(bbox, margin, px_width, dpi, _MOCKUP_BACKGROUND)

    extent = (bbox[0], bbox[2], bbox[1], bbox[3])
    rows = max(64, int(texture_px * height / span))
    cols = max(64, int(texture_px * width / span))
    seed = _seed_of(placed.slug)
    wood = _wood_image((rows, cols), extent, placed.material, seed)

    for part in placed.placed_parts():
        body = compound_path(part.outline, part.holes)
        _drop_shadow(ax, body, span)
        clip = PathPatch(body, facecolor="none", edgecolor="none")
        ax.add_patch(clip)
        image = ax.imshow(
            wood, extent=extent, origin="lower", interpolation="bilinear", zorder=2
        )
        image.set_clip_path(clip)

        deepest = max(placed.pocket_depths(), default=1.0)
        for pocket in part.pockets:
            recess = compound_path(pocket.ring, pocket.islands)
            patch = PathPatch(recess, facecolor="none", edgecolor="none")
            ax.add_patch(patch)
            darken = 0.88 - 0.12 * (pocket.depth / deepest if deepest else 1.0)
            shaded = ax.imshow(
                np.clip(wood * darken, 0, 1),
                extent=extent,
                origin="lower",
                interpolation="bilinear",
                zorder=3,
            )
            shaded.set_clip_path(patch)
            # An inner shadow band, clipped to the recess, does the work of
            # showing depth.  A darker fill alone is not enough on pale
            # material, where the tonal difference is small.
            inner = PathPatch(
                recess,
                facecolor="none",
                edgecolor=(0.18, 0.12, 0.07),
                alpha=0.16,
                linewidth=span / 55.0,
                zorder=4,
            )
            ax.add_patch(inner)
            inner.set_clip_path(patch)
            ax.add_patch(
                PathPatch(
                    recess,
                    facecolor="none",
                    edgecolor=(0.15, 0.10, 0.06),
                    alpha=0.45,
                    linewidth=span / 230.0,
                    zorder=5,
                )
            )
        for contour in part.engrave:
            points = np.asarray(
                list(contour.points) + ([contour.points[0]] if contour.closed else []),
                dtype=float,
            )
            ax.plot(
                points[:, 0],
                points[:, 1],
                color=(0.20, 0.13, 0.07),
                alpha=0.70,
                linewidth=span / 480.0,
                zorder=5,
            )
        for drill in part.drills:
            ax.add_patch(
                PathPatch(
                    compound_path(_circle_ring(drill.center, drill.radius)),
                    facecolor=(0.20, 0.14, 0.09),
                    edgecolor="none",
                    zorder=5,
                )
            )
        ax.add_patch(
            PathPatch(
                body,
                facecolor="none",
                edgecolor=(0.28, 0.19, 0.11),
                alpha=0.55,
                linewidth=span / 500.0,
                zorder=6,
            )
        )

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, facecolor=fig.get_facecolor())
    return out


def _drop_shadow(ax: "matplotlib.axes.Axes", body: MplPath, span: float) -> None:
    """Lay a soft shadow under a part.

    Built from stacked strokes of decreasing alpha rather than a blur, which
    keeps the render deterministic and avoids rasterising the silhouette.
    """
    offset = span * 0.006
    shifted = MplPath(body.vertices + np.array([offset, -offset]), body.codes)
    effects = [
        patheffects.withStroke(linewidth=span / 70.0, foreground=(0, 0, 0, 0.035)),
        patheffects.withStroke(linewidth=span / 130.0, foreground=(0, 0, 0, 0.045)),
        patheffects.withStroke(linewidth=span / 240.0, foreground=(0, 0, 0, 0.055)),
        patheffects.withStroke(linewidth=span / 500.0, foreground=(0, 0, 0, 0.070)),
    ]
    ax.add_patch(
        PathPatch(
            shifted,
            facecolor=(0.32, 0.28, 0.24),
            alpha=0.16,
            edgecolor="none",
            zorder=0,
            path_effects=effects,
        )
    )
