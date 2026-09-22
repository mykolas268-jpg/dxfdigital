"""Themed outlines built from parametric curves.

Every shape here is generated from an equation or from circles and Bezier
arcs combined with boolean operations.  Nothing is traced, sampled or copied
from existing artwork, which is what makes the output original and safe to
sell.  The maths used - the classic sine-cubed heart, superellipses, rose
curves, alternating-radius stars - is public and centuries old; the
proportions, the combinations and the machinability work are this project's.

Each function returns one counter-clockwise ring whose bounding box starts at
the origin and is ``width`` wide.  Shapes with inside corners are relieved to
suit the cutter before they are returned, so a star's notches and a tree's
tier junctions come back machinable rather than as sharp points that no round
cutter can reach.
"""

from __future__ import annotations

import math
from typing import Callable

from shapely.geometry import Polygon
from shapely.ops import unary_union

from ..core import geometry as geo
from ..core.geometry import Point, Ring

__all__ = [
    "SHAPES",
    "shape_names",
    "shape_ring",
    "detail_contours",
    "place_hang_hole",
    "seasonal_tags",
]


def _normalise(ring: Ring, width: float) -> Ring:
    """Scale a ring to ``width`` and move its bounding box to the origin."""
    cleaned = geo.dedupe(ring)
    x0, y0, x1, y1 = geo.bbox(cleaned)
    span = x1 - x0
    if span <= geo.EPS:
        raise ValueError("degenerate shape")
    # Deduplicate, scale, deduplicate again, and only then measure and shift.
    # Doing the translation first means the second dedupe can drop whichever
    # point was sitting exactly on the boundary, leaving the shape a few
    # microns off the origin.
    scaled = geo.dedupe(geo.scale(cleaned, factor := width / span, factor))
    bx0, by0, _, _ = geo.bbox(scaled)
    return geo.ensure_ccw(geo.translate(scaled, -bx0, -by0))


def _finish(ring: Ring, inside_radius: float, width: float) -> Ring:
    """Round the inside corners so a round cutter can follow the profile.

    Args:
        ring: The raw shape.
        inside_radius: Smallest inside radius the cutter needs, in mm.  Zero
            leaves the shape alone, which is what a laser wants.
        width: Target width in mm.

    Returns:
        The finished ring.
    """
    result = _normalise(ring, width)
    if inside_radius <= 0:
        return result
    try:
        relieved = geo.round_concave(result, inside_radius)
    except ValueError as exc:
        raise ValueError(
            f"this shape's detail is finer than a {inside_radius * 2:.2f} mm "
            f"cutter can follow at {width:g} mm across ({exc}); cut it on a "
            f"laser, make it bigger, or use a smaller cutter"
        ) from None
    return _normalise(relieved, width)


# --------------------------------------------------------------------------- #
# shapes
# --------------------------------------------------------------------------- #
def heart(width: float, inside_radius: float = 0.0, samples: int = 360) -> Ring:
    """The classic sine-cubed heart curve.

    ``x = 16 sin^3 t``, ``y = 13 cos t - 5 cos 2t - 2 cos 3t - cos 4t``.

    Args:
        width: Overall width in mm.
        inside_radius: Cutter radius to relieve the top cleft to, in mm.
        samples: Points around the curve.

    Returns:
        A closed ring.
    """
    points: Ring = []
    for index in range(samples):
        t = 2.0 * math.pi * index / samples
        points.append(
            (
                16.0 * math.sin(t) ** 3,
                13.0 * math.cos(t)
                - 5.0 * math.cos(2 * t)
                - 2.0 * math.cos(3 * t)
                - math.cos(4 * t),
            )
        )
    return _finish(points, inside_radius, width)


def star(
    width: float,
    inside_radius: float = 0.0,
    points_count: int = 5,
    inner_ratio: float = 0.48,
    tip_radius: float = 0.0,
) -> Ring:
    """An ``n``-pointed star of alternating radii.

    Args:
        width: Overall width in mm.
        inside_radius: Cutter radius to relieve the notches to, in mm.
        points_count: Number of points, 3 or more.
        inner_ratio: Inner radius as a fraction of the outer.
        tip_radius: Round the tips by this much, as a fraction of the width.

    Returns:
        A closed ring.

    Raises:
        ValueError: On fewer than 3 points or an inner ratio outside (0, 1).
    """
    if points_count < 3:
        raise ValueError(f"a star needs at least 3 points, got {points_count}")
    if not 0.0 < inner_ratio < 1.0:
        raise ValueError(f"inner_ratio must be in (0, 1), got {inner_ratio}")
    ring: Ring = []
    for index in range(2 * points_count):
        radius = 1.0 if index % 2 == 0 else inner_ratio
        angle = math.pi / 2.0 + math.pi * index / points_count
        ring.append((radius * math.cos(angle), radius * math.sin(angle)))
    if tip_radius > 0:
        ring = geo.round_convex(_normalise(ring, width), tip_radius * width)
    return _finish(ring, inside_radius, width)


def moon(width: float, inside_radius: float = 0.0, waning: float = 0.55) -> Ring:
    """A crescent, made by subtracting one circle from another.

    Args:
        width: Overall width of the crescent in mm.
        inside_radius: Cutter radius, unused here; the horns are convex.
        waning: How far the cutting circle is offset, as a fraction of the
            radius.  Larger gives a thinner crescent.

    Returns:
        A closed ring.

    Raises:
        ValueError: If the offset leaves no crescent.
    """
    if not 0.2 <= waning <= 0.95:
        raise ValueError(f"waning must be in [0.2, 0.95], got {waning}")
    outer = geo.polygon_from_ring(geo.circle_ring(0.0, 0.0, 1.0, 0.002))
    cutter = geo.polygon_from_ring(geo.circle_ring(waning, 0.12, 0.98, 0.002))
    crescent = outer.difference(cutter)
    if crescent.is_empty or crescent.geom_type != "Polygon":
        raise ValueError("the crescent did not close")
    return _finish(list(crescent.exterior.coords), inside_radius, width)


def pumpkin(width: float, inside_radius: float = 0.0, stem: float = 0.16) -> Ring:
    """A squat superellipse body with a stem on top.

    The lobes belong on the ENGRAVE layer rather than in the outline: a
    pumpkin's silhouette is a wide oval, and modulating the profile into a
    rose curve makes it read as a flower.

    Args:
        width: Overall width in mm.
        inside_radius: Cutter radius to relieve the stem junction to, in mm.
        stem: Stem height as a fraction of the body height.

    Returns:
        A closed ring.
    """
    body = geo.superellipse_ring(2.0, 1.55, 2.6, 0.0, 0.0, samples=240)
    stem_height = 1.55 / 2.0 * (1.0 + stem * 2.2)
    stalk = geo.rounded_rect_ring(0.30, stem_height, 0.12, -0.15, 0.0, tolerance=0.004)
    merged = unary_union(
        [geo.polygon_from_ring(body), geo.polygon_from_ring(stalk)]
    )
    if merged.geom_type != "Polygon":
        raise ValueError("the pumpkin stem did not merge with the body")
    return _finish(list(merged.exterior.coords), inside_radius, width)


def ghost(width: float, inside_radius: float = 0.0, scallops: int = 4) -> Ring:
    """A domed body on a scalloped hem.

    Args:
        width: Overall width in mm.
        inside_radius: Cutter radius to relieve the scallop junctions to, mm.
        scallops: Number of bumps along the hem, 2 or more.

    Returns:
        A closed ring.

    Raises:
        ValueError: On fewer than 2 scallops.
    """
    if scallops < 2:
        raise ValueError(f"a ghost needs at least 2 scallops, got {scallops}")
    half = 1.0
    body_height = 2.1
    ring: Ring = [(-half, 0.0)]
    ring.extend(
        geo.arc_points(0.0, body_height - half, half, math.pi, 0.0, tolerance=0.004)
    )
    ring.append((half, 0.0))
    # Hem: semicircular bumps walked right to left, so the ring stays CCW.
    step = 2.0 * half / scallops
    for index in range(scallops):
        centre = half - step * (index + 0.5)
        ring.extend(
            geo.arc_points(centre, 0.0, step / 2.0, 0.0, -math.pi, tolerance=0.004)
        )
    return _finish(geo.dedupe(ring), inside_radius, width)


def tree(width: float, inside_radius: float = 0.0, tiers: int = 3) -> Ring:
    """A tiered fir with a trunk.

    Args:
        width: Overall width in mm.
        inside_radius: Cutter radius to relieve the tier junctions to, mm.
        tiers: Number of branch tiers, 2 or more.

    Returns:
        A closed ring.

    Raises:
        ValueError: On fewer than 2 tiers.
    """
    if tiers < 2:
        raise ValueError(f"a tree needs at least 2 tiers, got {tiers}")
    trunk_w, trunk_h = 0.22, 0.28
    tier_h = 1.0 / tiers
    right: Ring = [(trunk_w / 2.0, 0.0), (trunk_w / 2.0, trunk_h)]
    for index in range(tiers):
        base = trunk_h + index * tier_h
        spread = 1.0 - index * (0.62 / max(tiers - 1, 1))
        right.append((spread / 2.0, base))
        right.append((spread / 2.0 * 0.50, base + tier_h * 0.72))
    right.append((0.0, trunk_h + tiers * tier_h))
    left = [(-x, y) for x, y in reversed(right)]
    return _finish(geo.dedupe(right + left), inside_radius, width)


#: Widest a single toe may be, as a fraction of the paw's overall width.
#: Past this a three-toed paw reads as a clover rather than a print.
MAX_TOE_WIDTH: float = 0.50


def paw(width: float, inside_radius: float = 0.0, toes: int = 4) -> Ring:
    """A paw print: a pad with toes splayed on an arc above it.

    The toes must overlap the pad, because the result has to be one closed
    contour that cuts as one piece.  Getting that overlap right is the whole
    problem: sink the toes far into the pad and the notches between them
    vanish, leaving something that reads as a flower; lift them clear and the
    piece falls into five.  So each toe is dropped onto the pad's own top edge
    and sunk a fixed, small amount into it, which keeps the notches as deep as
    a single contour allows, and the inner pair is lifted above the outer pair
    the way a real print splays.

    Args:
        width: Overall width in mm.
        inside_radius: Cutter radius to relieve the toe junctions to, mm.
        toes: Number of toes, 3 or more.

    Returns:
        A closed ring.

    Raises:
        ValueError: On fewer than 3 toes, or if the toes do not all merge with
            the pad - which would mean shipping a design that falls apart on
            the machine.
    """
    if toes < 3:
        raise ValueError(f"a paw needs at least 3 toes, got {toes}")
    pad_w, pad_h, pad_cy, exponent = 1.10, 1.00, -0.16, 2.6
    a, b = pad_w / 2.0, pad_h / 2.0
    pad = geo.superellipse_ring(pad_w, pad_h, exponent, 0.0, pad_cy, samples=220)
    shapes = [geo.polygon_from_ring(pad)]

    # Toes reach past the pad's own half width so the outer pair stands proud
    # of it rather than merging into its side, but not so far that it loses
    # contact.  Both conditions are one equation: the outermost toe's centre
    # sits OVERHANG toe-widths outside the pad edge, and the toes are spread
    # to leave a gap of a tenth of their pitch between neighbours.  Solving
    # the two together, rather than picking a reach and hoping, is what keeps
    # three, four and five toes all merging.
    overhang, fill = 0.35, 0.90
    reach = a / (1.0 - overhang * 2.0 * fill / (toes - 1))
    toe_w = 2.0 * fill * reach / (toes - 1)
    if toe_w > MAX_TOE_WIDTH:
        toe_w = MAX_TOE_WIDTH
        reach = a + overhang * toe_w
    toe_h = toe_w * 1.32
    # A real print splays: the outer toes ride lower than the inner ones.  The
    # pad's own curvature does most of that, and the rest is a smaller bite
    # into the pad for the inner toes - never so small that one floats free,
    # which is the failure this whole shape is one bad number away from.
    outer_sink, inner_sink = 0.10, 0.045

    def pad_top(x: float) -> float:
        """Top of the pad directly above ``x``, from the superellipse."""
        ratio = min(abs(x) / a, 1.0)
        return pad_cy + b * (1.0 - ratio**exponent) ** (1.0 / exponent)

    for index in range(toes):
        fraction = index / (toes - 1)
        offset = 1.0 - 2.0 * fraction
        cx = reach * offset
        sink = inner_sink + (outer_sink - inner_sink) * abs(offset)
        cy = pad_top(cx) + toe_h / 2.0 - sink
        shapes.append(
            geo.polygon_from_ring(
                geo.superellipse_ring(toe_w, toe_h, 2.3, cx, cy, samples=140)
            )
        )
    merged = unary_union(shapes)
    if merged.geom_type != "Polygon":
        raise ValueError(
            f"{toes} toes do not all merge with the pad; the paw would fall "
            f"into pieces"
        )
    return _finish(list(merged.exterior.coords), inside_radius, width)


def leaf(width: float, inside_radius: float = 0.0, fullness: float = 0.42) -> Ring:
    """A pointed leaf from two mirrored cubic Bezier curves.

    Args:
        width: Overall width in mm.
        inside_radius: Cutter radius, unused; both tips are convex.
        fullness: How far the control points push out, as a fraction of the
            length.

    Returns:
        A closed ring.
    """
    tip_a: Point = (0.0, 0.0)
    tip_b: Point = (1.0, 0.0)
    upper = geo.bezier_points(
        tip_a, (0.22, fullness), (0.78, fullness), tip_b, samples=90
    )
    lower = geo.bezier_points(
        tip_b, (0.78, -fullness), (0.22, -fullness), tip_a, samples=90
    )
    return _finish(geo.dedupe(upper + lower[1:]), inside_radius, width)


def snowflake(
    width: float, inside_radius: float = 0.0, arms: int = 6, arm_width: float = 0.15
) -> Ring:
    """A radially symmetric flake built from one arm, rotated.

    Thin arms make this a laser shape: a router will report it as
    unmachinable unless the arms are wider than the cutter.

    Args:
        width: Overall width in mm.
        inside_radius: Cutter radius to relieve the arm junctions to, mm.
        arms: Number of arms, 3 or more.
        arm_width: Arm width as a fraction of the overall width.

    Returns:
        A closed ring.

    Raises:
        ValueError: On fewer than 3 arms, or if the flake falls apart.
    """
    if arms < 3:
        raise ValueError(f"a snowflake needs at least 3 arms, got {arms}")
    half = arm_width / 2.0
    spine = geo.rect_ring(1.0, arm_width, 0.0, -half)
    branch_a = geo.rect_ring(0.30, arm_width * 0.75, 0.0, -half * 0.75)
    pieces = [geo.polygon_from_ring(spine)]
    for offset, angle in ((0.42, 55.0), (0.42, -55.0), (0.70, 40.0), (0.70, -40.0)):
        rotated = geo.rotate(branch_a, angle)
        pieces.append(geo.polygon_from_ring(geo.translate(rotated, offset, 0.0)))
    arm = unary_union(pieces)
    # The first arm points straight up, so the flake hangs level and the hub
    # sits on the vertical centreline where a hanging hole can find it.
    flake = unary_union(
        [
            geo.polygon_from_ring(
                geo.rotate(list(arm.exterior.coords), 90.0 + 360.0 * index / arms)
            )
            for index in range(arms)
        ]
    )
    if flake.geom_type != "Polygon":
        raise ValueError("the snowflake arms do not meet at the centre")
    return _finish(list(flake.exterior.coords), inside_radius, width)


def detail_contours(
    name: str, width: float, inset: float = 0.0
) -> list[tuple[list[Point], bool]]:
    """Engraved decoration belonging to a shape.

    Detail that would weaken or complicate the cut goes here instead of into
    the outline: a pumpkin's lobes and a leaf's veins are lines on the surface,
    not changes to the silhouette.

    Args:
        name: A shape name.
        width: The shape's overall width in mm.
        inset: Border inset in mm for shapes with no specific detail; 0 gives
            no border.

    Returns:
        ``[(points, closed), ...]`` ready for the ENGRAVE layer.

    Raises:
        KeyError: If the shape does not exist.
    """
    if name not in SHAPES:
        raise KeyError(f"unknown shape {name!r}")
    ring = shape_ring(name, width)
    x0, y0, x1, y1 = geo.bbox(ring)
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    height = y1 - y0
    region = geo.polygon_from_ring(ring)
    out: list[tuple[list[Point], bool]] = []

    def clipped(points: list[Point], margin: float) -> None:
        """Keep the part of a line that stays inside the shape."""
        from shapely.geometry import LineString

        inner = region.buffer(-margin)
        if inner.is_empty:
            return
        piece = inner.intersection(LineString(points))
        for part in getattr(piece, "geoms", [piece]):
            trimmed = geo.dedupe(list(part.coords), closed=False)
            if len(trimmed) >= 2:
                out.append((trimmed, False))

    margin = max(width * 0.06, 3.0)
    if name == "pumpkin":
        for offset in (-0.30, -0.10, 0.10, 0.30):
            arc = [
                (
                    cx + offset * width * (1.0 - 0.55 * (t / 40.0 - 0.5) ** 2 * 4),
                    y0 + height * t / 40.0,
                )
                for t in range(41)
            ]
            clipped(arc, margin)
    elif name == "leaf":
        clipped([(x0, cy), (x1, cy)], margin)
        for step in (0.25, 0.45, 0.65, 0.85):
            base = x0 + (x1 - x0) * step
            clipped([(base, cy), (base + width * 0.10, cy + height * 0.30)], margin)
            clipped([(base, cy), (base + width * 0.10, cy - height * 0.30)], margin)
    elif name == "ghost":
        for side in (-1.0, 1.0):
            out.append(
                (
                    geo.circle_ring(
                        cx + side * width * 0.17, y0 + height * 0.70, width * 0.075
                    ),
                    True,
                )
            )
        out.append(
            (
                geo.superellipse_ring(
                    width * 0.16, height * 0.10, 2.2, cx, y0 + height * 0.52
                ),
                True,
            )
        )
    elif inset > 0:
        for border in geo.offset_ring(ring, -inset, join="round"):
            out.append((border, True))
    return out


#: Every shape, by name.
SHAPES: dict[str, Callable[..., Ring]] = {
    "heart": heart,
    "star": star,
    "moon": moon,
    "pumpkin": pumpkin,
    "ghost": ghost,
    "tree": tree,
    "paw": paw,
    "leaf": leaf,
    "snowflake": snowflake,
}

#: Which season or occasion each shape belongs to, for naming and bundling.
seasonal_tags: dict[str, str] = {
    "heart": "valentine",
    "star": "christmas",
    "moon": "halloween",
    "pumpkin": "halloween",
    "ghost": "halloween",
    "tree": "christmas",
    "paw": "pets",
    "leaf": "autumn",
    "snowflake": "christmas",
}


def place_hang_hole(
    ring: Ring, diameter: float, min_wall: float, steps: int = 40
) -> Ring:
    """Find somewhere near the top of a shape for a hanging hole.

    The hole is walked down the vertical centreline from the top until it
    clears the profile by ``min_wall`` everywhere.  Fixing it at a nominal
    height instead means it falls outside a heart's cleft and off the end of a
    crescent.

    Args:
        ring: The shape outline.
        diameter: Hole diameter in mm.
        min_wall: Material required around the hole, in mm.
        steps: How many positions to try.

    Returns:
        A closed ring for the hole.

    Raises:
        ValueError: If no position works, which means the shape is too small
            or too thin to hang.
    """
    if diameter <= 0:
        raise ValueError(f"hang hole diameter must be > 0, got {diameter}")
    region = geo.polygon_from_ring(ring)
    boundary = region.exterior
    x0, y0, x1, y1 = geo.bbox(ring)
    radius = diameter / 2.0
    centre_x = (x0 + x1) / 2.0
    top = y1 - radius - min_wall
    bottom = y0 + (y1 - y0) * 0.30
    reach = max((x1 - x0) / 2.0 - radius - min_wall, 0.0)
    # Sweep across the width as well as down: a crescent's thick part is not
    # on its centreline, so a centreline-only search finds nowhere to hang it.
    offsets = sorted(
        {round(reach * k / 6.0, 6) for k in range(-6, 7)}, key=abs
    )
    for index in range(steps):
        y = top - (top - bottom) * index / max(steps - 1, 1)
        for offset in offsets:
            hole = geo.circle_ring(centre_x + offset, y, radius)
            disc = geo.polygon_from_ring(hole)
            if region.contains(disc) and boundary.distance(disc) >= min_wall - 1e-6:
                return hole
    raise ValueError(
        f"a {diameter:g} mm hanging hole does not fit in this shape with "
        f"{min_wall:g} mm of material around it; make it bigger or drop the hole"
    )


def shape_names() -> list[str]:
    """Every available shape name, sorted."""
    return sorted(SHAPES)


def shape_ring(name: str, width: float, inside_radius: float = 0.0, **kwargs) -> Ring:
    """Build one shape by name.

    Args:
        name: A name from :func:`shape_names`.
        width: Overall width in mm.
        inside_radius: Cutter radius to relieve inside corners to, in mm; 0
            for a laser.
        **kwargs: Shape-specific options.

    Returns:
        A closed ring, bounding box at the origin.

    Raises:
        KeyError: If the shape does not exist.
        ValueError: If the options are invalid for that shape.
    """
    try:
        builder = SHAPES[name]
    except KeyError:
        raise KeyError(
            f"unknown shape {name!r}; available: {', '.join(shape_names())}"
        ) from None
    return builder(width, inside_radius, **kwargs)
