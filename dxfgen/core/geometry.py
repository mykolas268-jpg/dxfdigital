"""Parametric 2-D geometry primitives for machine-ready artwork.

Conventions
-----------
* Every coordinate is a millimetre float.  There are no other units anywhere.
* A *ring* is a closed boundary stored as ``list[(x, y)]`` **without** a
  repeated closing point; exporters close the contour themselves.  Keeping the
  closing point implicit is what stops duplicate zero-length segments from
  ever reaching a DXF.
* Rings returned by this module are counter-clockwise (positive signed area),
  so "material interior is on the left of travel" holds everywhere and vertex
  convexity is well defined.
* Curves are tessellated into straight segments with a bounded chord
  deviation (:data:`ARC_TOLERANCE`).  No splines are ever produced, which is
  what keeps output free of "spline not supported" warnings in Vectric,
  LightBurn and Fusion.
"""

from __future__ import annotations

import math
from typing import Iterable, Literal, Sequence

from shapely.geometry import LinearRing, MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

Point = tuple[float, float]
Ring = list[Point]

#: Maximum chord deviation when tessellating a curve, in mm.
ARC_TOLERANCE: float = 0.05
#: Segments shorter than this are considered zero-length and removed.
MIN_SEGMENT: float = 0.01
#: Generic float comparison epsilon.
EPS: float = 1e-9
#: Quadrant segment count used for shapely buffer operations.
BUFFER_QUAD_SEGS: int = 32
#: Erosion relief used by :func:`opening`, in mm.  Tessellated arcs sit up to
#: :data:`ARC_TOLERANCE` inside the true curve, so a tighter slack than that
#: would measure the tessellation rather than the tool.
EROSION_SLACK: float = ARC_TOLERANCE

JoinStyle = Literal["round", "mitre", "bevel"]
ReliefStyle = Literal["dogbone", "tbone"]
CornerKind = Literal["convex", "concave", "all"]
RegionKind = Literal["cutout", "part"]

__all__ = [
    "ARC_TOLERANCE",
    "MIN_SEGMENT",
    "EPS",
    "EROSION_SLACK",
    "Point",
    "Ring",
    "signed_area",
    "is_ccw",
    "ensure_ccw",
    "ensure_cw",
    "bbox",
    "bbox_of",
    "size_of",
    "centroid",
    "perimeter",
    "dedupe",
    "translate",
    "rotate",
    "scale",
    "mirror_x",
    "mirror_y",
    "arc_segment_count",
    "arc_points",
    "circle_ring",
    "rect_ring",
    "rounded_rect_ring",
    "stadium_ring",
    "superellipse_ring",
    "bezier_points",
    "polygon_from_ring",
    "rings_of",
    "offset_ring",
    "classify_convex",
    "fillet_ring",
    "round_convex",
    "round_concave",
    "offset_centreline",
    "curved_slot",
    "relief_positions",
    "apply_relief",
    "slot_ring",
    "joint_slot_width",
    "suggest_finger_count",
    "finger_joint_edge",
    "kerf_compensate_ring",
    "opening",
    "closing",
    "excess_zones",
    "residual_region",
    "residual_thickness",
    "unreachable_zones",
    "text_polygons",
    "text_contours",
]


# --------------------------------------------------------------------------- #
# vector helpers
# --------------------------------------------------------------------------- #
def _sub(a: Point, b: Point) -> Point:
    return (a[0] - b[0], a[1] - b[1])


def _add(a: Point, b: Point) -> Point:
    return (a[0] + b[0], a[1] + b[1])


def _mul(a: Point, k: float) -> Point:
    return (a[0] * k, a[1] * k)


def _norm(a: Point) -> float:
    return math.hypot(a[0], a[1])


def _unit(a: Point) -> Point:
    n = _norm(a)
    if n < EPS:
        raise ValueError("cannot normalise a zero-length vector")
    return (a[0] / n, a[1] / n)


def _cross(a: Point, b: Point) -> float:
    return a[0] * b[1] - a[1] * b[0]


def _dot(a: Point, b: Point) -> float:
    return a[0] * b[0] + a[1] * b[1]


# --------------------------------------------------------------------------- #
# ring measurement
# --------------------------------------------------------------------------- #
def signed_area(ring: Sequence[Point]) -> float:
    """Return the signed area of a ring; positive means counter-clockwise.

    Args:
        ring: Ring vertices, closing point optional.

    Returns:
        Signed area in mm^2.
    """
    if len(ring) < 3:
        return 0.0
    total = 0.0
    n = len(ring)
    for i in range(n):
        x0, y0 = ring[i]
        x1, y1 = ring[(i + 1) % n]
        total += x0 * y1 - x1 * y0
    return total / 2.0


def is_ccw(ring: Sequence[Point]) -> bool:
    """Return ``True`` when ``ring`` winds counter-clockwise."""
    return signed_area(ring) > 0.0


def ensure_ccw(ring: Sequence[Point]) -> Ring:
    """Return ``ring`` wound counter-clockwise."""
    pts = list(ring)
    return pts if is_ccw(pts) else pts[::-1]


def ensure_cw(ring: Sequence[Point]) -> Ring:
    """Return ``ring`` wound clockwise."""
    pts = list(ring)
    return pts[::-1] if is_ccw(pts) else pts


def bbox(points: Iterable[Point]) -> tuple[float, float, float, float]:
    """Return ``(xmin, ymin, xmax, ymax)`` of ``points``.

    Raises:
        ValueError: If ``points`` is empty.
    """
    xs: list[float] = []
    ys: list[float] = []
    for x, y in points:
        xs.append(x)
        ys.append(y)
    if not xs:
        raise ValueError("bbox of an empty point set is undefined")
    return (min(xs), min(ys), max(xs), max(ys))


def bbox_of(rings: Iterable[Sequence[Point]]) -> tuple[float, float, float, float]:
    """Return the combined bounding box of several rings."""
    pts: list[Point] = []
    for ring in rings:
        pts.extend(ring)
    return bbox(pts)


def size_of(ring: Sequence[Point]) -> tuple[float, float]:
    """Return ``(width, height)`` of a ring's bounding box."""
    x0, y0, x1, y1 = bbox(ring)
    return (x1 - x0, y1 - y0)


def centroid(ring: Sequence[Point]) -> Point:
    """Return the area centroid of a ring.

    Falls back to the vertex average for degenerate (zero area) rings.
    """
    area = signed_area(ring)
    if abs(area) < EPS:
        n = max(len(ring), 1)
        return (sum(p[0] for p in ring) / n, sum(p[1] for p in ring) / n)
    cx = cy = 0.0
    n = len(ring)
    for i in range(n):
        x0, y0 = ring[i]
        x1, y1 = ring[(i + 1) % n]
        cross = x0 * y1 - x1 * y0
        cx += (x0 + x1) * cross
        cy += (y0 + y1) * cross
    factor = 1.0 / (6.0 * area)
    return (cx * factor, cy * factor)


def perimeter(ring: Sequence[Point], closed: bool = True) -> float:
    """Return the total edge length of a polyline or ring."""
    if len(ring) < 2:
        return 0.0
    total = 0.0
    for i in range(len(ring) - 1):
        total += _norm(_sub(ring[i + 1], ring[i]))
    if closed:
        total += _norm(_sub(ring[0], ring[-1]))
    return total


def dedupe(ring: Sequence[Point], tol: float = MIN_SEGMENT, closed: bool = True) -> Ring:
    """Drop consecutive duplicate points, and the closing duplicate.

    This is the single guard against zero-length segments, which CAM software
    reports as "duplicate vector" warnings.

    Args:
        ring: Input vertices.
        tol: Points closer together than this are merged.
        closed: When ``True`` the last point is also compared to the first.

    Returns:
        A cleaned vertex list.
    """
    out: Ring = []
    for p in ring:
        if not out or _norm(_sub(p, out[-1])) > tol:
            out.append((float(p[0]), float(p[1])))
    if closed:
        while len(out) > 1 and _norm(_sub(out[0], out[-1])) <= tol:
            out.pop()
    return out


# --------------------------------------------------------------------------- #
# transforms
# --------------------------------------------------------------------------- #
def translate(ring: Sequence[Point], dx: float, dy: float) -> Ring:
    """Return ``ring`` shifted by ``(dx, dy)``."""
    return [(x + dx, y + dy) for x, y in ring]


def rotate(ring: Sequence[Point], degrees: float, origin: Point = (0.0, 0.0)) -> Ring:
    """Return ``ring`` rotated about ``origin`` by ``degrees`` counter-clockwise."""
    rad = math.radians(degrees)
    c, s = math.cos(rad), math.sin(rad)
    ox, oy = origin
    out: Ring = []
    for x, y in ring:
        dx, dy = x - ox, y - oy
        out.append((ox + dx * c - dy * s, oy + dx * s + dy * c))
    return out


def scale(
    ring: Sequence[Point], kx: float, ky: float | None = None, origin: Point = (0.0, 0.0)
) -> Ring:
    """Return ``ring`` scaled about ``origin``.

    Args:
        ring: Input vertices.
        kx: X scale factor.
        ky: Y scale factor; defaults to ``kx`` for a uniform scale.
        origin: Fixed point of the scale.
    """
    ky = kx if ky is None else ky
    ox, oy = origin
    return [(ox + (x - ox) * kx, oy + (y - oy) * ky) for x, y in ring]


def mirror_x(ring: Sequence[Point], axis: float = 0.0) -> Ring:
    """Mirror across the vertical line ``x = axis`` (winding is reversed)."""
    return [(2 * axis - x, y) for x, y in ring][::-1]


def mirror_y(ring: Sequence[Point], axis: float = 0.0) -> Ring:
    """Mirror across the horizontal line ``y = axis`` (winding is reversed)."""
    return [(x, 2 * axis - y) for x, y in ring][::-1]


# --------------------------------------------------------------------------- #
# curve primitives
# --------------------------------------------------------------------------- #
def arc_segment_count(
    radius: float, sweep: float, tolerance: float = ARC_TOLERANCE
) -> int:
    """Segment count needed to approximate an arc within ``tolerance``.

    Args:
        radius: Arc radius in mm.
        sweep: Absolute sweep angle in radians.
        tolerance: Maximum chord deviation in mm.

    Returns:
        At least 1 segment.
    """
    sweep = abs(sweep)
    if radius <= EPS or sweep <= EPS:
        return 1
    ratio = max(-1.0, min(1.0, 1.0 - tolerance / radius))
    max_step = 2.0 * math.acos(ratio)
    if max_step <= EPS:
        return 1
    return max(1, int(math.ceil(sweep / max_step)))


def arc_points(
    cx: float,
    cy: float,
    radius: float,
    start: float,
    end: float,
    tolerance: float = ARC_TOLERANCE,
    include_last: bool = True,
) -> Ring:
    """Tessellate a circular arc.

    Args:
        cx: Centre x.
        cy: Centre y.
        radius: Radius in mm.
        start: Start angle in radians.
        end: End angle in radians; may be smaller than ``start`` for a
            clockwise sweep.
        tolerance: Maximum chord deviation in mm.
        include_last: Whether to emit the end point.

    Returns:
        Points along the arc from ``start`` to ``end``.
    """
    sweep = end - start
    n = arc_segment_count(radius, sweep, tolerance)
    pts: Ring = []
    last = n if include_last else n - 1
    for i in range(last + 1):
        a = start + sweep * (i / n)
        pts.append((cx + radius * math.cos(a), cy + radius * math.sin(a)))
    return pts


def circle_ring(
    cx: float, cy: float, radius: float, tolerance: float = ARC_TOLERANCE
) -> Ring:
    """Return a counter-clockwise polygonal approximation of a circle.

    Real circles are exported as DXF ``CIRCLE`` entities; this is for boolean
    work where a polygon is required.

    Raises:
        ValueError: If ``radius`` is not positive.
    """
    if radius <= 0:
        raise ValueError(f"circle radius must be > 0, got {radius}")
    return arc_points(cx, cy, radius, 0.0, 2.0 * math.pi, tolerance, include_last=False)


def rect_ring(width: float, height: float, x0: float = 0.0, y0: float = 0.0) -> Ring:
    """Return a counter-clockwise rectangle with its corner at ``(x0, y0)``.

    Raises:
        ValueError: If either dimension is not positive.
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"rectangle needs positive dimensions, got {width}x{height}")
    return [
        (x0, y0),
        (x0 + width, y0),
        (x0 + width, y0 + height),
        (x0, y0 + height),
    ]


def rounded_rect_ring(
    width: float,
    height: float,
    radius: float | Sequence[float],
    x0: float = 0.0,
    y0: float = 0.0,
    tolerance: float = ARC_TOLERANCE,
) -> Ring:
    """Return a counter-clockwise rectangle with rounded corners.

    Args:
        width: Overall width in mm.
        height: Overall height in mm.
        radius: One radius for all corners, or four radii ordered
            bottom-left, bottom-right, top-right, top-left.
        x0: Bounding box left edge.
        y0: Bounding box bottom edge.
        tolerance: Arc chord tolerance in mm.

    Returns:
        The ring, counter-clockwise, with no duplicate points.

    Raises:
        ValueError: On non-positive dimensions, negative radii, or radii that
            do not fit the rectangle.
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"rectangle needs positive dimensions, got {width}x{height}")
    if isinstance(radius, (int, float)):
        radii = [float(radius)] * 4
    else:
        radii = [float(r) for r in radius]
        if len(radii) != 4:
            raise ValueError("radius sequence must hold exactly four values")
    if any(r < 0 for r in radii):
        raise ValueError(f"corner radii must be >= 0, got {radii}")
    limit = min(width, height) / 2.0
    if any(r > limit + EPS for r in radii):
        raise ValueError(
            f"corner radius {max(radii)} does not fit a {width}x{height} rectangle "
            f"(max {limit})"
        )

    bl, br, tr, tl = radii
    # Corner arcs walked counter-clockwise, starting at the bottom-left.
    corners = [
        ((x0 + bl, y0 + bl), bl, math.pi, 1.5 * math.pi),
        ((x0 + width - br, y0 + br), br, 1.5 * math.pi, 2.0 * math.pi),
        ((x0 + width - tr, y0 + height - tr), tr, 0.0, math.pi / 2),
        ((x0 + tl, y0 + height - tl), tl, math.pi / 2, math.pi),
    ]
    ring: Ring = []
    for (cx, cy), r, a0, a1 in corners:
        if r <= EPS:
            ring.append((cx, cy))
        else:
            ring.extend(arc_points(cx, cy, r, a0, a1, tolerance))
    return dedupe(ring)


def stadium_ring(
    length: float,
    width: float,
    x0: float = 0.0,
    y0: float = 0.0,
    tolerance: float = ARC_TOLERANCE,
) -> Ring:
    """Return a fully rounded slot ("stadium") shape.

    This is the workhorse handle / slot outline: a rectangle capped with
    semicircles, so a router or laser never meets a sharp inside corner.

    Args:
        length: Overall length along x in mm, must exceed ``width``.
        width: Overall width along y in mm.
        x0: Bounding box left edge.
        y0: Bounding box bottom edge.
        tolerance: Arc chord tolerance in mm.

    Raises:
        ValueError: If ``length`` is smaller than ``width``.
    """
    if length < width - EPS:
        raise ValueError(f"stadium length {length} must be >= width {width}")
    return rounded_rect_ring(length, width, width / 2.0, x0, y0, tolerance)


def superellipse_ring(
    width: float,
    height: float,
    exponent: float = 2.5,
    cx: float = 0.0,
    cy: float = 0.0,
    samples: int = 180,
) -> Ring:
    """Return a superellipse (Lame curve) ring.

    ``exponent`` 2 gives an ellipse, larger values approach a rounded
    rectangle - the proportion knob used for organic tray and coaster
    outlines.

    Args:
        width: Overall width in mm.
        height: Overall height in mm.
        exponent: Curve exponent, must be > 0.
        cx: Centre x.
        cy: Centre y.
        samples: Vertex count, must be >= 16.

    Raises:
        ValueError: On non-positive dimensions, exponent, or too few samples.
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"superellipse needs positive size, got {width}x{height}")
    if exponent <= 0:
        raise ValueError(f"exponent must be > 0, got {exponent}")
    if samples < 16:
        raise ValueError(f"need at least 16 samples, got {samples}")
    a, b = width / 2.0, height / 2.0
    ring: Ring = []
    for i in range(samples):
        t = 2.0 * math.pi * i / samples
        ct, st = math.cos(t), math.sin(t)
        x = a * math.copysign(abs(ct) ** (2.0 / exponent), ct)
        y = b * math.copysign(abs(st) ** (2.0 / exponent), st)
        ring.append((cx + x, cy + y))
    return dedupe(ring)


def bezier_points(
    p0: Point, p1: Point, p2: Point, p3: Point, samples: int = 24
) -> Ring:
    """Tessellate a cubic Bezier curve from ``p0`` to ``p3``.

    Args:
        p0: Start point.
        p1: First control point.
        p2: Second control point.
        p3: End point.
        samples: Number of segments, must be >= 1.

    Returns:
        ``samples + 1`` points including both ends.

    Raises:
        ValueError: If ``samples`` < 1.
    """
    if samples < 1:
        raise ValueError(f"samples must be >= 1, got {samples}")
    pts: Ring = []
    for i in range(samples + 1):
        t = i / samples
        mt = 1.0 - t
        w0 = mt * mt * mt
        w1 = 3.0 * mt * mt * t
        w2 = 3.0 * mt * t * t
        w3 = t * t * t
        pts.append(
            (
                p0[0] * w0 + p1[0] * w1 + p2[0] * w2 + p3[0] * w3,
                p0[1] * w0 + p1[1] * w1 + p2[1] * w2 + p3[1] * w3,
            )
        )
    return pts


# --------------------------------------------------------------------------- #
# shapely bridge
# --------------------------------------------------------------------------- #
def polygon_from_ring(
    ring: Sequence[Point], holes: Sequence[Sequence[Point]] | None = None
) -> Polygon:
    """Build a shapely polygon from a ring and optional holes.

    Args:
        ring: Outer boundary.
        holes: Interior boundaries.

    Returns:
        A shapely :class:`~shapely.geometry.Polygon`, not necessarily valid -
        validity is the validator's business, not this function's.

    Raises:
        ValueError: If ``ring`` has fewer than 3 distinct points.
    """
    clean = dedupe(ring)
    if len(clean) < 3:
        raise ValueError(f"a polygon needs >= 3 distinct points, got {len(clean)}")
    hole_rings = [dedupe(h) for h in (holes or [])]
    return Polygon(clean, [h for h in hole_rings if len(h) >= 3])


def rings_of(geom: BaseGeometry) -> list[Ring]:
    """Flatten any shapely geometry to a list of rings.

    Exterior and interior rings are returned together; each is already
    de-duplicated and counter-clockwise for exteriors, clockwise for holes,
    matching the DXF convention this project exports.
    """
    out: list[Ring] = []
    if geom.is_empty:
        return out
    polys: list[Polygon]
    if isinstance(geom, Polygon):
        polys = [geom]
    elif isinstance(geom, MultiPolygon):
        polys = list(geom.geoms)
    elif hasattr(geom, "geoms"):
        polys = [g for g in geom.geoms if isinstance(g, Polygon)]
    else:
        return out
    for poly in polys:
        out.append(ensure_ccw(dedupe(list(poly.exterior.coords))))
        for interior in poly.interiors:
            out.append(ensure_cw(dedupe(list(interior.coords))))
    return out


def offset_ring(
    ring: Sequence[Point],
    distance: float,
    join: JoinStyle = "round",
    mitre_limit: float = 2.0,
) -> list[Ring]:
    """Offset a ring outward (positive) or inward (negative).

    Args:
        ring: Input boundary.
        distance: Offset in mm; positive grows the enclosed region.
        join: Corner treatment; ``"mitre"`` preserves sharp corners, which is
            what kerf compensation on a tabbed edge needs.
        mitre_limit: Mitre ratio clamp.

    Returns:
        Zero or more rings.  An inward offset can erase or split a region, so
        callers must handle an empty result.
    """
    poly = polygon_from_ring(ring)
    if not poly.is_valid:
        poly = poly.buffer(0)
    grown = poly.buffer(
        distance,
        quad_segs=BUFFER_QUAD_SEGS,
        join_style=join,
        mitre_limit=mitre_limit,
    )
    return rings_of(grown)


def kerf_compensate_ring(
    ring: Sequence[Point], kerf: float, outward: bool
) -> list[Ring]:
    """Apply laser kerf compensation to one ring.

    A laser removes ``kerf`` of material centred on the drawn line.  Growing
    an outer profile by ``kerf / 2`` and shrinking an interior cutout by the
    same amount means every cut edge lands on its nominal dimension, so
    nominal-to-nominal joints press-fit without per-feature fudge factors.

    Args:
        ring: Boundary to compensate.
        kerf: Laser kerf width in mm, must be >= 0.
        outward: ``True`` for an outer profile, ``False`` for an interior
            cutout.

    Returns:
        The compensated ring(s).

    Raises:
        ValueError: If ``kerf`` is negative.
    """
    if kerf < 0:
        raise ValueError(f"kerf must be >= 0, got {kerf}")
    if kerf < EPS:
        return [dedupe(ring)]
    delta = kerf / 2.0 if outward else -kerf / 2.0
    return offset_ring(ring, delta, join="mitre")


# --------------------------------------------------------------------------- #
# corners: fillets and tool relief
# --------------------------------------------------------------------------- #
def classify_convex(ring: Sequence[Point]) -> list[bool]:
    """Classify each vertex of a ring as convex or reflex.

    The ring is interpreted counter-clockwise, so "convex" means the interior
    angle at that vertex is less than 180 degrees.

    Args:
        ring: Ring vertices, at least 3.

    Returns:
        One boolean per vertex, ``True`` for convex.

    Raises:
        ValueError: If the ring has fewer than 3 points.
    """
    pts = ensure_ccw(dedupe(ring))
    n = len(pts)
    if n < 3:
        raise ValueError("need at least 3 points to classify convexity")
    flags: list[bool] = []
    for i in range(n):
        a, b, c = pts[i - 1], pts[i], pts[(i + 1) % n]
        flags.append(_cross(_sub(b, a), _sub(c, b)) > 0.0)
    return flags


def _corner_fillet(
    a: Point,
    b: Point,
    c: Point,
    radius: float,
    tolerance: float,
    clamp: bool,
    min_turn_deg: float = 1.0,
) -> Ring | None:
    """Return arc points replacing vertex ``b``, or ``None`` if impossible."""
    try:
        v1 = _unit(_sub(a, b))
        v2 = _unit(_sub(c, b))
    except ValueError:
        return None
    dot = max(-1.0, min(1.0, _dot(v1, v2)))
    phi = math.acos(dot)
    if phi < math.radians(1.0) or phi > math.pi - math.radians(min_turn_deg):
        return None  # spike or near-collinear: nothing sensible to fillet
    half = phi / 2.0
    r = radius
    tangent = r / math.tan(half)
    max_tangent = min(_norm(_sub(a, b)), _norm(_sub(c, b))) / 2.0
    if tangent > max_tangent:
        if not clamp:
            return None
        r = max_tangent * math.tan(half)
        if r < EPS:
            return None
        tangent = max_tangent
    bis = _unit(_add(v1, v2))
    centre = _add(b, _mul(bis, r / math.sin(half)))
    t1 = _add(b, _mul(v1, tangent))
    t2 = _add(b, _mul(v2, tangent))
    a1 = math.atan2(t1[1] - centre[1], t1[0] - centre[0])
    a2 = math.atan2(t2[1] - centre[1], t2[0] - centre[0])
    sweep = a2 - a1
    while sweep > math.pi:
        sweep -= 2.0 * math.pi
    while sweep < -math.pi:
        sweep += 2.0 * math.pi
    return arc_points(centre[0], centre[1], r, a1, a1 + sweep, tolerance)


def fillet_ring(
    ring: Sequence[Point],
    radius: float,
    corners: CornerKind = "all",
    tolerance: float = ARC_TOLERANCE,
    clamp: bool = True,
    min_turn_deg: float = 5.0,
) -> Ring:
    """Round the corners of a ring.

    Works on convex and reflex corners alike; a reflex corner receives a
    concave arc, which is exactly the inside radius a router bit can follow.
    Pre-filleting an inside corner to ``radius >= tool_radius`` is the
    preferred alternative to a dogbone wherever the corner is visible.

    Args:
        ring: Input boundary.
        radius: Fillet radius in mm, must be > 0.
        corners: Which corners to touch.
        tolerance: Arc chord tolerance in mm.
        clamp: When ``True`` a corner whose edges are too short gets the
            largest radius that fits instead of being skipped.
        min_turn_deg: Vertices that turn by less than this are left alone, so
            filleting an already-curved contour does not shred it.

    Returns:
        The filleted ring, counter-clockwise.

    Raises:
        ValueError: If ``radius`` is not positive.
    """
    if radius <= 0:
        raise ValueError(f"fillet radius must be > 0, got {radius}")
    pts = ensure_ccw(dedupe(ring))
    if len(pts) < 3:
        return pts
    convex = classify_convex(pts)
    n = len(pts)
    out: Ring = []
    for i in range(n):
        wanted = (
            corners == "all"
            or (corners == "convex" and convex[i])
            or (corners == "concave" and not convex[i])
        )
        if not wanted:
            out.append(pts[i])
            continue
        arc = _corner_fillet(
            pts[i - 1],
            pts[i],
            pts[(i + 1) % n],
            radius,
            tolerance,
            clamp,
            min_turn_deg,
        )
        if arc is None:
            out.append(pts[i])
        else:
            out.extend(arc)
    return dedupe(out)


def relief_positions(
    ring: Sequence[Point],
    tool_radius: float,
    style: ReliefStyle = "dogbone",
    corners: CornerKind = "convex",
    factor: float = 1.0,
    min_turn_deg: float = 20.0,
) -> list[tuple[Point, float]]:
    """Compute tool relief circle centres for the corners of a cut region.

    A ring is read as the boundary of the region being *removed*.  Its convex
    corners are the ones a round tool cannot reach, so those are relieved by
    default.

    Args:
        ring: Boundary of the region being removed, e.g. a slot.
        tool_radius: Cutter radius in mm, must be > 0.
        style: ``"dogbone"`` puts the circle on the corner bisector, which
            spreads the overcut evenly; ``"tbone"`` pushes it along one edge
            so the overcut hides inside a mating joint.
        corners: Which corners to relieve.
        factor: Scales how far the circle sits from the corner.  1.0 places
            the circle through the corner point, the classic dogbone.
        min_turn_deg: A vertex only counts as a corner if the direction of
            travel turns by at least this much *and* both adjacent edges are
            at least one tool radius long.  Both tests are needed: a
            tessellated curve is a long run of vertices that each turn a
            little, and relieving those would shred a smooth contour.

    Returns:
        ``[(centre, radius), ...]``.

    Raises:
        ValueError: If ``tool_radius`` is not positive.
    """
    if tool_radius <= 0:
        raise ValueError(f"tool_radius must be > 0, got {tool_radius}")
    pts = ensure_ccw(dedupe(ring))
    convex = classify_convex(pts)
    n = len(pts)
    out: list[tuple[Point, float]] = []
    for i in range(n):
        wanted = (
            corners == "all"
            or (corners == "convex" and convex[i])
            or (corners == "concave" and not convex[i])
        )
        if not wanted:
            continue
        a, b, c = pts[i - 1], pts[i], pts[(i + 1) % n]
        len_prev, len_next = _norm(_sub(a, b)), _norm(_sub(c, b))
        try:
            v1 = _unit(_sub(a, b))
            v2 = _unit(_sub(c, b))
        except ValueError:
            continue
        dot = max(-1.0, min(1.0, _dot(v1, v2)))
        phi = math.acos(dot)
        turn = math.pi - phi
        if turn < math.radians(min_turn_deg) or phi < math.radians(5.0):
            continue
        if min(len_prev, len_next) < tool_radius:
            continue
        if style == "dogbone":
            bis_raw = _add(v1, v2)
            if _norm(bis_raw) < EPS:
                continue
            centre = _add(b, _mul(_unit(bis_raw), tool_radius * factor))
        else:
            # T-bone: slide along the longer adjacent edge, staying on its line
            edge = v1 if _norm(_sub(a, b)) >= _norm(_sub(c, b)) else v2
            centre = _add(b, _mul(edge, tool_radius * factor))
        out.append((centre, tool_radius))
    return out


def apply_relief(
    ring: Sequence[Point],
    tool_radius: float,
    style: ReliefStyle = "dogbone",
    region: RegionKind = "cutout",
    factor: float = 1.0,
    tolerance: float = ARC_TOLERANCE,
    min_turn_deg: float = 20.0,
) -> Ring:
    """Merge tool relief into a contour, keeping it a single closed ring.

    Relief is booleaned into the profile rather than exported as separate
    overlapping circles.  Overlapping vectors are precisely what CAM software
    flags as duplicates, so a dogbone that is part of the outline is both
    cleaner and correct.

    Args:
        ring: Boundary to relieve.
        tool_radius: Cutter radius in mm.
        style: ``"dogbone"`` or ``"tbone"``.
        region: ``"cutout"`` when ``ring`` bounds removed material (relief is
            added to it); ``"part"`` when ``ring`` bounds kept material
            (relief is subtracted from it).
        factor: Relief distance scale, see :func:`relief_positions`.
        tolerance: Arc chord tolerance in mm.
        min_turn_deg: Corner detection threshold, see
            :func:`relief_positions`.

    Returns:
        One closed ring with the relief merged in.

    Raises:
        ValueError: If the boolean result is not a single simple polygon,
            which means the relief circles collided and the caller must widen
            the feature or shrink the tool.
    """
    corners: CornerKind = "convex" if region == "cutout" else "concave"
    circles = relief_positions(
        ring, tool_radius, style, corners, factor, min_turn_deg
    )
    if not circles:
        return dedupe(ring)
    base = polygon_from_ring(ring)
    if not base.is_valid:
        base = base.buffer(0)
    discs = [
        polygon_from_ring(circle_ring(cx, cy, r, tolerance)) for (cx, cy), r in circles
    ]
    merged = unary_union([base, *discs]) if region == "cutout" else base.difference(
        unary_union(discs)
    )
    if isinstance(merged, MultiPolygon):
        raise ValueError(
            "tool relief split the contour into multiple pieces; "
            "the feature is too small for this tool"
        )
    if not isinstance(merged, Polygon) or merged.is_empty:
        raise ValueError("tool relief erased the contour")
    if merged.interiors:
        raise ValueError("tool relief produced an unexpected interior hole")
    return ensure_ccw(dedupe(list(merged.exterior.coords)))


# --------------------------------------------------------------------------- #
# joints
# --------------------------------------------------------------------------- #
def round_convex(
    ring: Sequence[Point], radius: float, tolerance: float = ARC_TOLERANCE
) -> Ring:
    """Round every convex corner of a cut region so a round tool can clear it.

    This is a morphological opening, not a vertex-by-vertex fillet, and that
    difference matters.  Filleting clamps each corner's radius to the length of
    the edges meeting there, so a corner where a long straight edge meets a
    *tessellated curve* gets clamped to half a chord - a millimetre or two -
    and silently fails to do its job.  An opening has no such failure mode: the
    result is by definition the area a disc of ``radius`` can sweep, so
    :func:`unreachable_zones` on it is empty for any tool radius at or below
    ``radius``.

    Args:
        ring: Boundary of the region to be removed.
        radius: Corner radius in mm, must be > 0 and should be at least the
            tool radius.
        tolerance: Arc chord tolerance used to thin the result, in mm.

    Returns:
        One closed ring, counter-clockwise.

    Raises:
        ValueError: If ``radius`` is not positive, or the region is too small
            to survive the operation.
    """
    if radius <= 0:
        raise ValueError(f"radius must be > 0, got {radius}")
    poly = polygon_from_ring(ring)
    if not poly.is_valid:
        poly = poly.buffer(0)
    opened = opening(poly, radius, slack=0.0)
    if opened.is_empty:
        raise ValueError(
            f"rounding to r={radius} erased the region; it is smaller than the "
            f"corner radius"
        )
    if isinstance(opened, MultiPolygon):
        raise ValueError(f"rounding to r={radius} split the region in two")
    thinned = opened.simplify(tolerance * 0.4, preserve_topology=True)
    if thinned.is_empty or not isinstance(thinned, Polygon):
        thinned = opened
    return ensure_ccw(dedupe(list(thinned.exterior.coords)))


def round_concave(
    ring: Sequence[Point],
    radius: float,
    tolerance: float = ARC_TOLERANCE,
    max_area_gain: float = 0.05,
) -> Ring:
    """Round every concave corner of a part profile so a round tool can cut it.

    The dual of :func:`round_convex`: a morphological closing, which leaves the
    profile with no inside corner tighter than ``radius``, so
    :func:`excess_zones` on it is empty for any tool radius at or below
    ``radius``.

    Args:
        ring: Boundary of the material being kept.
        radius: Inside corner radius in mm, must be > 0.
        tolerance: Arc chord tolerance used to thin the result, in mm.
        max_area_gain: Guard against a radius large enough to swallow a whole
            concave feature, as a fraction of the original area.  A closing
            fills any notch narrower than twice the radius, and silently
            deleting a design feature is worse than refusing.

    Returns:
        One closed ring, counter-clockwise.

    Raises:
        ValueError: If ``radius`` is not positive, or the closing filled more
            than ``max_area_gain`` of the profile.
    """
    if radius <= 0:
        raise ValueError(f"radius must be > 0, got {radius}")
    poly = polygon_from_ring(ring)
    if not poly.is_valid:
        poly = poly.buffer(0)
    closed = closing(poly, radius)
    if isinstance(closed, MultiPolygon) or closed.is_empty:
        raise ValueError(f"rounding inside corners to r={radius} failed")
    gain = (closed.area - poly.area) / poly.area if poly.area > 0 else 0.0
    if gain > max_area_gain:
        raise ValueError(
            f"rounding inside corners to r={radius} filled "
            f"{gain * 100:.1f}% of the profile, which means it swallowed a "
            f"feature; use a smaller radius"
        )
    thinned = closed.simplify(tolerance * 0.4, preserve_topology=True)
    if thinned.is_empty or not isinstance(thinned, Polygon):
        thinned = closed
    return ensure_ccw(dedupe(list(thinned.exterior.coords)))


def offset_centreline(
    ring: Sequence[Point], distance: float
) -> list[Point]:
    """Return the curve running ``distance`` inside a profile.

    Used to lay a feature out along a rim: because every point of this curve is
    ``distance`` from the profile, a slot of width ``w`` centred on it keeps a
    uniform ``distance - w/2`` of wall all the way along, whatever shape the
    profile is.

    Args:
        ring: The outer profile.
        distance: Inward distance in mm, must be > 0.

    Returns:
        The offset ring's vertices.

    Raises:
        ValueError: If ``distance`` is not positive or the offset collapses.
    """
    if distance <= 0:
        raise ValueError(f"distance must be > 0, got {distance}")
    rings = offset_ring(ring, -distance, join="round")
    if len(rings) != 1:
        raise ValueError(
            f"offsetting {distance} mm inward does not leave a single curve"
        )
    return rings[0]


def curved_slot(
    centreline: Sequence[Point],
    at: Point,
    length: float,
    width: float,
    tolerance: float = ARC_TOLERANCE,
) -> Ring:
    """Build a slot of ``length`` along a centreline, centred nearest ``at``.

    The slot follows the centreline, so on a curved rim it comes out as a
    curved hand slot rather than a straight one that crowds the edge at its
    ends.  Because it is a constant-width sweep of a disc, every corner is a
    rounded cap and the wall thickness is uniform.

    Args:
        centreline: The curve to follow, from :func:`offset_centreline`.
        at: The slot is centred at the point of the curve closest to this.
        length: Slot length measured along the curve, in mm.
        width: Slot width in mm.
        tolerance: Arc chord tolerance used to thin the result, in mm.

    Returns:
        One closed ring.

    Raises:
        ValueError: On non-positive dimensions, or a curve too short to hold
            the slot.
    """
    from shapely.geometry import LineString, Point as _ShapelyPoint
    from shapely.ops import substring

    if length <= 0 or width <= 0:
        raise ValueError(f"slot needs positive size, got {length}x{width}")
    pts = dedupe(centreline)
    if len(pts) < 3:
        raise ValueError("centreline is degenerate")
    # Rotate the closed curve so the slot's midpoint sits far from the seam,
    # which lets a plain substring do the work without wrap-around logic.
    target = _ShapelyPoint(*at)
    nearest = min(range(len(pts)), key=lambda i: target.distance(_ShapelyPoint(*pts[i])))
    shift = (nearest - len(pts) // 2) % len(pts)
    rotated = pts[shift:] + pts[:shift]
    line = LineString(rotated + [rotated[0]])
    if line.length < length + 2.0:
        raise ValueError(
            f"centreline is {line.length:.0f} mm long, too short for a "
            f"{length:g} mm slot"
        )
    middle = line.project(target)
    start = max(0.0, min(middle - length / 2.0, line.length - length))
    piece = substring(line, start, start + length)
    slot = piece.buffer(width / 2.0, quad_segs=BUFFER_QUAD_SEGS, cap_style="round")
    if isinstance(slot, MultiPolygon) or slot.is_empty:
        raise ValueError("slot sweep produced unusable geometry")
    thinned = slot.simplify(tolerance * 0.4, preserve_topology=True)
    if thinned.is_empty or not isinstance(thinned, Polygon):
        thinned = slot
    return ensure_ccw(dedupe(list(thinned.exterior.coords)))


def slot_ring(
    length: float,
    width: float,
    x0: float = 0.0,
    y0: float = 0.0,
    vertical: bool = False,
    corner_radius: float = 0.0,
    tolerance: float = ARC_TOLERANCE,
) -> Ring:
    """Return a rectangular slot, optionally with rounded corners.

    Args:
        length: Slot length in mm.
        width: Slot width in mm.
        x0: Bounding box left edge.
        y0: Bounding box bottom edge.
        vertical: Swap length and width so the slot runs along y.
        corner_radius: Corner radius in mm; 0 keeps sharp corners for a
            press fit, where relief is added separately.
        tolerance: Arc chord tolerance in mm.
    """
    w, h = (width, length) if vertical else (length, width)
    if corner_radius <= EPS:
        return rect_ring(w, h, x0, y0)
    return rounded_rect_ring(w, h, corner_radius, x0, y0, tolerance)


def joint_slot_width(
    thickness: float,
    mode: str,
    clearance: float = 0.2,
    kerf: float = 0.15,
) -> float:
    """Return the width a slot must be *drawn* at for a slot-together joint.

    Router mode cuts on the drawn line, so the drawn width carries the fit
    clearance directly.  Laser mode burns ``kerf`` away, so the drawn width is
    reduced by one kerf and the finished cut lands on ``thickness +
    clearance``.

    Args:
        thickness: Material thickness in mm.
        mode: ``"router"`` or ``"laser"``.
        clearance: Desired finished clearance in mm; 0 gives a press fit.
        kerf: Laser kerf width in mm, ignored for router mode.

    Returns:
        The drawn slot width in mm.

    Raises:
        ValueError: On unknown ``mode``, non-positive ``thickness``, or a
            drawn width that would collapse to zero.
    """
    if thickness <= 0:
        raise ValueError(f"thickness must be > 0, got {thickness}")
    mode = mode.lower()
    if mode == "router":
        width = thickness + clearance
    elif mode == "laser":
        width = thickness + clearance - kerf
    else:
        raise ValueError(f"mode must be 'router' or 'laser', got {mode!r}")
    if width <= 0:
        raise ValueError(
            f"slot width collapsed to {width}; clearance/kerf exceed thickness"
        )
    return width


def suggest_finger_count(
    length: float,
    thickness: float,
    min_cell_factor: float = 1.5,
    max_cell_factor: float = 3.0,
) -> int:
    """Pick an odd finger count giving well proportioned tabs.

    Tabs read as deliberate design when each cell is roughly 1.5 to 3 times
    the material thickness.  An odd count makes an edge start and end with the
    same state, which keeps corners symmetric.

    Args:
        length: Edge length in mm.
        thickness: Material thickness in mm.
        min_cell_factor: Lower bound on cell width as a multiple of thickness.
        max_cell_factor: Upper bound on cell width as a multiple of thickness.

    Returns:
        An odd integer >= 3.

    Raises:
        ValueError: If the edge cannot hold three cells of the minimum width.
    """
    if thickness <= 0:
        raise ValueError(f"thickness must be > 0, got {thickness}")
    min_cell = thickness * min_cell_factor
    max_cell = thickness * max_cell_factor
    if length < 3 * min_cell:
        raise ValueError(
            f"edge {length} mm is too short for 3 fingers of >= {min_cell} mm"
        )
    ideal = length / ((min_cell + max_cell) / 2.0)
    count = int(round(ideal))
    if count % 2 == 0:
        count += 1
    count = max(3, count)
    while count > 3 and length / count < min_cell:
        count -= 2
    while length / count > max_cell:
        count += 2
    return count


def finger_joint_edge(
    start: Point,
    end: Point,
    thickness: float,
    fingers: int,
    male: bool = True,
    clearance: float = 0.0,
    tab_first: bool = True,
) -> Ring:
    """Generate the tabbed profile of one edge of a finger-jointed panel.

    Geometry is *nominal*: laser kerf is handled globally by
    :func:`kerf_compensate_ring`, not by fudging each tab.  The edge is walked
    from ``start`` to ``end``; tabs protrude toward the outside of a
    counter-clockwise ring and notches cut inward by exactly ``thickness``, so
    a male edge and a female edge built with identical arguments mate flush.

    Args:
        start: Edge start point.
        end: Edge end point.
        thickness: Mating material thickness in mm, the tab depth.
        fingers: Cell count; odd values keep the edge symmetric.
        male: ``True`` emits protruding tabs, ``False`` emits notches.
        clearance: Finished fit clearance in mm; tabs shrink and notches grow
            by half of it on each side.
        tab_first: Whether cell 0 carries the tab/notch.

    Returns:
        Points from ``start`` to ``end`` inclusive, ready to splice into a
        ring.

    Raises:
        ValueError: On a degenerate edge, non-positive thickness, or
            ``fingers`` < 3.
    """
    if thickness <= 0:
        raise ValueError(f"thickness must be > 0, got {thickness}")
    if fingers < 3:
        raise ValueError(f"need at least 3 fingers, got {fingers}")
    length = _norm(_sub(end, start))
    if length <= EPS:
        raise ValueError("finger joint edge has zero length")
    u = _unit(_sub(end, start))
    normal = (u[1], -u[0])  # outward for a counter-clockwise ring
    cell = length / fingers
    if cell <= abs(clearance):
        raise ValueError(f"cell width {cell} is smaller than the clearance")
    depth = thickness if male else -thickness
    trim = (clearance / 2.0) if male else (-clearance / 2.0)

    def at(t: float) -> Point:
        return _add(start, _mul(u, t))

    pts: Ring = [tuple(start)]
    for i in range(fingers):
        if ((i % 2) == 0) != tab_first:
            continue
        t0 = min(max(i * cell + trim, 0.0), length)
        t1 = min(max((i + 1) * cell - trim, 0.0), length)
        if t1 - t0 <= EPS:
            continue
        p0, p1 = at(t0), at(t1)
        off = _mul(normal, depth)
        pts.extend([p0, _add(p0, off), _add(p1, off), p1])
    pts.append(tuple(end))
    return dedupe(pts, closed=False)


# --------------------------------------------------------------------------- #
# manufacturability geometry
# --------------------------------------------------------------------------- #
def opening(
    geom: BaseGeometry, radius: float, slack: float = EROSION_SLACK
) -> BaseGeometry:
    """Morphological opening: the union of every tool position inside ``geom``.

    Eroding then dilating by the tool radius yields exactly the area a round
    cutter of that radius can clear; whatever is left over is unreachable.

    ``slack`` shrinks the erosion radius slightly.  A dogbone places the tool
    centre at *exactly* the erosion limit, and the contour is a tessellation
    that sits up to :data:`ARC_TOLERANCE` inside the true curve, so a slack
    below the arc tolerance measures the tessellation instead of the tool and
    silently discards entire valid relief circles.  The default equals the arc
    tolerance: 50 microns on a 6.35 mm cutter, far below any machine's
    resolution.

    Args:
        geom: Region to be removed by the tool.
        radius: Tool radius in mm, must be > 0.
        slack: Erosion relief in mm, must be >= 0 and < ``radius``.

    Returns:
        The reachable sub-region, possibly empty.

    Raises:
        ValueError: If ``radius`` is not positive or ``slack`` swallows it.
    """
    if radius <= 0:
        raise ValueError(f"radius must be > 0, got {radius}")
    if slack < 0 or slack >= radius:
        raise ValueError(f"slack must be in [0, {radius}), got {slack}")
    effective = radius - slack
    eroded = geom.buffer(-effective, quad_segs=BUFFER_QUAD_SEGS)
    if eroded.is_empty:
        return eroded
    return eroded.buffer(effective, quad_segs=BUFFER_QUAD_SEGS)


def residual_region(
    geom: BaseGeometry, radius: float, slack: float = EROSION_SLACK
) -> BaseGeometry:
    """Return the material inside ``geom`` a round tool cannot remove.

    Args:
        geom: Region to be removed.
        radius: Tool radius in mm.
        slack: See :func:`opening`.

    Returns:
        The unreachable region, empty when the tool clears everything.
    """
    reachable = opening(geom, radius, slack)
    if reachable.is_empty:
        return geom
    return geom.difference(reachable)


def residual_thickness(
    region: BaseGeometry, limit: float = 5.0, iterations: int = 22
) -> float:
    """Return the largest inscribed-circle diameter inside ``region``.

    This is the physically meaningful measure of leftover corner material: a
    sliver thinner than the joint clearance is invisible, while one thicker
    than it stops a mating part from seating.

    Args:
        region: Region to measure, typically from :func:`residual_region`.
        limit: Upper search bound in mm.
        iterations: Bisection steps; 22 resolves to about 1 micron.

    Returns:
        Thickness in mm, 0.0 for an empty region.
    """
    if region.is_empty:
        return 0.0
    lo, hi = 0.0, float(limit)
    for _ in range(iterations):
        mid = (lo + hi) / 2.0
        if region.buffer(-mid / 2.0).is_empty:
            hi = mid
        else:
            lo = mid
    return lo


def _zone_list(
    region: BaseGeometry, min_thickness: float
) -> list[tuple[Point, float, float]]:
    """Describe each piece of a residual region, thickest first."""
    if region.is_empty:
        return []
    if min_thickness > 0 and region.buffer(-min_thickness / 2.0).is_empty:
        return []
    zones: list[tuple[Point, float, float]] = []
    for piece in getattr(region, "geoms", [region]):
        if piece.is_empty:
            continue
        thickness = residual_thickness(piece)
        if thickness <= min_thickness:
            continue
        c = piece.centroid
        zones.append(((c.x, c.y), piece.area, thickness))
    zones.sort(key=lambda z: -z[2])
    return zones


def closing(
    geom: BaseGeometry, radius: float, slack: float = EROSION_SLACK
) -> BaseGeometry:
    """Morphological closing: the shape a round tool can actually leave behind.

    Dilating then eroding by the tool radius fills in every concave feature
    the cutter is too fat to enter.  Where the result differs from the input,
    the profile asks for an inside corner the tool cannot cut.

    Args:
        geom: The part profile, i.e. the material being kept.
        radius: Tool radius in mm, must be > 0.
        slack: Unused here, accepted for symmetry with :func:`opening`.

    Returns:
        The closed region, always a superset of ``geom``.

    Raises:
        ValueError: If ``radius`` is not positive.
    """
    if radius <= 0:
        raise ValueError(f"radius must be > 0, got {radius}")
    dilated = geom.buffer(radius, quad_segs=BUFFER_QUAD_SEGS)
    return dilated.buffer(-radius, quad_segs=BUFFER_QUAD_SEGS)


def excess_zones(
    geom: BaseGeometry,
    radius: float,
    min_thickness: float = 0.0,
    slack: float = EROSION_SLACK,
) -> list[tuple[Point, float, float]]:
    """Find concave profile features a round tool is too fat to cut.

    This is the dual of :func:`unreachable_zones`: that one asks what the tool
    cannot remove from *inside* a pocket, this one asks what it cannot remove
    from *outside* a part, which is what a notch narrower than the cutter or a
    square inside corner on an outer profile amounts to.

    Args:
        geom: The part profile, i.e. the material being kept.
        radius: Tool radius in mm.
        min_thickness: Ignore slivers thinner than this, in mm.
        slack: Tolerance absorbed before comparing, in mm.  Tessellation makes
            an exact set comparison meaningless below this scale.

    Returns:
        ``[(centroid, area, thickness), ...]`` sorted by descending thickness.
    """
    filled = closing(geom, radius)
    excess = filled.difference(geom.buffer(slack, quad_segs=BUFFER_QUAD_SEGS))
    return _zone_list(excess, min_thickness)


def unreachable_zones(
    geom: BaseGeometry,
    radius: float,
    min_thickness: float = 0.0,
    slack: float = EROSION_SLACK,
) -> list[tuple[Point, float, float]]:
    """Find places a round tool cannot reach inside a cut region.

    This one test subsumes "minimum inside radius", "minimum feature width"
    and "slot too narrow": all three ask whether a disc of the tool radius
    fits.

    Args:
        geom: Region to be removed.
        radius: Tool radius in mm.
        min_thickness: Ignore slivers thinner than this, in mm.  A round tool
            always leaves something in a relieved corner; this is how the
            caller says how much it can live with.
        slack: See :func:`opening`.

    Returns:
        ``[(centroid, area, thickness), ...]`` sorted by descending
        thickness.
    """
    return _zone_list(residual_region(geom, radius, slack), min_thickness)


# --------------------------------------------------------------------------- #
# text
# --------------------------------------------------------------------------- #
def text_polygons(
    text: str,
    height: float,
    font_family: str = "DejaVu Sans",
    weight: str = "bold",
    origin: Point = (0.0, 0.0),
    align: Literal["left", "center", "right"] = "left",
) -> list[Polygon]:
    """Convert a string to filled outline polygons.

    Glyph outlines come from the font's own Bezier curves via matplotlib's
    text path machinery and are tessellated, so exported text is closed
    contours rather than unsupported text entities.  Counters (the hole in an
    "O") come back as polygon interiors.

    Args:
        text: String to convert; must not be blank.
        height: Rendered cap-to-baseline height of the result in mm, measured
            from the actual glyph bounding box so the caller gets what it
            asked for.
        font_family: Any family matplotlib can resolve.
        weight: Font weight, e.g. ``"normal"`` or ``"bold"``.
        origin: Where to place the result's bounding box.
        align: How ``origin`` relates to the bounding box horizontally.

    Returns:
        Valid shapely polygons, one per glyph region.

    Raises:
        ValueError: If ``text`` is blank, ``height`` <= 0, or the font
            produced no outline.
    """
    if not text.strip():
        raise ValueError("text must not be blank")
    if height <= 0:
        raise ValueError(f"text height must be > 0, got {height}")

    from matplotlib.font_manager import FontProperties
    from matplotlib.textpath import TextPath

    prop = FontProperties(family=font_family, weight=weight)
    path = TextPath((0, 0), text, size=100.0, prop=prop)
    loops = [list(map(tuple, poly)) for poly in path.to_polygons(closed_only=True)]
    loops = [dedupe(loop) for loop in loops]
    loops = [loop for loop in loops if len(loop) >= 3]
    if not loops:
        raise ValueError(f"font {font_family!r} produced no outline for {text!r}")

    polys = [polygon_from_ring(loop) for loop in loops]
    polys.sort(key=lambda p: -p.area)
    built: list[Polygon] = []
    for poly in polys:
        placed = False
        for i, parent in enumerate(built):
            if parent.contains(poly.representative_point()):
                built[i] = parent.difference(poly)
                placed = True
                break
        if not placed:
            built.append(poly)
    merged = unary_union([p.buffer(0) for p in built])
    result = [g for g in getattr(merged, "geoms", [merged]) if isinstance(g, Polygon)]
    if not result:
        raise ValueError(f"could not build outlines for {text!r}")

    x0, y0, x1, y1 = bbox(
        [pt for g in result for pt in g.exterior.coords]
    )
    span_y = y1 - y0
    if span_y <= EPS:
        raise ValueError(f"degenerate text outline for {text!r}")
    k = height / span_y
    width = (x1 - x0) * k
    shift = {"left": 0.0, "center": -width / 2.0, "right": -width}[align]
    dx = origin[0] - x0 * k + shift
    dy = origin[1] - y0 * k
    out: list[Polygon] = []
    for glyph in result:
        ext = translate(scale(list(glyph.exterior.coords), k), dx, dy)
        holes = [
            translate(scale(list(interior.coords), k), dx, dy)
            for interior in glyph.interiors
        ]
        out.append(polygon_from_ring(ext, holes))
    return out


def text_contours(
    text: str,
    height: float,
    font_family: str = "DejaVu Sans",
    weight: str = "bold",
    origin: Point = (0.0, 0.0),
    align: Literal["left", "center", "right"] = "left",
) -> list[Ring]:
    """Convert a string to closed rings ready for export.

    See :func:`text_polygons` for the arguments; this wrapper flattens the
    polygons to individual closed contours.
    """
    rings: list[Ring] = []
    for poly in text_polygons(text, height, font_family, weight, origin, align):
        rings.extend(rings_of(poly))
    return rings
