"""Manufacturability checks.

Every design passes through here before a single file is written.  A check
either proves something about the geometry or it does not exist: there are no
"looks about right" heuristics, and anything reported as an ERROR blocks
export.

The checks fall into three groups:

**Contour hygiene**
    Closed rings, no repeated or zero-length segments, no self-intersections,
    no duplicate contours.  These are exactly the conditions that make
    LightBurn, Vectric and Fusion emit "open contour" or "duplicate vector"
    warnings.

**Layout sanity**
    Origin at the bounding box corner, parts inside the sheet, parts not
    overlapping, features inside their part, minimum wall thickness.

**Tool reality**
    Pocket depth against material thickness, minimum feature width against
    the cutter, whether a round cutter can reach every corner of every cut
    region, and whether it can follow every concave feature of the outer
    profile.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from enum import Enum
from itertools import combinations
from pathlib import Path
from typing import Iterable, Sequence

from shapely.geometry import LinearRing, Point as ShapelyPoint, Polygon

from . import geometry as geo
from .design import Design, Machine, Part
from .limits import ValidationConfig
from .geometry import Point, Ring
from .layers import (
    CUT_INSIDE,
    CUT_OUTSIDE,
    ENGRAVE,
    INFO,
    is_cut_layer,
    parse_pocket_depth,
)

__all__ = [
    "Severity",
    "Issue",
    "Report",
    "ValidationConfig",
    "ValidationError",
    "validate_design",
    "validate_dxf_file",
]


class Severity(str, Enum):
    """How badly a check failed."""

    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"


class ValidationError(Exception):
    """Raised when an invalid design is about to be exported."""

    def __init__(self, report: "Report") -> None:
        super().__init__(report.format())
        self.report = report


@dataclass(frozen=True)
class Issue:
    """One finding from one check.

    Attributes:
        code: Stable machine-readable code, e.g. ``"E_WALL_THIN"``.
        severity: How badly it failed.
        message: Human readable explanation including the offending numbers.
        part: Name of the part it was found in, if applicable.
        location: Where it was found, if applicable.
    """

    code: str
    severity: Severity
    message: str
    part: str | None = None
    location: Point | None = None

    def __str__(self) -> str:
        where = f" [{self.part}]" if self.part else ""
        at = (
            f" at ({self.location[0]:.1f}, {self.location[1]:.1f})"
            if self.location
            else ""
        )
        return f"{self.severity.value} {self.code}{where}: {self.message}{at}"


@dataclass
class Report:
    """The collected findings for one design or file."""

    issues: list[Issue] = field(default_factory=list)

    def add(
        self,
        code: str,
        severity: Severity,
        message: str,
        part: str | None = None,
        location: Point | None = None,
    ) -> None:
        """Record one finding."""
        self.issues.append(Issue(code, severity, message, part, location))

    def extend(self, issues: Iterable[Issue]) -> None:
        """Record several findings."""
        self.issues.extend(issues)

    @property
    def errors(self) -> list[Issue]:
        """Findings that block export."""
        return [i for i in self.issues if i.severity is Severity.ERROR]

    @property
    def warnings(self) -> list[Issue]:
        """Findings worth a human's attention that do not block export."""
        return [i for i in self.issues if i.severity is Severity.WARNING]

    @property
    def ok(self) -> bool:
        """``True`` when nothing blocks export."""
        return not self.errors

    def reason(self) -> str:
        """Short single-line reason for a skip log."""
        if self.ok:
            return "ok"
        first = self.errors[0]
        extra = f" (+{len(self.errors) - 1} more)" if len(self.errors) > 1 else ""
        return f"{first.code}: {first.message}{extra}"

    def format(self, include_info: bool = False) -> str:
        """Render the report as multi-line text."""
        shown = [
            i
            for i in self.issues
            if include_info or i.severity is not Severity.INFO
        ]
        if not shown:
            return "PASS: no issues"
        head = (
            f"{len(self.errors)} error(s), {len(self.warnings)} warning(s)"
            if self.errors
            else f"PASS with {len(self.warnings)} warning(s)"
        )
        return "\n".join([head, *(f"  {i}" for i in shown)])

    def raise_for_status(self) -> None:
        """Raise :class:`ValidationError` if any error was recorded."""
        if not self.ok:
            raise ValidationError(self)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _ring_signature(ring: Sequence[Point], tol: float) -> frozenset[tuple[float, float]]:
    """Order- and direction-independent fingerprint of a ring."""
    q = max(tol, 1e-6)
    return frozenset(
        (round(x / q) * q, round(y / q) * q) for x, y in geo.dedupe(ring)
    )


def _safe_polygon(ring: Sequence[Point]) -> Polygon:
    """Build a polygon, repairing self-intersections so later checks can run."""
    poly = geo.polygon_from_ring(ring)
    if not poly.is_valid:
        repaired = poly.buffer(0)
        if isinstance(repaired, Polygon) and not repaired.is_empty:
            return repaired
    return poly


def _safe_region(pocket) -> Polygon:
    """The material a pocket actually removes, islands excluded.

    Layout checks have to use this rather than the pocket's outer ring: a
    juice groove is a ring with the working surface standing inside it, and a
    hanging hole through that island removes nothing the groove also removes.
    """
    try:
        region = pocket.region()
    except ValueError:
        return _safe_polygon(pocket.ring)
    if not region.is_valid:
        repaired = region.buffer(0)
        if isinstance(repaired, Polygon) and not repaired.is_empty:
            return repaired
    return region


# --------------------------------------------------------------------------- #
# contour hygiene
# --------------------------------------------------------------------------- #
def _check_ring(
    report: Report,
    ring: Sequence[Point],
    layer: str,
    part_name: str,
    cfg: ValidationConfig,
    what: str,
) -> bool:
    """Check one closed ring. Returns ``False`` if it is unusable."""
    pts = list(ring)
    if len(pts) < 3:
        report.add(
            "E_DEGENERATE",
            Severity.ERROR,
            f"{what} on {layer} has only {len(pts)} point(s); a closed contour needs 3",
            part_name,
        )
        return False
    if geo._norm(geo._sub(pts[0], pts[-1])) <= cfg.min_segment:
        report.add(
            "E_CLOSING_DUPLICATE",
            Severity.ERROR,
            f"{what} on {layer} repeats its first point; rings must leave "
            f"closure implicit or CAM sees a zero-length segment",
            part_name,
            pts[0],
        )
        return False
    n = len(pts)
    for i in range(n):
        a, b = pts[i], pts[(i + 1) % n]
        if geo._norm(geo._sub(a, b)) <= cfg.min_segment:
            report.add(
                "E_ZERO_SEGMENT",
                Severity.ERROR,
                f"{what} on {layer} has a zero-length segment at vertex {i}",
                part_name,
                a,
            )
            return False
    if len(pts) > cfg.max_points_per_contour:
        report.add(
            "W_DENSE_CONTOUR",
            Severity.WARNING,
            f"{what} on {layer} has {len(pts)} vertices; consider a coarser "
            f"arc tolerance to keep the file small",
            part_name,
        )
    linear = LinearRing(pts + [pts[0]])
    if not linear.is_simple:
        report.add(
            "E_SELF_INTERSECT",
            Severity.ERROR,
            f"{what} on {layer} intersects itself",
            part_name,
            geo.centroid(pts),
        )
        return False
    poly = geo.polygon_from_ring(pts)
    if not poly.is_valid:
        report.add(
            "E_INVALID_RING",
            Severity.ERROR,
            f"{what} on {layer} is not a valid polygon",
            part_name,
            geo.centroid(pts),
        )
        return False
    if poly.area < cfg.min_feature_area:
        report.add(
            "W_TINY_FEATURE",
            Severity.WARNING,
            f"{what} on {layer} encloses only {poly.area:.2f} mm^2",
            part_name,
            geo.centroid(pts),
        )
    return True


def _check_part_contours(
    report: Report, part: Part, cfg: ValidationConfig
) -> None:
    """Contour hygiene plus duplicate detection for one part."""
    seen: dict[frozenset[tuple[float, float]], str] = {}
    for ring, layer in part.cut_rings():
        what = "outline" if layer == CUT_OUTSIDE else "contour"
        if not _check_ring(report, ring, layer, part.name, cfg, what):
            continue
        sig = _ring_signature(ring, cfg.duplicate_tolerance)
        if sig in seen:
            report.add(
                "E_DUPLICATE_CONTOUR",
                Severity.ERROR,
                f"contour on {layer} duplicates an earlier contour on "
                f"{seen[sig]}; CAM reports this as a duplicate vector",
                part.name,
                geo.centroid(ring),
            )
        else:
            seen[sig] = layer
    for contour in part.engrave:
        if len(contour.points) < 2:
            report.add(
                "E_DEGENERATE",
                Severity.ERROR,
                f"engrave contour on {contour.layer} has "
                f"{len(contour.points)} point(s)",
                part.name,
            )
        elif contour.closed:
            _check_ring(
                report, contour.points, contour.layer, part.name, cfg, "engrave contour"
            )
    for label in part.labels:
        if is_cut_layer(label.layer):
            report.add(
                "E_TEXT_ON_CUT_LAYER",
                Severity.ERROR,
                f"label {label.text!r} sits on cut layer {label.layer}; text "
                f"entities are not closed contours and would be machined",
                part.name,
                label.position,
            )


# --------------------------------------------------------------------------- #
# layout sanity
# --------------------------------------------------------------------------- #
def _check_part_layout(report: Report, part: Part, cfg: ValidationConfig) -> None:
    """Features must sit inside the part with enough wall left around them."""
    outline = _safe_polygon(part.outline)
    boundary = outline.exterior
    # Depth is carried alongside each feature so nesting can be judged: a
    # through hole counts as infinitely deep.
    features: list[tuple[str, Polygon, str, float]] = []
    for hole in part.holes:
        features.append(("hole", _safe_polygon(hole), CUT_INSIDE, math.inf))
    for pocket in part.pockets:
        features.append(
            (f"pocket {pocket.depth:g} mm", _safe_region(pocket), pocket.layer,
             pocket.depth)
        )

    for label, poly, layer, _depth in features:
        if poly.is_empty:
            continue
        if not outline.contains(poly):
            report.add(
                "E_FEATURE_OUTSIDE",
                Severity.ERROR,
                f"{label} on {layer} is not fully inside the part outline",
                part.name,
                (poly.centroid.x, poly.centroid.y),
            )
            continue
        gap = boundary.distance(poly)
        if gap < cfg.min_wall - 1e-6:
            report.add(
                "E_WALL_THIN",
                Severity.ERROR,
                f"{label} leaves only {gap:.2f} mm of wall to the part edge, "
                f"minimum is {cfg.min_wall:g} mm",
                part.name,
                (poly.centroid.x, poly.centroid.y),
            )

    for (la, pa, _, da), (lb, pb, _, db) in combinations(features, 2):
        if pa.is_empty or pb.is_empty:
            continue
        inter = pa.intersection(pb)
        if inter.area <= cfg.min_feature_area / 100.0:
            gap = pa.distance(pb)
            if gap < cfg.min_wall - 1e-6:
                report.add(
                    "E_WALL_THIN",
                    Severity.ERROR,
                    f"{la} and {lb} are only {gap:.2f} mm apart, minimum wall "
                    f"is {cfg.min_wall:g} mm",
                    part.name,
                    (pa.centroid.x, pa.centroid.y),
                )
            continue
        # One feature wholly inside another is a step, not a collision: a
        # through hole in a pocket floor is a counterbore, and a deeper pocket
        # inside a shallower one is a stepped recess.  Only the inner feature
        # being the shallower of the two makes no physical sense, because the
        # outer operation would already have removed it.
        if pa.contains(pb) or pb.contains(pa):
            outer, inner = ((la, da), (lb, db)) if pa.contains(pb) else ((lb, db), (la, da))
            if inner[1] < outer[1] - 1e-9:
                report.add(
                    "E_STEP_INVERTED",
                    Severity.ERROR,
                    f"{inner[0]} sits inside {outer[0]} but is shallower, so "
                    f"the outer operation removes it first",
                    part.name,
                    (inter.centroid.x, inter.centroid.y),
                )
            continue
        report.add(
            "E_FEATURE_OVERLAP",
            Severity.ERROR,
            f"{la} and {lb} overlap by {inter.area:.2f} mm^2 without one "
            f"containing the other, which leaves a ragged edge",
            part.name,
            (inter.centroid.x, inter.centroid.y),
        )

    for drill in part.drills:
        disc = ShapelyPoint(*drill.center).buffer(drill.radius, quad_segs=32)
        if not outline.contains(disc):
            report.add(
                "E_FEATURE_OUTSIDE",
                Severity.ERROR,
                f"drill dia {drill.diameter:g} mm is not fully inside the part",
                part.name,
                drill.center,
            )
        if drill.diameter < cfg.min_drill_diameter:
            report.add(
                "W_DRILL_SMALL",
                Severity.WARNING,
                f"drill dia {drill.diameter:g} mm is below the "
                f"{cfg.min_drill_diameter:g} mm minimum",
                part.name,
                drill.center,
            )


# --------------------------------------------------------------------------- #
# tool reality
# --------------------------------------------------------------------------- #
def _check_tooling(
    report: Report, part: Part, machine: Machine, cfg: ValidationConfig
) -> None:
    """Can the configured machine actually cut these features?"""
    regions: list[tuple[str, Polygon]] = []
    for hole in part.holes:
        regions.append((CUT_INSIDE, _safe_polygon(hole)))
    for pocket in part.pockets:
        regions.append((pocket.layer, pocket.region()))

    if machine.is_laser:
        floor = max(2.0 * machine.kerf, 0.3)
        for layer, poly in regions:
            if poly.is_empty:
                continue
            if poly.buffer(-floor / 2.0).is_empty:
                report.add(
                    "E_FEATURE_NARROW",
                    Severity.ERROR,
                    f"cut region on {layer} is narrower than {floor:.2f} mm, "
                    f"twice the {machine.kerf:g} mm kerf; it would burn away",
                    part.name,
                    (poly.centroid.x, poly.centroid.y),
                )
        return

    r = machine.tool_radius
    min_width = cfg.feature_factor * machine.tool_diameter
    for layer, poly in regions:
        if poly.is_empty:
            continue
        if poly.buffer(-min_width / 2.0).is_empty:
            report.add(
                "E_FEATURE_NARROW",
                Severity.ERROR,
                f"cut region on {layer} is narrower than "
                f"{min_width:.2f} mm ({cfg.feature_factor:g} x the "
                f"{machine.tool_diameter:g} mm cutter); there is no chip "
                f"clearance",
                part.name,
                (poly.centroid.x, poly.centroid.y),
            )
            continue
        if poly.buffer(-(cfg.tight_factor * machine.tool_diameter) / 2.0).is_empty:
            report.add(
                "W_FEATURE_TIGHT",
                Severity.WARNING,
                f"cut region on {layer} is narrower than "
                f"{cfg.tight_factor:g} x the {machine.tool_diameter:g} mm "
                f"cutter, so the cutter runs at full engagement with no room "
                f"for a finishing pass; expect a rougher wall",
                part.name,
                (poly.centroid.x, poly.centroid.y),
            )
        if not cfg.check_reachability:
            continue
        zones = geo.unreachable_zones(poly, r, cfg.max_corner_residual)
        if zones:
            where, _area, thickness = zones[0]
            report.add(
                "E_TOOL_UNREACHABLE",
                Severity.ERROR,
                f"a {machine.tool_diameter:g} mm cutter leaves "
                f"{thickness:.2f} mm of material in {len(zones)} corner(s) on "
                f"{layer} (max allowed {cfg.max_corner_residual:g} mm); add "
                f"dogbone relief or pre-fillet to r >= {r:.2f} mm",
                part.name,
                where,
            )


def _check_profile(
    report: Report, part: Part, machine: Machine, cfg: ValidationConfig
) -> None:
    """Can the cutter follow the outer profile's concave features?"""
    if machine.is_laser or not cfg.check_reachability:
        return
    outline = _safe_polygon(part.outline)
    if outline.is_empty:
        return
    zones = geo.excess_zones(outline, machine.tool_radius, cfg.max_corner_residual)
    if zones:
        where, _area, thickness = zones[0]
        report.add(
            "E_PROFILE_TOO_TIGHT",
            Severity.ERROR,
            f"the outer profile has {len(zones)} concave feature(s) a "
            f"{machine.tool_diameter:g} mm cutter cannot enter, leaving "
            f"{thickness:.2f} mm of material; widen the notch or fillet the "
            f"inside corner to r >= {machine.tool_radius:.2f} mm",
            part.name,
            where,
        )


def _check_depths(report: Report, design: Design, cfg: ValidationConfig) -> None:
    """Pockets must leave a floor under them."""
    limit = design.thickness - cfg.pocket_floor
    for part in design.parts:
        for pocket in part.pockets:
            if pocket.depth > limit + 1e-9:
                report.add(
                    "E_POCKET_DEPTH",
                    Severity.ERROR,
                    f"pocket {pocket.depth:g} mm deep leaves "
                    f"{design.thickness - pocket.depth:.2f} mm of floor in "
                    f"{design.thickness:g} mm material; minimum floor is "
                    f"{cfg.pocket_floor:g} mm",
                    part.name,
                    geo.centroid(pocket.ring),
                )


def _check_design_layout(
    report: Report, design: Design, cfg: ValidationConfig
) -> None:
    """Origin, sheet fit and part collisions."""
    parts = design.placed_parts()
    x0, y0, x1, y1 = geo.bbox_of(p.outline for p in parts)
    if abs(x0) > cfg.origin_tolerance or abs(y0) > cfg.origin_tolerance:
        report.add(
            "E_ORIGIN",
            Severity.ERROR,
            f"bounding box starts at ({x0:.3f}, {y0:.3f}); every file must "
            f"place the bottom-left of the bounding box at the origin",
            location=(x0, y0),
        )
    width, height = x1 - x0, y1 - y0
    sw, sh = design.sheet_size()
    fits = (width <= sw + 1e-6 and height <= sh + 1e-6) or (
        width <= sh + 1e-6 and height <= sw + 1e-6
    )
    if not fits:
        report.add(
            "E_SHEET_OVERFLOW",
            Severity.ERROR,
            f"layout is {width:.1f}x{height:.1f} mm and does not fit the "
            f"declared {sw:g}x{sh:g} mm sheet in either orientation",
        )
    for a, b in combinations(parts, 2):
        pa, pb = _safe_polygon(a.outline), _safe_polygon(b.outline)
        inter = pa.intersection(pb)
        if inter.area > 1e-6:
            report.add(
                "E_PART_OVERLAP",
                Severity.ERROR,
                f"parts {a.name!r} and {b.name!r} overlap by "
                f"{inter.area:.2f} mm^2",
                location=(inter.centroid.x, inter.centroid.y),
            )


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #
def validate_design(
    design: Design, config: ValidationConfig | None = None
) -> Report:
    """Run every check against a design.

    Args:
        design: The design to judge.
        config: Limits to apply.  Defaults to the limits the design itself
            carries, so a design is held to the standard it was built to
            rather than to a global default.

    Returns:
        A :class:`Report`.  Call :meth:`Report.raise_for_status` to turn
        errors into an exception, which is what the exporters do.
    """
    cfg = config or design.limits
    report = Report()
    _check_design_layout(report, design, cfg)
    _check_depths(report, design, cfg)
    for part in design.placed_parts():
        _check_part_contours(report, part, cfg)
        _check_part_layout(report, part, cfg)
        _check_tooling(report, part, design.machine, cfg)
        _check_profile(report, part, design.machine, cfg)
    return report


def _check_file_reachability(
    report: Report,
    ring: Sequence[Point],
    layer: str,
    machine: Machine,
    cfg: ValidationConfig,
) -> None:
    """Can the given cutter reach every corner of one contour from a file?

    Which side the material is on comes from the layer name, since a file has
    no part structure to ask: a cut-inside or pocket contour is a void the
    cutter works within, so its own corners must be reachable; a cut-outside
    contour is a part the cutter works around, so its concave features must
    admit the tool.

    Args:
        report: Where findings go.
        ring: The closed contour.
        layer: The layer it sits on.
        machine: The machine to judge against.
        cfg: The limits.
    """
    if machine.is_laser or not cfg.check_reachability:
        return
    inside = layer == CUT_INSIDE or parse_pocket_depth(layer) is not None
    if not inside and layer != CUT_OUTSIDE:
        return
    poly = _safe_polygon(ring)
    if poly.is_empty:
        return
    radius = machine.tool_radius
    if inside:
        zones = geo.unreachable_zones(poly, radius, cfg.max_corner_residual)
        if zones:
            where, _area, thickness = zones[0]
            report.add(
                "E_TOOL_UNREACHABLE",
                Severity.ERROR,
                f"a {machine.tool_diameter:g} mm cutter leaves "
                f"{thickness:.2f} mm of material in {len(zones)} corner(s) on "
                f"{layer} (max allowed {cfg.max_corner_residual:g} mm); this "
                f"file needs a cutter of {radius * 2:.2f} mm or less, or "
                f"dogbone relief",
                location=where,
            )
        return
    zones = geo.excess_zones(poly, radius, cfg.max_corner_residual)
    if zones:
        where, _area, thickness = zones[0]
        report.add(
            "E_PROFILE_TOO_TIGHT",
            Severity.ERROR,
            f"the profile on {layer} has {len(zones)} concave feature(s) a "
            f"{machine.tool_diameter:g} mm cutter cannot enter, leaving "
            f"{thickness:.2f} mm of material; it needs a cutter of "
            f"{radius * 2:.2f} mm or less",
            location=where,
        )


def validate_dxf_file(
    path: str | Path,
    machine: Machine | None = None,
    config: ValidationConfig | None = None,
) -> Report:
    """Run the checks against an arbitrary DXF file on disk.

    This works on third-party files too, which is the point: it answers "will
    this import cleanly and can my machine cut it" without opening CAD.  The
    file's own layer names are honoured where they follow the dxfgen standard
    and reported where they do not.

    Args:
        path: DXF file to inspect.
        machine: Machine to judge feature sizes against.  If omitted it is
            read from the file's own record of what it was written for, which
            only files this tool wrote carry; failing that nothing is assumed
            and the checks needing a cutter are skipped rather than run
            against a guess.  Given one, in router mode every closed contour
            on a cut layer is tested for corners the cutter cannot reach,
            which is what makes ``--tool`` worth setting before trusting
            somebody else's file.
        config: Limits to apply.

    Returns:
        A :class:`Report`.

    Note:
        A file on disk carries no part structure, so each contour is judged on
        its own and its layer decides which side the material is on.  An
        island standing inside a cutout is therefore analysed as if the cutout
        were solid, which can over-report on a design that nests one contour
        inside another on the same cut layer.  :func:`validate_design` knows
        the structure and does not have that limitation.
    """
    import ezdxf
    from ezdxf.lldxf.const import DXFError

    cfg = config or ValidationConfig()
    report = Report()
    try:
        doc = ezdxf.readfile(str(path))
    except (OSError, DXFError) as exc:
        report.add("E_UNREADABLE", Severity.ERROR, f"cannot read DXF: {exc}")
        return report

    # A DXF has nowhere standard to record which machine it was cut for, and
    # guessing wrong is worse than not guessing: router reachability rules
    # applied to a laser file flag every finger joint as unmachinable.  So the
    # machine comes from the caller, else from the file's own provenance if
    # this tool wrote it, else nothing is assumed and the checks that need a
    # cutter are skipped - and said to be skipped.
    from .export_dxf import machine_from_document

    mach = machine or machine_from_document(doc)
    if mach is None:
        report.add(
            "INFO_NO_MACHINE",
            Severity.INFO,
            "the file does not record which machine it was cut for and none "
            "was given, so cutter reachability was not checked; pass a tool "
            "diameter to check it",
        )
        mach = Machine()
        cfg = replace(cfg, check_reachability=False)

    auditor = doc.audit()
    for err in auditor.errors:
        report.add("E_AUDIT", Severity.ERROR, f"ezdxf audit: {err}")
    for fix in auditor.fixes:
        report.add("W_AUDIT_FIX", Severity.WARNING, f"ezdxf auto-fixed: {fix}")

    insunits = doc.header.get("$INSUNITS", 0)
    if insunits != 4:
        report.add(
            "E_UNITS",
            Severity.ERROR if insunits not in (0, 4) else Severity.WARNING,
            f"$INSUNITS is {insunits}, expected 4 (millimetres); "
            f"CAM will guess the scale",
        )

    msp = doc.modelspace()
    rings: list[tuple[Ring, str]] = []
    all_points: list[Point] = []
    counts: dict[str, int] = {}
    for entity in msp:
        etype = entity.dxftype()
        counts[etype] = counts.get(etype, 0) + 1
        layer = entity.dxf.layer
        if etype == "LWPOLYLINE":
            pts = [(float(p[0]), float(p[1])) for p in entity.get_points("xy")]
            all_points.extend(pts)
            has_bulge = any(abs(p[4]) > 1e-9 for p in entity.get_points())
            if has_bulge:
                report.add(
                    "W_BULGE",
                    Severity.WARNING,
                    f"polyline on {layer} uses bulge arcs; geometric checks "
                    f"treat it as straight segments",
                )
            if not entity.closed:
                # Engraving is legitimately open: a ray, a hatch line or a
                # signature is a stroke, not a boundary.  Warning about those
                # would bury the one case that matters - an open contour on a
                # layer something is supposed to cut through.
                if layer != ENGRAVE:
                    report.add(
                        "E_OPEN_CONTOUR",
                        Severity.ERROR if is_cut_layer(layer) else Severity.WARNING,
                        f"polyline on {layer} is not closed",
                        location=pts[0] if pts else None,
                    )
            else:
                rings.append((geo.dedupe(pts), layer))
        elif etype == "CIRCLE":
            c = entity.dxf.center
            all_points.extend(
                [
                    (c.x - entity.dxf.radius, c.y - entity.dxf.radius),
                    (c.x + entity.dxf.radius, c.y + entity.dxf.radius),
                ]
            )
        elif etype in ("SPLINE", "ELLIPSE"):
            report.add(
                "E_SPLINE",
                Severity.ERROR,
                f"{etype} on {layer} is not a polyline; many CAM packages "
                f"cannot offset it reliably",
            )
        elif etype in ("LINE", "ARC"):
            report.add(
                "W_OPEN_PRIMITIVE",
                Severity.WARNING,
                f"{etype} on {layer} is an open primitive, not a closed "
                f"contour; it will import as an open vector",
            )
        elif etype == "POLYLINE":
            report.add(
                "W_LEGACY_POLYLINE",
                Severity.WARNING,
                f"legacy POLYLINE on {layer}; prefer LWPOLYLINE",
            )
        elif etype in ("TEXT", "MTEXT", "ATTRIB"):
            if is_cut_layer(layer):
                report.add(
                    "E_TEXT_ON_CUT_LAYER",
                    Severity.ERROR,
                    f"{etype} on cut layer {layer} would be machined",
                )
        elif etype in ("DIMENSION", "LEADER", "HATCH", "INSERT", "POINT"):
            report.add(
                "W_NON_GEOMETRY",
                Severity.WARNING,
                f"{etype} on {layer} carries no cuttable geometry",
            )

    if not rings:
        report.add(
            "W_NO_CLOSED_CONTOURS",
            Severity.WARNING,
            f"no closed contours found; entity counts: {counts}",
        )

    standard = {CUT_OUTSIDE, CUT_INSIDE, INFO, "ENGRAVE", "DRILL"}
    for name in doc.layers:
        layer_name = name.dxf.name
        if layer_name in ("0", "Defpoints"):
            continue
        if layer_name not in standard and parse_pocket_depth(layer_name) is None:
            report.add(
                "W_NONSTANDARD_LAYER",
                Severity.WARNING,
                f"layer {layer_name!r} is not part of the dxfgen standard",
            )

    seen: dict[frozenset[tuple[float, float]], str] = {}
    for ring, layer in rings:
        if len(ring) < 3:
            report.add(
                "E_DEGENERATE",
                Severity.ERROR,
                f"closed polyline on {layer} has {len(ring)} distinct points",
            )
            continue
        if not LinearRing(ring + [ring[0]]).is_simple:
            report.add(
                "E_SELF_INTERSECT",
                Severity.ERROR,
                f"closed polyline on {layer} intersects itself",
                location=geo.centroid(ring),
            )
        _check_file_reachability(report, ring, layer, mach, cfg)
        sig = _ring_signature(ring, cfg.duplicate_tolerance)
        if sig in seen:
            report.add(
                "E_DUPLICATE_CONTOUR",
                Severity.ERROR,
                f"closed polyline on {layer} duplicates one on {seen[sig]}",
                location=geo.centroid(ring),
            )
        else:
            seen[sig] = layer

    if all_points:
        x0, y0, x1, y1 = geo.bbox(all_points)
        if abs(x0) > 0.5 or abs(y0) > 0.5:
            report.add(
                "W_ORIGIN",
                Severity.WARNING,
                f"geometry starts at ({x0:.2f}, {y0:.2f}) rather than the "
                f"origin; the operator must re-zero",
            )
        report.add(
            "INFO_EXTENTS",
            Severity.INFO,
            f"extents {x1 - x0:.1f}x{y1 - y0:.1f} mm, entities: {counts}",
        )
    return report
