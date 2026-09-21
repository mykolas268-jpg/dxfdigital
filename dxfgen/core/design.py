"""The in-memory description of one manufacturable design.

A :class:`Design` is a machine-independent, unit-checked description of a
product: one or more :class:`Part` objects plus the material and machine
settings needed to cut them.  Generators produce it, the validator judges it
and the exporters render it.  Nothing here knows about DXF, SVG or PNG.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Iterable, Literal, Sequence

from shapely.geometry import Polygon
from shapely.ops import unary_union

from . import geometry as geo
from .geometry import Point, Ring
from .limits import ValidationConfig
from .layers import (
    CUT_INSIDE,
    CUT_OUTSIDE,
    DRILL,
    ENGRAVE,
    INFO,
    is_cut_layer,
    layer_def,
    pocket_layer_name,
)

__all__ = [
    "Mode",
    "Machine",
    "Pocket",
    "Drill",
    "Contour",
    "Label",
    "Part",
    "Design",
    "ValidationConfig",
    "ROUTER_SHEET",
    "LASER_SHEET",
]

#: Default stock sheet for router work (mm), a full 4x8 ft panel.
ROUTER_SHEET: tuple[float, float] = (1220.0, 2440.0)
#: Default stock sheet for laser work (mm), a common 600x400 bed.
LASER_SHEET: tuple[float, float] = (600.0, 400.0)


class Mode(str, Enum):
    """Which kind of machine a design is cut on."""

    ROUTER = "router"
    LASER = "laser"


@dataclass(frozen=True)
class Machine:
    """Machine and fit settings for one design.

    Attributes:
        mode: Router or laser.
        tool_diameter: Router cutter diameter in mm.  Ignored in laser mode
            except as a sanity floor on feature size.
        kerf: Laser kerf width in mm.  Ignored in router mode, where CAM
            applies the cutter offset itself.
        clearance: Intended finished fit clearance for slot-together joints in
            mm.  0 gives a press fit.
    """

    mode: Mode = Mode.ROUTER
    tool_diameter: float = 6.35
    kerf: float = 0.15
    clearance: float = 0.2

    def __post_init__(self) -> None:
        if self.tool_diameter <= 0:
            raise ValueError(f"tool_diameter must be > 0, got {self.tool_diameter}")
        if self.kerf < 0:
            raise ValueError(f"kerf must be >= 0, got {self.kerf}")
        if self.clearance < 0:
            raise ValueError(f"clearance must be >= 0, got {self.clearance}")

    @property
    def tool_radius(self) -> float:
        """Half the cutter diameter, in mm."""
        return self.tool_diameter / 2.0

    @property
    def is_laser(self) -> bool:
        """``True`` in laser mode."""
        return self.mode is Mode.LASER

    def slot_width(self, thickness: float) -> float:
        """Drawn width of a slot that accepts ``thickness`` material.

        See :func:`dxfgen.core.geometry.joint_slot_width`.
        """
        return geo.joint_slot_width(
            thickness, self.mode.value, self.clearance, self.kerf
        )

    def default_sheet(self) -> tuple[float, float]:
        """Default stock sheet size in mm for this mode."""
        return LASER_SHEET if self.is_laser else ROUTER_SHEET


@dataclass
class Pocket:
    """A cleared recess at one depth.

    Attributes:
        ring: Closed boundary of the cleared area.
        depth: Depth below the top surface in mm.
        islands: Closed boundaries of areas left standing inside the pocket.
    """

    ring: Ring
    depth: float
    islands: list[Ring] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.depth <= 0:
            raise ValueError(f"pocket depth must be > 0, got {self.depth}")

    @property
    def layer(self) -> str:
        """Layer name encoding this pocket's depth."""
        return pocket_layer_name(self.depth)

    def region(self) -> Polygon:
        """Shapely polygon of the cleared area, islands as holes."""
        return geo.polygon_from_ring(self.ring, self.islands)


@dataclass
class Drill:
    """A drill point.

    Attributes:
        center: Hole centre.
        diameter: Hole diameter in mm.
    """

    center: Point
    diameter: float

    def __post_init__(self) -> None:
        if self.diameter <= 0:
            raise ValueError(f"drill diameter must be > 0, got {self.diameter}")

    @property
    def radius(self) -> float:
        """Half the hole diameter, in mm."""
        return self.diameter / 2.0


@dataclass
class Contour:
    """A polyline on a named layer.

    Attributes:
        points: Vertices; a closed contour must not repeat its first point.
        layer: Target layer name.
        closed: Whether the contour closes back on itself.
    """

    points: Ring
    layer: str
    closed: bool = True

    def __post_init__(self) -> None:
        layer_def(self.layer)  # raises on a non-standard layer name


@dataclass
class Label:
    """A single-line text annotation.

    Text is exported as a DXF ``TEXT`` entity, which is appropriate for
    annotation.  Machinable lettering should instead be generated as closed
    outlines with :func:`dxfgen.core.geometry.text_contours`, because CAM
    packages substitute fonts for ``TEXT`` entities.

    Attributes:
        text: The string.
        position: Insertion point.
        height: Cap height in mm.
        layer: Target layer, normally ``INFO``.
        rotation: Rotation in degrees counter-clockwise.
        align: Horizontal alignment relative to ``position``.
    """

    text: str
    position: Point
    height: float = 4.0
    layer: str = INFO
    rotation: float = 0.0
    align: Literal["left", "center", "right"] = "left"

    def __post_init__(self) -> None:
        if self.height <= 0:
            raise ValueError(f"label height must be > 0, got {self.height}")


@dataclass
class Part:
    """One physically separate piece of material.

    Attributes:
        name: Short identifier, used for INFO labels and nesting reports.
        outline: Closed outer profile, cut on ``CUT_OUTSIDE``.
        holes: Closed through cuts, cut on ``CUT_INSIDE``.
        pockets: Cleared recesses.
        drills: Drill points.
        engrave: Decoration on ``ENGRAVE``.
        labels: Annotation, normally on ``INFO``.
        origin: Translation applied when the part is placed on a sheet.
        rotation: Rotation in degrees applied before ``origin``.
        quantity: How many copies to cut.  Nesting expands these into
            separate placed parts.
    """

    name: str
    outline: Ring
    holes: list[Ring] = field(default_factory=list)
    pockets: list[Pocket] = field(default_factory=list)
    drills: list[Drill] = field(default_factory=list)
    engrave: list[Contour] = field(default_factory=list)
    labels: list[Label] = field(default_factory=list)
    origin: Point = (0.0, 0.0)
    rotation: float = 0.0
    quantity: int = 1

    def __post_init__(self) -> None:
        if self.quantity < 1:
            raise ValueError(f"quantity must be >= 1, got {self.quantity}")
        if len(geo.dedupe(self.outline)) < 3:
            raise ValueError(f"part {self.name!r} outline needs >= 3 distinct points")

    # ---------------------------------------------------------------- placing
    def placed(self) -> "Part":
        """Return a copy with ``rotation`` and ``origin`` baked into geometry.

        Exporters and the validator always work on placed parts so that
        nesting is invisible to them.
        """
        if abs(self.rotation) < geo.EPS and self.origin == (0.0, 0.0):
            return self
        deg, (dx, dy) = self.rotation, self.origin

        def xf(ring: Sequence[Point]) -> Ring:
            return geo.translate(geo.rotate(ring, deg), dx, dy)

        def xf_pt(p: Point) -> Point:
            return xf([p])[0]

        return Part(
            name=self.name,
            outline=xf(self.outline),
            holes=[xf(h) for h in self.holes],
            pockets=[
                Pocket(xf(p.ring), p.depth, [xf(i) for i in p.islands])
                for p in self.pockets
            ],
            drills=[Drill(xf_pt(d.center), d.diameter) for d in self.drills],
            engrave=[Contour(xf(c.points), c.layer, c.closed) for c in self.engrave],
            labels=[
                Label(
                    l.text,
                    xf_pt(l.position),
                    l.height,
                    l.layer,
                    l.rotation + deg,
                    l.align,
                )
                for l in self.labels
            ],
            quantity=self.quantity,
        )

    def translated(self, dx: float, dy: float) -> "Part":
        """Return a copy shifted by ``(dx, dy)`` on top of its current origin."""
        return replace(self, origin=(self.origin[0] + dx, self.origin[1] + dy))

    # ------------------------------------------------------------- geometry
    def region(self) -> Polygon:
        """Shapely polygon of the material that remains, holes removed."""
        return geo.polygon_from_ring(self.outline, self.holes)

    def cut_rings(self) -> list[tuple[Ring, str]]:
        """Return every closed cut ring with its layer."""
        out: list[tuple[Ring, str]] = [(self.outline, CUT_OUTSIDE)]
        out.extend((h, CUT_INSIDE) for h in self.holes)
        for pocket in self.pockets:
            out.append((pocket.ring, pocket.layer))
            out.extend((island, pocket.layer) for island in pocket.islands)
        return out

    def all_rings(self) -> list[Ring]:
        """Every closed ring in the part, in no particular order."""
        rings = [ring for ring, _ in self.cut_rings()]
        rings.extend(c.points for c in self.engrave if c.closed)
        return rings

    def bbox(self) -> tuple[float, float, float, float]:
        """Bounding box of the part's outer profile."""
        return geo.bbox(self.outline)

    def size(self) -> tuple[float, float]:
        """``(width, height)`` of the part's outer profile."""
        return geo.size_of(self.outline)

    def area(self) -> float:
        """Material area of the part in mm^2, holes excluded."""
        return self.region().area


@dataclass
class Design:
    """A complete, exportable design.

    Attributes:
        slug: Filesystem-safe identifier, unique within a niche.
        name: Human readable title for listings and READMEs.
        niche: Niche key, e.g. ``"trays"``.
        description: One-paragraph description for the listing and README.
        parts: The parts to cut.
        machine: Machine and fit settings.
        material: Material description, e.g. ``"19 mm hardwood"``.
        thickness: Material thickness in mm.
        sheet: Stock sheet size in mm, or ``None`` to use the machine default.
        params: The generator parameters that produced this design, recorded
            verbatim so any file can be regenerated.
        notes: Extra README lines.
        cutting_order: Ordered operation names for the README.
        limits: The manufacturing limits this design was built to, used by
            the validator unless it is given others.
        seed: Seed used by the variant generator, when applicable.
    """

    slug: str
    name: str
    niche: str
    description: str
    parts: list[Part]
    machine: Machine = field(default_factory=Machine)
    material: str = "plywood"
    thickness: float = 18.0
    sheet: tuple[float, float] | None = None
    params: dict[str, object] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    cutting_order: list[str] = field(default_factory=list)
    limits: ValidationConfig = field(default_factory=ValidationConfig)
    seed: int | None = None

    def __post_init__(self) -> None:
        if not self.parts:
            raise ValueError(f"design {self.slug!r} has no parts")
        if self.thickness <= 0:
            raise ValueError(f"thickness must be > 0, got {self.thickness}")

    # ------------------------------------------------------------- placement
    def placed_parts(self) -> list[Part]:
        """Return every part with its placement transform applied."""
        return [part.placed() for part in self.parts]

    def sheet_size(self) -> tuple[float, float]:
        """Stock sheet size in mm, falling back to the machine default."""
        return self.sheet or self.machine.default_sheet()

    def bbox(self) -> tuple[float, float, float, float]:
        """Bounding box across all placed parts."""
        return geo.bbox_of(part.outline for part in self.placed_parts())

    def size(self) -> tuple[float, float]:
        """Overall ``(width, height)`` in mm."""
        x0, y0, x1, y1 = self.bbox()
        return (x1 - x0, y1 - y0)

    def normalized(self) -> "Design":
        """Return a copy whose overall bounding box starts at the origin.

        Every exported file puts the bottom-left of the bounding box at
        ``(0, 0)``, so an operator can zero the machine on the stock corner.
        """
        x0, y0, _, _ = self.bbox()
        if abs(x0) < geo.EPS and abs(y0) < geo.EPS:
            return self
        moved = [part.translated(-x0, -y0) for part in self.parts]
        return replace(self, parts=moved)

    # ---------------------------------------------------------------- layers
    def pocket_depths(self) -> list[float]:
        """Sorted distinct pocket depths in mm."""
        depths = {p.depth for part in self.parts for p in part.pockets}
        return sorted(depths)

    def layers_used(self) -> list[str]:
        """Sorted names of every layer this design writes to."""
        names: set[str] = set()
        for part in self.placed_parts():
            names.add(CUT_OUTSIDE)
            if part.holes:
                names.add(CUT_INSIDE)
            for pocket in part.pockets:
                names.add(pocket.layer)
            if part.drills:
                names.add(DRILL)
            for contour in part.engrave:
                names.add(contour.layer)
            for label in part.labels:
                names.add(label.layer)
        return sorted(names, key=lambda n: (not is_cut_layer(n), n))

    def region(self) -> Polygon:
        """Union of the material of every placed part."""
        return unary_union([part.region() for part in self.placed_parts()])

    def total_cut_length(self) -> float:
        """Total length of all cut contours in mm, a rough runtime proxy."""
        total = 0.0
        for part in self.placed_parts():
            for ring, _ in part.cut_rings():
                total += geo.perimeter(ring)
            for drill in part.drills:
                total += 2.0 * 3.141592653589793 * drill.radius
        return total

    def summary(self) -> str:
        """One-line human summary used in logs and contact sheets."""
        w, h = self.size()
        return (
            f"{self.name} [{self.slug}] {w:.1f}x{h:.1f} mm, "
            f"{len(self.parts)} part(s), {self.material} {self.thickness:g} mm, "
            f"{self.machine.mode.value}"
        )
