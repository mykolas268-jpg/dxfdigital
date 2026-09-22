"""Serving, valet and snack trays cut from a single board.

One board, one outer profile, one or more recessed compartments, optional
handles.  The shape rules that keep the results looking designed rather than
generated:

* The outline defaults to the golden ratio and its corner radius scales with
  the tray's width, so a small valet tray and a large serving tray look like
  members of the same family.
* Compartments are cut out of a uniform inward offset of the outline, so the
  rim is exactly the same width all the way round whatever the outline shape.
* Compartment corners are pre-filleted to a radius the cutter can follow.  On
  a surface the buyer looks at, that is the right answer; dogbone relief is
  for hidden mating joints.
* Handle cutouts are sized for a hand - at least 90 x 30 mm - and the rim is
  widened automatically to carry them with full wall thickness on both sides.
"""

from __future__ import annotations

import random
from enum import Enum
from typing import Sequence

from pydantic import Field, model_validator
from shapely.geometry import Polygon

from ..core import geometry as geo
from ..core.design import Contour, Design, Label, Part, Pocket
from ..core.geometry import Point, Ring
from ..core.layers import ENGRAVE
from .base import GOLDEN, Generator, GeneratorParams, cutting_order_for, slugify, smallest_stock

__all__ = ["TrayStyle", "HandleStyle", "Layout", "TrayParams", "TrayGenerator"]


class TrayStyle(str, Enum):
    """Outer profile family."""

    ROUNDED = "rounded"
    """Rounded rectangle, the workhorse."""
    SOFT = "soft"
    """Superellipse, a softer organic outline."""
    PILL = "pill"
    """Fully rounded ends."""


class HandleStyle(str, Enum):
    """How the tray is picked up."""

    CUTOUT = "cutout"
    """A through slot in the rim at each end."""
    SCALLOP = "scallop"
    """A grip scallop cut into the outer edge at each end."""
    NONE = "none"
    """No handle feature."""


class Layout(str, Enum):
    """How the recessed area is divided."""

    SINGLE = "single"
    HALVES = "halves"
    THIRDS = "thirds"
    GOLDEN = "golden"
    GRID = "grid"
    ONE_PLUS_TWO = "one_plus_two"


#: Relative cell sizes along the primary and secondary axis for each layout.
_LAYOUT_WEIGHTS: dict[Layout, tuple[Sequence[float], Sequence[float]]] = {
    Layout.SINGLE: ((1.0,), (1.0,)),
    Layout.HALVES: ((1.0, 1.0), (1.0,)),
    Layout.THIRDS: ((1.0, 1.0, 1.0), (1.0,)),
    Layout.GOLDEN: ((GOLDEN, 1.0), (1.0,)),
    Layout.GRID: ((1.0, 1.0), (1.0, 1.0)),
}

#: Smallest compartment side that still reads as a compartment, in mm.
MIN_CELL = 45.0
#: The recess must span at least this fraction of the tray's width.  Below it
#: the rim dominates and the tray reads as a picture frame; this is what stops
#: a hand-sized handle cutout from being forced onto a tray too small for it.
MIN_RECESS_FRACTION = 0.62
#: A compartment longer than this many times its width reads as a slot.
MAX_CELL_ASPECT = 4.5
#: How much of an end's straight run a grip scallop may occupy.  A scallop is
#: a detail bitten out of a flat edge; when it is as wide as the flat itself
#: it stops reading as a grip and starts reading as a waist.
SCALLOP_RUN_FRACTION = 0.9

#: Widest scallop a superellipse outline can take, as a fraction of the tray's
#: width.  Measured, not derived: the nearly-flat stretch at the middle of a
#: superellipse end runs to about a sixth of the width, and a scallop wider
#: than that runs off into the curve and is refused.
SOFT_SCALLOP_FRACTION = 0.16
#: Smallest rim that can carry an INFO label, in mm.
MIN_LABEL_RIM = 12.0


class TrayParams(GeneratorParams):
    """Parameters for a tray.

    Dimensions left as ``None`` are derived from the others using the
    proportion rules described in the module docstring.
    """

    length: float = Field(450.0, ge=120.0, le=900.0, description="overall length in mm")
    width: float | None = Field(
        None, ge=90.0, le=700.0, description="overall width in mm; default length/aspect"
    )
    aspect: float = Field(
        GOLDEN, ge=1.0, le=2.6, description="length/width ratio when width is omitted"
    )
    style: TrayStyle = Field(TrayStyle.ROUNDED, description="outer profile family")
    superellipse_exponent: float = Field(
        3.2, ge=2.0, le=8.0, description="curve exponent for the soft style"
    )
    corner_radius: float | None = Field(
        None, ge=0.0, le=350.0, description="outline corner radius in mm; default 10% of width"
    )
    border: float | None = Field(
        None, ge=8.0, le=200.0, description="rim width in mm; default fits the handle"
    )
    layout: Layout = Field(Layout.SINGLE, description="compartment layout")
    divider: float | None = Field(
        None, ge=6.0, le=100.0, description="wall between compartments in mm"
    )
    pocket_depth: float = Field(8.0, gt=0.0, le=40.0, description="recess depth in mm")
    depth_step: float = Field(
        0.0,
        ge=0.0,
        le=12.0,
        description="how much shallower secondary compartments are, in mm",
    )
    pocket_radius: float | None = Field(
        None, ge=0.0, le=200.0, description="compartment corner radius in mm"
    )
    handle: HandleStyle = Field(HandleStyle.CUTOUT, description="handle feature")
    handle_length: float = Field(
        110.0, ge=90.0, le=320.0, description="handle cutout length in mm (>= 90 for a hand)"
    )
    handle_width: float = Field(
        32.0, ge=30.0, le=60.0, description="handle cutout width in mm (>= 30 for a hand)"
    )
    scallop_depth: float = Field(
        14.0, ge=6.0, le=40.0, description="how far a grip scallop cuts into the edge, mm"
    )
    scallop_width: float = Field(
        70.0, ge=40.0, le=220.0, description="grip scallop opening width in mm"
    )
    engrave_border: bool = Field(False, description="add a decorative engraved border line")
    engrave_inset: float = Field(
        6.0, ge=2.0, le=40.0, description="engraved border inset from the edge in mm"
    )
    label: bool = Field(True, description="add an INFO size label on the rim")

    @model_validator(mode="after")
    def _check_depth_step(self) -> "TrayParams":
        if self.depth_step >= self.pocket_depth:
            raise ValueError(
                f"depth_step {self.depth_step} must be less than pocket_depth "
                f"{self.pocket_depth}"
            )
        return self

    # ------------------------------------------------------------- derivations
    def resolved_width(self) -> float:
        """Overall width in mm, derived from ``aspect`` when not given.

        A derived width is rounded to the millimetre: a listing that says
        "278 mm" reads like a product, one that says "278.115 mm" reads like a
        spreadsheet.
        """
        if self.width is not None:
            return self.width
        return float(round(self.length / self.aspect))

    def resolved_corner_radius(self) -> float:
        """Outline corner radius in mm."""
        width = self.resolved_width()
        limit = min(self.length, width) / 2.0
        if self.style is TrayStyle.PILL:
            return limit
        if self.corner_radius is not None:
            return min(self.corner_radius, limit)
        return min(max(6.0, round(width * 0.10)), limit)

    def required_border(self) -> float:
        """Smallest rim that carries the chosen handle with full wall."""
        if self.handle is HandleStyle.CUTOUT:
            return self.handle_width + 2.0 * self.min_wall
        if self.handle is HandleStyle.SCALLOP:
            return self.scallop_depth + self.min_wall
        return self.min_wall + 4.0

    def resolved_border(self) -> float:
        """Rim width in mm.

        Raises:
            ValueError: If an explicit border cannot carry the chosen handle.
        """
        width = self.resolved_width()
        needed = self.required_border()
        if self.border is None:
            border = max(needed, min(max(10.0, round(width * 0.09)), width / 3.0))
        elif self.border < needed - 1e-9:
            raise ValueError(
                f"border {self.border:g} mm is too narrow for a "
                f"{self.handle.value} handle; needs at least {needed:g} mm"
            )
        else:
            border = self.border
        recess = (width - 2.0 * border) / width
        if recess < MIN_RECESS_FRACTION:
            raise ValueError(
                f"a {border:g} mm rim leaves the recess at only "
                f"{recess * 100:.0f}% of the {width:g} mm width, below the "
                f"{MIN_RECESS_FRACTION * 100:.0f}% a tray needs to look like a "
                f"tray; widen the tray or use a {HandleStyle.SCALLOP.value} handle"
            )
        return border

    def resolved_divider(self) -> float:
        """Wall between compartments in mm."""
        if self.divider is not None:
            if self.divider < self.min_wall - 1e-9:
                raise ValueError(
                    f"divider {self.divider:g} mm is below min_wall "
                    f"{self.min_wall:g} mm"
                )
            return self.divider
        return max(self.min_wall + 4.0, 12.0)

    def resolved_pocket_radius(self) -> float:
        """Compartment corner radius in mm, never below what the cutter needs."""
        floor = max(6.0, self.tool_diameter * 0.8)
        return max(floor, self.pocket_radius or 0.0)


def _outline(params: TrayParams) -> Ring:
    """Build the outer profile.

    Args:
        params: Tray parameters.

    Returns:
        A counter-clockwise ring with its bounding box at the origin.
    """
    length, width = params.length, params.resolved_width()
    if params.style is TrayStyle.SOFT:
        ring = geo.superellipse_ring(
            length, width, params.superellipse_exponent, length / 2.0, width / 2.0
        )
        return geo.dedupe(ring)
    return geo.rounded_rect_ring(length, width, params.resolved_corner_radius())


def _scalloped(outline: Ring, params: TrayParams) -> Ring:
    """Cut a grip scallop into each short end of the outline.

    The scallop is a circular bite whose radius follows from the requested
    depth and opening width, so the buyer's thumb lands on a smooth arc rather
    than a corner.  The two junctions where the arc meets the edge are
    filleted, because a round cutter cannot machine the sharp inside corner
    they would otherwise form.

    Args:
        outline: The outline to modify.
        params: Tray parameters.

    Returns:
        The modified ring.

    Raises:
        ValueError: If the scallop would not leave a machinable profile.
    """
    depth, opening = params.scallop_depth, params.scallop_width
    width = params.resolved_width()
    radius = (opening**2 / 4.0 + depth**2) / (2.0 * depth)
    if radius < params.tool_diameter:
        raise ValueError(
            f"a {opening:g} x {depth:g} mm scallop has radius {radius:.1f} mm, "
            f"too tight for a {params.tool_diameter:g} mm cutter"
        )
    if opening > width - 4.0 * params.min_wall:
        raise ValueError(
            f"scallop opening {opening:g} mm leaves too little edge on a "
            f"{width:g} mm wide tray"
        )
    run = _end_straight_run(outline)
    if opening > run * SCALLOP_RUN_FRACTION:
        raise ValueError(
            f"a {opening:g} mm scallop needs a straight end to bite into, but "
            f"this outline only runs straight for {run:.0f} mm; a scallop wider "
            f"than the flat it sits in eats the whole end and the tray reads as "
            f"a dog bone. Use a narrower scallop, the "
            f"{TrayStyle.ROUNDED.value} style, or a different handle"
        )
    poly = geo.polygon_from_ring(outline)
    for cx in (depth - radius, params.length - depth + radius):
        bite = geo.polygon_from_ring(geo.circle_ring(cx, width / 2.0, radius))
        poly = poly.difference(bite)
    if not isinstance(poly, Polygon) or poly.is_empty or poly.interiors:
        raise ValueError("scallops split the tray outline")
    ring = geo.ensure_ccw(geo.dedupe(list(poly.exterior.coords)))
    return geo.round_concave(ring, max(4.0, params.tool_diameter * 0.8))


def _end_straight_run(ring: Ring, tolerance: float = 0.5) -> float:
    """Length of the flat section at the narrow end of an outline.

    Measured as the span of the boundary that lies within ``tolerance`` of the
    outline's minimum x.  A rounded rectangle runs straight for most of its
    end; a pill is curved the whole way and returns almost nothing.

    Args:
        ring: The outline.
        tolerance: How far from the extreme edge still counts as flat, in mm.

    Returns:
        The straight run in mm, 0.0 if there is no flat at all.
    """
    x0 = min(point[0] for point in ring)
    ys = [point[1] for point in ring if point[0] <= x0 + tolerance]
    return max(ys) - min(ys) if len(ys) > 1 else 0.0


def _split(lo: float, hi: float, weights: Sequence[float], gap: float) -> list[tuple[float, float]]:
    """Divide ``[lo, hi]`` into weighted intervals separated by ``gap``.

    Args:
        lo: Interval start.
        hi: Interval end.
        weights: Relative sizes; length determines the cell count.
        gap: Space left between cells.

    Returns:
        One ``(start, end)`` pair per weight.

    Raises:
        ValueError: If the gaps consume the whole interval.
    """
    usable = (hi - lo) - gap * (len(weights) - 1)
    if usable <= 0:
        raise ValueError(
            f"{len(weights)} cells with {gap:g} mm dividers do not fit in "
            f"{hi - lo:g} mm"
        )
    total = float(sum(weights))
    out: list[tuple[float, float]] = []
    cursor = lo
    for weight in weights:
        size = usable * weight / total
        out.append((cursor, cursor + size))
        cursor += size + gap
    return out


def _cell_rects(
    bounds: tuple[float, float, float, float], layout: Layout, gap: float
) -> list[tuple[tuple[float, float, float, float], int]]:
    """Divide the recess envelope's bounding box into compartment cells.

    The primary axis is whichever is longer, so a layout reads the same way on
    a landscape or portrait tray.

    Args:
        bounds: ``(x0, y0, x1, y1)`` of the recess envelope.
        layout: Which division to apply.
        gap: Divider width in mm.

    Returns:
        ``[(rect, rank), ...]`` where rank 0 marks the primary compartment.

    Raises:
        ValueError: If the layout does not fit.
    """
    x0, y0, x1, y1 = bounds
    horizontal = (x1 - x0) >= (y1 - y0)

    def rect(a0: float, a1: float, b0: float, b1: float) -> tuple[float, float, float, float]:
        return (a0, b0, a1, b1) if horizontal else (b0, a0, b1, a1)

    pa0, pa1 = (x0, x1) if horizontal else (y0, y1)
    sa0, sa1 = (y0, y1) if horizontal else (x0, x1)

    if layout is Layout.ONE_PLUS_TWO:
        primary = _split(pa0, pa1, (GOLDEN, 1.0), gap)
        cells = [(rect(primary[0][0], primary[0][1], sa0, sa1), 0)]
        for b0, b1 in _split(sa0, sa1, (1.0, 1.0), gap):
            cells.append((rect(primary[1][0], primary[1][1], b0, b1), 1))
        return cells

    primary_w, secondary_w = _LAYOUT_WEIGHTS[layout]
    cells = []
    for index, (a0, a1) in enumerate(_split(pa0, pa1, primary_w, gap)):
        for jndex, (b0, b1) in enumerate(_split(sa0, sa1, secondary_w, gap)):
            cells.append((rect(a0, a1, b0, b1), 0 if index == 0 and jndex == 0 else 1))
    return cells


def _pocket_ring(
    envelope: Polygon,
    rect: tuple[float, float, float, float],
    bounds: tuple[float, float, float, float],
    radius: float,
) -> Ring:
    """Clip one compartment out of the recess envelope and fillet its corners.

    Cell rectangles are grown outwards past the envelope before intersecting,
    so a cell edge never lands exactly on the envelope boundary where shapely
    would produce slivers.

    Args:
        envelope: The recess envelope polygon.
        rect: The cell rectangle.
        bounds: The envelope's bounding box.
        radius: Corner radius to apply, in mm.

    Returns:
        A closed ring for the compartment, with every convex corner rounded to
        a radius the cutter can follow.

    Raises:
        ValueError: If the clipped cell is not a single simple region, or is
            smaller than a compartment should be.
    """
    x0, y0, x1, y1 = rect
    bx0, by0, bx1, by1 = bounds
    pad = 50.0
    grown = (
        x0 - pad if abs(x0 - bx0) < 1e-6 else x0,
        y0 - pad if abs(y0 - by0) < 1e-6 else y0,
        x1 + pad if abs(x1 - bx1) < 1e-6 else x1,
        y1 + pad if abs(y1 - by1) < 1e-6 else y1,
    )
    clip = geo.polygon_from_ring(
        geo.rect_ring(grown[2] - grown[0], grown[3] - grown[1], grown[0], grown[1])
    )
    cell = envelope.intersection(clip)
    if not isinstance(cell, Polygon) or cell.is_empty or cell.interiors:
        raise ValueError("a compartment did not clip to a single simple region")
    width, height = geo.size_of(list(cell.exterior.coords))
    if min(width, height) < MIN_CELL:
        raise ValueError(
            f"compartment {width:.0f} x {height:.0f} mm is below the "
            f"{MIN_CELL:g} mm minimum"
        )
    aspect = max(width, height) / min(width, height)
    if aspect > MAX_CELL_ASPECT:
        raise ValueError(
            f"compartment {width:.0f} x {height:.0f} mm is {aspect:.1f}:1, "
            f"which reads as a slot rather than a compartment"
        )
    ring = geo.ensure_ccw(geo.dedupe(list(cell.exterior.coords)))
    return geo.round_convex(ring, min(radius, min(width, height) / 2.0 - 0.01))


def _handle_slots(params: TrayParams, outline: Ring) -> list[Ring]:
    """Build the two handle cutouts as slots swept along the rim.

    Sweeping a disc along the curve that runs half a rim inside the profile
    gives a slot of exactly ``handle_width`` whose wall thickness is
    ``(border - handle_width) / 2`` at every point, on a straight rim or a
    curved one.  The rim width is derived from the handle for precisely this
    reason, so the wall comes out at ``min_wall`` with nothing left to check.

    Args:
        params: Tray parameters.
        outline: The outer profile.

    Returns:
        Two closed rings.

    Raises:
        ValueError: If two ergonomic handles do not fit on the rim.
    """
    width = params.resolved_width()
    border = params.resolved_border()
    length = min(params.handle_length, max(90.0, width * 0.55))
    if length < 90.0:
        raise ValueError(
            f"a {width:g} mm wide tray cannot carry a 90 mm handle cutout"
        )
    centreline = geo.offset_centreline(outline, border / 2.0)
    perimeter = geo.perimeter(centreline)
    if 2.0 * length + 4.0 * params.min_wall > perimeter:
        raise ValueError(
            f"two {length:g} mm handles do not fit a {perimeter:.0f} mm rim"
        )
    slots = [
        geo.curved_slot(centreline, anchor, length, params.handle_width)
        for anchor in ((-20.0, width / 2.0), (params.length + 20.0, width / 2.0))
    ]
    first, second = (geo.polygon_from_ring(ring) for ring in slots)
    if first.distance(second) < params.min_wall - 1e-6:
        raise ValueError(
            f"the two handle cutouts come within "
            f"{first.distance(second):.1f} mm of each other"
        )
    return slots


class TrayGenerator(Generator):
    """Generates serving, valet and snack trays."""

    niche = "trays"
    title = "Trays"
    summary = "Serving, valet and snack trays with recessed compartments and handles"
    params_model = TrayParams

    def generate(self, params: TrayParams) -> Design:  # type: ignore[override]
        """Build one tray.

        Args:
            params: Tray parameters.

        Returns:
            The design, with the bounding box at the origin.

        Raises:
            ValueError: If the parameters are geometrically incompatible.  The
                message names the offending dimension.
        """
        width = params.resolved_width()
        border = params.resolved_border()
        divider = params.resolved_divider()

        # The recess is laid out from the base outline, before any scallop is
        # bitten out of it.  Offsetting the scalloped outline instead makes the
        # end compartments inherit the scallop's curve and come out waisted,
        # which reads as a mistake rather than a detail.  The rim is wide
        # enough that the scallop still leaves min_wall to the recess, because
        # required_border() derives it that way.
        base_outline = _outline(params)
        outline = (
            _scalloped(base_outline, params)
            if params.handle is HandleStyle.SCALLOP
            else base_outline
        )

        envelopes = geo.offset_ring(base_outline, -border, join="round")
        if len(envelopes) != 1:
            raise ValueError(
                f"a {border:g} mm rim leaves no single recess area on a "
                f"{params.length:g} x {width:g} mm tray"
            )
        envelope = geo.polygon_from_ring(envelopes[0])
        bounds = geo.bbox(envelopes[0])
        if min(bounds[2] - bounds[0], bounds[3] - bounds[1]) < MIN_CELL:
            raise ValueError(
                f"a {border:g} mm rim leaves only "
                f"{min(bounds[2] - bounds[0], bounds[3] - bounds[1]):.0f} mm "
                f"of recess; minimum is {MIN_CELL:g} mm"
            )

        radius = params.resolved_pocket_radius()
        secondary_depth = params.pocket_depth - params.depth_step
        if secondary_depth < 2.0:
            raise ValueError(
                f"secondary compartments would be only {secondary_depth:g} mm deep"
            )
        pockets: list[Pocket] = []
        for rect, rank in _cell_rects(bounds, params.layout, divider):
            ring = _pocket_ring(envelope, rect, bounds, radius)
            pockets.append(Pocket(ring, params.pocket_depth if rank == 0 else secondary_depth))

        holes: list[Ring] = []
        if params.handle is HandleStyle.CUTOUT:
            holes = _handle_slots(params, outline)

        engrave: list[Contour] = []
        if params.engrave_border:
            inset = min(params.engrave_inset, border - params.tool_diameter)
            if inset < 2.0:
                raise ValueError(
                    f"a {border:g} mm rim has no room for an engraved border"
                )
            for ring in geo.offset_ring(outline, -inset, join="round"):
                engrave.append(Contour(ring, ENGRAVE, closed=True))

        labels: list[Label] = []
        text = f"{params.length:g} x {width:g} x {params.thickness:g} mm  {params.material}"
        height = min(6.0, border * 0.35)
        if params.label and border >= MIN_LABEL_RIM and 0.62 * height * len(text) < params.length - 2 * border:
            labels.append(Label(text, (params.length / 2.0, border * 0.42), height, align="center"))

        part = Part(
            name="tray",
            outline=outline,
            holes=holes,
            pockets=pockets,
            engrave=engrave,
            labels=labels,
        )
        # Scallops shorten the outline, so every name, slug and stock size is
        # taken from what the part actually measures.  A listing that quotes a
        # dimension the file does not have is a returned order.
        real_length, real_width = (round(v) for v in geo.size_of(outline))
        design = Design(
            slug=self._slug(params, real_length, real_width),
            name=self._name(params, real_length, real_width, len(pockets)),
            niche=self.niche,
            description=self._description(params, real_length, real_width, pockets),
            parts=[part],
            machine=params.machine(),
            material=params.material,
            thickness=params.thickness,
            sheet=smallest_stock(real_length, real_width),
            params=params.model_dump(mode="json"),
            notes=self._notes(params, border, divider, pockets),
            limits=params.validation_config(),
        )
        design.cutting_order = cutting_order_for(design)
        return design

    # ------------------------------------------------------------------ naming
    @staticmethod
    def _kind(length: float, width: float, cells: int) -> str:
        """Name the product category from its size and compartment count."""
        if cells >= 3:
            return "snack tray"
        if length * width < 90_000:
            return "valet tray"
        return "serving tray"

    def _slug(self, params: TrayParams, length: float, width: float) -> str:
        return slugify(
            "tray",
            params.style.value,
            params.layout.value,
            f"{length:g}x{width:g}",
            params.handle.value,
            f"d{params.pocket_depth:g}",
        )

    def _name(self, params: TrayParams, length: float, width: float, cells: int) -> str:
        kind = self._kind(length, width, cells).title()
        compartments = "single compartment" if cells == 1 else f"{cells} compartments"
        return (
            f"{params.style.value.title()} {kind}, {length:g} x {width:g} mm, "
            f"{compartments}"
        )

    def _description(
        self, params: TrayParams, length: float, width: float, pockets: list[Pocket]
    ) -> str:
        depths = sorted({p.depth for p in pockets})
        depth_text = (
            f"{depths[0]:g} mm deep"
            if len(depths) == 1
            else f"{depths[-1]:g} mm and {depths[0]:g} mm deep"
        )
        handle_text = {
            HandleStyle.CUTOUT: f"handle cutouts {params.handle_length:g} x {params.handle_width:g} mm",
            HandleStyle.SCALLOP: f"grip scallops {params.scallop_width:g} mm wide",
            HandleStyle.NONE: "no handles",
        }[params.handle]
        return (
            f"A {self._kind(length, width, len(pockets))} measuring "
            f"{length:g} x {width:g} mm in {params.thickness:g} mm "
            f"{params.material}, with {len(pockets)} recessed "
            f"compartment{'s' if len(pockets) != 1 else ''} {depth_text}, and "
            f"{handle_text}. Cut with a {params.tool_diameter:g} mm cutter; "
            f"compartment corners are pre-filleted so no corner relief is needed."
        )

    def _notes(
        self, params: TrayParams, border: float, divider: float, pockets: list[Pocket]
    ) -> list[str]:
        deepest = max(p.depth for p in pockets)
        notes = [
            f"Rim width {border:g} mm, divider width {divider:g} mm.",
            f"Deepest recess {deepest:g} mm leaves a "
            f"{params.thickness - deepest:g} mm floor.",
        ]
        if params.handle is HandleStyle.CUTOUT:
            notes.append(
                f"Handle cutouts are {params.handle_width:g} mm wide, sized for a hand."
            )
        return notes

    # ------------------------------------------------------------------ variants
    def sample_params(self, rng: random.Random, index: int) -> TrayParams:
        """Draw one tray variant.

        Sizes snap to 10 mm and proportions come from a curated list, so every
        draw is a plausible product rather than a random rectangle.
        """
        aspect = rng.choice([GOLDEN, GOLDEN, 1.5, 1.4, 1.33, 2.0, 1.25])
        length = float(rng.randrange(20, 60) * 10)
        width = round(length / aspect / 10.0) * 10.0
        style = rng.choice(
            [TrayStyle.ROUNDED, TrayStyle.ROUNDED, TrayStyle.ROUNDED, TrayStyle.SOFT, TrayStyle.PILL]
        )
        layout = rng.choice(
            [
                Layout.SINGLE,
                Layout.SINGLE,
                Layout.HALVES,
                Layout.THIRDS,
                Layout.GOLDEN,
                Layout.GRID,
                Layout.ONE_PLUS_TWO,
            ]
        )
        # A hand needs 30 mm of slot, and the rim carrying it must still leave
        # the recess dominant, so only trays above a certain width can wear a
        # cutout handle at all.
        widest_handle = (1.0 - MIN_RECESS_FRACTION) / 2.0 * width - 2.0 * 8.0
        # A scallop needs a straight run at the end to bite into.  A rounded
        # rectangle has one the length of its flat edge; a superellipse has
        # only the nearly-flat stretch at its middle, which measures out at
        # about a sixth of the tray's width and disappears below 240 mm; a
        # pill has none at all.  The earlier comment here claimed the soft
        # outline had a flat end, and a twentieth of sampled trays were
        # rejected for believing it.
        widest_scallop = (
            width * SOFT_SCALLOP_FRACTION if style is TrayStyle.SOFT else width
        )
        scallop_ok = style is not TrayStyle.PILL and widest_scallop >= 40.0
        if widest_handle >= 30.0:
            choices = [HandleStyle.CUTOUT] * 3 + [HandleStyle.NONE]
            if scallop_ok:
                choices.append(HandleStyle.SCALLOP)
        else:
            choices = [HandleStyle.NONE]
            if scallop_ok:
                choices += [HandleStyle.SCALLOP, HandleStyle.SCALLOP]
        handle = rng.choice(choices)
        # Floored at the schema minimum: this value is passed whatever the
        # handle is, and a tray with a cutout handle must not be rejected over
        # a scallop width it never uses.  scallop_ok above has already ruled
        # out choosing a scallop where 40 mm will not fit.
        scallop_width = float(
            max(
                40.0,
                min(
                    rng.randrange(40, 100, 10)
                    if style is TrayStyle.ROUNDED
                    else rng.randrange(40, 56, 4),
                    widest_scallop,
                ),
            )
        )
        handle_width = float(rng.randrange(30, max(32, int(min(44, widest_handle))) + 1, 2))
        thickness = rng.choice([19.0, 19.0, 25.0, 18.0])
        depth = rng.choice([6.0, 8.0, 8.0, 10.0, 12.0])
        depth = min(depth, thickness - 5.0 - 1.0)
        return TrayParams(
            length=length,
            width=width,
            aspect=aspect,
            style=style,
            superellipse_exponent=rng.choice([2.6, 3.2, 4.0]),
            corner_radius=None if rng.random() < 0.7 else float(rng.randrange(6, 40, 2)),
            layout=layout,
            pocket_depth=depth,
            depth_step=rng.choice([0.0, 0.0, 2.0, 3.0]) if layout is not Layout.SINGLE else 0.0,
            handle=handle,
            handle_length=float(rng.randrange(90, 160, 10)),
            handle_width=handle_width,
            scallop_depth=float(rng.randrange(10, 20, 2)),
            scallop_width=scallop_width,
            engrave_border=rng.random() < 0.35,
            engrave_inset=float(rng.randrange(4, 12, 2)),
            material=rng.choice(["oak", "walnut", "maple", "birch plywood", "cherry"]),
            thickness=thickness,
        )
