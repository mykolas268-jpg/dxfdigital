"""Flat-pack slot-together furniture cut from sheet material.

Two forms, both assembled without fasteners:

``shelf``
    Two uprights and a set of shelves whose tabs pass through mortises in the
    uprights and finish flush with the outside faces.

``table``
    Two legs cross-lapped into each other, with tabs through the top.  A
    cross-lap works here, unlike in the phone dock, because both legs are
    vertical planes: each carries a slot half its height, one open at the top
    and one at the bottom, and they slide together.

Parts are nested onto stock sheets by the shared shelf packer and labelled on
INFO, because a sheet of eight similar panels is unusable otherwise.  When a
design needs more than one sheet they are laid out side by side in the same
file, which is how flat-pack files are normally shipped.
"""

from __future__ import annotations

import math
import random
from enum import Enum
from typing import Sequence

from pydantic import Field, model_validator

from ..core import geometry as geo
from ..core.assembly import Assembly, Placement, Plane
from ..core.design import Design, Label, Part
from ..core.geometry import Point, Ring
from .base import (
    Generator,
    GeneratorParams,
    cutting_order_for,
    label_parts,
    material_phrase,
    nest_parts,
    slugify,
)

__all__ = ["FurnitureForm", "FurnitureParams", "FurnitureGenerator"]

#: Standard structural sheet, in mm.
SHEET = (1220.0, 2440.0)
#: Space left between nested sheets when a design needs more than one.
SHEET_GAP = 60.0

#: Upper bound on sampled shelf counts; the schema allows more, but past
#: this the shelves are closer together than anything you would stand on one.
MAX_SHELVES: int = 6

#: Material left beyond a wedge slot so the tenon does not split out.
END_GRAIN_MARGIN: float = 10.0


class FurnitureForm(str, Enum):
    """Which piece to build."""

    SHELF = "shelf"
    TABLE = "table"


class FurnitureParams(GeneratorParams):
    """Parameters for a flat-pack piece.

    Defaults are structural: 18 mm sheet, a 6.35 mm cutter and a full
    1220 x 2440 mm panel.
    """

    form: FurnitureForm = Field(FurnitureForm.SHELF, description="which piece to build")
    thickness: float = Field(18.0, gt=0, le=40, description="sheet thickness in mm")
    material: str = Field(
        "birch plywood", min_length=1, max_length=60, description="material description"
    )
    width: float = Field(760.0, ge=250.0, le=1200.0, description="overall width in mm")
    depth: float = Field(320.0, ge=150.0, le=800.0, description="overall depth in mm")
    height: float = Field(900.0, ge=180.0, le=2000.0, description="overall height in mm")
    shelves: int = Field(3, ge=1, le=8, description="shelf count, for the shelf form")
    wedges: bool = Field(
        False, description="wedge the shelf tenons instead of relying on friction"
    )
    wedge_slot: float = Field(
        16.0, ge=8.0, le=40.0, description="wedge slot length along the tenon, mm"
    )
    wedge_bite: float = Field(
        1.5, ge=0.5, le=6.0,
        description="how far the slot reaches back inside the upright, mm",
    )
    tabs: int = Field(2, ge=1, le=5, description="tabs per joint")
    tab_width: float = Field(60.0, ge=25.0, le=200.0, description="tab width in mm")
    foot_arch: float = Field(
        60.0, ge=0.0, le=200.0, description="height of the arch cut into each foot, mm"
    )
    foot_inset: float = Field(
        70.0, ge=20.0, le=300.0, description="how far the arch is inset from each side, mm"
    )
    corner_radius: float = Field(
        12.0, ge=0.0, le=60.0, description="corner radius on the panels in mm"
    )
    relief: bool = Field(True, description="dogbone the mortises and inside corners")
    sheet_gap: float = Field(
        14.0, ge=6.0, le=60.0, description="space between nested parts in mm"
    )

    @model_validator(mode="after")
    def _check_proportions(self) -> "FurnitureParams":
        if self.form is FurnitureForm.SHELF and self.height < self.depth:
            raise ValueError(
                f"a shelf unit {self.height:g} mm tall and {self.depth:g} mm deep "
                f"is a sideboard; swap the dimensions"
            )
        if self.foot_arch > 0 and self.foot_inset * 2 >= min(self.width, self.depth):
            raise ValueError(
                f"a {self.foot_inset:g} mm foot inset leaves no arch on a "
                f"{min(self.width, self.depth):g} mm panel"
            )
        return self

    def relief_radius(self) -> float:
        """Radius of the relief disc used on inside corners, in mm."""
        return self.tool_diameter / 2.0 * geo.RELIEF_OVERSIZE

    def relief_margin(self) -> float:
        """How far past a mortise the relief reaches, in mm."""
        if not self.relief or self.machine().is_laser:
            return geo.ARC_TOLERANCE
        return self.relief_radius() * 0.35 + geo.ARC_TOLERANCE

    def mortise_size(self) -> tuple[float, float]:
        """``(length, width)`` of a mortise for one tab, in mm."""
        machine = self.machine()
        return (machine.slot_width(self.tab_width), machine.slot_width(self.thickness))


def widest_tab(params: FurnitureParams, length: float) -> float:
    """The widest tab that fits along one joint, in mm.

    Solves the inequality :func:`_tab_spans` checks rather than restating it,
    so the two cannot drift apart.

    Args:
        params: Furniture parameters.
        length: The edge length the tabs are spread along, in mm.

    Returns:
        The widest workable tab, which may be zero or less on an edge too
        short for this tab count.
    """
    edge = (
        max(params.min_wall, params.corner_radius + params.tool_diameter / 2.0)
        + params.relief_margin()
        + params.clearance / 2.0
    )
    span = length - 2.0 * edge
    return (span - (params.tabs - 1) * params.min_wall) / params.tabs


def _tab_spans(params: FurnitureParams, length: float) -> list[tuple[float, float]]:
    """Where the tabs sit along a joint, in the panel's coordinates.

    Args:
        params: Furniture parameters.
        length: The edge length the tabs are spread along, in mm.

    Returns:
        ``[(start, end), ...]``.

    Raises:
        ValueError: If the tabs do not fit.
    """
    # The margin must clear the panel's corner radius by a whole cutter
    # radius.  Relief will not fire on an edge shorter than that, so a tab root
    # left sitting just past the end of a corner arc - with a 1 mm scrap of
    # straight edge beside it - silently stays unmachinable.
    edge = (
        max(params.min_wall, params.corner_radius + params.tool_diameter / 2.0)
        + params.relief_margin()
        + params.clearance / 2.0
    )
    span = length - 2.0 * edge
    needed = params.tabs * params.tab_width + (params.tabs - 1) * params.min_wall
    if needed > span:
        raise ValueError(
            f"{params.tabs} tabs {params.tab_width:g} mm wide need {needed:.0f} mm "
            f"along a {length:g} mm edge, which leaves {span:.0f} mm"
        )
    if params.tabs == 1:
        start = (length - params.tab_width) / 2.0
        return [(start, start + params.tab_width)]
    gap = (span - params.tabs * params.tab_width) / (params.tabs - 1)
    return [
        (edge + i * (params.tab_width + gap), edge + i * (params.tab_width + gap) + params.tab_width)
        for i in range(params.tabs)
    ]


def _mortise(params: FurnitureParams, x0: float, y0: float, length: float) -> Ring:
    """One mortise, relieved where the machine needs it."""
    _, width = params.mortise_size()
    ring = geo.rect_ring(length, width, x0, y0)
    if params.machine().is_laser or not params.relief:
        return ring
    return geo.apply_relief(ring, params.tool_diameter / 2.0, style="dogbone")


def tenon_reach(params: FurnitureParams) -> float:
    """How far a shelf's tenon protrudes past the upright, in mm.

    A friction tenon stops flush with the outside of the upright.  A wedged
    one has to carry its slot plus enough material beyond it not to split, so
    it stands proud - which is what a wedged joint looks like, and is meant to.

    Args:
        params: Furniture parameters.

    Returns:
        The protrusion measured from the upright's inner face.
    """
    if not params.wedges:
        return params.thickness
    return params.thickness - params.wedge_bite + params.wedge_slot + END_GRAIN_MARGIN


def _wedge_slots(
    params: FurnitureParams, length: float, spans: list[tuple[float, float]]
) -> list[Ring]:
    """The slot in each tenon that its wedge is driven through.

    The slot starts ``wedge_bite`` *inside* the upright's outer face.  That is
    the whole mechanism: the wedge's straight side bears on the upright, its
    tapered side on the far end of the slot, so driving it pushes the slot
    outwards and pulls the shelf's shoulder tight.  A slot starting flush with
    the upright has nothing left to pull with.

    Args:
        params: Furniture parameters.
        length: Panel length between the tab shoulders, in mm.
        spans: Tab positions along the panel's height.

    Returns:
        Closed rings, in panel coordinates, for both edges.
    """
    if not params.wedges:
        return []
    thickness = params.thickness
    opening = thickness + params.clearance
    near = thickness - params.wedge_bite
    rings: list[Ring] = []
    for start, end in spans:
        middle = (start + end) / 2.0
        for sign, base in ((1.0, length), (-1.0, 0.0)):
            x0 = base + near * sign if sign > 0 else base - (near + params.wedge_slot)
            rings.append(
                geo.rect_ring(params.wedge_slot, opening, x0, middle - opening / 2.0)
            )
    # A hole is relieved the other way round from a profile: the cutter works
    # inside it, so its corners get dogbones, exactly as a mortise does.
    if params.machine().is_laser or not params.relief:
        return rings
    return [
        geo.apply_relief(ring, params.tool_diameter / 2.0, style="dogbone")
        for ring in rings
    ]


def _tabbed_edge(
    params: FurnitureParams, length: float, height: float, spans: list[tuple[float, float]]
) -> Ring:
    """A panel with tabs protruding from its left and right edges.

    Args:
        params: Furniture parameters.
        length: Panel length between the tab shoulders, in mm.
        height: Panel height in mm.
        spans: Tab positions along the height.

    Returns:
        A counter-clockwise ring.
    """
    reach = tenon_reach(params)
    ring: list[Point] = [(0.0, 0.0), (length, 0.0)]
    for start, end in spans:
        ring.extend(
            [
                (length, start),
                (length + reach, start),
                (length + reach, end),
                (length, end),
            ]
        )
    ring.append((length, height))
    ring.append((0.0, height))
    for start, end in reversed(spans):
        ring.extend([(0.0, end), (-reach, end), (-reach, start), (0.0, start)])
    return _relieve(params, geo.dedupe(ring))


def _relieve(params: FurnitureParams, ring: Ring) -> Ring:
    """Relieve every inside corner on a profile.

    Tab roots, arch junctions and the open end of a cross-lap slot are all
    inside corners, and a round cutter cannot cut one.  Applying this once at
    the end of every profile is the only way to be sure none is missed - doing
    it inside the arch routine meant a panel without an arch got none.

    Args:
        params: Furniture parameters.
        ring: The profile.

    Returns:
        The relieved profile, unchanged on a laser or with relief switched off.
    """
    if params.machine().is_laser or not params.relief:
        return ring
    return geo.apply_relief(
        ring, params.relief_radius(), style="dogbone", region="part"
    )


def _arched(params: FurnitureParams, ring: Ring, width: float) -> Ring:
    """Cut a standing arch into the bottom edge of an upright.

    Two feet look intentional and sit flat on an uneven floor; a full edge
    rocks.  The arch is a stadium, so its ends are already radiused, and the
    two junctions with the bottom edge are relieved like any other inside
    corner.
    """
    if params.foot_arch <= 0:
        return ring
    span = width - 2.0 * params.foot_inset
    if span < params.foot_arch * 1.2:
        raise ValueError(
            f"a {span:.0f} mm arch opening is too narrow for a "
            f"{params.foot_arch:g} mm rise"
        )
    arch = geo.stadium_ring(span, params.foot_arch * 2.0, params.foot_inset, -params.foot_arch)
    cut = geo.polygon_from_ring(ring).difference(geo.polygon_from_ring(arch))
    if cut.is_empty or cut.geom_type != "Polygon":
        raise ValueError("the foot arch cut the upright in two")
    return geo.ensure_ccw(geo.dedupe(list(cut.exterior.coords)))


def shelf_span(params: FurnitureParams) -> tuple[float, float]:
    """The lowest and highest a shelf's underside may sit, in mm.

    Args:
        params: Furniture parameters.

    Returns:
        ``(lowest, highest)``.

    Raises:
        ValueError: If the arch and the top leave no room for a shelf.
    """
    thickness = params.thickness
    # The lowest shelf has to clear the foot arch: a mortise cut into the arch
    # leaves a sliver of material between the two.
    margin = params.min_wall + params.relief_margin()
    lowest = max(thickness, params.foot_arch + margin)
    # The top shelf's mortise must finish inside the upright, not run off it.
    highest = params.height - thickness - params.clearance - margin
    if highest <= lowest:
        raise ValueError(
            f"a {params.height:g} mm upright with a {params.foot_arch:g} mm "
            f"arch has no room between the feet and the top for a shelf"
        )
    return lowest, highest


def shelves_for_spacing(params: FurnitureParams, spacing: float) -> int:
    """How many shelves land about ``spacing`` apart in this upright.

    Shelves are spread between the lowest and highest a mortise may sit, so
    ``n`` shelves leave ``n - 1`` gaps across that span, not ``n``.  Sizing by
    height over spacing instead is how a 1560 mm upright ends up with two
    shelves 1400 mm apart.

    Args:
        params: Furniture parameters; only the upright matters, not the
            shelf count already on them.
        spacing: Wanted clear distance between shelves, in mm.

    Returns:
        A shelf count of at least two, capped at :data:`MAX_SHELVES`.

    Raises:
        ValueError: If the upright has no room for a shelf at all.
    """
    if spacing <= 0.0:
        raise ValueError(f"spacing must be > 0, got {spacing}")
    lowest, highest = shelf_span(params)
    return max(2, min(MAX_SHELVES, 1 + round((highest - lowest) / spacing)))


def shelf_levels(params: FurnitureParams) -> list[float]:
    """The height of each shelf's underside above the floor, in mm.

    Both the upright's mortises and the assembly drawing read this, so a
    shelf cannot be drawn at a height its mortise was not cut for.

    Args:
        params: Furniture parameters.

    Returns:
        One height per shelf, lowest first.

    Raises:
        ValueError: If the shelves do not fit between the feet and the top.
    """
    lowest, highest = shelf_span(params)
    thickness = params.thickness
    if params.shelves == 1:
        return [lowest]
    step = (highest - lowest) / (params.shelves - 1)
    if step < thickness + 4.0 * params.min_wall:
        raise ValueError(
            f"{params.shelves} shelves in a {params.height:g} mm upright "
            f"leaves only {step:.0f} mm between them"
        )
    return [lowest + index * step for index in range(params.shelves)]


def _wedge_part(params: FurnitureParams, quantity: int) -> Part:
    """The tapered key driven through a tenon's slot.

    Cut from the same sheet, so its thickness is the material's and the slot
    it passes through is one material thickness plus the fit clearance wide.
    It tapers along its length, so tapping it further always tightens the
    joint; a parallel key would only ever be as tight as it was cut.

    Args:
        params: Furniture parameters.
        quantity: How many to cut.

    Returns:
        The part.

    Raises:
        ValueError: If the taper leaves no tip.
    """
    slot = params.wedge_slot
    # Wide enough at the head to still be driving after the joint has taken
    # up, narrow enough at the tip to start by hand.
    head, tip = slot * 0.92, slot * 0.42
    length = max(52.0, params.thickness * 3.4)
    if tip < 4.0:
        raise ValueError(
            f"a {slot:g} mm wedge slot tapers to a {tip:.1f} mm tip, too fine "
            f"to cut or to tap"
        )
    collar = slot * 1.35
    shoulder = length * 0.16
    ring: Ring = [
        (0.0, 0.0),
        (tip, 0.0),
        (head, length - shoulder),
        (collar, length - shoulder),
        (collar, length),
        (0.0, length),
    ]
    return Part(
        name="wedge",
        outline=_relieve(params, geo.dedupe(ring)),
        quantity=quantity,
    )


def _shelf_unit(params: FurnitureParams) -> list[Part]:
    """Two uprights and a stack of shelves."""
    thickness = params.thickness
    inner_width = params.width - 2.0 * thickness
    spans = _tab_spans(params, params.depth)
    length_mortise, _ = params.mortise_size()

    heights = shelf_levels(params)

    holes: list[Ring] = []
    for level in heights:
        for start, end in spans:
            holes.append(
                _mortise(
                    params,
                    start - params.clearance / 2.0,
                    level - params.clearance / 2.0,
                    (end - start) + params.clearance,
                )
            )
    upright = geo.rect_ring(params.depth, params.height)
    if params.corner_radius > 0:
        upright = geo.round_convex(upright, params.corner_radius)
    upright = _relieve(params, _arched(params, upright, params.depth))

    shelf = Part(
        name="shelf",
        outline=_tabbed_edge(params, inner_width, params.depth, spans),
        holes=_wedge_slots(params, inner_width, spans),
        quantity=params.shelves,
    )
    parts = [
        Part(name="upright", outline=list(upright), holes=list(holes), quantity=2),
        shelf,
    ]
    if params.wedges:
        # One per tenon: every shelf, both ends, every tab.
        parts.append(
            _wedge_part(params, params.shelves * 2 * params.tabs)
        )
    return parts


def _cross_table(params: FurnitureParams) -> list[Part]:
    """Two cross-lapped legs and a top.

    Raises:
        ValueError: On an odd tab count, which no cross-lapped table can take.
    """
    if params.tabs % 2:
        raise ValueError(
            f"a cross-lapped table needs an even tab count, not {params.tabs}: "
            f"the legs cross at the centre of the top, and an odd count puts a "
            f"tab there too, so its mortise lands on top of the other leg's"
        )
    thickness = params.thickness
    machine = params.machine()
    slot = machine.slot_width(thickness)
    leg_height = params.height - thickness

    def leg(span: float, slot_from_top: bool) -> Ring:
        spans = _tab_spans(params, span)
        # The panel is rounded first and the tabs are added afterwards.  Doing
        # it the other way round rounds the tabs too, and a tab with a 12 mm
        # radius on it no longer matches the square mortise it goes into.
        base = geo.rect_ring(span, leg_height)
        if params.corner_radius > 0:
            base = geo.round_convex(base, params.corner_radius)
        poly = geo.polygon_from_ring(base)
        for start, end in spans:
            poly = poly.union(
                geo.polygon_from_ring(
                    geo.rect_ring(
                        end - start, thickness + 1.0, start, leg_height - 1.0
                    )
                )
            )
        if poly.geom_type != "Polygon" or poly.interiors:
            raise ValueError("the leg tabs did not merge cleanly with the panel")
        outline = geo.ensure_ccw(geo.dedupe(list(poly.exterior.coords)))
        outline = _arched(params, outline, span)
        # The cross-lap slot: half the leg height, open at one edge.
        x0 = (span - slot) / 2.0
        y0 = leg_height / 2.0 if slot_from_top else -thickness
        notch = geo.rect_ring(slot, leg_height / 2.0 + thickness, x0, y0)
        cut = geo.polygon_from_ring(outline).difference(geo.polygon_from_ring(notch))
        if cut.is_empty or cut.geom_type != "Polygon":
            raise ValueError("the cross-lap slot cut a leg in two")
        return _relieve(params, geo.ensure_ccw(geo.dedupe(list(cut.exterior.coords))))

    top = geo.rounded_rect_ring(params.width, params.depth, max(params.corner_radius, 8.0))
    holes: list[Ring] = []
    length_mortise, width_mortise = params.mortise_size()
    for axis, span, extent in (("x", params.width, params.depth), ("y", params.depth, params.width)):
        for start, end in _tab_spans(params, span):
            if axis == "x":
                x0, y0 = start - params.clearance / 2.0, (params.depth - width_mortise) / 2.0
                holes.append(
                    _mortise(params, x0, y0, (end - start) + params.clearance)
                )
            else:
                ring = geo.rect_ring(
                    width_mortise,
                    (end - start) + params.clearance,
                    (params.width - width_mortise) / 2.0,
                    start - params.clearance / 2.0,
                )
                if not machine.is_laser and params.relief:
                    ring = geo.apply_relief(
                        ring, params.tool_diameter / 2.0, style="dogbone"
                    )
                holes.append(ring)

    return [
        Part(name="top", outline=top, holes=holes),
        Part(name="leg-long", outline=leg(params.width - 2 * params.corner_radius, True)),
        Part(name="leg-short", outline=leg(params.depth - 2 * params.corner_radius, False)),
    ]


def _wedge_placements(
    params: FurnitureParams, levels: Sequence[float]
) -> list[Placement]:
    """Where each wedge stands, driven through its tenon outside the upright.

    A wedge stands on edge in the plane of the shelf's length, so its face is
    the one you tap.  It is drawn at the slot it passes through, one per
    tenon.

    Args:
        params: Furniture parameters.
        levels: Shelf heights.

    Returns:
        One placement per wedge, named to match the parts nesting produced.
    """
    thickness = params.thickness
    reach = tenon_reach(params)
    spans = _tab_spans(params, params.depth)
    # A wedge's own zero is its tip, and it is driven downwards, so it starts
    # below the shelf by about a third of its length.
    drop = max(52.0, thickness * 3.4)
    number = 0
    placements: list[Placement] = []
    for level in levels:
        for side, base in ((1.0, width_of(params) - thickness), (-1.0, thickness)):
            for start, end in spans:
                number += 1
                x = base + (reach - params.wedge_slot - END_GRAIN_MARGIN) * side
                if side < 0:
                    x = base - reach + END_GRAIN_MARGIN
                placements.append(
                    Placement(
                        f"wedge-{number}",
                        Plane.FRONT,
                        (x, (start + end) / 2.0 - thickness / 2.0,
                         level + thickness - drop * 0.55),
                    )
                )
    return placements


def width_of(params: FurnitureParams) -> float:
    """The piece's overall width in mm, for readability at the call site."""
    return params.width


def _assembly(params: FurnitureParams) -> Assembly:
    """Where every panel stands in the finished piece.

    Part names here are the ones nesting leaves behind: a part cut twice is
    expanded into ``upright-1`` and ``upright-2``, so the assembly names those
    rather than the single ``upright`` the builder created.

    Args:
        params: Furniture parameters.

    Returns:
        The assembly.

    Raises:
        ValueError: If the shelves do not fit, which shelf_levels reports.
    """
    thickness = params.thickness
    width, depth, height = params.width, params.depth, params.height

    if params.form is FurnitureForm.SHELF:
        placements = [
            Placement("upright-1", Plane.SIDE, (0.0, 0.0, 0.0)),
            Placement("upright-2", Plane.SIDE, (width - thickness, 0.0, 0.0)),
        ]
        levels = shelf_levels(params)
        if params.wedges:
            placements += _wedge_placements(params, levels)
        # A shelf's tabs reach out past its body into the uprights' mortises,
        # so its own origin sits one thickness in and the tabs land on zero.
        placements += [
            Placement(f"shelf-{index + 1}", Plane.FLAT, (thickness, 0.0, level))
            for index, level in enumerate(levels)
        ]
        caption = (
            f"{width:g} x {depth:g} x {height:g} mm shelf unit, "
            f"{params.shelves} shelves"
        )
        return Assembly(tuple(placements), caption)

    # Cross-lapped table: each leg spans one axis, inset by the corner radius
    # it was shortened by, and they pass through each other at the centre.
    radius = params.corner_radius
    placements = [
        Placement("leg-long", Plane.FRONT, (radius, (depth - thickness) / 2.0, 0.0)),
        Placement("leg-short", Plane.SIDE, ((width - thickness) / 2.0, radius, 0.0)),
        Placement("top", Plane.FLAT, (0.0, 0.0, height - thickness)),
    ]
    return Assembly(
        tuple(placements),
        f"{width:g} x {depth:g} x {height:g} mm cross-leg table",
    )


class FurnitureGenerator(Generator):
    """Generates flat-pack slot-together furniture."""

    niche = "furniture"
    title = "Furniture"
    summary = "Flat-pack slot-together shelves and tables, nested onto sheet stock"
    params_model = FurnitureParams

    def generate(self, params: FurnitureParams) -> Design:  # type: ignore[override]
        """Build one piece and nest it onto stock sheets.

        Args:
            params: Furniture parameters.

        Returns:
            The design, parts nested and labelled.

        Raises:
            ValueError: If the parameters are geometrically incompatible, or a
                part does not fit the stock sheet.
        """
        if params.form is FurnitureForm.SHELF:
            parts = _shelf_unit(params)
        else:
            parts = _cross_table(params)

        sheets = nest_parts(parts, SHEET, gap=params.sheet_gap)
        placed: list[Part] = []
        for index, sheet_parts in enumerate(sheets):
            offset = index * (SHEET[0] + SHEET_GAP)
            for part in sheet_parts:
                part.origin = (part.origin[0] + offset, part.origin[1])
                placed.append(part)
        label_parts(placed, height=max(8.0, params.thickness * 0.6))

        # The sheet marker goes inside the area that sheet's parts occupy.
        # Anywhere else and it falls outside the drawing extents, where it is
        # cropped out of previews and ignored by CAD's zoom-to-fit.
        for index, sheet_parts in enumerate(sheets):
            anchor = sheet_parts[0]
            x0, _, x1, y1 = geo.bbox_of(p.placed().outline for p in sheet_parts)
            anchor.labels.append(
                Label(
                    f"SHEET {index + 1} OF {len(sheets)}  "
                    f"{SHEET[0]:g} x {SHEET[1]:g} mm stock",
                    (
                        (x0 + x1) / 2.0 - anchor.origin[0],
                        y1 - 26.0 - anchor.origin[1],
                    ),
                    22.0,
                    align="center",
                )
            )

        extent = geo.size_of([p for part in placed for p in part.placed().outline])
        design = Design(
            slug=slugify(
                "furniture", params.form.value,
                f"{params.width:g}x{params.depth:g}x{params.height:g}",
                f"t{params.thickness:g}",
                "wedged" if params.wedges else "friction",
            ),
            name=self._name(params),
            niche=self.niche,
            description=self._description(params, len(sheets)),
            parts=placed,
            machine=params.machine(),
            material=params.material,
            thickness=params.thickness,
            sheet=(
                len(sheets) * SHEET[0] + (len(sheets) - 1) * SHEET_GAP,
                SHEET[1],
            ),
            params=params.model_dump(mode="json"),
            notes=self._notes(params, len(sheets), len(placed)),
            limits=params.validation_config(),
            assembly=_assembly(params),
        )
        design.cutting_order = cutting_order_for(design)
        return design

    def _name(self, params: FurnitureParams) -> str:
        if params.form is FurnitureForm.SHELF:
            return (
                f"Flat-pack Shelf Unit, {params.width:g} x {params.depth:g} x "
                f"{params.height:g} mm, {params.shelves} shelves"
            )
        return (
            f"Flat-pack Cross-leg Table, {params.width:g} x {params.depth:g} x "
            f"{params.height:g} mm"
        )

    def _description(self, params: FurnitureParams, sheets: int) -> str:
        stock = material_phrase(params.thickness, params.material)
        if params.form is FurnitureForm.SHELF:
            body = (
                f"Two uprights and {params.shelves} shelves, joined by tabs "
                f"through the uprights that finish flush with the outside faces"
            )
        else:
            body = (
                "Two cross-lapped legs and a top, joined by tabs through the "
                "top; the legs slot into each other"
            )
        return (
            f"{self._name(params)} in {stock}. {body}. No fasteners and no "
            f"glue needed. Parts are nested onto {sheets} sheet"
            f"{'s' if sheets != 1 else ''} of "
            f"{SHEET[0]:g} x {SHEET[1]:g} mm stock and labelled on the INFO "
            f"layer. Cut with a {params.tool_diameter:g} mm cutter."
        )

    def _notes(self, params: FurnitureParams, sheets: int, parts: int) -> list[str]:
        length, width = params.mortise_size()
        if params.wedges:
            over = tenon_reach(params) - params.thickness
            notes = [
                f"Wedged through-tenons. The {params.width:g} mm width is the "
                f"carcass; the tenons stand {over:.0f} mm proud of each upright, "
                f"so the piece measures {params.width + 2 * over:.0f} mm across "
                f"the wedges.",
                "Drive each wedge with a mallet until the shelf shoulder pulls "
                "tight against the upright. Tap them again after a week; that "
                "is the point of a wedge and the reason there is no glue here.",
                "Wedges are cut from the same sheet, so their thickness is the "
                "material's. Cut a spare or two; they are the part that gets "
                "lost.",
            ]
        else:
            notes = []
        notes += [
            f"{parts} parts nested onto {sheets} sheet"
            f"{'s' if sheets != 1 else ''} of {SHEET[0]:g} x {SHEET[1]:g} mm.",
            f"Mortises are drawn {length:.2f} x {width:.2f} mm for "
            f"{params.tab_width:g} mm tabs in {params.thickness:g} mm sheet, a "
            f"{params.clearance:g} mm fit.",
            "Tabs come through their mortises by one sheet thickness and finish "
            "flush; sand any proud edges rather than forcing the joint.",
            "The joints are a friction fit. For a piece that will be moved "
            "often, glue the tabs or fit wedges.",
        ]
        if sheets > 1:
            notes.append(
                f"The {sheets} sheets are laid out side by side in the same "
                f"file, {SHEET_GAP:g} mm apart. Cut them one at a time."
            )
        if params.relief and not params.machine().is_laser:
            notes.append(
                "Mortises and inside corners carry dogbone relief so square "
                "tabs and shoulders seat fully."
            )
        return notes

    def sample_params(self, rng: random.Random, index: int) -> FurnitureParams:
        """Draw one furniture variant."""
        form = rng.choice([FurnitureForm.SHELF, FurnitureForm.SHELF, FurnitureForm.TABLE])
        thickness = rng.choice([18.0, 18.0, 15.0, 21.0])
        if form is FurnitureForm.SHELF:
            width = float(rng.randrange(60, 100) * 10)
            depth = float(rng.randrange(24, 40) * 10)
            height = float(rng.randrange(70, 180) * 10)
            # Shelf count follows the height.  Drawn independently it gives a
            # 1560 mm upright with two shelves 1400 mm apart, which is a frame
            # rather than a bookcase - obvious the moment the piece is drawn
            # assembled, invisible while it is a nest of flat panels.
            spacing = float(rng.randrange(280, 420, 20))
            shelves = 2  # replaced below, once the upright is known
        else:
            width = float(rng.randrange(45, 110) * 10)
            depth = float(rng.randrange(35, 70) * 10)
            height = float(rng.randrange(34, 76) * 10)
            shelves = 1
        # The arch is cut into the span left between the two feet, so it has
        # to fit there.  Drawn independently it asks for a 160 mm arch across
        # a 40 mm span, which is not a shallow arch, it is no panel at all.
        foot_inset = float(rng.randrange(50, 100, 10))
        shortest = min(depth, width if form is FurnitureForm.TABLE else depth)
        span = shortest - 2.0 * foot_inset
        foot_arch = rng.choice([0.0, 50.0, 60.0, 80.0])
        if foot_arch * 2.0 > span:
            foot_arch = max(0.0, float(int(span / 2.0 / 10.0) * 10.0))
        tabs = (
            rng.choice([2, 2, 4]) if form is FurnitureForm.TABLE
            else rng.choice([2, 2, 3])
        )
        tab_width = float(rng.randrange(40, 90, 10))
        drawn = FurnitureParams(
            form=form,
            thickness=thickness,
            width=width,
            depth=min(depth, width - 80.0),
            height=height,
            shelves=shelves,
            tabs=tabs,
            tab_width=tab_width,
            # Wedges are shelf work: a cross-lapped table is held by its own
            # geometry, so there is nothing for a wedge to pull tight.
            wedges=form is FurnitureForm.SHELF and rng.random() < 0.45,
            wedge_slot=float(rng.randrange(14, 24, 2)),
            foot_arch=foot_arch,
            foot_inset=foot_inset,
            corner_radius=float(rng.randrange(0, 20, 4)),
            material=rng.choice(["birch plywood", "poplar plywood", "oak veneer ply"]),
        )
        # A tab has to fit the shortest joint it runs along, so the panels are
        # drawn first and the tab sized to them.
        joints = (
            [drawn.depth] if form is FurnitureForm.SHELF
            else [drawn.width - 2 * drawn.corner_radius,
                  drawn.depth - 2 * drawn.corner_radius]
        )
        widest = min(widest_tab(drawn, length) for length in joints)
        drawn = drawn.model_copy(
            update={"tab_width": max(25.0, min(tab_width, widest * 0.85))}
        )
        if form is not FurnitureForm.SHELF:
            return drawn
        # The shelf count needs the upright it goes in, so it is settled once
        # the rest is drawn rather than guessed beforehand.
        return drawn.model_copy(
            update={"shelves": shelves_for_spacing(drawn, spacing)}
        )
