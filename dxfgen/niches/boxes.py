"""Finger-jointed laser boxes.

Four sides interlock at the vertical corners with finger joints; the floor is a
separate panel whose tabs pass through mortises in the side faces, so it sits
on a small plinth rather than trying to join three panels at one vertex.  That
choice is the whole design: a floor finger-jointed to the sides as well would
put three panels in competition for the same corner cube, which is where box
generators earn their reputation.

Corner geometry, in one paragraph.  Every panel is a plain rectangle of the
box's outer dimensions, and the joint is cut *into* it rather than added on:
along a vertical corner the front panel keeps the corner for some cells and
gives it up for the others, and the side panel does the exact opposite.  The
two patterns are generated from one call with ``male`` flipped, so they cannot
drift apart.

Kerf is handled once, on the whole part, for the reason set out in
:meth:`dxfgen.core.design.Part.kerf_compensated`: correcting the slots alone
leaves every joint a kerf loose, because the beam takes the same kerf off the
fingers.
"""

from __future__ import annotations

import random
from enum import Enum
from typing import Sequence

from pydantic import Field, model_validator

from ..core import geometry as geo
from ..core.assembly import Assembly, Placement, Plane
from ..core.design import Design, Label, Mode, Part
from ..core.geometry import Point, Ring
from .base import (
    Generator,
    GeneratorParams,
    apply_kerf,
    arrange_for_stock,
    cutting_order_for,
    material_phrase,
    slugify,
    smallest_stock,
)

#: Sheet sizes a laser cutter actually has, smallest first.  The router
#: stock list runs up to a 1220 x 2440 mm panel, which no hobby laser bed
#: takes, and offering it would pick layouts nobody can cut.
LASER_STOCK: tuple[tuple[float, float], ...] = (
    (300.0, 200.0),
    (400.0, 300.0),
    (600.0, 400.0),
    (900.0, 600.0),
)

__all__ = ["LidStyle", "BoxParams", "BoxGenerator"]


class LidStyle(str, Enum):
    """What closes the box."""

    NONE = "none"
    """An open-topped box."""
    CAP = "cap"
    """A two-part lift-off lid: a top plate with a locator plate under it."""


#: Smallest inside dimension worth calling a box, in mm.
MIN_INSIDE = 30.0


class BoxParams(GeneratorParams):
    """Parameters for a finger-jointed box.

    The machine defaults are overridden here: a finger-jointed box is a laser
    object in thin sheet, and inheriting a 19 mm router default would make
    every sensible box refuse itself.
    """

    mode: Mode = Field(Mode.LASER, description="router or laser; boxes are laser work")
    material: str = Field(
        "birch ply", min_length=1, max_length=60, description="material description"
    )
    thickness: float = Field(3.0, gt=0, le=25, description="material thickness in mm")
    min_wall: float = Field(
        4.5, ge=1, le=50, description="minimum material between features in mm"
    )
    pocket_floor: float = Field(
        1.0, ge=0.2, le=30, description="material that must remain under a pocket in mm"
    )
    width: float = Field(160.0, ge=50.0, le=600.0, description="outside width in mm")
    depth: float = Field(110.0, ge=50.0, le=600.0, description="outside depth in mm")
    height: float = Field(70.0, ge=30.0, le=400.0, description="outside height in mm")
    fingers_long: int | None = Field(
        None, ge=3, le=31, description="finger count on the long edges; odd, auto by default"
    )
    fingers_short: int | None = Field(
        None, ge=3, le=31, description="finger count on the short edges; odd, auto by default"
    )
    floor_offset: float | None = Field(
        None, ge=0.0, le=60.0, description="floor height above the bottom edge in mm"
    )
    floor_tabs: int = Field(2, ge=1, le=6, description="floor tabs per side")
    columns: int = Field(
        1, ge=1, le=6, description="compartments across the width"
    )
    rows: int = Field(1, ge=1, le=6, description="compartments across the depth")
    divider_tabs: int = Field(
        2, ge=1, le=4, description="tabs holding each divider into its walls"
    )
    floor_tab_width: float = Field(
        20.0, ge=8.0, le=60.0, description="floor tab width in mm"
    )
    lid: LidStyle = Field(LidStyle.NONE, description="lid style")
    lid_clearance: float = Field(
        0.4, ge=0.0, le=2.0, description="gap around the lid locator in mm"
    )
    gap: float = Field(8.0, ge=3.0, le=40.0, description="spacing between parts on the sheet, mm")
    label: bool = Field(True, description="add an INFO label to the floor")

    @model_validator(mode="after")
    def _check_inside(self) -> "BoxParams":
        for name, value in (("width", self.width), ("depth", self.depth)):
            if value - 2.0 * self.thickness < MIN_INSIDE:
                raise ValueError(
                    f"{name} {value:g} mm leaves only "
                    f"{value - 2 * self.thickness:.0f} mm inside in "
                    f"{self.thickness:g} mm material"
                )
        return self

    def resolved_floor_offset(self) -> float:
        """Height of the floor's underside above the bottom edge, in mm."""
        if self.floor_offset is not None:
            return self.floor_offset
        return max(self.thickness, self.min_wall)

    def widest_floor_tab(self, box_length: float) -> float:
        """The widest floor tab that fits along one side, in mm.

        Solves the inequality :func:`_tab_spans` checks rather than restating
        it, so the two cannot drift apart.  The shortest side governs, since
        one tab width serves all four.

        Args:
            box_length: The box dimension along the side, in mm.

        Returns:
            The widest workable tab, which may be zero or less on a side too
            short for this tab count.
        """
        span = box_length - 2.0 * (self.thickness + self.min_wall)
        return (span - (self.floor_tabs - 1) * self.min_wall) / self.floor_tabs

    def fingers_for(self, length: float, override: int | None) -> int:
        """Finger count for an edge, odd so the pattern stays symmetric."""
        if override is not None:
            return override if override % 2 else override + 1
        return geo.suggest_finger_count(length, self.thickness)

    def inside(self) -> tuple[float, float, float]:
        """Inside ``(width, depth, height)`` in mm."""
        return (
            self.width - 2.0 * self.thickness,
            self.depth - 2.0 * self.thickness,
            self.height - self.resolved_floor_offset() - self.thickness,
        )


def _panel(
    width: float,
    height: float,
    thickness: float,
    left_male: bool,
    right_male: bool,
    fingers: int,
) -> Ring:
    """Build a side panel with finger joints on its left and right edges.

    The joint is cut into the rectangle rather than added to it: a cell where
    this panel keeps the corner stays at the panel's full width, and a cell
    where the neighbour keeps it is set back by one thickness.  The mating
    panel is built with the same call and ``male`` flipped, so the two
    patterns are complementary by construction.

    Args:
        width: Panel width in mm.
        height: Panel height in mm.
        thickness: Material thickness in mm.
        left_male: Whether this panel keeps the corner on its left edge's
            first cell.
        right_male: The same for the right edge.
        fingers: Cell count along the vertical edges.

    Returns:
        A counter-clockwise ring.
    """
    cell = height / fingers
    ring: list[Point] = []
    # Left edge, walked upwards; x alternates between 0 and thickness.
    for index in range(fingers):
        keeps = (index % 2 == 0) == left_male
        x = 0.0 if keeps else thickness
        ring.append((x, index * cell))
        ring.append((x, (index + 1) * cell))
    ring.append((width, height))
    # Right edge, walked downwards.
    for index in reversed(range(fingers)):
        keeps = (index % 2 == 0) == right_male
        x = width if keeps else width - thickness
        ring.append((x, (index + 1) * cell))
        ring.append((x, index * cell))
    ring.append((0.0, 0.0))
    return geo.dedupe(ring)


def _tab_spans(params: BoxParams, box_length: float, count: int) -> list[tuple[float, float]]:
    """Where the floor's tabs sit along one side, in *box* coordinates.

    Both the tabs and the mortises they pass through are derived from this one
    function.  Computing them separately, one in panel coordinates and one in
    floor coordinates, is how they end up a material thickness apart.

    Args:
        params: Box parameters.
        box_length: The box dimension along this side, in mm.
        count: How many tabs.

    Returns:
        ``[(start, end), ...]`` in box coordinates.

    Raises:
        ValueError: If the tabs do not fit.
    """
    tab = params.floor_tab_width
    lo = params.thickness + params.min_wall
    hi = box_length - params.thickness - params.min_wall
    span = hi - lo
    needed = count * tab + (count - 1) * params.min_wall
    if needed > span:
        raise ValueError(
            f"{count} floor tabs {tab:g} mm wide need {needed:.0f} mm along a "
            f"{box_length:g} mm side, which leaves {span:.0f} mm"
        )
    if count == 1:
        start = (box_length - tab) / 2.0
        return [(start, start + tab)]
    gap = (span - count * tab) / (count - 1)
    return [(lo + i * (tab + gap), lo + i * (tab + gap) + tab) for i in range(count)]


def _floor_mortises(
    params: BoxParams, panel_width: float, count: int
) -> list[Ring]:
    """Mortises in a side panel's face for the floor's tabs.

    The mortise is the tab plus the fit clearance, centred on it in both
    directions, so the floor drops in without being forced.

    Args:
        params: Box parameters.
        panel_width: The box dimension this panel spans, in mm.
        count: How many mortises.

    Returns:
        Closed rings in panel coordinates, which are box coordinates here.

    Raises:
        ValueError: If the mortises do not fit across the panel.
    """
    offset = params.resolved_floor_offset()
    slop = params.clearance
    height = params.thickness + slop
    rings: list[Ring] = []
    for start, end in _tab_spans(params, panel_width, count):
        rings.append(
            geo.rect_ring(
                (end - start) + slop, height, start - slop / 2.0, offset - slop / 2.0
            )
        )
    return rings


def _floor(params: BoxParams) -> Part:
    """Build the floor panel with tabs on all four edges."""
    thickness = params.thickness
    inner_w = params.width - 2.0 * thickness
    inner_d = params.depth - 2.0 * thickness

    def tab_spans(box_length: float, count: int) -> list[tuple[float, float]]:
        """Tab spans in floor-local coordinates, shifted off the box ones."""
        return [
            (start - thickness, end - thickness)
            for start, end in _tab_spans(params, box_length, count)
        ]

    ring: list[Point] = []
    # Bottom edge, left to right, tabs pointing down.
    for start, end in tab_spans(params.width, params.floor_tabs):
        ring.extend([(start, 0.0), (start, -thickness), (end, -thickness), (end, 0.0)])
    # Right edge, bottom to top, tabs pointing right.
    ring.append((inner_w, 0.0))
    for start, end in tab_spans(params.depth, params.floor_tabs):
        ring.extend(
            [
                (inner_w, start),
                (inner_w + thickness, start),
                (inner_w + thickness, end),
                (inner_w, end),
            ]
        )
    ring.append((inner_w, inner_d))
    # Top edge, right to left, tabs pointing up.
    for start, end in reversed(tab_spans(params.width, params.floor_tabs)):
        ring.extend(
            [
                (end, inner_d),
                (end, inner_d + thickness),
                (start, inner_d + thickness),
                (start, inner_d),
            ]
        )
    ring.append((0.0, inner_d))
    # Left edge, top to bottom, tabs pointing left.
    for start, end in reversed(tab_spans(params.depth, params.floor_tabs)):
        ring.extend([(0.0, end), (-thickness, end), (-thickness, start), (0.0, start)])
    ring.append((0.0, 0.0))

    labels: list[Label] = []
    if params.label:
        inside = params.inside()
        text = f"{inside[0]:.0f} x {inside[1]:.0f} x {inside[2]:.0f} mm inside"
        if 0.62 * 4.0 * len(text) < inner_w * 0.85:
            labels.append(
                Label(text, (inner_w / 2.0, inner_d / 2.0), 4.0, align="center")
            )
    return Part(name="floor", outline=geo.dedupe(ring), labels=labels)


def _divided(params: BoxParams) -> bool:
    """Whether this box has any dividers in it."""
    return params.columns > 1 or params.rows > 1


def divider_positions(params: BoxParams) -> tuple[list[float], list[float]]:
    """Where the dividers stand, in box coordinates.

    A divider's position is the centre of its thickness, measured from the
    outside of the box, so the walls and the assembly drawing can both place
    it from the same number.

    Args:
        params: Box parameters.

    Returns:
        ``(column_x, row_y)`` - the x of each divider that splits the width,
        and the y of each that splits the depth.  Either may be empty.
    """
    thickness = params.thickness
    inner_w = params.width - 2.0 * thickness
    inner_d = params.depth - 2.0 * thickness
    columns = [
        thickness + inner_w * index / params.columns
        for index in range(1, params.columns)
    ]
    rows = [
        thickness + inner_d * index / params.rows for index in range(1, params.rows)
    ]
    return columns, rows


def _divider_tab_spans(params: BoxParams) -> list[tuple[float, float]]:
    """Where a divider's tabs sit up its height, in divider coordinates.

    The lowest tab clears the floor's own mortises, which sit at the floor
    offset: two slots a couple of millimetres apart in the same wall is a
    wall that snaps.

    Args:
        params: Box parameters.

    Returns:
        ``[(bottom, top), ...]`` measured from the divider's bottom edge.

    Raises:
        ValueError: If the tabs do not fit up the divider.
    """
    height = divider_height(params)
    # Both ends carry the clearance too: the slot below is the floor's own
    # mortise, grown half a clearance, and above is the wall's top edge.
    margin = params.min_wall + params.clearance
    span = height - 2.0 * margin
    count = params.divider_tabs
    # The wall left between two tabs is the gap less the fit clearance, since
    # each mortise is drawn half a clearance proud at both ends.  Budgeting
    # only min_wall here leaves min_wall minus clearance in the wall, which
    # the validator rejects by exactly that margin.
    between = params.min_wall + params.clearance
    length = min(28.0, (span - (count - 1) * between) / count)
    if length < 8.0:
        raise ValueError(
            f"a {height:.0f} mm divider has no room for {count} tabs; it needs "
            f"about {count * (8.0 + params.min_wall) + 2 * margin:.0f} mm of height"
        )
    # Divided by count - 1: n tabs leave n-1 gaps between them, not n.  With
    # count over the whole leftover the tabs bunch up and leave half the wall
    # they should, which the validator catches as a 2.2 mm web.
    gap = max(between, (span - count * length) / (count - 1)) if count > 1 else 0.0
    return [
        (margin + index * (length + gap), margin + index * (length + gap) + length)
        for index in range(count)
    ]


def divider_height(params: BoxParams) -> float:
    """How tall a divider stands, in mm.

    From the top of the floor to the top of the walls, so a divided box has
    its compartments closed right up to the rim.

    Args:
        params: Box parameters.

    Returns:
        The height in mm.
    """
    return params.height - params.resolved_floor_offset() - params.thickness


def _divider_wall_mortises(
    params: BoxParams, positions: Sequence[float]
) -> list[Ring]:
    """Slots in a wall for the dividers that land on it.

    Args:
        params: Box parameters.
        positions: Divider centres along this wall, in box coordinates.

    Returns:
        Closed rings in the wall's own coordinates, which match the box's
        along the wall and measure height from the wall's bottom edge.
    """
    slop = params.clearance
    width = params.thickness + slop
    base = params.resolved_floor_offset() + params.thickness
    rings: list[Ring] = []
    for centre in positions:
        for bottom, top in _divider_tab_spans(params):
            rings.append(
                geo.rect_ring(
                    width,
                    (top - bottom) + slop,
                    centre - width / 2.0,
                    base + bottom - slop / 2.0,
                )
            )
    return rings


def _divider_part(
    params: BoxParams,
    name: str,
    box_length: float,
    crossings: Sequence[float],
    slot_from_top: bool,
) -> Part:
    """Build one divider: tabs at both ends, cross-lap slots where it meets
    the dividers running the other way.

    Args:
        params: Box parameters.
        name: Part name.
        box_length: The box dimension this divider spans, in mm.
        crossings: Positions of the dividers it crosses, in box coordinates.
        slot_from_top: Whether its cross-lap slots are cut down from its top
            edge.  The two directions must disagree, or they cannot interlock.

    Returns:
        The part.

    Raises:
        ValueError: If a cross-lap slot cuts the divider in two.
    """
    thickness = params.thickness
    height = divider_height(params)
    inner = box_length - 2.0 * thickness

    ring: list[Point] = [(0.0, 0.0), (inner, 0.0)]
    # Right edge, bottom to top, tabs pointing right into the wall.
    for bottom, top in _divider_tab_spans(params):
        ring.extend(
            [(inner, bottom), (inner + thickness, bottom),
             (inner + thickness, top), (inner, top)]
        )
    ring.extend([(inner, height), (0.0, height)])
    # Left edge, top to bottom, tabs pointing left.
    for bottom, top in reversed(_divider_tab_spans(params)):
        ring.extend(
            [(0.0, top), (-thickness, top), (-thickness, bottom), (0.0, bottom)]
        )
    panel = geo.polygon_from_ring(geo.dedupe(ring))

    slot = params.machine().slot_width(thickness)
    for centre in crossings:
        # Crossings are given in box coordinates; the divider's own zero sits
        # one thickness in from the box's.
        local = centre - thickness
        y0 = height / 2.0 if slot_from_top else -1.0
        notch = geo.rect_ring(slot, height / 2.0 + 1.0, local - slot / 2.0, y0)
        panel = panel.difference(geo.polygon_from_ring(notch))
        if panel.is_empty or panel.geom_type != "Polygon":
            raise ValueError(
                f"a cross-lap slot cut the {name} divider in two; the "
                f"compartments are too small for this material"
            )
    return Part(name=name, outline=geo.ensure_ccw(geo.dedupe(list(panel.exterior.coords))))


def _divider_parts(params: BoxParams) -> list[Part]:
    """Every divider panel, or none when the box is undivided."""
    columns, rows = divider_positions(params)
    parts: list[Part] = []
    # A divider splitting the width spans the depth, and vice versa.  The two
    # directions take their cross-lap slots from opposite edges so they can be
    # dropped into each other.
    for index, _x in enumerate(columns, 1):
        parts.append(
            _divider_part(params, f"divider-col-{index}", params.depth, rows, True)
        )
    for index, _y in enumerate(rows, 1):
        parts.append(
            _divider_part(params, f"divider-row-{index}", params.width, columns, False)
        )
    return parts


def _lid_parts(params: BoxParams) -> list[Part]:
    """Build the two lid plates, if a lid was asked for."""
    if params.lid is LidStyle.NONE:
        return []
    clearance = params.lid_clearance
    top = geo.rounded_rect_ring(params.width, params.depth, params.thickness)
    locator = geo.rect_ring(
        params.width - 2.0 * params.thickness - 2.0 * clearance,
        params.depth - 2.0 * params.thickness - 2.0 * clearance,
    )
    return [
        Part(name="lid-top", outline=top),
        Part(name="lid-locator", outline=locator),
    ]


def _assembly(params: BoxParams) -> Assembly:
    """Where every panel sits in the finished box.

    The walls each span the full outside dimension and overlap at the corners,
    which is not a clash: the fingers alternate up the height, so at any one
    height only one wall has material there.  That is the joint working, and
    it is why the placements below can all start at zero.

    Args:
        params: Box parameters.

    Returns:
        The assembly.
    """
    thickness = params.thickness
    width, depth, height = params.width, params.depth, params.height
    floor_z = params.resolved_floor_offset()
    placements = [
        Placement("floor", Plane.FLAT, (thickness, thickness, floor_z)),
        Placement("front", Plane.FRONT, (0.0, 0.0, 0.0)),
        Placement("back", Plane.FRONT, (0.0, depth - thickness, 0.0)),
        Placement("left", Plane.SIDE, (0.0, 0.0, 0.0)),
        Placement("right", Plane.SIDE, (width - thickness, 0.0, 0.0)),
    ]
    columns, rows = divider_positions(params)
    base = params.resolved_floor_offset() + thickness
    half = thickness / 2.0
    # A divider's own zero sits one thickness in from the box's, because its
    # tabs reach back out through the walls to reach it.
    for index, x in enumerate(columns, 1):
        placements.append(
            Placement(f"divider-col-{index}", Plane.SIDE, (x - half, thickness, base))
        )
    for index, y in enumerate(rows, 1):
        placements.append(
            Placement(f"divider-row-{index}", Plane.FRONT, (thickness, y - half, base))
        )
    if params.lid is not LidStyle.NONE:
        clearance = params.lid_clearance
        placements += [
            Placement(
                "lid-locator",
                Plane.FLAT,
                (thickness + clearance, thickness + clearance, height - thickness),
            ),
            Placement("lid-top", Plane.FLAT, (0.0, 0.0, height)),
        ]
    compartments = params.columns * params.rows
    caption = f"{width:g} x {depth:g} x {height:g} mm box"
    if compartments > 1:
        caption = (
            f"{width:g} x {depth:g} x {height:g} mm box, "
            f"{params.columns} x {params.rows} compartments"
        )
    if params.lid is not LidStyle.NONE:
        caption = (
            f"{width:g} x {depth:g} mm box, {height:g} mm body and its lid on top"
        )
    return Assembly(tuple(placements), caption)


class BoxGenerator(Generator):
    """Generates finger-jointed laser boxes."""

    niche = "boxes"
    title = "Boxes"
    summary = "Finger-jointed laser boxes with a tabbed floor and an optional lid"
    params_model = BoxParams

    def generate(self, params: BoxParams) -> Design:  # type: ignore[override]
        """Build one box.

        Args:
            params: Box parameters.

        Returns:
            The design, parts laid out on the sheet.

        Raises:
            ValueError: If the parameters are geometrically incompatible.
        """
        thickness = params.thickness
        long_fingers = params.fingers_for(params.height, params.fingers_long)
        short_fingers = long_fingers  # both meet on the same vertical edges

        # Front and back keep the corner on their first cell; left and right
        # give it up, so the two patterns interlock.
        front = _panel(params.width, params.height, thickness, True, True, long_fingers)
        side = _panel(params.depth, params.height, thickness, False, False, short_fingers)

        # A divider splitting the width spans the depth, so its tabs come out
        # through the front and back walls; the front and back walls run along
        # the width, so that is where their slots sit.
        columns, rows = divider_positions(params)
        long_wall_slots = _divider_wall_mortises(params, columns)
        short_wall_slots = _divider_wall_mortises(params, rows)
        parts = [
            Part(
                name="front",
                outline=list(front),
                holes=_floor_mortises(params, params.width, params.floor_tabs)
                + long_wall_slots,
            ),
            Part(
                name="back",
                outline=list(front),
                holes=_floor_mortises(params, params.width, params.floor_tabs)
                + long_wall_slots,
            ),
            Part(
                name="left",
                outline=list(side),
                holes=_floor_mortises(params, params.depth, params.floor_tabs)
                + short_wall_slots,
            ),
            Part(
                name="right",
                outline=list(side),
                holes=_floor_mortises(params, params.depth, params.floor_tabs)
                + short_wall_slots,
            ),
            _floor(params),
            *_divider_parts(params),
            *_lid_parts(params),
        ]
        parts = apply_kerf(parts, params)
        parts, layout = arrange_for_stock(parts, params.gap, sizes=LASER_STOCK)
        width, height = geo.size_of(
            [point for part in parts for point in part.placed().outline]
        )
        design = Design(
            slug=slugify(
                "box", f"{params.width:g}x{params.depth:g}x{params.height:g}",
                f"t{params.thickness:g}",
                f"{params.columns}x{params.rows}" if _divided(params) else "open",
                f"lid-{params.lid.value}",
                params.mode.value,
            ),
            name=(
                (
                    f"Finger-jointed Compartment Box {params.width:g} x "
                    f"{params.depth:g} x {params.height:g} mm, "
                    f"{params.columns * params.rows} compartments"
                    if _divided(params)
                    else f"Finger-jointed Box {params.width:g} x {params.depth:g} x "
                    f"{params.height:g} mm"
                )
                + (" with lid" if params.lid is not LidStyle.NONE else "")
            ),
            niche=self.niche,
            description=self._description(params, long_fingers),
            parts=parts,
            machine=params.machine(),
            material=params.material,
            thickness=thickness,
            sheet=smallest_stock(width, height),
            params=params.model_dump(mode="json"),
            notes=[
                *self._notes(params, long_fingers),
                f"The parts are laid out as {layout}, chosen because it needs "
                f"the smallest sheet of the layouts tried.",
            ],
            limits=params.validation_config(),
            assembly=_assembly(params),
        )
        design.cutting_order = cutting_order_for(design)
        return design

    def _description(self, params: BoxParams, fingers: int) -> str:
        inside = params.inside()
        machine = params.machine()
        fit = (
            f"every outline is grown and every cutout shrunk by half the "
            f"{params.kerf:g} mm kerf, so fingers and slots both finish on size"
            if machine.is_laser
            else f"slots carry a {params.clearance:g} mm fit clearance"
        )
        return (
            f"A finger-jointed box measuring {params.width:g} x {params.depth:g} "
            f"x {params.height:g} mm outside and {inside[0]:.0f} x "
            f"{inside[1]:.0f} x {inside[2]:.0f} mm inside, in "
            f"{material_phrase(params.thickness, params.material)}. Corners "
            f"interlock with "
            f"{fingers} fingers; the floor sits on tabs through the sides. No "
            f"glue needed for assembly: {fit}."
        )

    def _notes(self, params: BoxParams, fingers: int) -> list[str]:
        inside = params.inside()
        notes = [
            f"Inside dimensions {inside[0]:.0f} x {inside[1]:.0f} x "
            f"{inside[2]:.0f} mm.",
            f"Corner joints use {fingers} fingers of "
            f"{params.height / fingers:.1f} mm each.",
            f"The floor sits {params.resolved_floor_offset():g} mm above the "
            f"bottom edge, so the box stands on a plinth rather than on the "
            f"floor panel.",
            "Assemble the four sides first, then drop the floor in from above "
            "and push its tabs through.",
        ]
        if params.machine().is_laser:
            notes.append(
                f"Kerf compensated at {params.kerf:g} mm across the whole part. "
                f"Cut one corner as a test if your machine runs wider."
            )
        if _divided(params):
            compartment = params.inside()
            notes.append(
                f"The {params.columns * params.rows} compartments come out at "
                f"about {compartment[0] / params.columns:.0f} x "
                f"{compartment[1] / params.rows:.0f} mm each."
            )
            notes.append(
                "Drop the dividers in after the floor. The ones running each "
                "way half-lap into each other, so start them together and "
                "press them down as one."
            )
        if params.lid is not LidStyle.NONE:
            notes.append(
                f"The {params.height:g} mm height is the body. The lid sits on "
                f"top of the walls, so with it on the box stands "
                f"{params.height + params.thickness:g} mm."
            )
        if params.lid is LidStyle.CAP:
            notes.append(
                f"Glue the locator plate centred under the lid top; it drops "
                f"into the box with {params.lid_clearance:g} mm of play."
            )
        return notes

    def sample_params(self, rng: random.Random, index: int) -> BoxParams:
        """Draw one box variant."""
        thickness = rng.choice([3.0, 3.0, 4.0, 6.0])
        width = float(rng.randrange(90, 280, 10))
        # A third of boxes are divided, because a compartment box is a
        # different product rather than a variation on the plain one.
        divided = rng.random() < 0.34
        depth = float(rng.randrange(70, min(int(width), 220) + 1, 10))
        height = float(rng.randrange(40, 140, 10))
        tabs = rng.choice([2, 2, 3])
        # The tab has to fit the shortest side it runs along, so the box is
        # drawn first and the tab sized to it.  Drawn independently, one box
        # in eight proposed three 22 mm tabs along a 70 mm side.
        min_wall = max(4.0, thickness * 1.5)
        probe = BoxParams(
            width=width, depth=depth, height=height, floor_tabs=tabs,
            thickness=thickness, mode="laser", min_wall=min_wall,
        )
        widest = min(probe.widest_floor_tab(width), probe.widest_floor_tab(depth))
        # Compartments must stay big enough to put something in, so the
        # divider count follows the box rather than being drawn beside it.
        columns = rows = 1
        if divided:
            inner_w = width - 2.0 * thickness
            inner_d = depth - 2.0 * thickness
            columns = max(1, min(4, int(inner_w // 55.0)))
            rows = max(1, min(3, int(inner_d // 55.0)))
            if columns * rows == 1:
                columns = 2 if inner_w >= inner_d else 1
                rows = 1 if columns == 2 else 2
            elif rng.random() < 0.45:
                rows = 1
        # A short divider has room for one tab, not two.  Derived from the
        # height for the same reason every other count here is.
        standing = height - max(thickness, min_wall) - thickness
        divider_tabs = 2 if standing >= 2 * (8.0 + min_wall) + 2 * min_wall else 1
        tab_width = float(max(8.0, min(rng.randrange(14, 30, 2), widest * 0.85)))
        return BoxParams(
            width=width,
            depth=depth,
            height=height,
            floor_tabs=tabs,
            floor_tab_width=tab_width,
            columns=columns,
            rows=rows,
            divider_tabs=divider_tabs,
            lid=rng.choice([LidStyle.NONE, LidStyle.NONE, LidStyle.CAP]),
            mode="laser",
            thickness=thickness,
            kerf=rng.choice([0.1, 0.15, 0.2]),
            material=rng.choice(["birch ply", "poplar ply", "acrylic", "MDF"]),
            min_wall=min_wall,
            pocket_floor=1.0,
        )
