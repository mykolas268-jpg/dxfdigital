"""Slot-together phone and tablet docks.

Three flat parts and no fasteners: a base that lies on the desk, a back the
device leans against, and a front lip that stops it sliding off.  The two
uprights carry tabs on their bottom edges which pass through mortises in the
base and finish flush with its underside.

This is the first design here with a real joint, and the first place dogbone
relief belongs.  A tray's pockets are surfaces the buyer looks at, so their
corners are pre-filleted; a mortise is hidden inside the joint, so it gets
relief instead and the square-cornered tab seats properly.

The lean is deliberately not a parameter.  A device standing in a channel
leans back by an angle that depends on its own thickness, so the design states
the channel width, which is a fact about the stand, rather than an angle that
would only be true for one phone.
"""

from __future__ import annotations

import math
import random
from enum import Enum

from pydantic import Field, model_validator

from ..core import geometry as geo
from ..core.assembly import Assembly, Placement, Plane
from ..core.design import Design, Label, Part
from ..core.geometry import Ring
from .base import (
    Generator,
    GeneratorParams,
    apply_kerf,
    arrange_grid,
    cutting_order_for,
    material_phrase,
    slugify,
    smallest_stock,
)

__all__ = ["BackStyle", "StandSize", "StandParams", "StandGenerator"]


class StandSize(str, Enum):
    """What the dock is meant to hold."""

    PHONE = "phone"
    TABLET = "tablet"


#: Preset (base depth, back height, lip height, width) per size, in mm.
_PRESETS: dict[StandSize, tuple[float, float, float, float]] = {
    StandSize.PHONE: (100.0, 100.0, 28.0, 85.0),
    StandSize.TABLET: (140.0, 150.0, 38.0, 130.0),
}


class BackStyle(str, Enum):
    """The shape of the back panel's top edge.

    Only the back is shaped.  The lip is short, mostly hidden behind the
    device, and carries the cable notch, so a profile on it would be work
    nobody sees and geometry the notch has to dodge.
    """

    SQUARE = "square"
    """A flat top, square across."""
    ARCH = "arch"
    """A circular arc, highest in the middle."""
    TAPER = "taper"
    """Straight shoulders running in to a narrower flat top."""
    PEAK = "peak"
    """A shallow gable, highest in the middle."""


class StandParams(GeneratorParams):
    """Parameters for a slot-together dock."""

    size: StandSize = Field(StandSize.PHONE, description="phone or tablet proportions")
    width: float | None = Field(
        None, ge=50.0, le=260.0, description="dock width in mm; default per size"
    )
    base_depth: float | None = Field(
        None, ge=60.0, le=300.0, description="base depth in mm; default per size"
    )
    back_height: float | None = Field(
        None, ge=50.0, le=320.0, description="back height above the base in mm"
    )
    lip_height: float | None = Field(
        None, ge=14.0, le=70.0, description="front lip height above the base in mm"
    )
    device_gap: float = Field(
        14.0, ge=9.0, le=32.0, description="channel width the device stands in, mm"
    )
    tabs: int = Field(2, ge=2, le=4, description="tabs per upright")
    tab_width: float = Field(
        24.0, ge=12.0, le=80.0, description="tab width in mm"
    )
    back_style: BackStyle = Field(
        BackStyle.SQUARE, description="shape of the back panel's top edge"
    )
    back_shape_rise: float = Field(
        28.0, ge=6.0, le=120.0, description="how far the shaped top rises, mm"
    )
    corner_radius: float = Field(
        6.0, ge=0.0, le=40.0, description="corner radius on the parts in mm"
    )
    cable_slot: bool = Field(True, description="cut a cable notch in the lip")
    cable_width: float = Field(
        16.0, ge=8.0, le=40.0, description="cable notch width in mm"
    )
    relief: bool = Field(
        True,
        description=(
            "dogbone the router mortises; a no-op on a laser, and a router "
            "design with it off will not pass validation because a round "
            "cutter cannot cut a square mortise"
        ),
    )
    gap: float = Field(12.0, ge=5.0, le=60.0, description="spacing between parts on the sheet, mm")
    label: bool = Field(True, description="add an INFO label to the base")

    @model_validator(mode="after")
    def _check_channel(self) -> "StandParams":
        if self.device_gap < self.thickness * 0.55:
            raise ValueError(
                f"a {self.device_gap:g} mm channel is too narrow to be worth "
                f"cutting in {self.thickness:g} mm material"
            )
        return self

    def preset(self) -> tuple[float, float, float, float]:
        """Preset dimensions for this size."""
        return _PRESETS[self.size]

    def resolved_width(self) -> float:
        """Dock width in mm."""
        return self.width if self.width is not None else self.preset()[3]

    def resolved_base_depth(self) -> float:
        """Base depth in mm."""
        return self.base_depth if self.base_depth is not None else self.preset()[0]

    def resolved_back_height(self) -> float:
        """Back height above the base in mm."""
        return self.back_height if self.back_height is not None else self.preset()[1]

    def resolved_lip_height(self) -> float:
        """Lip height above the base in mm."""
        return self.lip_height if self.lip_height is not None else self.preset()[2]

    def relief_margin(self) -> float:
        """Extra clearance a dogboned mortise needs beyond its own rectangle.

        A dogbone at the classic position reaches ``r * (1 - cos 45)`` past
        each wall, about 0.93 mm for a 6.35 mm cutter.  That overhang is real
        material removed, so it has to be budgeted for when a mortise is
        placed near an edge, or the wall comes out thinner than asked for.
        """
        if self.machine().is_laser or not self.relief:
            return geo.ARC_TOLERANCE
        return self.tool_diameter / 2.0 * geo.RELIEF_OVERSIZE * 0.35 + geo.ARC_TOLERANCE

    def feature_margin(self) -> float:
        """How far a mortise must sit from the edge of the part it is in.

        Three things eat into a nominal ``min_wall``:

        * the mortise is longer than its tab by the fit clearance, half of it
          at each end;
        * dogbone relief reaches past the mortise corners;
        * a rounded part corner cuts the corner off, so a feature sitting
          ``min_wall`` from both nominal edges is closer than that to the arc.
          For a corner of radius ``R``, a feature ``d`` from both edges is
          ``R - sqrt(2)(R - d)`` from the arc, which gives the ``d`` below.

        Returns:
            The margin in mm.
        """
        margin = self.min_wall + self.relief_margin() + self.clearance / 2.0
        radius = self.corner_radius
        if radius > self.min_wall:
            corner = radius - (radius - self.min_wall) / math.sqrt(2.0)
            margin = max(
                margin,
                corner + self.relief_margin() + self.clearance / 2.0 + geo.ARC_TOLERANCE,
            )
        return margin

    def upright_positions(self) -> tuple[float, float]:
        """Where the two uprights stand along the base, in mm.

        Measured to the centre of each upright's thickness, from the front
        edge of the base.  The base's mortises and the assembly drawing both
        read this, so a change to the channel cannot move one without the
        other.

        Returns:
            ``(lip_x, back_x)``.
        """
        half = self.mortise_width() / 2.0
        lip_x = self.feature_margin() + half
        return lip_x, lip_x + self.thickness + self.device_gap

    def mortise_width(self) -> float:
        """Drawn width of a mortise across the upright's thickness, in mm."""
        return self.machine().slot_width(self.thickness)

    def mortise_length(self) -> float:
        """Drawn length of a mortise along the tab, in mm."""
        return self.machine().slot_width(self.tab_width)

    def widest_tab(self) -> float:
        """The widest tab this upright can carry, in mm.

        Solves the same inequality :meth:`tab_positions` checks, rather than
        restating it: tabs on an even pitch need ``min_wall`` plus the fit
        clearance plus relief at both corners between each pair.  A sampler
        that draws a tab width independently of the panel it goes in produces
        a quarter of its variants unbuildable, which is how this got noticed.

        Returns:
            The widest workable tab, which may be zero or less when the panel
            is too narrow for this tab count at all.
        """
        span = self.resolved_width() - 2.0 * self.feature_margin()
        between = self.min_wall + self.clearance + 2.0 * self.relief_margin()
        return (span + between) / self.tabs - between

    def tab_positions(self) -> list[float]:
        """Left edge of each tab, measured across the upright's width.

        Raises:
            ValueError: If the tabs do not fit across the width.
        """
        width = self.resolved_width()
        edge = self.feature_margin()
        span = width - 2.0 * edge
        # Tabs are placed on an even pitch, so the material left between two
        # mortises is the pitch less the tab width, less the fit clearance and
        # the relief overhang at both corners.
        between = self.min_wall + self.clearance + 2.0 * self.relief_margin()
        needed = self.tabs * (self.tab_width + between)
        if needed > span + between:
            raise ValueError(
                f"{self.tabs} tabs {self.tab_width:g} mm wide need "
                f"{needed - between:.0f} mm across a {width:g} mm upright, "
                f"leaving no material between them"
            )
        if self.tabs == 1:
            return [(width - self.tab_width) / 2.0]
        # Spread the tabs to the extremes of the usable span: tabs near the
        # edges resist racking, tabs clustered in the middle do not.
        gap = (span - self.tabs * self.tab_width) / (self.tabs - 1)
        return [edge + index * (self.tab_width + gap) for index in range(self.tabs)]


def _top_edge(
    width: float, top: float, style: BackStyle, rise: float
) -> list[tuple[float, float]]:
    """The profile across the top of an upright, right to left.

    Walking the ring counter-clockwise the top is traversed from the right
    side to the left, so that is the order here.  Every style is convex, which
    is what keeps it cuttable: a round cutter travelling outside the profile
    needs no relief on a corner that points outwards.

    Args:
        width: Panel width in mm.
        top: Height of the panel's highest point, in mm.
        style: Which profile to draw.
        rise: How far the shaping rises above the shoulders, in mm.

    Returns:
        The points from the right shoulder across to the left shoulder.

    Raises:
        ValueError: If the rise leaves no panel below it.
    """
    if style is BackStyle.SQUARE:
        return [(width, top), (0.0, top)]
    rise = min(rise, top * 0.45)
    if rise <= 0.0:
        raise ValueError(f"a {top:g} mm upright has no room for a shaped top")
    shoulder = top - rise
    if style is BackStyle.PEAK:
        return [(width, shoulder), (width / 2.0, top), (0.0, shoulder)]
    if style is BackStyle.TAPER:
        inset = min(width * 0.22, width / 2.0 - 8.0)
        return [(width, shoulder), (width - inset, top), (inset, top), (0.0, shoulder)]
    # An arc through the two shoulders and the midpoint of the top.
    half = width / 2.0
    radius = (half * half + rise * rise) / (2.0 * rise)
    centre = top - radius
    span = math.asin(min(1.0, half / radius))
    # Even, so the apex is a sampled point: an odd count steps either side of
    # it and the arch finishes a fraction below the height it says it is.
    steps = max(8, int(math.degrees(span * 2.0) / 4.0))
    steps += steps % 2
    points = []
    for index in range(steps + 1):
        angle = span - 2.0 * span * index / steps
        points.append((half + radius * math.sin(angle), centre + radius * math.cos(angle)))
    return points


def _upright_ring(
    params: StandParams,
    height: float,
    style: BackStyle = BackStyle.SQUARE,
) -> Ring:
    """Build an upright outline with tabs on its bottom edge.

    The tabs protrude by exactly the material thickness, so they come through
    the base and finish flush with its underside.

    Args:
        params: Stand parameters.
        height: Height of the upright above the base, in mm.
        style: Shape of the top edge.

    Returns:
        A counter-clockwise ring whose bounding box starts at ``y = 0``.

    Raises:
        ValueError: If the shaped top does not fit the panel.
    """
    width = params.resolved_width()
    depth = params.thickness
    ring: list[tuple[float, float]] = [(0.0, depth)]
    for left in params.tab_positions():
        right = left + params.tab_width
        ring.extend([(left, depth), (left, 0.0), (right, 0.0), (right, depth)])
    ring.append((width, depth))
    ring.extend(_top_edge(width, depth + height, style, params.back_shape_rise))
    return _relieve_profile(geo.dedupe(ring), params)


def _relieve_profile(ring: Ring, params: StandParams) -> Ring:
    """Relieve the inside corners of an upright's profile.

    A tab root is an inside corner, and a round cutter cannot cut one: it
    leaves material exactly where the shoulder needs to sit flat against the
    base.  A dogbone notch at each root is the standard answer and is hidden
    inside the joint once assembled.  A laser needs none of this.

    Args:
        ring: The upright profile.
        params: Stand parameters.

    Returns:
        The relieved profile, or the input unchanged in laser mode.
    """
    if params.machine().is_laser or not params.relief:
        return ring
    return geo.apply_relief(
        ring,
        params.tool_diameter / 2.0 * geo.RELIEF_OVERSIZE,
        style="dogbone",
        region="part",
    )


def _mortise(params: StandParams, x_centre: float, tab_left: float) -> Ring:
    """Build one mortise in the base, with relief where the machine needs it.

    Args:
        params: Stand parameters.
        x_centre: Where the upright sits along the base's depth, in mm.
        tab_left: Left edge of the matching tab, across the width.

    Returns:
        A closed ring.
    """
    length = params.mortise_length()
    width = params.mortise_width()
    ring = geo.rect_ring(
        width,
        length,
        x_centre - width / 2.0,
        tab_left - (length - params.tab_width) / 2.0,
    )
    if params.machine().is_laser or not params.relief:
        return ring
    return geo.apply_relief(ring, params.tool_diameter / 2.0, style="dogbone")


def _base_part(params: StandParams) -> Part:
    """Build the base plate.

    Raises:
        ValueError: If the base is too shallow to hold both uprights.
    """
    depth = params.resolved_base_depth()
    width = params.resolved_width()
    thickness = params.thickness
    edge = params.feature_margin()
    half = params.mortise_width() / 2.0
    lip_x, back_x = params.upright_positions()
    if back_x + half + edge > depth:
        raise ValueError(
            f"a {depth:g} mm base cannot hold a {params.device_gap:g} mm channel "
            f"with {params.min_wall:g} mm of material front and back; it needs "
            f"at least {back_x + half + edge:.0f} mm"
        )
    holes: list[Ring] = []
    for x_centre in (lip_x, back_x):
        holes.extend(_mortise(params, x_centre, left) for left in params.tab_positions())

    labels: list[Label] = []
    if params.label:
        text = f"{params.size.value} dock  {thickness:g} mm"
        height = 4.0
        if 0.62 * height * len(text) < (depth - back_x) * 0.9:
            labels.append(
                Label(text, ((back_x + depth) / 2.0, width / 2.0), height, align="center")
            )
    outline = geo.rect_ring(depth, width)
    if params.corner_radius > 0:
        outline = geo.round_convex(outline, params.corner_radius)
    return Part(name="base", outline=outline, holes=holes, labels=labels)


def _lip_part(params: StandParams) -> Part:
    """Build the front lip, with an optional cable notch."""
    ring = _upright_ring(params, params.resolved_lip_height())
    if params.cable_slot:
        width = params.resolved_width()
        top = params.thickness + params.resolved_lip_height()
        notch = geo.stadium_ring(
            params.cable_width * 1.4,
            params.cable_width,
            (width - params.cable_width * 1.4) / 2.0,
            top - params.cable_width / 2.0,
        )
        merged = geo.polygon_from_ring(ring).difference(geo.polygon_from_ring(notch))
        if merged.is_empty or merged.geom_type != "Polygon":
            raise ValueError("the cable notch cut the lip in two")
        ring = geo.ensure_ccw(geo.dedupe(list(merged.exterior.coords)))
        ring = _relieve_profile(ring, params)
    return Part(name="lip", outline=ring)


def _assembly(params: StandParams) -> Assembly:
    """Where the three parts stand in the finished dock.

    The base lies down and the two uprights stand across it, their tabs
    passing through and finishing flush with its underside - which is why both
    uprights start at z = 0 rather than on top of the base.

    Args:
        params: Stand parameters.

    Returns:
        The assembly.
    """
    lip_x, back_x = params.upright_positions()
    half = params.thickness / 2.0
    return Assembly(
        (
            Placement("base", Plane.FLAT, (0.0, 0.0, 0.0)),
            Placement("back", Plane.SIDE, (back_x - half, 0.0, 0.0)),
            Placement("lip", Plane.SIDE, (lip_x - half, 0.0, 0.0)),
        ),
        f"{params.size.value} dock, {params.device_gap:g} mm channel",
    )


#: Back profiles in the order a bundle works through them.  Drawn
#: independently they cluster - one seed gave eight arches in nine docks -
#: so the style rotates on the variant index, as the seasonal shapes do.
#: The arch appears twice because it suits the most sizes.
_BACK_CYCLE: tuple[BackStyle, ...] = (
    BackStyle.ARCH,
    BackStyle.TAPER,
    BackStyle.SQUARE,
    BackStyle.ARCH,
    BackStyle.PEAK,
)


class StandGenerator(Generator):
    """Generates slot-together phone and tablet docks."""

    niche = "stands"
    title = "Stands"
    summary = "Three-part slot-together phone and tablet docks, no glue"
    params_model = StandParams

    def generate(self, params: StandParams) -> Design:  # type: ignore[override]
        """Build one dock.

        Args:
            params: Stand parameters.

        Returns:
            The design, parts laid out side by side.

        Raises:
            ValueError: If the parameters are geometrically incompatible.
        """
        parts = apply_kerf(
            [
                _base_part(params),
                Part(
                    name="back",
                    outline=_upright_ring(
                        params, params.resolved_back_height(), params.back_style
                    ),
                ),
                _lip_part(params),
            ],
            params,
        )
        arrange_grid(parts, params.gap, columns=2)
        width, height = geo.size_of(
            [point for part in parts for point in part.placed().outline]
        )
        design = Design(
            slug=slugify(
                "dock", params.size.value, params.back_style.value,
                f"{params.resolved_width():g}w",
                f"gap{params.device_gap:g}", params.mode.value,
                f"t{params.thickness:g}",
            ),
            name=(
                f"Slot-together {params.size.value.title()} Dock, "
                f"{params.back_style.value} back, "
                f"{params.resolved_width():g} mm wide, "
                f"{params.device_gap:g} mm channel"
            ),
            niche=self.niche,
            description=self._description(params),
            parts=parts,
            machine=params.machine(),
            material=params.material,
            thickness=params.thickness,
            sheet=smallest_stock(width, height),
            params=params.model_dump(mode="json"),
            notes=self._notes(params),
            limits=params.validation_config(),
            assembly=_assembly(params),
        )
        design.cutting_order = cutting_order_for(design)
        return design

    def _description(self, params: StandParams) -> str:
        machine = params.machine()
        if machine.is_laser:
            fit = (
                f"every outline is grown and every cutout shrunk by half the "
                f"{params.kerf:g} mm kerf, so tabs and mortises both finish on "
                f"size and the joint closes to {params.clearance:g} mm"
            )
        else:
            fit = (
                f"mortises are drawn {params.mortise_width():.2f} mm wide for a "
                f"{params.clearance:g} mm fit clearance, with dogbone relief so "
                f"the square tabs seat fully"
            )
        return (
            f"A three-part slot-together {params.size.value} dock in "
            f"{material_phrase(params.thickness, params.material)}. The device "
            f"stands in a "
            f"{params.device_gap:g} mm channel between the lip and the back. No "
            f"glue and no fasteners: the {fit}."
        )

    def _notes(self, params: StandParams) -> list[str]:
        notes = [
            "Push the two uprights down through the base from above; the tabs "
            f"come through by {params.thickness:g} mm and finish flush with the "
            "underside.",
            f"Mortises are drawn {params.mortise_width():.2f} x "
            f"{params.mortise_length():.2f} mm for {params.thickness:g} mm "
            f"material and {params.tab_width:g} mm tabs.",
            f"The {params.device_gap:g} mm channel suits a device up to about "
            f"{params.device_gap - 2:g} mm thick, case included.",
        ]
        if params.machine().is_laser:
            notes.append(
                f"Kerf is compensated at {params.kerf:g} mm across the whole "
                f"part, tabs included. Cut one mortise as a test first if your "
                f"machine runs wider."
            )
        if params.cable_slot:
            notes.append(
                f"The {params.cable_width:g} mm notch in the lip passes a "
                f"charging cable."
            )
        return notes

    def sample_params(self, rng: random.Random, index: int) -> StandParams:
        """Draw one dock variant."""
        size = rng.choice([StandSize.PHONE, StandSize.PHONE, StandSize.TABLET])
        laser = rng.random() < 0.4
        thickness = (
            rng.choice([3.0, 6.0, 6.0]) if laser else rng.choice([12.0, 18.0, 18.0])
        )
        depth, back, lip, width = _PRESETS[size]
        panel_width = float(rng.randrange(int(width * 0.9), int(width * 1.25), 5))
        tabs = rng.choice([2, 2, 3])
        # The tab has to fit the panel it is cut into, so the panel is drawn
        # first and the tab is sized to it.  A margin below the maximum keeps
        # a real wall between the mortises rather than the minimum the
        # validator will just about accept.
        probe = StandParams(
            size=size, width=panel_width, tabs=tabs, thickness=thickness,
            mode="laser" if laser else "router",
        )
        widest = probe.widest_tab()
        tab_width = float(max(14.0, min(rng.randrange(18, 34, 2), widest * 0.8)))
        return StandParams(
            size=size,
            width=panel_width,
            base_depth=float(rng.randrange(int(depth * 0.95), int(depth * 1.3), 5)),
            back_height=float(rng.randrange(int(back * 0.85), int(back * 1.25), 5)),
            lip_height=float(rng.randrange(int(lip * 0.8), int(lip * 1.3), 2)),
            device_gap=max(float(rng.randrange(11, 20)), thickness * 0.6),
            tabs=tabs,
            tab_width=tab_width,
            back_style=_BACK_CYCLE[index % len(_BACK_CYCLE)],
            back_shape_rise=float(rng.randrange(18, 46, 4)),
            corner_radius=float(rng.randrange(0, 10, 2)),
            cable_slot=rng.random() < 0.7,
            cable_width=float(rng.randrange(12, 22, 2)),
            mode="laser" if laser else "router",
            thickness=thickness,
            material=(
                rng.choice(["birch ply", "acrylic"])
                if laser
                else rng.choice(["birch plywood", "oak", "walnut"])
            ),
            min_wall=5.0 if laser else 8.0,
            pocket_floor=2.0 if laser else 5.0,
        )
