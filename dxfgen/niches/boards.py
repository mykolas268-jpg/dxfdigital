"""Cutting and charcuterie boards.

A board is a simpler object than a tray but has three features a tray does
not, and each one constrains the others:

* A **juice groove** is an annular channel, not a pocket: a closed ring with a
  second ring standing inside it as an island.  It has to be at least a
  cutter-and-a-bit wide or there is nowhere for chips to go.
* A **hang hole** needs somewhere to live.  On a board with a juice groove the
  rim is only as wide as the groove inset, which is nowhere near enough for a
  22 mm hole plus two walls, so the hole goes in a paddle tongue or it is
  refused.  Putting it in the middle of the cutting surface is not a design,
  it is a bug.
* A **paddle tongue** meets the body at two concave corners, which are inside
  corners as far as the cutter is concerned and get rounded to suit it.
"""

from __future__ import annotations

import random
from enum import Enum

from pydantic import Field, model_validator
from shapely.geometry import Polygon

from ..core import geometry as geo
from ..core.design import Contour, Design, Label, Part, Pocket
from ..core.geometry import Ring
from ..core.layers import ENGRAVE
from .base import GOLDEN, Generator, GeneratorParams, cutting_order_for, slugify, smallest_stock

__all__ = [
    "Handle","BoardStyle", "HangHole", "BoardParams", "BoardGenerator"]


class BoardStyle(str, Enum):
    """Outer profile family."""

    ROUNDED = "rounded"
    """Rounded rectangle."""
    SOFT = "soft"
    """Superellipse, a softer organic outline."""
    PADDLE = "paddle"
    """Rounded rectangle with a handle tongue at one end."""
    ROUND = "round"
    """Circular, for a cheese board."""


class Handle(str, Enum):
    """Hand holds cut through the board."""

    NONE = "none"
    ENDS = "ends"
    """A slot at each short end, for lifting a loaded board with both hands."""


class HangHole(str, Enum):
    """Where, if anywhere, the board hangs from."""

    NONE = "none"
    END = "end"
    """A hole in the rim at one end; needs a board with no juice groove."""
    TONGUE = "tongue"
    """A hole in the paddle tongue."""


#: Smallest working surface inside a juice groove, in mm.
MIN_WORKING_SIDE = 120.0


class BoardParams(GeneratorParams):
    """Parameters for a board."""

    length: float = Field(400.0, ge=150.0, le=900.0, description="overall length in mm")
    width: float | None = Field(
        None, ge=100.0, le=600.0, description="overall width in mm; default length/aspect"
    )
    aspect: float = Field(
        GOLDEN, ge=1.0, le=2.6, description="length/width ratio when width is omitted"
    )
    style: BoardStyle = Field(BoardStyle.ROUNDED, description="outer profile family")
    superellipse_exponent: float = Field(
        3.4, ge=2.0, le=8.0, description="curve exponent for the soft style"
    )
    corner_radius: float | None = Field(
        None, ge=0.0, le=300.0, description="corner radius in mm; default 8% of width"
    )
    juice_groove: bool = Field(True, description="cut a juice groove")
    groove_inset: float = Field(
        22.0, ge=10.0, le=120.0, description="groove distance from the edge in mm"
    )
    groove_width: float = Field(
        9.0, ge=5.0, le=30.0, description="groove channel width in mm"
    )
    groove_depth: float = Field(
        5.0, gt=0.0, le=15.0, description="groove depth in mm"
    )
    groove_radius: float | None = Field(
        None, ge=0.0, le=200.0, description="groove corner radius in mm; default 25"
    )
    handle: Handle = Field(Handle.NONE, description="hand hold cutouts")
    handle_length: float = Field(
        120.0, ge=90.0, le=320.0, description="handle cutout length in mm (>= 90 for a hand)"
    )
    handle_width: float = Field(
        32.0, ge=30.0, le=60.0, description="handle cutout width in mm (>= 30 for a hand)"
    )
    hang_hole: HangHole = Field(HangHole.NONE, description="hanging hole placement")
    hang_hole_diameter: float = Field(
        22.0, ge=8.0, le=45.0, description="hanging hole diameter in mm"
    )
    tongue_length: float = Field(
        110.0, ge=60.0, le=300.0, description="paddle tongue length in mm"
    )
    tongue_width: float = Field(
        90.0, ge=50.0, le=250.0, description="paddle tongue width in mm"
    )
    engrave_border: bool = Field(False, description="add an engraved border line")
    engrave_inset: float = Field(
        10.0, ge=3.0, le=60.0, description="engraved border inset in mm"
    )
    label: bool = Field(True, description="add an INFO size label")

    @model_validator(mode="after")
    def _check_combination(self) -> "BoardParams":
        if self.hang_hole is HangHole.TONGUE and self.style is not BoardStyle.PADDLE:
            raise ValueError(
                f"a tongue hang hole needs the {BoardStyle.PADDLE.value} style, "
                f"not {self.style.value}"
            )
        if self.style is BoardStyle.ROUND and self.width is not None:
            if abs(self.width - self.length) > 1e-9:
                raise ValueError("a round board must have width equal to length")
        return self

    def resolved_groove_inset(self) -> float:
        """How far in the juice groove really sits.

        A handle needs the rim to itself: the cutout runs from ``min_wall`` to
        ``min_wall + handle_width`` in from the edge, and the groove has to
        clear it by another wall.  A groove that opens into a hand hold drains
        onto the floor, so the groove yields rather than the handle - the same
        rule the trays use, where the rim is derived from the handle instead of
        being checked against it afterwards.

        Returns:
            The groove inset in mm, widened if handles demand it.
        """
        if self.handle is Handle.NONE:
            return self.groove_inset
        return max(self.groove_inset, self.handle_width + 2.0 * self.min_wall)

    def resolved_width(self) -> float:
        """Overall width in mm, rounded to the millimetre when derived."""
        if self.style is BoardStyle.ROUND:
            return self.length
        if self.width is not None:
            return self.width
        return float(round(self.length / self.aspect))

    def resolved_groove_radius(self) -> float:
        """Corner radius of the juice groove, never tighter than the cutter."""
        floor = self.tool_diameter * 0.8 + self.groove_width
        return max(floor, self.groove_radius if self.groove_radius is not None else 25.0)

    def body_length(self) -> float:
        """Length of the board body, excluding any paddle tongue."""
        if self.style is BoardStyle.PADDLE:
            return self.length - self.tongue_length
        return self.length

    def resolved_corner_radius(self) -> float:
        """Corner radius in mm."""
        width = self.resolved_width()
        limit = min(self.body_length(), width) / 2.0
        if self.style is BoardStyle.ROUND:
            return limit
        if self.corner_radius is not None:
            return min(self.corner_radius, limit)
        return min(max(8.0, round(width * 0.08)), limit)


def _body_ring(params: BoardParams) -> Ring:
    """The board body, before any tongue is added."""
    width = params.resolved_width()
    length = params.body_length()
    if params.style is BoardStyle.ROUND:
        return geo.circle_ring(length / 2.0, width / 2.0, length / 2.0)
    if params.style is BoardStyle.SOFT:
        return geo.dedupe(
            geo.superellipse_ring(
                length, width, params.superellipse_exponent, length / 2.0, width / 2.0
            )
        )
    return geo.rounded_rect_ring(length, width, params.resolved_corner_radius())


def _outline(params: BoardParams) -> Ring:
    """Build the outer profile, tongue included.

    Raises:
        ValueError: If the tongue does not fit the body.
    """
    body = _body_ring(params)
    if params.style is not BoardStyle.PADDLE:
        return body
    width = params.resolved_width()
    if params.tongue_width > width - 4.0 * params.min_wall:
        raise ValueError(
            f"a {params.tongue_width:g} mm tongue is too wide for a "
            f"{width:g} mm board"
        )
    overlap = min(40.0, params.body_length() / 4.0)
    tongue = geo.rounded_rect_ring(
        params.tongue_length + overlap,
        params.tongue_width,
        params.tongue_width / 2.0,
        params.body_length() - overlap,
        (width - params.tongue_width) / 2.0,
    )
    merged = geo.polygon_from_ring(body).union(geo.polygon_from_ring(tongue))
    if not isinstance(merged, Polygon) or merged.interiors:
        raise ValueError("the tongue did not merge cleanly with the board body")
    ring = geo.ensure_ccw(geo.dedupe(list(merged.exterior.coords)))
    # The two junctions are inside corners; round them to something the cutter
    # can follow, which also makes the paddle look deliberate.
    radius = min(25.0, params.tongue_width / 3.0)
    return geo.round_concave(ring, max(radius, params.tool_diameter))


def _juice_groove(params: BoardParams, base: Ring) -> Pocket:
    """Build the juice groove as a channel with a standing island.

    Args:
        params: Board parameters.
        base: The board body, without any paddle tongue.  A groove that ran
            down the tongue would be a groove down the handle.

    Returns:
        A pocket whose ring is the outside of the channel and whose island is
        the working surface inside it.

    Raises:
        ValueError: If the groove is too narrow for the cutter, or leaves no
            working surface.
    """
    minimum = params.tool_diameter * 1.15
    if params.groove_width < minimum:
        raise ValueError(
            f"a {params.groove_width:g} mm groove is too narrow for a "
            f"{params.tool_diameter:g} mm cutter; it needs at least "
            f"{minimum:.1f} mm"
        )
    inset = params.resolved_groove_inset()
    outer_rings = geo.offset_ring(base, -inset, join="round")
    inner_rings = geo.offset_ring(
        base, -(inset + params.groove_width), join="round"
    )
    if len(outer_rings) != 1 or len(inner_rings) != 1:
        raise ValueError(
            f"a groove {inset:g} mm in does not close on this board"
        )
    # Offsetting a rounded rectangle inward by more than its corner radius
    # leaves square corners, which a round cutter cannot reach inside a 9 mm
    # channel.  Rounding both rings by the same amount keeps the channel a
    # constant width while giving it a radius the cutter can follow - and it
    # is a no-op on outlines that are already curved.
    radius = params.resolved_groove_radius()
    outer = geo.round_convex(outer_rings[0], radius)
    inner = geo.round_convex(
        inner_rings[0], max(radius - params.groove_width, params.tool_diameter * 0.6)
    )
    working_w, working_h = geo.size_of(inner)
    if min(working_w, working_h) < MIN_WORKING_SIDE:
        raise ValueError(
            f"the groove leaves a {working_w:.0f} x {working_h:.0f} mm working "
            f"surface, below the {MIN_WORKING_SIDE:g} mm minimum"
        )
    return Pocket(outer, params.groove_depth, islands=[inner])


def _handle_slots(
    params: BoardParams, outline: Ring, groove: "Pocket | None"
) -> list[Ring]:
    """Cut a hand hold through each short end of the board.

    The slot is swept along the curve running half a handle-width plus a wall
    inside the profile, the same construction the trays use: sweeping a disc
    along an inset centreline gives a slot of exactly ``handle_width`` whose
    wall to the edge is constant, on a straight end or a round one.  That is
    what makes a handle on a circular cheese board work at all.

    Both dimensions are floors, not defaults: a cutout under 90 x 30 mm is not
    a hand hold, it is a decoration you cannot lift a loaded board by.

    Args:
        params: Board parameters.
        outline: The outer profile.
        groove: The juice groove, if the board has one; a handle may not break
            into it, because a groove that drains into a hole is not a groove.

    Returns:
        Two closed rings.

    Raises:
        ValueError: If two ergonomic handles do not fit, or one would open
            into the juice groove.
    """
    inset = params.handle_width / 2.0 + params.min_wall
    width, height = geo.size_of(outline)
    length = min(params.handle_length, max(90.0, height * 0.55))
    if length < 90.0:
        raise ValueError(
            f"a {height:g} mm deep board cannot carry a 90 mm handle cutout"
        )

    centreline = geo.offset_centreline(outline, inset)
    perimeter = geo.perimeter(centreline)
    if 2.0 * length + 4.0 * params.min_wall > perimeter:
        raise ValueError(
            f"two {length:g} mm handles do not fit a {perimeter:.0f} mm perimeter"
        )

    x0, y0, x1, y1 = geo.bbox(outline)
    middle = (y0 + y1) / 2.0
    slots = [
        geo.curved_slot(centreline, anchor, length, params.handle_width)
        for anchor in ((x0 - 20.0, middle), (x1 + 20.0, middle))
    ]
    first, second = (geo.polygon_from_ring(ring) for ring in slots)
    if first.distance(second) < params.min_wall - 1e-6:
        raise ValueError(
            f"the two handle cutouts come within {first.distance(second):.1f} mm "
            f"of each other"
        )
    if groove is not None:
        region = groove.region()
        for slot in slots:
            gap = region.distance(geo.polygon_from_ring(slot))
            if gap < params.min_wall - 1e-6:
                raise ValueError(
                    f"a handle cutout comes within {gap:.1f} mm of the juice "
                    f"groove; move the groove in or drop the handles"
                )
    return slots


def _hang_hole(params: BoardParams, outline: Ring, groove: Pocket | None) -> Ring:
    """Place the hanging hole and check it clears everything.

    Args:
        params: Board parameters.
        outline: The outer profile.
        groove: The juice groove, if there is one.

    Returns:
        A closed ring for the hole.

    Raises:
        ValueError: If the hole cannot be placed with full wall thickness.
    """
    width = params.resolved_width()
    radius = params.hang_hole_diameter / 2.0
    boundary = geo.polygon_from_ring(outline).exterior
    if params.hang_hole is HangHole.TONGUE:
        centre_y = width / 2.0
        far = params.length - params.min_wall - radius
        near = params.length - params.tongue_length + radius + params.min_wall
        candidates = [
            (x, centre_y) for x in _steps(far, near, -5.0)
        ]
    else:
        centre_y = width / 2.0
        candidates = [
            (x, centre_y)
            for x in _steps(radius + params.min_wall + 4.0, params.length / 3.0, 5.0)
        ]
    for centre in candidates:
        ring = geo.circle_ring(centre[0], centre[1], radius)
        disc = geo.polygon_from_ring(ring)
        if boundary.distance(disc) < params.min_wall - 1e-6:
            continue
        if groove is not None:
            if groove.region().distance(disc) < params.min_wall - 1e-6:
                continue
            # An END hole must sit in the rim, outside the channel.  Inside the
            # channel's island it would be a hole in the middle of the working
            # surface, which is not a design anyone wants.
            if params.hang_hole is HangHole.END and not geo.polygon_from_ring(
                groove.ring
            ).disjoint(disc):
                continue
        return ring
    raise ValueError(
        f"a {params.hang_hole_diameter:g} mm hang hole does not fit with "
        f"{params.min_wall:g} mm of wall"
        + (
            "; a juice groove leaves only the rim, so use the paddle style"
            if groove is not None
            else ""
        )
    )


def _steps(start: float, stop: float, step: float) -> list[float]:
    """Inclusive float range used to try candidate positions."""
    values: list[float] = []
    current = start
    while (step > 0 and current <= stop) or (step < 0 and current >= stop):
        values.append(current)
        current += step
    return values or [start]


class BoardGenerator(Generator):
    """Generates cutting and charcuterie boards."""

    niche = "boards"
    title = "Boards"
    summary = "Cutting and charcuterie boards with juice grooves, handles and hang holes"
    params_model = BoardParams

    def generate(self, params: BoardParams) -> Design:  # type: ignore[override]
        """Build one board.

        Args:
            params: Board parameters.

        Returns:
            The design, with the bounding box at the origin.

        Raises:
            ValueError: If the parameters are geometrically incompatible.
        """
        outline = _outline(params)
        pockets: list[Pocket] = []
        groove = None
        if params.juice_groove:
            groove = _juice_groove(params, _body_ring(params))
            pockets.append(groove)

        holes: list[Ring] = []
        if params.handle is Handle.ENDS:
            holes.extend(_handle_slots(params, outline, groove))
        if params.hang_hole is not HangHole.NONE:
            holes.append(_hang_hole(params, outline, groove))

        engrave: list[Contour] = []
        if params.engrave_border:
            limit = (params.resolved_groove_inset() - params.tool_diameter) if groove else 40.0
            inset = min(params.engrave_inset, max(limit, 0.0))
            if inset < 3.0:
                raise ValueError(
                    "there is no room for an engraved border inside the groove inset"
                )
            engrave = [
                Contour(ring, ENGRAVE, closed=True)
                for ring in geo.offset_ring(outline, -inset, join="round")
            ]

        real_length, real_width = (round(v) for v in geo.size_of(outline))
        labels: list[Label] = []
        text = (
            f"{real_length:g} x {real_width:g} x {params.thickness:g} mm  "
            f"{params.material}"
        )
        if params.label:
            labels = self._label_for(params, outline, groove, text)

        part = Part(
            name="board",
            outline=outline,
            holes=holes,
            pockets=pockets,
            engrave=engrave,
            labels=labels,
        )
        design = Design(
            slug=slugify(
                "board", params.style.value, f"{real_length:g}x{real_width:g}",
                "groove" if params.juice_groove else "flat",
                f"hang-{params.hang_hole.value}",
            ),
            name=self._name(params, real_length, real_width),
            niche=self.niche,
            description=self._description(params, real_length, real_width),
            parts=[part],
            machine=params.machine(),
            material=params.material,
            thickness=params.thickness,
            sheet=smallest_stock(real_length, real_width),
            params=params.model_dump(mode="json"),
            notes=self._notes(params),
            limits=params.validation_config(),
        )
        design.cutting_order = cutting_order_for(design)
        return design

    @staticmethod
    def _label_for(
        params: BoardParams, outline: Ring, groove: Pocket | None, text: str
    ) -> list[Label]:
        """Place the INFO label on the rim, if there is room for it."""
        width = params.resolved_width()
        rim = params.resolved_groove_inset() if groove else 18.0
        height = min(5.0, rim * 0.35)
        if rim < 12.0 or 0.62 * height * len(text) > params.body_length() * 0.8:
            return []
        return [Label(text, (params.body_length() / 2.0, rim * 0.45), height, align="center")]

    @staticmethod
    def _kind(params: BoardParams) -> str:
        """Name the product category."""
        if params.style is BoardStyle.PADDLE:
            return "paddle serving board"
        if params.style is BoardStyle.ROUND:
            return "round cheese board"
        return "cutting board" if params.juice_groove else "charcuterie board"

    def _name(self, params: BoardParams, length: float, width: float) -> str:
        kind = self._kind(params).title()
        return f"{kind}, {length:g} x {width:g} mm"

    def _description(self, params: BoardParams, length: float, width: float) -> str:
        bits = [
            f"A {self._kind(params)} measuring {length:g} x {width:g} mm in "
            f"{params.thickness:g} mm {params.material}"
        ]
        if params.juice_groove:
            bits.append(
                f"with a {params.groove_width:g} mm juice groove "
                f"{params.groove_depth:g} mm deep set {params.resolved_groove_inset():g} mm "
                f"in from the edge"
            )
        if params.handle is Handle.ENDS:
            bits.append(
                f"with a hand hold {params.handle_length:g} x "
                f"{params.handle_width:g} mm cut through each end"
            )
        if params.hang_hole is not HangHole.NONE:
            bits.append(
                f"and a {params.hang_hole_diameter:g} mm hanging hole in the "
                f"{params.hang_hole.value}"
            )
        return (
            ", ".join(bits)
            + f". Cut with a {params.tool_diameter:g} mm cutter."
        )

    def _notes(self, params: BoardParams) -> list[str]:
        notes = [f"Finish with food-safe oil before use."]
        if params.handle is Handle.ENDS:
            notes.append(
                "The hand holds are cut right through. Sand their inside edges "
                "well; they are the part the board is picked up by."
            )
            if params.juice_groove:
                notes.append(
                    f"The groove sits {params.resolved_groove_inset():g} mm in "
                    f"rather than the usual rim, so it clears the hand holds "
                    f"and cannot drain into them."
                )
        if params.juice_groove:
            notes.append(
                f"The groove is drawn as a flat-bottomed channel "
                f"{params.groove_width:g} mm wide. A round-nose bit of the same "
                f"width gives the usual rounded profile; keep the depth at "
                f"{params.groove_depth:g} mm either way."
            )
            notes.append(
                f"Groove leaves {params.thickness - params.groove_depth:g} mm "
                f"under the channel."
            )
        if params.style is BoardStyle.PADDLE:
            notes.append(
                "The tongue meets the body on a radius so a straight cutter can "
                "follow it without stopping."
            )
        return notes

    def sample_params(self, rng: random.Random, index: int) -> BoardParams:
        """Draw one board variant."""
        style = rng.choice(
            [BoardStyle.ROUNDED, BoardStyle.ROUNDED, BoardStyle.PADDLE,
             BoardStyle.SOFT, BoardStyle.ROUND]
        )
        if style is BoardStyle.ROUND:
            length = float(rng.randrange(28, 46) * 10)
            width = length
            aspect = 1.0
        else:
            aspect = rng.choice([GOLDEN, GOLDEN, 1.5, 1.4, 1.33, 1.75])
            length = float(rng.randrange(30, 65) * 10)
            width = round(length / aspect / 10.0) * 10.0
        tongue_length = float(rng.randrange(90, 160, 10))
        tongue_width = min(float(rng.randrange(70, 130, 10)), width - 40.0)
        # A round cheese board takes a juice groove as happily as any other -
        # an inward offset of a circle is a circle - and excluding it left the
        # contact sheet showing bare discs with one hole in them.
        groove = rng.random() < 0.6
        if style is BoardStyle.PADDLE:
            hang = rng.choice([HangHole.TONGUE, HangHole.TONGUE, HangHole.NONE])
        elif groove:
            hang = HangHole.NONE
        else:
            hang = rng.choice([HangHole.END, HangHole.NONE])
        # Hand holds need a wide rim and a board big enough to want carrying,
        # and the paddle already has a handle.
        handle = Handle.NONE
        if style is not BoardStyle.PADDLE and min(length, width) >= 280.0:
            handle = rng.choice([Handle.ENDS, Handle.NONE, Handle.NONE])
        border = rng.random() < 0.3
        # A board with no groove, no handle, no hanging hole and no engraving
        # is a rounded rectangle, and nobody buys a file of a rounded
        # rectangle.  Give the plainest draws their one feature back.
        if not (groove or border or handle is not Handle.NONE
                or hang is not HangHole.NONE or style is BoardStyle.PADDLE):
            if min(length, width) >= 280.0:
                handle = Handle.ENDS
            else:
                border = True
        thickness = rng.choice([19.0, 19.0, 25.0, 20.0])
        return BoardParams(
            length=length,
            width=width,
            aspect=aspect,
            style=style,
            superellipse_exponent=rng.choice([2.8, 3.4, 4.2]),
            juice_groove=groove,
            groove_inset=float(rng.randrange(18, 34, 2)),
            groove_width=float(rng.randrange(8, 14)),
            groove_depth=rng.choice([4.0, 5.0, 6.0]),
            hang_hole=hang,
            hang_hole_diameter=float(rng.randrange(14, 30, 2)),
            tongue_length=tongue_length,
            tongue_width=tongue_width,
            handle=handle,
            handle_length=float(rng.randrange(100, 160, 10)),
            handle_width=float(rng.randrange(30, 42, 2)),
            engrave_border=border,
            engrave_inset=float(rng.randrange(6, 16, 2)),
            material=rng.choice(["oak", "walnut", "maple", "cherry", "beech"]),
            thickness=thickness,
        )
