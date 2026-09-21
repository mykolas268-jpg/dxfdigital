"""Coaster sets, with an optional holder.

The first design in this project made of more than one part, and the first
that uses the ENGRAVE layer for anything.  Two things are worth knowing:

* The engraved patterns are generated from the coaster's own outline - rings
  are inward offsets of it, rays are chords of it - so a hexagonal coaster
  gets hexagonal rings rather than a circle stamped on a hexagon.
* The holder is a plate with a recess the stack drops into, and a push hole
  through its floor so the stack can be lifted out.  That hole is a
  counterbore, not a collision, which is why the validator had to learn the
  difference.
"""

from __future__ import annotations

import math
import random
from enum import Enum

from pydantic import Field, model_validator

from ..core import geometry as geo
from ..core.design import Contour, Design, Label, Part, Pocket
from ..core.geometry import Point, Ring
from ..core.layers import ENGRAVE
from .base import Generator, GeneratorParams, arrange_grid, cutting_order_for, slugify, smallest_stock

__all__ = ["CoasterShape", "Pattern", "CoasterParams", "CoasterGenerator"]


class CoasterShape(str, Enum):
    """Coaster outline."""

    ROUND = "round"
    SQUARE = "square"
    HEX = "hex"
    OCTAGON = "octagon"
    SOFT = "soft"


class Pattern(str, Enum):
    """Engraved decoration."""

    NONE = "none"
    BORDER = "border"
    RINGS = "rings"
    RAYS = "rays"
    GRID = "grid"


#: Smallest coaster that still holds a glass, in mm.
MIN_SIZE = 80.0


class CoasterParams(GeneratorParams):
    """Parameters for a coaster set."""

    size: float = Field(100.0, ge=MIN_SIZE, le=160.0, description="coaster size across in mm")
    count: int = Field(4, ge=1, le=12, description="how many coasters in the set")
    shape: CoasterShape = Field(CoasterShape.ROUND, description="coaster outline")
    corner_radius: float | None = Field(
        None, ge=0.0, le=60.0, description="corner radius for the square shape in mm"
    )
    superellipse_exponent: float = Field(
        3.6, ge=2.0, le=8.0, description="curve exponent for the soft shape"
    )
    recess: bool = Field(False, description="cut a shallow recess to catch condensation")
    recess_inset: float = Field(
        10.0, ge=5.0, le=40.0, description="recess distance from the edge in mm"
    )
    recess_depth: float = Field(2.0, gt=0.0, le=10.0, description="recess depth in mm")
    pattern: Pattern = Field(Pattern.RINGS, description="engraved decoration")
    pattern_lines: int = Field(5, ge=1, le=24, description="lines or rings in the pattern")
    pattern_inset: float = Field(
        8.0, ge=2.0, le=40.0, description="pattern distance from the edge in mm"
    )
    holder: bool = Field(True, description="include a holder plate")
    holder_clearance: float = Field(
        1.0, ge=0.2, le=4.0, description="gap around the stack in the holder, mm"
    )
    holder_wall: float = Field(
        14.0, ge=8.0, le=50.0, description="holder material around the recess, mm"
    )
    holder_depth: float | None = Field(
        None, ge=1.0, le=40.0, description="holder recess depth in mm; default half the stack"
    )
    holder_push_hole: float = Field(
        32.0, ge=0.0, le=80.0, description="push-out hole diameter in the holder floor, 0 for none"
    )
    gap: float = Field(12.0, ge=5.0, le=60.0, description="spacing between parts on the sheet, mm")
    label: bool = Field(True, description="add an INFO label to the holder")

    @model_validator(mode="after")
    def _check_recess(self) -> "CoasterParams":
        if self.recess and self.recess_depth > self.thickness - self.pocket_floor:
            raise ValueError(
                f"a {self.recess_depth:g} mm recess leaves too little floor in "
                f"{self.thickness:g} mm material"
            )
        return self

    def stack_height(self) -> float:
        """Total height of the stack of coasters, in mm."""
        return self.count * self.thickness

    def resolved_holder_depth(self) -> float:
        """Holder recess depth in mm."""
        if self.holder_depth is not None:
            return self.holder_depth
        return min(self.stack_height() * 0.5, self.thickness - self.pocket_floor)

    def resolved_corner_radius(self) -> float:
        """Corner radius for the square shape."""
        if self.corner_radius is not None:
            return min(self.corner_radius, self.size / 2.0)
        return round(self.size * 0.12)


def _polygon_ring(sides: int, size: float, cx: float, cy: float, rotate: float) -> Ring:
    """A regular polygon inscribed in a circle of diameter ``size``."""
    radius = size / 2.0
    return [
        (
            cx + radius * math.cos(rotate + 2 * math.pi * i / sides),
            cy + radius * math.sin(rotate + 2 * math.pi * i / sides),
        )
        for i in range(sides)
    ]


def _coaster_ring(params: CoasterParams) -> Ring:
    """Build one coaster outline, centred on the origin of its own box."""
    size = params.size
    half = size / 2.0
    if params.shape is CoasterShape.ROUND:
        return geo.circle_ring(half, half, half)
    if params.shape is CoasterShape.SQUARE:
        return geo.rounded_rect_ring(size, size, params.resolved_corner_radius())
    if params.shape is CoasterShape.SOFT:
        return geo.dedupe(
            geo.superellipse_ring(size, size, params.superellipse_exponent, half, half)
        )
    sides = 6 if params.shape is CoasterShape.HEX else 8
    rotate = math.pi / 6.0 if sides == 6 else math.pi / 8.0
    ring = _polygon_ring(sides, size, half, half, rotate)
    # A cutter cannot turn a true point; rounding the corners is both
    # machinable and kinder to the hand.
    return geo.round_convex(ring, max(4.0, params.tool_diameter * 0.7))


def _pattern_contours(params: CoasterParams, outline: Ring) -> list[Contour]:
    """Build the engraved decoration for one coaster.

    Args:
        params: Coaster parameters.
        outline: The coaster outline.

    Returns:
        Engrave contours, possibly empty.

    Raises:
        ValueError: If the pattern does not fit inside the coaster.
    """
    if params.pattern is Pattern.NONE:
        return []
    inset = params.pattern_inset
    rings = geo.offset_ring(outline, -inset, join="round")
    if not rings:
        raise ValueError(
            f"a {inset:g} mm pattern inset leaves nothing on a "
            f"{params.size:g} mm coaster"
        )
    first = rings[0]
    if params.pattern is Pattern.BORDER:
        return [Contour(first, ENGRAVE, closed=True)]

    cx, cy = params.size / 2.0, params.size / 2.0
    if params.pattern is Pattern.RINGS:
        out: list[Contour] = []
        step = (min(geo.size_of(first)) / 2.0 - 6.0) / max(params.pattern_lines, 1)
        if step < 3.0:
            raise ValueError(
                f"{params.pattern_lines} rings do not fit on a "
                f"{params.size:g} mm coaster"
            )
        for index in range(params.pattern_lines):
            for ring in geo.offset_ring(first, -index * step, join="round"):
                out.append(Contour(ring, ENGRAVE, closed=True))
        return out

    area = geo.polygon_from_ring(first)
    segments: list[list[Point]] = []
    if params.pattern is Pattern.RAYS:
        radius = params.size
        for index in range(params.pattern_lines):
            angle = math.pi * index / params.pattern_lines
            segments.append(
                [
                    (cx - radius * math.cos(angle), cy - radius * math.sin(angle)),
                    (cx + radius * math.cos(angle), cy + radius * math.sin(angle)),
                ]
            )
    else:  # GRID
        span = params.size
        step = span / (params.pattern_lines + 1)
        for index in range(1, params.pattern_lines + 1):
            offset = index * step
            segments.append([(offset, -span), (offset, 2 * span)])
            segments.append([(-span, offset), (2 * span, offset)])

    from shapely.geometry import LineString

    out = []
    for segment in segments:
        clipped = area.intersection(LineString(segment))
        for piece in getattr(clipped, "geoms", [clipped]):
            points = geo.dedupe(list(piece.coords), closed=False)
            if len(points) >= 2:
                out.append(Contour(points, ENGRAVE, closed=False))
    if not out:
        raise ValueError("the pattern fell entirely outside the coaster")
    return out


def _coaster_part(params: CoasterParams, index: int) -> Part:
    """Build one coaster."""
    outline = _coaster_ring(params)
    pockets: list[Pocket] = []
    if params.recess:
        rings = geo.offset_ring(outline, -params.recess_inset, join="round")
        if not rings:
            raise ValueError(
                f"a {params.recess_inset:g} mm recess inset leaves nothing on a "
                f"{params.size:g} mm coaster"
            )
        pockets.append(
            Pocket(geo.round_convex(rings[0], params.tool_diameter * 0.8), params.recess_depth)
        )
    engrave = _pattern_contours(params, outline)
    if params.recess and engrave:
        # Decoration belongs on the floor of the recess or outside it, never
        # straddling the wall.
        engrave = [c for c in engrave if _inside(c, pockets[0])]
    return Part(name=f"coaster-{index + 1}", outline=outline, pockets=pockets, engrave=engrave)


def _inside(contour: Contour, pocket: Pocket) -> bool:
    """Whether an engrave contour lies wholly inside a pocket."""
    from shapely.geometry import LineString

    region = pocket.region()
    points = list(contour.points) + ([contour.points[0]] if contour.closed else [])
    return region.covers(LineString(points))


def _holder_part(params: CoasterParams, coaster: Ring) -> Part:
    """Build the holder plate.

    Args:
        params: Coaster parameters.
        coaster: The coaster outline the recess must accept.

    Returns:
        The holder part.

    Raises:
        ValueError: If the recess is too deep for the material, or the push
            hole does not leave enough floor around it.
    """
    depth = params.resolved_holder_depth()
    if depth > params.thickness - params.pocket_floor:
        raise ValueError(
            f"a {depth:g} mm holder recess leaves too little floor in "
            f"{params.thickness:g} mm material"
        )
    grown = geo.offset_ring(coaster, params.holder_clearance, join="round")
    if len(grown) != 1:
        raise ValueError("the holder recess did not offset cleanly")
    recess = geo.round_convex(grown[0], params.tool_diameter * 0.8)
    outer_rings = geo.offset_ring(recess, params.holder_wall, join="round")
    if len(outer_rings) != 1:
        raise ValueError("the holder outline did not offset cleanly")
    outline = geo.round_convex(outer_rings[0], max(6.0, params.tool_diameter))

    holes: list[Ring] = []
    if params.holder_push_hole > 0:
        radius = params.holder_push_hole / 2.0
        centre = geo.centroid(recess)
        hole = geo.circle_ring(centre[0], centre[1], radius)
        region = geo.polygon_from_ring(recess)
        if not region.contains(geo.polygon_from_ring(hole)):
            raise ValueError(
                f"a {params.holder_push_hole:g} mm push hole does not fit "
                f"inside the holder recess"
            )
        if region.exterior.distance(geo.polygon_from_ring(hole)) < params.min_wall:
            raise ValueError(
                f"a {params.holder_push_hole:g} mm push hole leaves less than "
                f"{params.min_wall:g} mm of recess floor around it"
            )
        holes.append(hole)

    labels: list[Label] = []
    if params.label:
        x0, y0, x1, y1 = geo.bbox(outline)
        text = f"{params.count} x {params.size:g} mm  {params.material}"
        height = min(4.5, params.holder_wall * 0.35)
        if 0.62 * height * len(text) < (x1 - x0) * 0.9:
            labels.append(
                Label(text, ((x0 + x1) / 2.0, y0 + params.holder_wall * 0.45), height, align="center")
            )
    return Part(
        name="holder",
        outline=outline,
        holes=holes,
        pockets=[Pocket(recess, depth)],
        labels=labels,
    )


class CoasterGenerator(Generator):
    """Generates coaster sets with an optional holder."""

    niche = "coasters"
    title = "Coasters"
    summary = "Coaster sets in five shapes with engraved patterns and a holder"
    params_model = CoasterParams

    def generate(self, params: CoasterParams) -> Design:  # type: ignore[override]
        """Build one coaster set.

        Args:
            params: Coaster parameters.

        Returns:
            The design, parts laid out in a grid on the sheet.

        Raises:
            ValueError: If the parameters are geometrically incompatible.
        """
        coaster = _coaster_ring(params)
        parts = [_coaster_part(params, index) for index in range(params.count)]
        if params.holder:
            parts.append(_holder_part(params, coaster))
        arrange_grid(parts, params.gap)

        width, height = geo.size_of(
            [point for part in parts for point in part.placed().outline]
        )
        design = Design(
            slug=slugify(
                "coaster", params.shape.value, f"{params.size:g}mm",
                f"set{params.count}", params.pattern.value,
                "holder" if params.holder else "plain",
            ),
            name=(
                f"{params.shape.value.title()} Coaster Set of {params.count}, "
                f"{params.size:g} mm"
                + (" with holder" if params.holder else "")
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
        )
        design.cutting_order = cutting_order_for(design)
        return design

    def _description(self, params: CoasterParams) -> str:
        bits = [
            f"A set of {params.count} {params.shape.value} coasters "
            f"{params.size:g} mm across, cut from {params.thickness:g} mm "
            f"{params.material}"
        ]
        if params.pattern is not Pattern.NONE:
            bits.append(f"engraved with a {params.pattern.value} pattern")
        if params.recess:
            bits.append(f"with a {params.recess_depth:g} mm drip recess")
        if params.holder:
            bits.append(
                f"and a holder that takes the whole stack "
                f"{params.resolved_holder_depth():g} mm deep"
            )
        return ", ".join(bits) + f". Cut with a {params.tool_diameter:g} mm cutter."

    def _notes(self, params: CoasterParams) -> list[str]:
        notes = [
            f"All {params.count} coasters are identical; cut as many as you like.",
        ]
        if params.holder:
            notes.append(
                f"The holder recess is {params.holder_clearance:g} mm larger "
                f"than a coaster all round, so the stack drops in without "
                f"forcing."
            )
            if params.holder_push_hole > 0:
                notes.append(
                    f"The {params.holder_push_hole:g} mm hole in the holder "
                    f"floor is there to push the stack back out."
                )
        if params.pattern is not Pattern.NONE:
            notes.append(
                "Engraving is single-line: run it with a V-bit or a small "
                "engraving cutter, not as a pocket."
            )
        return notes

    def sample_params(self, rng: random.Random, index: int) -> CoasterParams:
        """Draw one coaster set variant."""
        shape = rng.choice(list(CoasterShape))
        thickness = rng.choice([9.0, 12.0, 19.0, 19.0])
        pattern = rng.choice(
            [Pattern.RINGS, Pattern.RAYS, Pattern.GRID, Pattern.BORDER, Pattern.NONE]
        )
        recess = rng.random() < 0.35 and thickness >= 12.0
        return CoasterParams(
            size=float(rng.randrange(90, 130, 5)),
            count=rng.choice([4, 4, 6, 2]),
            shape=shape,
            superellipse_exponent=rng.choice([3.0, 3.6, 4.5]),
            recess=recess,
            recess_inset=float(rng.randrange(8, 16, 2)),
            recess_depth=rng.choice([2.0, 3.0]),
            pattern=pattern,
            pattern_lines=rng.choice([3, 4, 5, 6, 8, 12]),
            pattern_inset=float(rng.randrange(6, 14, 2)),
            holder=rng.random() < 0.7,
            holder_wall=float(rng.randrange(12, 22, 2)),
            holder_push_hole=rng.choice([0.0, 30.0, 32.0, 36.0]),
            material=rng.choice(["oak", "walnut", "maple", "bamboo", "cherry"]),
            thickness=thickness,
        )
