"""Cigar and whiskey trays.

The layout is the design: a tray of this kind is two zones end to end, a
recess that takes the base of a tumbler at one end and cigar rests at the
other, with an optional ash well between them.  Everything is positioned from
the glass outwards, and each zone has to leave full wall thickness to its
neighbour, so the tray length follows from the contents rather than the other
way round.  A tray too short to hold a real cigar rest is refused rather than
shipped with a 10 mm groove in it.
"""

from __future__ import annotations

import math
import random
from enum import Enum

from pydantic import Field

from ..core import geometry as geo
from ..core.design import Contour, Design, Label, Part, Pocket
from ..core.geometry import Ring
from ..core.layers import ENGRAVE
from .base import GOLDEN, Generator, GeneratorParams, cutting_order_for, slugify, smallest_stock

__all__ = ["TrayShape", "AshtrayParams", "AshtrayGenerator"]


class TrayShape(str, Enum):
    """Outer profile family."""

    ROUNDED = "rounded"
    SOFT = "soft"
    PILL = "pill"


#: A cigar rest shorter than this does not cradle a cigar, in mm.
MIN_REST_LENGTH = 55.0
#: Typical tumbler base diameters run 70-90 mm; below this nothing sits in it.
MIN_GLASS_DIAMETER = 60.0


class AshtrayParams(GeneratorParams):
    """Parameters for a cigar and whiskey tray."""

    length: float = Field(320.0, ge=200.0, le=700.0, description="overall length in mm")
    width: float | None = Field(
        None, ge=120.0, le=400.0, description="overall width in mm; default length/aspect"
    )
    aspect: float = Field(
        GOLDEN, ge=1.1, le=2.4, description="length/width ratio when width is omitted"
    )
    shape: TrayShape = Field(TrayShape.ROUNDED, description="outer profile family")
    superellipse_exponent: float = Field(
        3.4, ge=2.0, le=8.0, description="curve exponent for the soft shape"
    )
    corner_radius: float | None = Field(
        None, ge=0.0, le=200.0, description="corner radius in mm; default 12% of width"
    )
    rim: float | None = Field(
        None, ge=8.0, le=80.0, description="material around the features in mm"
    )
    glass_diameter: float = Field(
        88.0, ge=MIN_GLASS_DIAMETER, le=140.0, description="glass recess diameter in mm"
    )
    glass_depth: float = Field(
        6.0, gt=0.0, le=20.0, description="glass recess depth in mm"
    )
    ash_well: bool = Field(False, description="add an ash well between the zones")
    well_diameter: float = Field(
        60.0, ge=30.0, le=140.0, description="ash well diameter in mm"
    )
    well_depth: float = Field(
        12.0, gt=0.0, le=30.0, description="ash well depth in mm"
    )
    rests: int = Field(2, ge=1, le=4, description="how many cigar rests")
    rest_width: float = Field(
        22.0, ge=14.0, le=40.0, description="cigar rest groove width in mm"
    )
    rest_depth: float = Field(
        11.0, gt=0.0, le=25.0, description="cigar rest groove depth in mm"
    )
    rest_length: float | None = Field(
        None,
        ge=MIN_REST_LENGTH,
        le=260.0,
        description="cigar rest groove length in mm; default 80",
    )
    engrave_border: bool = Field(False, description="add an engraved border line")
    engrave_inset: float = Field(
        8.0, ge=3.0, le=40.0, description="engraved border inset in mm"
    )
    label: bool = Field(True, description="add an INFO size label")

    def resolved_width(self) -> float:
        """Overall width in mm, rounded to the millimetre when derived."""
        if self.width is not None:
            return self.width
        return float(round(self.length / self.aspect))

    def resolved_rest_length(self) -> float:
        """Cigar rest length in mm.

        A rest is a notch the cigar sits in, not a trough it lies in: the
        cigar overhangs at both ends.  Running the groove the whole way to the
        end of the tray turns it into something that looks like a pen rest.
        """
        return self.rest_length if self.rest_length is not None else 80.0

    def resolved_rim(self) -> float:
        """Material left around every feature, in mm."""
        if self.rim is not None:
            return self.rim
        return max(self.min_wall + 4.0, round(self.resolved_width() * 0.09))

    def resolved_corner_radius(self) -> float:
        """Corner radius in mm."""
        width = self.resolved_width()
        limit = min(self.length, width) / 2.0
        if self.shape is TrayShape.PILL:
            return limit
        if self.corner_radius is not None:
            return min(self.corner_radius, limit)
        return min(max(8.0, round(width * 0.12)), limit)


def _outline(params: AshtrayParams) -> Ring:
    """Build the outer profile."""
    length, width = params.length, params.resolved_width()
    if params.shape is TrayShape.SOFT:
        return geo.dedupe(
            geo.superellipse_ring(
                length, width, params.superellipse_exponent, length / 2.0, width / 2.0
            )
        )
    return geo.rounded_rect_ring(length, width, params.resolved_corner_radius())


def _layout(params: AshtrayParams) -> tuple[float, float, float, float]:
    """Place the zones along the tray.

    Everything is measured out from the glass recess, which is the one feature
    with a fixed real-world size.

    Args:
        params: Tray parameters.

    Returns:
        ``(glass_x, well_x, rest_start_x, rest_end_x)`` in mm.

    Raises:
        ValueError: If the tray is too short to hold the zones with full wall
            thickness between them.
    """
    rim = params.resolved_rim()
    wall = params.min_wall
    glass_x = rim + params.glass_diameter / 2.0
    cursor = glass_x + params.glass_diameter / 2.0
    well_x = 0.0
    if params.ash_well:
        well_x = cursor + wall + params.well_diameter / 2.0
        cursor = well_x + params.well_diameter / 2.0
    zone_start = cursor + wall
    zone_end = params.length - rim
    available = zone_end - zone_start
    if available < MIN_REST_LENGTH:
        needed = zone_start + MIN_REST_LENGTH + rim
        raise ValueError(
            f"a {params.length:g} mm tray leaves only "
            f"{max(available, 0.0):.0f} mm for the cigar rests, below the "
            f"{MIN_REST_LENGTH:g} mm a cigar needs; it would have to be at "
            f"least {needed:.0f} mm long"
        )
    # Centre the rests in the zone they have, rather than stretching them
    # across it.
    length = min(params.resolved_rest_length(), available)
    rest_start = zone_start + (available - length) / 2.0
    return glass_x, well_x, rest_start, rest_start + length


def _rest_rings(params: AshtrayParams, start: float, end: float) -> list[Ring]:
    """Build the cigar rest grooves, spread across the tray width.

    Args:
        params: Tray parameters.
        start: Where the rests begin along the length, in mm.
        end: Where they end, in mm.

    Returns:
        One closed ring per rest.

    Raises:
        ValueError: If the rests do not fit across the width.
    """
    width = params.resolved_width()
    rim = params.resolved_rim()
    span = width - 2.0 * rim
    needed = params.rests * params.rest_width + (params.rests - 1) * params.min_wall
    if needed > span:
        raise ValueError(
            f"{params.rests} rests {params.rest_width:g} mm wide need "
            f"{needed:.0f} mm across, but only {span:.0f} mm is available"
        )
    pitch = span / params.rests
    rings: list[Ring] = []
    for index in range(params.rests):
        centre = rim + pitch * (index + 0.5)
        rings.append(
            geo.stadium_ring(
                end - start,
                params.rest_width,
                start,
                centre - params.rest_width / 2.0,
            )
        )
    return rings


class AshtrayGenerator(Generator):
    """Generates cigar and whiskey trays."""

    niche = "ashtrays"
    title = "Cigar trays"
    summary = "Cigar and whiskey trays with a glass recess, cigar rests and an ash well"
    params_model = AshtrayParams

    def generate(self, params: AshtrayParams) -> Design:  # type: ignore[override]
        """Build one cigar tray.

        Args:
            params: Tray parameters.

        Returns:
            The design, with the bounding box at the origin.

        Raises:
            ValueError: If the parameters are geometrically incompatible.
        """
        width = params.resolved_width()
        outline = _outline(params)
        glass_x, well_x, rest_start, rest_end = _layout(params)

        pockets = [
            Pocket(
                geo.circle_ring(glass_x, width / 2.0, params.glass_diameter / 2.0),
                params.glass_depth,
            )
        ]
        if params.ash_well:
            if params.well_depth <= params.glass_depth:
                raise ValueError(
                    f"an ash well {params.well_depth:g} mm deep is no deeper "
                    f"than the {params.glass_depth:g} mm glass recess"
                )
            pockets.append(
                Pocket(
                    geo.circle_ring(well_x, width / 2.0, params.well_diameter / 2.0),
                    params.well_depth,
                )
            )
        pockets.extend(
            Pocket(ring, params.rest_depth)
            for ring in _rest_rings(params, rest_start, rest_end)
        )

        engrave: list[Contour] = []
        if params.engrave_border:
            inset = min(params.engrave_inset, params.resolved_rim() - params.tool_diameter)
            if inset < 3.0:
                raise ValueError("there is no room for an engraved border on this rim")
            engrave = [
                Contour(ring, ENGRAVE, closed=True)
                for ring in geo.offset_ring(outline, -inset, join="round")
            ]

        labels: list[Label] = []
        text = f"{params.length:g} x {width:g} x {params.thickness:g} mm  {params.material}"
        rim = params.resolved_rim()
        height = min(5.0, rim * 0.35)
        if params.label and rim >= 12.0 and 0.62 * height * len(text) < params.length * 0.8:
            labels.append(Label(text, (params.length / 2.0, rim * 0.42), height, align="center"))

        part = Part(
            name="tray", outline=outline, pockets=pockets, engrave=engrave, labels=labels
        )
        design = Design(
            slug=slugify(
                "cigar-tray", params.shape.value,
                f"{params.length:g}x{width:g}", f"{params.rests}rest",
                "well" if params.ash_well else "plain",
            ),
            name=(
                f"{params.shape.value.title()} Cigar and Whiskey Tray, "
                f"{params.length:g} x {width:g} mm, {params.rests} rest"
                f"{'s' if params.rests != 1 else ''}"
            ),
            niche=self.niche,
            description=self._description(params, width),
            parts=[part],
            machine=params.machine(),
            material=params.material,
            thickness=params.thickness,
            sheet=smallest_stock(params.length, width),
            params=params.model_dump(mode="json"),
            notes=self._notes(params),
            limits=params.validation_config(),
        )
        design.cutting_order = cutting_order_for(design)
        return design

    def _description(self, params: AshtrayParams, width: float) -> str:
        bits = [
            f"A cigar and whiskey tray measuring {params.length:g} x {width:g} mm "
            f"in {params.thickness:g} mm {params.material}",
            f"with a {params.glass_diameter:g} mm glass recess "
            f"{params.glass_depth:g} mm deep",
        ]
        if params.ash_well:
            bits.append(
                f"a {params.well_diameter:g} mm ash well {params.well_depth:g} mm deep"
            )
        bits.append(
            f"and {params.rests} cigar rest{'s' if params.rests != 1 else ''} "
            f"{params.rest_width:g} mm wide"
        )
        return ", ".join(bits) + f". Cut with a {params.tool_diameter:g} mm cutter."

    def _notes(self, params: AshtrayParams) -> list[str]:
        deepest = max(
            params.glass_depth,
            params.rest_depth,
            params.well_depth if params.ash_well else 0.0,
        )
        return [
            f"The glass recess takes a tumbler base up to "
            f"{params.glass_diameter - 2:g} mm across.",
            f"Deepest cut {deepest:g} mm leaves "
            f"{params.thickness - deepest:g} mm of floor.",
            "Rests are drawn flat-bottomed; a round-nose bit of the same width "
            "gives the usual cradle profile at the same depth.",
        ]

    def sample_params(self, rng: random.Random, index: int) -> AshtrayParams:
        """Draw one cigar tray variant."""
        glass = float(rng.randrange(78, 106, 4))
        well = rng.random() < 0.4
        well_diameter = float(rng.randrange(45, 75, 5))
        rests = rng.choice([1, 2, 2, 3])
        rest_width = float(rng.randrange(18, 30, 2))
        # Length follows from the contents: glass, optional well, then a rest
        # long enough to be a rest, each separated by a full wall.
        rim = 20.0
        needed = (
            rim + glass + (8.0 + well_diameter if well else 0.0) + 8.0
            + MIN_REST_LENGTH + 15.0 + rim
        )
        length = float(max(math.ceil(needed / 10.0) * 10.0, rng.randrange(28, 52) * 10))
        aspect = rng.choice([GOLDEN, 1.5, 1.8, 2.0])
        width = max(
            round(length / aspect / 10.0) * 10.0,
            rests * rest_width + (rests + 1) * 14.0,
        )
        thickness = rng.choice([19.0, 25.0, 25.0])
        return AshtrayParams(
            length=length,
            width=width,
            aspect=aspect,
            shape=rng.choice([TrayShape.ROUNDED, TrayShape.ROUNDED, TrayShape.SOFT, TrayShape.PILL]),
            superellipse_exponent=rng.choice([3.0, 3.4, 4.2]),
            glass_diameter=glass,
            glass_depth=rng.choice([5.0, 6.0, 8.0]),
            ash_well=well,
            well_diameter=well_diameter,
            well_depth=min(rng.choice([10.0, 12.0, 14.0]), thickness - 6.0),
            rests=rests,
            rest_width=rest_width,
            rest_depth=min(rng.choice([9.0, 11.0, 13.0]), thickness - 6.0),
            rest_length=float(rng.randrange(60, 110, 10)),
            engrave_border=rng.random() < 0.3,
            engrave_inset=float(rng.randrange(5, 12)),
            material=rng.choice(["walnut", "oak", "cherry", "mahogany", "maple"]),
            thickness=thickness,
        )
