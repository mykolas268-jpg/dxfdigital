"""Seasonal and themed pieces built on the parametric shape library.

Three forms share one set of outlines: a flat plaque, a shaped serving tray
with a recess, and a set of shaped coasters.  Nothing here is traced - every
silhouette comes from :mod:`dxfgen.niches.shapes`, which builds them from
equations - so the results are original and safe to sell.
"""

from __future__ import annotations

import math
import random
from enum import Enum
from typing import ClassVar

from pydantic import Field, model_validator

from ..core import geometry as geo
from ..core.design import Contour, Design, Label, Part, Pocket
from ..core.geometry import Ring
from ..core.layers import ENGRAVE
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
from .shapes import detail_contours, place_hang_hole, seasonal_tags, shape_names, shape_ring

__all__ = ["SeasonalForm", "SeasonalParams", "SeasonalGenerator"]


class SeasonalForm(str, Enum):
    """What the shape is made into."""

    PLAQUE = "plaque"
    """A flat shaped panel, engraved, optionally hung."""
    TRAY = "tray"
    """A shaped tray with a recess."""
    COASTER = "coaster"
    """A set of small shaped coasters."""


#: Sensible widths per form, in mm.
_FORM_WIDTH: dict[SeasonalForm, tuple[float, float]] = {
    SeasonalForm.PLAQUE: (180.0, 420.0),
    SeasonalForm.TRAY: (260.0, 480.0),
    SeasonalForm.COASTER: (95.0, 130.0),
}


class SeasonalParams(GeneratorParams):
    """Parameters for a seasonal piece."""

    SPACING_FIELD: ClassVar[str | None] = "gap"
    NOMINAL_SPACING: ClassVar[float] = 12.0

    shape: str = Field("pumpkin", description=f"one of: {', '.join(shape_names())}")
    form: SeasonalForm = Field(SeasonalForm.PLAQUE, description="what to make of it")
    width: float = Field(300.0, ge=80.0, le=600.0, description="overall width in mm")
    count: int = Field(1, ge=1, le=8, description="how many, for the coaster form")
    engrave_detail: bool = Field(True, description="engrave the shape's own detail")
    border_inset: float = Field(
        9.0, ge=3.0, le=40.0, description="engraved border inset in mm"
    )
    hang_hole: bool = Field(False, description="add a hanging hole")
    hang_hole_diameter: float = Field(
        8.0, ge=3.0, le=30.0, description="hanging hole diameter in mm"
    )
    pocket_inset: float = Field(
        22.0, ge=10.0, le=90.0, description="tray recess inset from the edge in mm"
    )
    pocket_depth: float = Field(
        8.0, gt=0.0, le=30.0, description="tray recess depth in mm"
    )
    gap: float | None = Field(
        None,
        ge=5.0,
        le=60.0,
        description="spacing between parts in mm; unset = 12, or the "
        "cutter + 5 on a router if that is more",
    )
    label: bool = Field(True, description="add an INFO label")

    @model_validator(mode="after")
    def _check_shape(self) -> "SeasonalParams":
        if self.shape not in shape_names():
            raise ValueError(
                f"unknown shape {self.shape!r}; available: "
                f"{', '.join(shape_names())}"
            )
        if self.form is not SeasonalForm.COASTER and self.count != 1:
            raise ValueError(
                f"only the {SeasonalForm.COASTER.value} form comes in sets, "
                f"got count={self.count} for a {self.form.value}"
            )
        return self

    def inside_radius(self) -> float:
        """Radius the shape's inside corners are relieved to, in mm."""
        return 0.0 if self.machine().is_laser else self.tool_diameter / 2.0

    def season(self) -> str:
        """Which occasion this shape belongs to."""
        return seasonal_tags[self.shape]


#: How much wigglier than its own outline a recess may be before the inward
#: offset has stopped following the shape and started dissolving it.  Measured
#: across every shape at 400 mm: the paw comes out at 1.40, because its toes
#: are small lobes that an offset turns into blobs, and everything that trays
#: well sits between 0.78 and 1.05.  The gap is wide enough that the exact
#: threshold does not matter much.
MAX_RECESS_DISTORTION = 1.20


def _shape_factor(ring: Ring) -> float:
    """Perimeter squared over area: how wiggly a shape is, regardless of size.

    Args:
        ring: A closed ring.

    Returns:
        The dimensionless factor; a circle gives about 12.6 and it rises the
        more convoluted the boundary gets.
    """
    poly = geo.polygon_from_ring(ring)
    return poly.length ** 2 / poly.area if poly.area > 0 else float("inf")


def _check_recess_follows(outline: Ring, recess: Ring, shape: str) -> None:
    """Refuse a recess that the inward offset has distorted out of shape.

    A tray's recess should be the outline moved inwards, not a different
    shape.  On an outline with small lobes - a paw's toes - offsetting eats
    them and rounding blobs what is left, and the result reads as a puddle
    inside a paw rather than as a paw-shaped tray.  Comparing how wiggly each
    is, relative to its own size, catches that without naming any shape.

    Args:
        outline: The part's outer profile.
        recess: The recess ring.
        shape: Shape name, for the message.

    Raises:
        ValueError: If the recess no longer follows the outline.
    """
    ratio = _shape_factor(recess) / _shape_factor(outline)
    if ratio > MAX_RECESS_DISTORTION:
        raise ValueError(
            f"a recess set into a {shape} does not follow its outline: the "
            f"offset leaves a shape {ratio:.2f} times as convoluted, which "
            f"reads as a puddle rather than a {shape}-shaped tray"
        )


def _shaped_part(params: SeasonalParams, index: int) -> Part:
    """Build one shaped part.

    Raises:
        ValueError: If the recess or hanging hole does not fit.
    """
    outline = shape_ring(params.shape, params.width, params.inside_radius())
    holes: list[Ring] = []
    pockets: list[Pocket] = []

    if params.form is SeasonalForm.TRAY:
        rings = geo.offset_ring(outline, -params.pocket_inset, join="round")
        if len(rings) != 1:
            raise ValueError(
                f"a {params.pocket_inset:g} mm rim leaves no single recess in a "
                f"{params.width:g} mm {params.shape}"
            )
        recess = rings[0]
        if min(geo.size_of(recess)) < 60.0:
            raise ValueError(
                f"a {params.pocket_inset:g} mm rim leaves only "
                f"{min(geo.size_of(recess)):.0f} mm of recess"
            )
        floor = max(params.tool_diameter * 0.8, 6.0)
        recess = geo.round_convex(recess, floor)
        _check_recess_follows(outline, recess, params.shape)
        pockets.append(Pocket(recess, params.pocket_depth))

    if params.hang_hole:
        holes.append(
            place_hang_hole(outline, params.hang_hole_diameter, params.min_wall)
        )

    engrave: list[Contour] = []
    if params.engrave_detail:
        inset = params.border_inset
        if params.form is SeasonalForm.TRAY:
            inset = min(inset, params.pocket_inset - params.tool_diameter)
        if inset >= 3.0:
            region = geo.polygon_from_ring(outline)
            for points, closed in detail_contours(params.shape, params.width, inset):
                contour = Contour(list(points), ENGRAVE, closed=closed)
                engrave.append(contour)
            if pockets:
                # Decoration on a tray belongs on the rim, not across the
                # recess wall where the cut would step through it.
                rim = region.difference(pockets[0].region())
                engrave = [c for c in engrave if _covered(rim, c)]

    name = "piece" if params.count == 1 else f"{params.shape}-{index + 1}"
    return Part(name=name, outline=outline, holes=holes, pockets=pockets, engrave=engrave)


def _covered(region, contour: Contour) -> bool:
    """Whether an engrave contour lies wholly within a region."""
    from shapely.geometry import LineString, Polygon

    points = list(contour.points) + ([contour.points[0]] if contour.closed else [])
    return region.covers(LineString(points))


class SeasonalGenerator(Generator):
    """Generates seasonal plaques, shaped trays and coaster sets."""

    niche = "seasonal"
    title = "Seasonal"
    summary = "Themed plaques, shaped trays and coaster sets from parametric curves"
    params_model = SeasonalParams

    def generate(self, params: SeasonalParams) -> Design:  # type: ignore[override]
        """Build one seasonal piece.

        Args:
            params: Seasonal parameters.

        Returns:
            The design.

        Raises:
            ValueError: If the shape cannot be made at this size on this
                machine.
        """
        parts = [_shaped_part(params, index) for index in range(params.count)]
        parts = apply_kerf(parts, params)
        if params.label and params.count == 1:
            x0, y0, x1, y1 = parts[0].bbox()
            text = (
                f"{x1 - x0:.0f} x {y1 - y0:.0f} mm  "
                f"{material_phrase(params.thickness, params.material)}"
            )
            if 0.62 * 4.0 * len(text) < (x1 - x0) * 0.7:
                parts[0].labels.append(
                    Label(text, ((x0 + x1) / 2.0, y0 + (y1 - y0) * 0.12), 4.0, align="center")
                )
        arrange_grid(parts, params.part_gap())
        width, height = geo.size_of(
            [point for part in parts for point in part.placed().outline]
        )
        design = Design(
            slug=slugify(
                params.season(), params.shape, params.form.value,
                f"{params.width:g}mm",
                f"x{params.count}" if params.count > 1 else "",
            ),
            name=self._name(params),
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

    def _name(self, params: SeasonalParams) -> str:
        shape = params.shape.title()
        if params.form is SeasonalForm.COASTER:
            return f"{shape} Coaster Set of {params.count}, {params.width:g} mm"
        form = "Serving Tray" if params.form is SeasonalForm.TRAY else "Plaque"
        return f"{shape} {form}, {params.width:g} mm"

    def _description(self, params: SeasonalParams) -> str:
        stock = material_phrase(params.thickness, params.material)
        bits = [
            f"A {params.shape} {params.form.value} {params.width:g} mm across in "
            f"{stock}, for {params.season()}"
        ]
        if params.form is SeasonalForm.TRAY:
            bits.append(
                f"with a {params.pocket_depth:g} mm recess set "
                f"{params.pocket_inset:g} mm in from the edge"
            )
        if params.engrave_detail:
            bits.append("and engraved detail")
        if params.hang_hole:
            bits.append(f"and a {params.hang_hole_diameter:g} mm hanging hole")
        return (
            ", ".join(bits)
            + ". The outline is generated from parametric curves, not traced "
            "from artwork."
        )

    def _notes(self, params: SeasonalParams) -> list[str]:
        notes = [
            "The silhouette is built from equations, so it is original work "
            "and free of any third-party artwork.",
        ]
        if not params.machine().is_laser:
            notes.append(
                f"Inside corners are relieved for a "
                f"{params.tool_diameter:g} mm cutter. A larger cutter will not "
                f"follow them."
            )
        if params.engrave_detail:
            notes.append(
                "Engraving is single-line: run it with a V-bit or a small "
                "engraving cutter, not as a pocket."
            )
        return notes

    def sample_params(self, rng: random.Random, index: int) -> SeasonalParams:
        """Draw one seasonal variant.

        The shape is taken in strict rotation rather than drawn at random.
        Twenty independent draws from nine shapes gives five pumpkins and one
        tree, and a buyer looking at the contact sheet counts the pumpkins: a
        seasonal bundle has to cover its occasions.  Rotating on the variant
        index guarantees the first nine variants are one of each.

        The rotation deliberately ignores ``rng``.  Each variant is seeded
        from ``(seed, index)`` alone, so a "random offset" drawn here would be
        a fresh uniform draw every time and no rotation at all - which is
        exactly the bug this replaced.  The seed still varies everything else:
        form, size, material, machine and decoration.
        """
        names = shape_names()
        shape = names[index % len(names)]
        forms = [SeasonalForm.PLAQUE, SeasonalForm.PLAQUE, SeasonalForm.TRAY,
                 SeasonalForm.COASTER]
        # A snowflake has no rim to sink a recess into, and its arms are finer
        # than any router bit, so it is laser-only and never a tray.
        if shape == "snowflake":
            forms = [f for f in forms if f is not SeasonalForm.TRAY]
        form = rng.choice(forms)
        laser = True if shape == "snowflake" else rng.random() < 0.35
        # A tray is a recess in thick stock.  Drawn independently of the
        # machine it lands as a 6 mm recess in 3 mm laser ply, which is not a
        # tray and not cuttable; the form wins and the work goes to a router.
        if form is SeasonalForm.TRAY:
            laser = False
        # A hanging hole a router cannot enter is not a hole.  The floor is the
        # cutter itself, with the validator's own feature margin on top.
        tool = 6.35
        smallest_hole = 4.0 if laser else math.ceil(tool * 1.15)
        hang_hole_diameter = float(
            max(smallest_hole, rng.randrange(6, 16, 2))
        )
        low, high = _FORM_WIDTH[form]
        width = float(rng.randrange(int(low), int(high), 10))
        thickness = 3.0 if laser else rng.choice([19.0, 19.0, 25.0])
        return SeasonalParams(
            shape=shape,
            form=form,
            width=width,
            count=rng.choice([4, 4, 6]) if form is SeasonalForm.COASTER else 1,
            engrave_detail=rng.random() < 0.8,
            border_inset=float(rng.randrange(6, 16, 2)),
            hang_hole=form is SeasonalForm.PLAQUE and rng.random() < 0.6,
            hang_hole_diameter=hang_hole_diameter,
            pocket_inset=float(rng.randrange(18, 34, 2)),
            pocket_depth=rng.choice([6.0, 8.0, 10.0]),
            mode="laser" if laser else "router",
            thickness=thickness,
            material=(
                rng.choice(["birch ply", "acrylic"])
                if laser
                else rng.choice(["oak", "walnut", "maple", "cherry"])
            ),
            min_wall=4.0 if laser else 8.0,
            pocket_floor=1.0 if laser else 5.0,
        )
