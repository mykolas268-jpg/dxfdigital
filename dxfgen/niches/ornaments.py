"""Laser-cut ornaments and keychains.

Small pieces from the same parametric shape library the seasonal work uses,
sized for a hanging thread or a split ring, engraved, and nested as a set on
one sheet.  The hole is the whole product here: an ornament you cannot hang is
a coaster, so its placement is searched for rather than assumed, and a shape
too small or too thin to carry one is refused.
"""

from __future__ import annotations

import random
from enum import Enum

from pydantic import Field, model_validator

from ..core import geometry as geo
from ..core.design import Contour, Design, Mode, Part
from ..core.geometry import Ring
from ..core.layers import ENGRAVE
from .base import (
    Generator,
    GeneratorParams,
    apply_kerf,
    cutting_order_for,
    label_parts,
    material_phrase,
    nest_parts,
    slugify,
    smallest_stock,
)
from .shapes import detail_contours, place_hang_hole, seasonal_tags, shape_names, shape_ring

__all__ = ["OrnamentUse", "OrnamentParams", "OrnamentGenerator"]


class OrnamentUse(str, Enum):
    """What the piece hangs from."""

    THREAD = "thread"
    """A small hole for thread or ribbon."""
    RING = "ring"
    """A larger hole for a split ring or lanyard clip."""


#: Hole diameters per use, in mm.
_HOLE: dict[OrnamentUse, float] = {OrnamentUse.THREAD: 4.0, OrnamentUse.RING: 6.0}
#: Stock sheet ornaments are nested on, in mm.
ORNAMENT_SHEET = (600.0, 400.0)


class OrnamentParams(GeneratorParams):
    """Parameters for an ornament or keychain set.

    Machine defaults are laser in thin sheet, which is what these are.
    """

    mode: Mode = Field(Mode.LASER, description="router or laser; ornaments are laser work")
    material: str = Field(
        "birch ply", min_length=1, max_length=60, description="material description"
    )
    thickness: float = Field(3.0, gt=0, le=12, description="material thickness in mm")
    min_wall: float = Field(
        3.0, ge=1, le=20, description="minimum material between features in mm"
    )
    pocket_floor: float = Field(
        1.0, ge=0.2, le=10, description="material that must remain under a pocket in mm"
    )
    shapes: list[str] = Field(
        default_factory=lambda: ["star", "tree", "snowflake"],
        min_length=1,
        max_length=9,
        description=f"shapes in the set, from: {', '.join(shape_names())}",
    )
    width: float = Field(70.0, ge=35.0, le=150.0, description="ornament width in mm")
    copies: int = Field(1, ge=1, le=12, description="copies of each shape")
    use: OrnamentUse = Field(OrnamentUse.THREAD, description="thread hole or split ring")
    hole_diameter: float | None = Field(
        None, ge=2.0, le=14.0, description="hanging hole diameter in mm; default per use"
    )
    engrave_detail: bool = Field(True, description="engrave the shape's own detail")
    border_inset: float = Field(
        6.0, ge=2.0, le=20.0, description="engraved border inset in mm"
    )
    gap: float = Field(6.0, ge=2.0, le=30.0, description="spacing between parts, mm")

    @model_validator(mode="after")
    def _check_shapes(self) -> "OrnamentParams":
        unknown = [name for name in self.shapes if name not in shape_names()]
        if unknown:
            raise ValueError(
                f"unknown shape(s) {', '.join(unknown)}; available: "
                f"{', '.join(shape_names())}"
            )
        if len(set(self.shapes)) != len(self.shapes):
            raise ValueError("the set repeats a shape; use copies instead")
        return self

    def resolved_hole(self) -> float:
        """Hanging hole diameter in mm."""
        return self.hole_diameter if self.hole_diameter is not None else _HOLE[self.use]

    def inside_radius(self) -> float:
        """Radius the shape's inside corners are relieved to, in mm."""
        return 0.0 if self.machine().is_laser else self.tool_diameter / 2.0


def _ornament(params: OrnamentParams, shape: str, index: int) -> Part:
    """Build one ornament.

    Raises:
        ValueError: If the shape cannot carry a hanging hole at this size.
    """
    outline = shape_ring(shape, params.width, params.inside_radius())
    hole = place_hang_hole(outline, params.resolved_hole(), params.min_wall)
    engrave: list[Contour] = []
    if params.engrave_detail:
        from shapely.geometry import LineString

        # Clip the decoration around the hanging hole rather than discarding
        # any contour that grazes it: on a 75 mm heart the inset border runs
        # straight past the hole, and dropping it leaves the piece bare.
        keep = geo.polygon_from_ring(outline).difference(
            geo.polygon_from_ring(hole).buffer(params.min_wall / 2.0)
        )
        for points, closed in detail_contours(shape, params.width, params.border_inset):
            line = LineString(list(points) + ([points[0]] if closed else []))
            clipped = keep.intersection(line)
            for piece in getattr(clipped, "geoms", [clipped]):
                if piece.is_empty or piece.geom_type != "LineString":
                    continue
                trimmed = geo.dedupe(list(piece.coords), closed=False)
                if len(trimmed) >= 2:
                    engrave.append(Contour(trimmed, ENGRAVE, closed=False))
    return Part(name=f"{shape}-{index + 1}", outline=outline, holes=[hole], engrave=engrave)


class OrnamentGenerator(Generator):
    """Generates laser ornament and keychain sets."""

    niche = "ornaments"
    title = "Ornaments"
    summary = "Laser ornaments and keychains from parametric shapes, nested as a set"
    params_model = OrnamentParams

    def generate(self, params: OrnamentParams) -> Design:  # type: ignore[override]
        """Build one ornament set.

        Args:
            params: Ornament parameters.

        Returns:
            The design, parts nested on one sheet.

        Raises:
            ValueError: If a shape cannot be made at this size, or the set
                does not fit the sheet.
        """
        parts: list[Part] = []
        for index, shape in enumerate(params.shapes):
            part = _ornament(params, shape, index)
            part.quantity = params.copies
            parts.append(part)
        parts = apply_kerf(parts, params)

        sheets = nest_parts(parts, ORNAMENT_SHEET, gap=params.gap, margin=params.gap)
        if len(sheets) > 1:
            raise ValueError(
                f"{len(params.shapes)} shapes x {params.copies} copies at "
                f"{params.width:g} mm need {len(sheets)} sheets of "
                f"{ORNAMENT_SHEET[0]:g} x {ORNAMENT_SHEET[1]:g} mm; reduce the "
                f"count or the size"
            )
        placed = sheets[0]
        label_parts(placed, height=max(3.0, params.width * 0.05))
        # Names go on INFO so a sheet of similar silhouettes can be sorted.
        width, height = geo.size_of(
            [point for part in placed for point in part.placed().outline]
        )
        design = Design(
            slug=slugify(
                "ornament", params.use.value, "-".join(params.shapes[:3]),
                f"{params.width:g}mm",
                f"x{params.copies}" if params.copies > 1 else "",
            ),
            name=self._name(params),
            niche=self.niche,
            description=self._description(params),
            parts=placed,
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

    def _name(self, params: OrnamentParams) -> str:
        kind = "Keychain" if params.use is OrnamentUse.RING else "Ornament"
        total = len(params.shapes) * params.copies
        seasons = {seasonal_tags[s] for s in params.shapes}
        theme = seasons.pop().title() if len(seasons) == 1 else "Mixed"
        return f"{theme} {kind} Set of {total}, {params.width:g} mm"

    def _description(self, params: OrnamentParams) -> str:
        total = len(params.shapes) * params.copies
        stock = material_phrase(params.thickness, params.material)
        hang = (
            f"a {params.resolved_hole():g} mm hole for a split ring"
            if params.use is OrnamentUse.RING
            else f"a {params.resolved_hole():g} mm hole for thread or ribbon"
        )
        return (
            f"{total} laser-cut pieces {params.width:g} mm across in {stock}: "
            f"{', '.join(params.shapes)}. Each has {hang} and engraved detail. "
            f"Every silhouette is generated from parametric curves, not traced "
            f"from artwork. Nested on one "
            f"{ORNAMENT_SHEET[0]:g} x {ORNAMENT_SHEET[1]:g} mm sheet."
        )

    def _notes(self, params: OrnamentParams) -> list[str]:
        return [
            f"Kerf compensated at {params.kerf:g} mm; cut one piece as a test "
            f"if your machine runs wider.",
            f"Hanging holes are {params.resolved_hole():g} mm, placed where the "
            f"shape has material around them rather than at a fixed height.",
            "Engraving is single-line. Run it as a separate low-power pass "
            "before the cut, or the pieces will drop out first.",
            "The silhouettes are built from equations, so they are original "
            "work and free of any third-party artwork.",
        ]

    def sample_params(self, rng: random.Random, index: int) -> OrnamentParams:
        """Draw one ornament set variant."""
        use = rng.choice([OrnamentUse.THREAD, OrnamentUse.THREAD, OrnamentUse.RING])
        width = float(rng.randrange(45, 95, 5)) if use is OrnamentUse.THREAD else float(
            rng.randrange(38, 62, 4)
        )
        pool = shape_names()
        season = rng.choice(["christmas", "halloween", "mixed"])
        if season != "mixed":
            pool = [n for n in pool if seasonal_tags[n] == season] or pool
        # A small snowflake's hub is too small to hang from; the shape library
        # will refuse it, so do not keep proposing it.
        if width < 65.0:
            pool = [n for n in pool if n != "snowflake"] or ["star"]
        count = min(len(pool), rng.choice([2, 3, 3, 4]))
        return OrnamentParams(
            # Sorted, because rng.sample returns an ordered sample and the
            # same three shapes drawn in a different order made a different
            # slug, slipped past the duplicate check, and shipped the same
            # set of keychains twice in one bundle.  The nester decides the
            # layout anyway, so the draw order carries no information.
            shapes=sorted(rng.sample(pool, count)),
            width=width,
            copies=rng.choice([1, 1, 2, 3]),
            use=use,
            engrave_detail=rng.random() < 0.85,
            border_inset=float(rng.randrange(3, 9)),
            thickness=rng.choice([3.0, 3.0, 4.0, 6.0]),
            kerf=rng.choice([0.1, 0.15, 0.2]),
            material=rng.choice(["birch ply", "poplar ply", "acrylic", "bamboo"]),
        )
