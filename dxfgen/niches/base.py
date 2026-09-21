"""The generator contract every niche implements.

A generator is two things: a pydantic schema describing what can be varied,
and a :meth:`Generator.generate` method turning one instance of that schema
into a :class:`~dxfgen.core.design.Design`.  Everything else - CLI parsing,
seeded variant sampling, validation, skip logging - is shared here so that
adding a niche means writing geometry, not plumbing.

Variant generation is deterministic by construction: variant *i* is seeded
from ``(seed, i)`` alone, so the same seed always yields the same designs, and
asking for more variants never changes the ones already produced.
"""

from __future__ import annotations

import logging
import math
import random
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import Any, ClassVar, Iterable, Mapping, Sequence

from annotated_types import Ge, Gt, Le, Lt, MaxLen, MinLen
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ..core.design import Design, Label, Machine, Mode, Part
from ..core.layers import (
    CUT_INSIDE,
    CUT_OUTSIDE,
    DRILL,
    ENGRAVE,
    INFO,
    is_pocket_layer,
    parse_pocket_depth,
)
from ..core import geometry as geo
from ..core.limits import ValidationConfig
from ..core.validate import Report, validate_design

log = logging.getLogger("dxfgen")

__all__ = [
    "GeneratorParams",
    "Generator",
    "ParamDoc",
    "STOCK_SIZES",
    "smallest_stock",
    "apply_kerf",
    "arrange_grid",
    "nest_parts",
    "label_parts",
    "cutting_order_for",
    "material_phrase",
    "slugify",
    "GOLDEN",
]

#: The golden ratio, the default proportion for outlines that need one.
GOLDEN: float = 1.6180339887498949

#: Common stock panel sizes in mm, smallest first.  Declaring the smallest one
#: a design fits tells the buyer what to go and buy.
STOCK_SIZES: tuple[tuple[float, float], ...] = (
    (300.0, 200.0),
    (400.0, 300.0),
    (600.0, 400.0),
    (900.0, 600.0),
    (1200.0, 600.0),
    (1220.0, 2440.0),
)


def smallest_stock(
    width: float, height: float, sizes: Sequence[tuple[float, float]] = STOCK_SIZES
) -> tuple[float, float]:
    """Return the smallest listed stock panel a design fits on.

    Either orientation counts, since the operator can turn the board.

    Args:
        width: Design width in mm.
        height: Design height in mm.
        sizes: Candidate panel sizes, smallest first.

    Returns:
        The first panel that fits, or the largest listed panel if none does -
        the sheet check in the validator then reports the overflow.
    """
    for sw, sh in sizes:
        if (width <= sw and height <= sh) or (width <= sh and height <= sw):
            return (sw, sh)
    return sizes[-1]


def material_phrase(thickness: float, material: str) -> str:
    """Describe stock without saying the thickness twice.

    Materials are often written with the thickness already in them, as "3 mm
    birch ply", and prefixing that again gives "3 mm 3 mm birch ply".

    Args:
        thickness: Material thickness in mm.
        material: Free-text material description.

    Returns:
        A phrase such as ``"19 mm oak"`` or ``"3 mm birch ply"``.
    """
    if re.match(r"^\s*\d+(\.\d+)?\s*mm\b", material, flags=re.IGNORECASE):
        return material.strip()
    return f"{thickness:g} mm {material}"


def slugify(*parts: object) -> str:
    """Build a filesystem-safe slug from arbitrary parts.

    Args:
        *parts: Values to join; numbers are formatted compactly.

    Returns:
        A lowercase, hyphen-separated slug.
    """
    chunks: list[str] = []
    for part in parts:
        if isinstance(part, float):
            text = f"{part:g}"
        else:
            text = str(part)
        text = re.sub(r"[^A-Za-z0-9.]+", "-", text).strip("-").lower()
        text = text.replace(".", "p")
        if text:
            chunks.append(text)
    return "-".join(chunks)


def apply_kerf(parts: Sequence["Part"], params: "GeneratorParams") -> list["Part"]:
    """Compensate every part for the laser kerf; a no-op on a router.

    Applied before parts are arranged, because compensation changes their
    sizes.  See :meth:`dxfgen.core.design.Part.kerf_compensated` for why this
    is done to the whole part rather than to individual slots.

    Args:
        parts: The parts to compensate.
        params: The generator parameters, for the machine settings.

    Returns:
        The compensated parts, or the originals on a router.
    """
    machine = params.machine()
    if not machine.is_laser or machine.kerf <= 0:
        return list(parts)
    return [part.kerf_compensated(machine.kerf) for part in parts]


def arrange_grid(
    parts: Sequence["Part"], gap: float, columns: int | None = None
) -> None:
    """Lay parts out in a grid by setting their origins, in place.

    A deliberately simple placement: parts are put in row-major order on a
    lattice sized by the widest and tallest part.  Real nesting, which packs
    parts of different sizes together, belongs to the furniture work; this is
    enough for a set of identical coasters plus a holder.

    Args:
        parts: The parts to place.  Their ``origin`` is overwritten.
        gap: Space left between parts, in mm.
        columns: Column count; defaults to a roughly square arrangement.

    Raises:
        ValueError: If ``parts`` is empty or ``gap`` is negative.
    """
    if not parts:
        raise ValueError("nothing to arrange")
    if gap < 0:
        raise ValueError(f"gap must be >= 0, got {gap}")
    if columns is None:
        columns = max(1, int(round(math.sqrt(len(parts)))))
    cell_w = max(part.size()[0] for part in parts) + gap
    cell_h = max(part.size()[1] for part in parts) + gap
    for index, part in enumerate(parts):
        x0, y0, _, _ = part.bbox()
        row, column = divmod(index, columns)
        part.origin = (column * cell_w - x0, row * cell_h - y0)
        part.rotation = 0.0


def nest_parts(
    parts: Sequence["Part"],
    sheet: tuple[float, float],
    gap: float = 12.0,
    margin: float = 10.0,
    allow_rotation: bool = True,
) -> list[list["Part"]]:
    """Pack parts onto sheets and set their placement, in place.

    A first-fit-decreasing shelf packer: parts are sorted tallest first and
    laid in rows, each row as tall as its first part.  It is not optimal - no
    practical packer is - but it is stable, quick, and leaves a layout an
    operator can read, which matters more on a full sheet of plywood than the
    last few percent of yield.

    Parts taller than they are wide are turned on their side when that helps
    them fit, and a part that fits neither way is reported rather than
    silently dropped.

    Args:
        parts: The parts to place.  Their ``origin`` and ``rotation`` are
            overwritten, and ``quantity`` is expanded into separate copies.
        sheet: ``(width, height)`` of the stock in mm.
        gap: Space between parts, in mm.
        margin: Unused border around the sheet, in mm.
        allow_rotation: Whether parts may be turned 90 degrees.

    Returns:
        One list of placed parts per sheet.

    Raises:
        ValueError: If a part does not fit on a sheet in either orientation.
    """
    sheet_w, sheet_h = sheet
    usable_w = sheet_w - 2 * margin
    usable_h = sheet_h - 2 * margin
    if usable_w <= 0 or usable_h <= 0:
        raise ValueError(f"a {margin} mm margin leaves no usable area on {sheet}")

    expanded: list[Part] = []
    for part in parts:
        for index in range(part.quantity):
            clone = part.copy(quantity=1)
            if part.quantity > 1:
                clone.name = f"{part.name}-{index + 1}"
            expanded.append(clone)

    def footprint(part: Part, turned: bool) -> tuple[float, float]:
        width, height = part.size()
        return (height, width) if turned else (width, height)

    prepared: list[tuple[float, float, bool, Part]] = []
    for part in expanded:
        width, height = footprint(part, False)
        turned = False
        if width > usable_w or height > usable_h:
            if allow_rotation and height <= usable_w and width <= usable_h:
                turned = True
                width, height = height, width
            else:
                raise ValueError(
                    f"part {part.name!r} is {width:.0f} x {height:.0f} mm and "
                    f"does not fit a {sheet_w:g} x {sheet_h:g} mm sheet with a "
                    f"{margin:g} mm margin"
                )
        prepared.append((width, height, turned, part))

    prepared.sort(key=lambda item: (-item[1], -item[0], item[3].name))

    sheets: list[list[Part]] = []
    cursor_x = margin
    cursor_y = margin
    row_height = 0.0
    current: list[Part] = []
    for width, height, turned, part in prepared:
        if current and cursor_x + width > sheet_w - margin:
            cursor_x = margin
            cursor_y += row_height + gap
            row_height = 0.0
        if current and cursor_y + height > sheet_h - margin:
            sheets.append(current)
            current = []
            cursor_x = margin
            cursor_y = margin
            row_height = 0.0
        part.rotation = 90.0 if turned else 0.0
        x0, y0, _, _ = geo.bbox(geo.rotate(part.outline, part.rotation))
        part.origin = (cursor_x - x0, cursor_y - y0)
        current.append(part)
        cursor_x += width + gap
        row_height = max(row_height, height)
    if current:
        sheets.append(current)
    return sheets


def label_parts(parts: Sequence["Part"], height: float = 6.0) -> None:
    """Put each part's name on its INFO layer, in place.

    A nested sheet of twenty similar panels is unusable without this.

    Args:
        parts: The parts to label.
        height: Text height in mm.
    """
    for part in parts:
        x0, y0, x1, y1 = part.bbox()
        if min(x1 - x0, y1 - y0) < height * 3:
            continue
        part.labels.append(
            Label(part.name, ((x0 + x1) / 2.0, (y0 + y1) / 2.0), height, align="center")
        )


def cutting_order_for(design: Design) -> list[str]:
    """Derive a sensible operation order for a design's README.

    Pockets first while the stock is still whole, then interior through cuts,
    then drilling, then the outer profile last so the part stays held down for
    as long as possible.  Engraving goes before the profile for the same
    reason.
    """
    layers = design.layers_used()
    steps: list[str] = []
    pockets = sorted(
        (layer for layer in layers if is_pocket_layer(layer)),
        key=lambda name: parse_pocket_depth(name) or 0.0,
    )
    for layer in pockets:
        depth = parse_pocket_depth(layer) or 0.0
        steps.append(f"Pocket {layer} to {depth:g} mm deep")
    if ENGRAVE in layers:
        steps.append(f"Engrave {ENGRAVE}")
    if DRILL in layers:
        steps.append(f"Drill {DRILL} holes")
    if CUT_INSIDE in layers:
        steps.append(f"Cut {CUT_INSIDE} through, tool inside the line")
    steps.append(f"Cut {CUT_OUTSIDE} through last, tool outside the line")
    steps.append(f"{INFO} is annotation only - do not machine it")
    return steps


class GeneratorParams(BaseModel):
    """Parameters common to every niche.

    Subclasses add their own geometry fields.  ``extra="forbid"`` means a typo
    on the command line is an error rather than a silently ignored setting.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    mode: Mode = Field(
        Mode.ROUTER, description="router cuts on the line, laser compensates kerf"
    )
    material: str = Field(
        "hardwood", min_length=1, max_length=60, description="material description"
    )
    thickness: float = Field(19.0, gt=0, le=60, description="material thickness in mm")
    tool_diameter: float = Field(
        6.35, gt=0, le=25.4, description="router cutter diameter in mm"
    )
    kerf: float = Field(0.15, ge=0, le=2.0, description="laser kerf width in mm")
    clearance: float = Field(
        0.2, ge=0, le=2.0, description="finished joint fit clearance in mm"
    )
    min_wall: float = Field(
        8.0, ge=1, le=50, description="minimum material between features in mm"
    )
    pocket_floor: float = Field(
        5.0, ge=1, le=30, description="material that must remain under a pocket in mm"
    )

    def machine(self) -> Machine:
        """Build the :class:`Machine` these parameters describe."""
        return Machine(
            mode=self.mode,
            tool_diameter=self.tool_diameter,
            kerf=self.kerf,
            clearance=self.clearance,
        )

    def validation_config(self) -> ValidationConfig:
        """Build the validator limits these parameters imply.

        The generator and the validator must agree on what "minimum wall"
        means, so the limits are derived from the same parameters rather than
        configured twice.
        """
        return ValidationConfig(min_wall=self.min_wall, pocket_floor=self.pocket_floor)


@dataclass(frozen=True)
class ParamDoc:
    """One row of ``dxfgen list`` output."""

    name: str
    type_name: str
    default: object
    constraints: str
    description: str


def _describe_constraints(metadata: Iterable[object]) -> str:
    """Render pydantic field constraints as a compact string."""
    bits: list[str] = []
    for item in metadata:
        if isinstance(item, Gt):
            bits.append(f"> {item.gt:g}")
        elif isinstance(item, Ge):
            bits.append(f">= {item.ge:g}")
        elif isinstance(item, Lt):
            bits.append(f"< {item.lt:g}")
        elif isinstance(item, Le):
            bits.append(f"<= {item.le:g}")
        elif isinstance(item, MinLen):
            bits.append(f"len >= {item.min_length}")
        elif isinstance(item, MaxLen):
            bits.append(f"len <= {item.max_length}")
    return ", ".join(bits)


class Generator(ABC):
    """Base class for every niche generator."""

    #: Niche key used on the command line and as the output directory name.
    niche: ClassVar[str]
    #: Human readable niche title.
    title: ClassVar[str]
    #: One-line description of what this niche produces.
    summary: ClassVar[str] = ""
    #: The parameter schema this generator accepts.
    params_model: ClassVar[type[GeneratorParams]]

    # ------------------------------------------------------------- subclasses
    @abstractmethod
    def generate(self, params: GeneratorParams) -> Design:
        """Build one design from one fully specified parameter set.

        Implementations must raise :class:`ValueError` with a specific message
        when the parameters are geometrically incompatible, rather than
        silently producing something that will not cut.  They should also pass
        ``params.validation_config()`` as the design's ``limits``; if they do
        not, :meth:`make` and :meth:`variants` fill it in.
        """

    @abstractmethod
    def sample_params(self, rng: random.Random, index: int) -> GeneratorParams:
        """Draw one variant's parameters.

        Args:
            rng: Seeded generator; all randomness must come from here so the
                result is reproducible.
            index: Variant index, useful for spreading choices deterministically.

        Returns:
            A parameter instance.
        """

    # ------------------------------------------------------------------ shared
    def parse(self, values: Mapping[str, Any] | None = None) -> GeneratorParams:
        """Validate a mapping of overrides into a parameter instance.

        String values are coerced by pydantic, so command line ``key=value``
        pairs can be passed straight through.

        Args:
            values: Overrides; omitted keys take their default.

        Returns:
            A parameter instance.

        Raises:
            pydantic.ValidationError: If a key is unknown or a value is out of
                range.
        """
        return self.params_model.model_validate(dict(values or {}))

    def make(self, **overrides: Any) -> Design:
        """Parse overrides and build one design.

        Args:
            **overrides: Parameter overrides.

        Returns:
            The design, normalised to the origin.
        """
        params = self.parse(overrides)
        return self._finish(self.generate(params), params)

    @staticmethod
    def _finish(design: Design, params: GeneratorParams) -> Design:
        """Normalise a design and give it the limits its parameters imply."""
        design = design.normalized()
        if design.limits == ValidationConfig():
            design = replace(design, limits=params.validation_config())
        return design

    def parameter_docs(self) -> list[ParamDoc]:
        """Describe this generator's parameters for ``dxfgen list``."""
        docs: list[ParamDoc] = []
        for name, field in self.params_model.model_fields.items():
            annotation = field.annotation
            type_name = getattr(annotation, "__name__", str(annotation))
            if isinstance(annotation, type) and issubclass(annotation, Mode):
                type_name = "router|laser"
            docs.append(
                ParamDoc(
                    name=name,
                    type_name=type_name,
                    default=field.default,
                    constraints=_describe_constraints(field.metadata),
                    description=field.description or "",
                )
            )
        return docs

    def variants(
        self,
        count: int,
        seed: int = 0,
        config: ValidationConfig | None = None,
        attempt_factor: int = 6,
    ) -> list[Design]:
        """Generate ``count`` distinct, validated variants.

        Variant *i* depends only on ``(seed, i)``, so the same seed reproduces
        the same designs and raising ``count`` never disturbs earlier ones.
        Anything that fails to build or fails validation is skipped with its
        reason logged, and never exported.

        Args:
            count: How many valid variants are wanted.
            seed: Master seed.
            config: Validator limits; defaults to those implied by each
                variant's own parameters.
            attempt_factor: How many samples per requested variant to try
                before giving up.

        Returns:
            Up to ``count`` designs.  Fewer means the parameter space could
            not produce more valid, distinct results.

        Raises:
            ValueError: If ``count`` is not positive.
        """
        if count < 1:
            raise ValueError(f"count must be >= 1, got {count}")
        designs: list[Design] = []
        seen: set[str] = set()
        index = 0
        limit = count * max(1, attempt_factor)
        while len(designs) < count and index < limit:
            rng = random.Random(seed * 1_000_003 + index)
            current, index = index, index + 1
            try:
                params = self.sample_params(rng, current)
                design = self._finish(self.generate(params), params)
            except (ValueError, ValidationError) as exc:
                log.info("%s: variant %d skipped, %s", self.niche, current, exc)
                continue
            if design.slug in seen:
                continue
            report = self.check(design, config)
            if not report.ok:
                log.info(
                    "%s: variant %d (%s) skipped, %s",
                    self.niche,
                    current,
                    design.slug,
                    report.reason(),
                )
                continue
            seen.add(design.slug)
            designs.append(design)
        if len(designs) < count:
            log.warning(
                "%s: produced %d of %d requested variants after %d attempts",
                self.niche,
                len(designs),
                count,
                index,
            )
        return designs

    def check(
        self, design: Design, config: ValidationConfig | None = None
    ) -> Report:
        """Validate a design against its own limits, or the ones given."""
        return validate_design(design, config)
