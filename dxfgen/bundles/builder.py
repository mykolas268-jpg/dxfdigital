"""Assembling the files that make up one sellable design.

A buyer gets a folder, not a file.  The DXF is for CAM, the SVG is for laser
software, the PDF is for anyone without either, the preview shows what the
file contains and the mockup shows what it becomes.  The README is what stops
the support emails.
"""

from __future__ import annotations

import logging
import tempfile
import zipfile
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Callable, Iterable, Sequence

from ..core.design import Design
from ..core.export_dxf import write_dxf
from ..core.export_pdf import A4, PaperSize, plan_pages, write_pdf
from ..core.export_svg import write_svg
from ..core.layers import layer_def
from ..core.limits import ValidationConfig
from ..core.preview import render_mockup, render_preview
from ..core.validate import Report
from ..niches import Generator, get_generator, niche_names
from ..niches.base import Skip, VariantRun, material_phrase
from .contact_sheet import Tile, render_contact_sheet

log = logging.getLogger("dxfgen")

__all__ = [
    "LICENSE_SUMMARY",
    "DEFAULT_SELLER",
    "CONTACT_SHEET_NAME",
    "LICENSE_NAME",
    "INDEX_NAME",
    "DesignOutput",
    "Bundle",
    "readme_text",
    "write_design",
    "license_text",
    "index_text",
    "build_bundle",
    "build_all",
    "zip_bundle",
]

#: Whose licence it is.  A seller replaces this with their shop name.
DEFAULT_SELLER: str = "the seller"

CONTACT_SHEET_NAME: str = "CONTACT_SHEET.png"
LICENSE_NAME: str = "LICENSE.txt"
INDEX_NAME: str = "INDEX.txt"

#: Plain words for the skip stages a generator reports.
_SKIP_WORDS: dict[str, str] = {
    "build": "parameter combinations that cannot be built as geometry",
    "validate": "designs that failed the manufacturing checks",
    "duplicate": "repeats of a design already in the set",
}

LICENSE_SUMMARY = """LICENSE SUMMARY
  You MAY cut this design and sell the physical items you make from it,
  including commercially and in unlimited quantity.
  You MAY modify the files for your own production.
  You MAY NOT resell, share, sub-license or redistribute the digital files
  themselves, modified or not, and you may not include them in another
  digital product or file bundle.
  See LICENSE.txt for the full terms."""


@dataclass
class DesignOutput:
    """The files written for one design.

    Attributes:
        design: The design that was written.
        directory: The folder holding its files.
        files: Written paths, keyed by kind.
        report: The validation report the export was gated on.
    """

    design: Design
    directory: Path
    files: dict[str, Path] = field(default_factory=dict)
    report: Report | None = None

    @property
    def preview(self) -> Path | None:
        """The clean preview image, if one was written."""
        return self.files.get("preview")


def _layer_table(design: Design) -> list[str]:
    """Describe every layer in the file, with its depth where it has one."""
    rows: list[str] = []
    for name in design.layers_used():
        spec = layer_def(name)
        if spec.depth is not None:
            detail = f"pocket, {spec.depth:g} mm deep"
        else:
            detail = spec.description
        rows.append(f"  {name:<16} {detail}")
    return rows


def readme_text(design: Design, paper: PaperSize = A4) -> str:
    """Build the README that ships beside a design.

    Args:
        design: The design being documented.
        paper: The paper the template PDF was laid out for.

    Returns:
        The README contents.
    """
    width, height = design.size()
    machine = design.machine
    deepest = max(design.pocket_depths(), default=0.0)
    plan = plan_pages(design, paper)

    lines: list[str] = [
        design.name,
        "=" * len(design.name),
        "",
        design.description,
        "",
        "MATERIAL",
        f"  Stock          {design.material}, {design.thickness:g} mm thick",
        f"  Finished size  {width:.1f} x {height:.1f} mm",
        f"  Board needed   {design.sheet_size()[0]:g} x {design.sheet_size()[1]:g} mm minimum",
    ]
    if deepest:
        lines.append(
            f"  Floor left     {design.thickness - deepest:.1f} mm under the "
            f"deepest recess"
        )
    lines += [
        "",
        "MACHINE",
        f"  Mode           {machine.mode.value}",
    ]
    if machine.is_laser:
        lines.append(f"  Kerf           {machine.kerf:g} mm, already compensated")
    else:
        lines.append(
            f"  Cutter         {machine.tool_diameter:g} mm diameter "
            f"({machine.tool_radius:g} mm radius)"
        )
        lines.append(
            "  Inside corners are pre-filleted to suit this cutter. A larger "
            "cutter will not fit them."
        )
    lines += [
        f"  Units          millimetres (DXF $INSUNITS = 4)",
        f"  Origin         bottom-left of the bounding box",
        "",
        "LAYERS",
        *_layer_table(design),
        "",
        "CUTTING ORDER",
    ]
    lines += [f"  {index}. {step}" for index, step in enumerate(design.cutting_order, 1)]

    if design.notes:
        lines += ["", "NOTES", *(f"  - {note}" for note in design.notes)]

    lines += [
        "",
        "FILES",
        f"  {design.slug}.dxf            DXF R2010, mm, closed polylines, for CAM",
        f"  {design.slug}.svg            SVG in mm, for laser software",
        f"  {design.slug}_template.pdf   1:1 print template, "
        f"{plan.total} sheet(s) of {plan.sheet.name}",
        f"  {design.slug}_preview.png    what the file contains",
        f"  {design.slug}_mockup.png     what it looks like cut",
        "",
        "PRINTING THE TEMPLATE",
        "  Print at 100% scale with page scaling and 'fit to page' turned off,",
        "  then check the 100 mm bar on the page with a ruler before cutting.",
    ]
    if plan.total > 1:
        lines.append(
            f"  The template covers {plan.columns} x {plan.rows} sheets that "
            f"overlap by {plan.overlap_mm:g} mm;"
        )
        lines.append("  align them on the corner crosses and tape them together.")
    lines += ["", LICENSE_SUMMARY, ""]
    return "\n".join(lines)


def write_design(
    design: Design,
    root: str | Path = "output",
    formats: tuple[str, ...] = ("dxf", "svg", "pdf", "preview", "mockup", "readme"),
    paper: PaperSize = A4,
    config: ValidationConfig | None = None,
) -> DesignOutput:
    """Write every file for one design into its own folder.

    The DXF export validates first and refuses to write invalid geometry, so
    an invalid design raises here before any file is created.

    Args:
        design: The design to write.
        root: Output root; files land in ``root/<niche>/<slug>/``.
        formats: Which outputs to produce.
        paper: Paper size for the template PDF.
        config: Validation limits; defaults to the design's own.

    Returns:
        A :class:`DesignOutput` describing what was written.

    Raises:
        ValidationError: If the design fails validation.
        ValueError: If ``formats`` names something unknown.
    """
    unknown = set(formats) - {"dxf", "svg", "pdf", "preview", "mockup", "readme"}
    if unknown:
        raise ValueError(f"unknown output format(s): {', '.join(sorted(unknown))}")

    placed = design.normalized()
    directory = Path(root) / placed.niche / placed.slug
    directory.mkdir(parents=True, exist_ok=True)
    out = DesignOutput(design=placed, directory=directory)

    if "dxf" in formats:
        path, report = write_dxf(placed, directory / f"{placed.slug}.dxf", config)
        out.files["dxf"] = path
        out.report = report
    if "svg" in formats:
        path, report = write_svg(placed, directory / f"{placed.slug}.svg", config)
        out.files["svg"] = path
        out.report = out.report or report
    if "pdf" in formats:
        path, report = write_pdf(
            placed, directory / f"{placed.slug}_template.pdf", paper=paper, config=config
        )
        out.files["pdf"] = path
        out.report = out.report or report
    if "preview" in formats:
        out.files["preview"] = render_preview(
            placed, directory / f"{placed.slug}_preview.png"
        )
    if "mockup" in formats:
        out.files["mockup"] = render_mockup(
            placed, directory / f"{placed.slug}_mockup.png"
        )
    if "readme" in formats:
        path = directory / "README.txt"
        path.write_text(readme_text(placed, paper), encoding="utf-8")
        out.files["readme"] = path
    return out


# --------------------------------------------------------------------------- #
# batch mode: a whole niche, packaged
# --------------------------------------------------------------------------- #
def license_text(seller: str = DEFAULT_SELLER, year: int | None = None) -> str:
    """Build the licence that ships with a bundle.

    Plain language on purpose.  The buyer is a person with a CNC router in a
    garage, and a licence they cannot read is a licence they will not follow.
    It is a template for the seller to issue, not legal advice.

    Args:
        seller: Who is granting the licence; appears throughout the text.
        year: Copyright year; defaults to the current one.

    Returns:
        The full licence text.
    """
    stamp = date.today().year if year is None else year
    return f"""DESIGN FILE LICENCE
{'=' * 19}

Copyright (c) {stamp} {seller}. All rights reserved.

These design files are licensed, not sold. Buying them gives you the rights
set out below, for as many machines as you personally operate.


WHAT YOU MAY DO
---------------
  * Cut, carve, engrave and laser these designs in any material you like.
  * SELL the physical items you make from them. Commercially, in unlimited
    quantity, at any price, in any shop or market, under your own brand.
    No royalty, no attribution, no reporting, no separate commercial licence.
  * Modify the files for your own production: rescale them, change the
    material thickness, swap the joinery, combine them with your own work.
  * Keep using them if this licence is later withdrawn from sale.


WHAT YOU MAY NOT DO
-------------------
  * Resell, redistribute, share, give away, sub-license, lend or publish the
    digital files, in any format, modified or not. That includes DXF, SVG, PDF, the preview
    images and any file you export from them that is still a design file
    rather than a physical object.
  * Include them in another digital product, file bundle, subscription,
    membership library, course, template pack or design marketplace listing.
  * Upload them to a file-sharing site, a public repository, a Discord or
    Telegram group, a cloud folder others can reach, or an AI training set.
  * Sell them as a "design service" where the customer receives the file.
  * Claim authorship of the designs themselves.

In short: sell what you MAKE, not what you DOWNLOADED.


ONE SEAT
--------
One purchase covers one person or one business. A workshop with several
machines under one owner is one seat. Passing the files to another business,
including a franchisee or a subcontractor who keeps them, is not.


NO WARRANTY
-----------
These files are provided "as is". They are generated parametrically and
checked against the manufacturing rules stated in each design's README, but
{seller} cannot test them on your machine, your material or your cutter.
Cutting is done at your own risk: dry-run the toolpath, check the fit on
offcuts before committing good stock, and wear eye protection.
{seller} is not liable for wasted material, damaged tools, machine downtime,
injury, or lost business arising from the use of these files.


ORIGINALITY
-----------
Every shape in these files is generated from parameters and equations. No
existing design has been copied, traced or reproduced, and no third-party
logo, character or trademarked shape is present. Anything you add to them
yourself is your responsibility.


QUESTIONS
---------
If you are unsure whether something is allowed, assume it needs asking, and
ask {seller} before doing it.
"""


def _stale_folders(directory: Path, keep: Iterable[str]) -> list[str]:
    """Design folders in a bundle directory that this bundle did not write.

    A bundle is never assembled by walking its directory, because a directory
    accumulates: an earlier run with a different seed leaves folders behind,
    and silently shipping them would put unlisted designs in the zip.  They
    are reported instead, so the operator decides.

    Args:
        directory: The bundle directory.
        keep: Slugs this bundle owns.

    Returns:
        Sorted names of leftover design folders.
    """
    wanted = set(keep)
    return sorted(
        child.name
        for child in directory.iterdir()
        if child.is_dir() and child.name not in wanted and not child.name.startswith(".")
    )


def index_text(bundle: "Bundle") -> str:
    """Build the manifest that lists everything in a bundle.

    Args:
        bundle: The assembled bundle.

    Returns:
        The INDEX.txt contents.
    """
    title = f"{bundle.title} - {len(bundle.designs)} designs"
    lines = [
        title,
        "=" * len(title),
        "",
        f"Generated by dxfgen with seed {bundle.seed}. The same seed and count",
        "reproduce this exact set of designs.",
        "",
        "CONTENTS",
    ]
    for number, output in enumerate(bundle.designs, 1):
        design = output.design
        width, height = design.size()
        lines.append(f"  {number:>3}. {design.slug}")
        lines.append(f"       {design.name}")
        lines.append(
            f"       {width:.0f} x {height:.0f} mm, "
            f"{material_phrase(design.thickness, design.material)}, "
            f"{design.machine.mode.value}"
        )
    lines += [
        "",
        "EVERY DESIGN FOLDER HOLDS",
        "  <slug>.dxf            DXF R2010 in mm, closed polylines, for CAM",
        "  <slug>.svg            SVG in mm, for laser software",
        "  <slug>_template.pdf   1:1 print template",
        "  <slug>_preview.png    what the file contains",
        "  <slug>_mockup.png     what it looks like cut",
        "  README.txt            material, depths, cutting order, licence summary",
        "",
        "ALSO IN THIS BUNDLE",
        f"  {CONTACT_SHEET_NAME:<21} every design on one page",
        f"  {LICENSE_NAME:<21} what you may and may not do with these files",
        "",
    ]
    if bundle.short:
        lines += [
            "NOTE",
            f"  {bundle.requested} designs were requested and {len(bundle.designs)} "
            f"were produced.",
            "  The rest were rejected by the manufacturing checks and deliberately",
            "  not shipped. Reasons:",
        ]
        for stage, count in sorted(bundle.skip_counts().items()):
            lines.append(f"    {count} x {_SKIP_WORDS.get(stage, stage)}")
        lines.append("")
    lines += [LICENSE_SUMMARY, ""]
    return "\n".join(lines)


@dataclass
class Bundle:
    """Everything produced by one batch run of one niche.

    Attributes:
        niche: The niche key.
        title: Human readable heading, used on the contact sheet.
        directory: The folder the designs were written into.
        designs: One entry per design actually written, in order.
        run: The generator run behind it, holding every rejection.
        contact_sheet: The grid image, if one was made.
        license_path: The written LICENSE.txt, if one was written.
        index_path: The written INDEX.txt, if one was written.
        archive: The zip, if one was made.
        stale: Design folders already in ``directory`` that this run did not
            write; they are excluded from the zip and reported.
    """

    niche: str
    title: str
    directory: Path
    designs: list[DesignOutput] = field(default_factory=list)
    run: VariantRun | None = None
    contact_sheet: Path | None = None
    license_path: Path | None = None
    index_path: Path | None = None
    archive: Path | None = None
    stale: list[str] = field(default_factory=list)

    @property
    def seed(self) -> int:
        """The master seed the designs were generated from."""
        return self.run.seed if self.run else 0

    @property
    def requested(self) -> int:
        """How many designs were asked for."""
        return self.run.requested if self.run else len(self.designs)

    @property
    def short(self) -> int:
        """How many fewer designs were produced than requested."""
        return max(0, self.requested - len(self.designs))

    @property
    def skips(self) -> list[Skip]:
        """Every rejected variant, with its reason."""
        return list(self.run.skips) if self.run else []

    def skip_counts(self) -> dict[str, int]:
        """Rejections per stage."""
        return self.run.skip_counts() if self.run else {}

    def files(self) -> list[Path]:
        """Every file this bundle owns, design files included."""
        found = [
            path
            for output in self.designs
            for path in sorted(output.directory.rglob("*"))
            if path.is_file()
        ]
        for extra in (self.contact_sheet, self.license_path, self.index_path):
            if extra is not None:
                found.append(extra)
        return found

    def summary(self) -> str:
        """One line describing what was produced."""
        parts = [f"{self.niche}: {len(self.designs)}/{self.requested} designs"]
        if self.short:
            counts = ", ".join(
                f"{count} {stage}" for stage, count in sorted(self.skip_counts().items())
            )
            parts.append(f"{self.short} short ({counts})")
        if self.archive:
            parts.append(self.archive.name)
        return "; ".join(parts)


def _tiles_for(bundle: Bundle, workspace: Path) -> list[Tile]:
    """Render the thumbnails a contact sheet is built from.

    They are re-rendered rather than taken from the shipped previews because
    a thumbnail wants the built-in caption and the INFO labels off; see
    :mod:`dxfgen.bundles.contact_sheet`.  The geometry is identical.

    Args:
        bundle: The bundle whose designs to draw.
        workspace: A scratch directory the thumbnails are written into.

    Returns:
        One tile per design, in bundle order.
    """
    tiles: list[Tile] = []
    for output in bundle.designs:
        design = output.design
        width, height = design.size()
        image = render_preview(
            design,
            workspace / f"{design.slug}.png",
            px_width=900,
            caption=False,
            labels=False,
        )
        tiles.append(
            Tile(
                image=image,
                caption=design.name,
                subcaption=(
                    f"{width:.0f} x {height:.0f} mm  |  "
                    f"{material_phrase(design.thickness, design.material)}"
                ),
            )
        )
    return tiles


def zip_bundle(bundle: Bundle, path: str | Path | None = None) -> Path:
    """Package a bundle into a zip ready to upload.

    Only the files this run wrote go in.  The archive is never built by
    walking the output directory, because that directory outlives the run: a
    previous bundle with a different seed leaves folders behind, and they must
    not end up in someone's download.

    Args:
        bundle: The bundle to package.
        path: Destination zip; defaults to ``<directory>/../<niche>_bundle.zip``,
            which keeps it outside the folder it is archiving.

    Returns:
        The written path.

    Raises:
        ValueError: If the bundle has no designs.
    """
    if not bundle.designs:
        raise ValueError(f"nothing to archive: {bundle.niche} bundle is empty")
    target = (
        bundle.directory.parent / f"{bundle.niche}_bundle.zip"
        if path is None
        else Path(path)
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    root = bundle.directory
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        for file in bundle.files():
            archive.write(file, arcname=str(Path(bundle.niche) / file.relative_to(root)))
    bundle.archive = target
    return target


def build_bundle(
    niche: str | Generator,
    count: int,
    seed: int = 0,
    root: str | Path = "output",
    formats: tuple[str, ...] = ("dxf", "svg", "pdf", "preview", "mockup", "readme"),
    paper: PaperSize = A4,
    config: ValidationConfig | None = None,
    seller: str = DEFAULT_SELLER,
    contact_sheet: bool = True,
    archive: bool = True,
    progress: Callable[[str], None] | None = None,
) -> Bundle:
    """Generate, validate and package a whole niche.

    Invalid variants never reach the disk: the generator rejects them before
    export and the reasons are carried on the returned bundle, so a short
    bundle says why it is short instead of quietly shipping fewer files.

    Args:
        niche: Niche key or an already constructed generator.
        count: How many designs are wanted.
        seed: Master seed; the same seed and count reproduce the same set.
        root: Output root; the bundle lands in ``root/<niche>/``.
        formats: Which per-design outputs to write.
        paper: Paper size for the template PDFs.
        config: Validator limits; defaults to each design's own.
        seller: Name used throughout the licence.
        contact_sheet: Render the grid image.
        archive: Build the zip.
        progress: Called with a short status line as each design is written.

    Returns:
        The assembled :class:`Bundle`.

    Raises:
        KeyError: If the niche name is unknown.
        ValueError: If ``count`` is not positive.
    """
    generator = get_generator(niche) if isinstance(niche, str) else niche
    run = generator.sample_variants(count, seed=seed, config=config)

    directory = Path(root) / generator.niche
    directory.mkdir(parents=True, exist_ok=True)
    bundle = Bundle(
        niche=generator.niche,
        title=generator.title,
        directory=directory,
        run=run,
    )

    total = len(run.designs)
    for number, design in enumerate(run.designs, 1):
        if progress:
            progress(f"{generator.niche} {number}/{total}  {design.slug}")
        bundle.designs.append(
            write_design(design, root=root, formats=formats, paper=paper, config=config)
        )

    if not bundle.designs:
        log.warning("%s: no valid designs, nothing written", generator.niche)
        return bundle

    bundle.stale = _stale_folders(directory, (o.design.slug for o in bundle.designs))
    if bundle.stale:
        log.warning(
            "%s: %d folder(s) in %s are not part of this bundle and were left out "
            "of the archive: %s",
            generator.niche,
            len(bundle.stale),
            directory,
            ", ".join(bundle.stale),
        )

    if contact_sheet and "preview" in formats:
        if progress:
            progress(f"{generator.niche}: contact sheet")
        with tempfile.TemporaryDirectory(prefix="dxfgen-tiles-") as scratch:
            bundle.contact_sheet = render_contact_sheet(
                _tiles_for(bundle, Path(scratch)),
                directory / CONTACT_SHEET_NAME,
                title=f"{generator.title}  -  dxfgen",
                subtitle=f"{len(bundle.designs)} designs  |  seed {seed}",
            )

    bundle.license_path = directory / LICENSE_NAME
    bundle.license_path.write_text(license_text(seller), encoding="utf-8")
    bundle.index_path = directory / INDEX_NAME
    bundle.index_path.write_text(index_text(bundle), encoding="utf-8")

    if archive:
        if progress:
            progress(f"{generator.niche}: archiving")
        zip_bundle(bundle)
    return bundle


def build_all(
    count: int,
    seed: int = 0,
    root: str | Path = "output",
    niches: Sequence[str] | None = None,
    **kwargs: object,
) -> list[Bundle]:
    """Build one bundle per niche.

    Each niche gets the same seed, which is what makes a whole run
    reproducible; the niches do not interfere because each seeds its own
    generator.

    Args:
        count: Designs wanted per niche.
        seed: Master seed, shared by every niche.
        root: Output root.
        niches: Which niches to build; defaults to all registered ones.
        **kwargs: Passed through to :func:`build_bundle`.

    Returns:
        One bundle per niche, in the order requested.
    """
    names = list(niches) if niches is not None else niche_names()
    return [
        build_bundle(name, count=count, seed=seed, root=root, **kwargs)  # type: ignore[arg-type]
        for name in names
    ]
