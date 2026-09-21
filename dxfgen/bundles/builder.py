"""Assembling the files that make up one sellable design.

A buyer gets a folder, not a file.  The DXF is for CAM, the SVG is for laser
software, the PDF is for anyone without either, the preview shows what the
file contains and the mockup shows what it becomes.  The README is what stops
the support emails.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..core.design import Design
from ..core.export_dxf import write_dxf
from ..core.export_pdf import A4, PaperSize, plan_pages, write_pdf
from ..core.export_svg import write_svg
from ..core.layers import layer_def
from ..core.limits import ValidationConfig
from ..core.preview import render_mockup, render_preview
from ..core.validate import Report

__all__ = ["LICENSE_SUMMARY", "DesignOutput", "readme_text", "write_design"]

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
