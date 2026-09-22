"""1:1 printable template PDF.

The point of this file is that a buyer without CAD can still use the design:
print it at 100%, check the scale bar with a ruler, glue the sheets to the
board and cut to the lines.

That imposes three requirements the rest of the exporters do not have.  The
drawing must be exactly life size on paper, which means the axes are placed in
inches computed from millimetres rather than left to matplotlib's layout.  A
design larger than the paper must tile across sheets, with overlap and
registration marks so the sheets can be aligned and glued.  And every page
carries a scale bar, because the single most common way this goes wrong is a
printer quietly applying "fit to page".

Layers are distinguished by dash pattern as well as colour, so the template
still reads when printed in black and white.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

from datetime import datetime, timezone  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.patches import PathPatch  # noqa: E402

from .design import Design  # noqa: E402
from .limits import ValidationConfig  # noqa: E402
from .preview import PREVIEW_COLORS, compound_path  # noqa: E402
from .validate import Report, validate_design  # noqa: E402

__all__ = [
    "PaperSize",
    "PagePlan",
    "A4",
    "A3",
    "LETTER",
    "PAPER_SIZES",
    "plan_pages",
    "build_page",
    "write_pdf",
]

MM_PER_INCH = 25.4


@dataclass(frozen=True)
class PaperSize:
    """A paper size in millimetres."""

    name: str
    width: float
    height: float

    def landscape(self) -> "PaperSize":
        """Return the same sheet turned on its side."""
        return PaperSize(f"{self.name} landscape", self.height, self.width)


A4 = PaperSize("A4", 210.0, 297.0)
A3 = PaperSize("A3", 297.0, 420.0)
LETTER = PaperSize("Letter", 215.9, 279.4)

#: Paper sizes selectable by name on the command line.
PAPER_SIZES: dict[str, PaperSize] = {"a4": A4, "a3": A3, "letter": LETTER}

#: How far adjacent sheets overlap, in mm, so they can be aligned and glued.
OVERLAP_MM = 12.0
#: Distance from the paper edge to the scale bar block, in mm.  Held clear of
#: the unprintable border that every desktop printer has, independently of the
#: drawing margin, so a tight layout does not push the bar off the page.
SCALE_BAR_EDGE_MM = 4.0


@dataclass(frozen=True)
class Bands:
    """How much of a sheet is given to the title block and the scale bar.

    Two presets are costed against each other when planning.  The roomy one
    reads better; the compact one exists because 34 mm of furniture is enough
    to push a design that would otherwise fit onto a second sheet, and making
    the buyer print, align and glue two pages for a small tray is a worse
    outcome than a tighter title block.
    """

    margin: float
    header: float
    footer: float


#: Preferred layout, used whenever it costs no extra sheets.
STANDARD_BANDS = Bands(margin=10.0, header=16.0, footer=18.0)
#: Tighter layout, used only when it saves sheets.
COMPACT_BANDS = Bands(margin=7.0, header=6.0, footer=10.0)

#: Line style per role.  Dashes carry the meaning when colour does not survive
#: a black and white printer.
_STYLES = {
    "outline": {"color": PREVIEW_COLORS["outline"], "lw": 0.7, "ls": "solid"},
    "inside": {"color": PREVIEW_COLORS["inside"], "lw": 0.6, "ls": "solid"},
    "pocket": {"color": PREVIEW_COLORS["pocket_edge"], "lw": 0.5, "ls": (0, (6, 3))},
    "engrave": {"color": PREVIEW_COLORS["engrave"], "lw": 0.45, "ls": (0, (4, 2, 1, 2))},
    "drill": {"color": PREVIEW_COLORS["drill"], "lw": 0.5, "ls": "solid"},
}


def _tile_counts(
    design_mm: float, window_mm: float, overlap_mm: float
) -> int:
    """How many sheets are needed along one axis.

    Args:
        design_mm: Design extent along this axis.
        window_mm: Usable drawing width on one sheet.
        overlap_mm: Overlap between adjacent sheets.

    Returns:
        A count of at least 1.

    Raises:
        ValueError: If the overlap leaves no forward progress.
    """
    if design_mm <= window_mm:
        return 1
    step = window_mm - overlap_mm
    if step <= 0:
        raise ValueError(
            f"an overlap of {overlap_mm} mm leaves no usable width on a "
            f"{window_mm:.0f} mm sheet"
        )
    return int(math.ceil((design_mm - overlap_mm) / step))


def _draw_geometry(ax, design: Design) -> None:
    """Draw every cut contour onto a 1:1 axes."""
    for part in design.placed_parts():
        for pocket in part.pockets:
            ax.add_patch(
                PathPatch(
                    compound_path(pocket.ring, pocket.islands),
                    facecolor="none",
                    edgecolor=_STYLES["pocket"]["color"],
                    linewidth=_STYLES["pocket"]["lw"],
                    linestyle=_STYLES["pocket"]["ls"],
                )
            )
        for contour in part.engrave:
            points = np.asarray(
                list(contour.points) + ([contour.points[0]] if contour.closed else []),
                dtype=float,
            )
            ax.plot(
                points[:, 0],
                points[:, 1],
                color=_STYLES["engrave"]["color"],
                linewidth=_STYLES["engrave"]["lw"],
                linestyle=_STYLES["engrave"]["ls"],
            )
        for hole in part.holes:
            ax.add_patch(
                PathPatch(
                    compound_path(hole),
                    facecolor="none",
                    edgecolor=_STYLES["inside"]["color"],
                    linewidth=_STYLES["inside"]["lw"],
                )
            )
        for drill in part.drills:
            cx, cy = drill.center
            circle = matplotlib.patches.Circle(
                (cx, cy),
                drill.radius,
                facecolor="none",
                edgecolor=_STYLES["drill"]["color"],
                linewidth=_STYLES["drill"]["lw"],
            )
            ax.add_patch(circle)
            reach = drill.radius * 1.8
            ax.plot(
                [cx - reach, cx + reach],
                [cy, cy],
                color=_STYLES["drill"]["color"],
                linewidth=0.3,
            )
            ax.plot(
                [cx, cx],
                [cy - reach, cy + reach],
                color=_STYLES["drill"]["color"],
                linewidth=0.3,
            )
        ax.add_patch(
            PathPatch(
                compound_path(part.outline),
                facecolor="none",
                edgecolor=_STYLES["outline"]["color"],
                linewidth=_STYLES["outline"]["lw"],
            )
        )


def _draw_scale_bar(fig: Figure, paper: PaperSize, bands: Bands) -> None:
    """Draw a 100 mm scale bar in the footer, exactly 100 mm wide on paper.

    The bar is the only defence against a printer applying "fit to page", so
    it is drawn on every sheet with the instruction next to it.
    """
    width = paper.width - 2 * bands.margin
    band = bands.footer * 0.78
    ax = fig.add_axes(
        (
            bands.margin / paper.width,
            SCALE_BAR_EDGE_MM / paper.height,
            width / paper.width,
            band / paper.height,
        )
    )
    ax.set_xlim(0, width)
    ax.set_ylim(0, band)
    ax.axis("off")
    y = band * 0.42
    height = 2.2
    for index in range(10):
        ax.add_patch(
            matplotlib.patches.Rectangle(
                (index * 10.0, y),
                10.0,
                height,
                facecolor="black" if index % 2 == 0 else "white",
                edgecolor="black",
                linewidth=0.4,
            )
        )
    for tick, text in ((0.0, "0"), (50.0, "50"), (100.0, "100 mm")):
        ax.text(tick, y - 0.8, text, ha="center", va="top", fontsize=5.5)
    note = (
        "This bar must measure exactly 100 mm. If it does not, reprint at 100% "
        "with page scaling off."
    )
    if width - 106.0 > 120.0:
        ax.text(
            108.0, y + height / 2, note, ha="left", va="center", fontsize=6,
            color="#8a2b22",
        )
    else:
        ax.text(
            width, y + height / 2, note.replace(". If", ".\nIf"), ha="right",
            va="center", fontsize=5.5, color="#8a2b22",
        )


def _draw_registration(ax, window: tuple[float, float, float, float]) -> None:
    """Draw corner crosses so adjacent sheets can be lined up."""
    x0, y0, x1, y1 = window
    arm = 6.0
    for cx, cy in ((x0, y0), (x1, y0), (x0, y1), (x1, y1)):
        # clip_on=False so the whole cross shows; half a cross is no use for
        # lining two sheets up.
        ax.plot(
            [cx - arm, cx + arm], [cy, cy],
            color="#999999", linewidth=0.4, clip_on=False,
        )
        ax.plot(
            [cx, cx], [cy - arm, cy + arm],
            color="#999999", linewidth=0.4, clip_on=False,
        )


@dataclass(frozen=True)
class PagePlan:
    """How a design is laid out across printed sheets.

    Attributes:
        sheet: The paper size chosen, possibly rotated to landscape.
        columns: Sheets across.
        rows: Sheets down.
        window_w: Usable drawing width on one sheet, in mm.
        window_h: Usable drawing height on one sheet, in mm.
        step_x: Distance between the left edges of adjacent sheets, in mm.
        step_y: Distance between the bottom edges of adjacent sheets, in mm.
        bands: The margin and title/scale-bar allowance in use.
        overlap_mm: How much adjacent sheets share.
    """

    sheet: PaperSize
    columns: int
    rows: int
    window_w: float
    window_h: float
    step_x: float
    step_y: float
    bands: Bands
    overlap_mm: float

    @property
    def margin_mm(self) -> float:
        """Unprinted margin on every edge, in mm."""
        return self.bands.margin

    @property
    def total(self) -> int:
        """Total sheet count."""
        return self.rows * self.columns

    def window(self, row: int, column: int) -> tuple[float, float, float, float]:
        """The design-space rectangle one sheet shows, in mm."""
        x0 = column * self.step_x
        y0 = self.origin_y - (row + 1) * self.step_y if self.rows > 1 else 0.0
        return (x0, y0, x0 + self.window_w, y0 + self.window_h)

    origin_y: float = 0.0


#: Timestamp written into every PDF, so the same design gives the same
#: file.  See where it is used for why a fixed date is the right one.
EPOCH = datetime(2000, 1, 1, tzinfo=timezone.utc)


def plan_pages(
    design: Design,
    paper: PaperSize = A4,
    margin_mm: float | None = None,
    overlap_mm: float = OVERLAP_MM,
) -> PagePlan:
    """Work out the sheet layout for a design.

    Portrait and landscape are both costed and the one needing fewer sheets
    wins, because a tray is usually wider than it is deep and turning the paper
    can halve the print.

    Args:
        design: The design to print.
        paper: Sheet size to consider, in both orientations.
        margin_mm: Margin override; ``None`` lets the band presets decide.
        overlap_mm: How much adjacent sheets share.

    Returns:
        The chosen :class:`PagePlan`.

    Raises:
        ValueError: If the margins leave no printable area.
    """
    width, height = design.normalized().size()
    presets = [STANDARD_BANDS, COMPACT_BANDS]
    if margin_mm is not None:
        presets = [
            Bands(margin_mm, b.header, b.footer) for b in presets
        ]
    best: tuple[tuple[int, int], PaperSize, float, float, Bands] | None = None
    for rank, bands in enumerate(presets):
        for candidate in (paper, paper.landscape()):
            window_w = candidate.width - 2 * bands.margin
            window_h = candidate.height - 2 * bands.margin - bands.header - bands.footer
            if window_w <= 0 or window_h <= 0:
                continue
            sheets = _tile_counts(width, window_w, overlap_mm) * _tile_counts(
                height, window_h, overlap_mm
            )
            # Fewest sheets wins; the roomier preset breaks ties.
            key = (sheets, rank)
            if best is None or key < best[0]:
                best = (key, candidate, window_w, window_h, bands)
    if best is None:
        raise ValueError(f"the margins leave no printable area on {paper.name}")
    _, sheet, window_w, window_h, bands = best
    columns = _tile_counts(width, window_w, overlap_mm)
    rows = _tile_counts(height, window_h, overlap_mm)
    return PagePlan(
        sheet=sheet,
        columns=columns,
        rows=rows,
        window_w=window_w,
        window_h=window_h,
        step_x=window_w - overlap_mm if columns > 1 else window_w,
        step_y=window_h - overlap_mm if rows > 1 else window_h,
        bands=bands,
        overlap_mm=overlap_mm,
        origin_y=height,
    )


def build_page(design: Design, plan: PagePlan, row: int, column: int) -> Figure:
    """Build one sheet of the template.

    The axes rectangle is computed from millimetres rather than left to
    matplotlib's layout, which is what makes the print exactly life size:
    ``axes width in inches * 25.4`` equals the millimetre span of the x limits.

    Args:
        design: The design to draw, already normalised.
        plan: The sheet layout.
        row: Sheet row, 0 at the top.
        column: Sheet column, 0 at the left.

    Returns:
        A matplotlib figure sized exactly to the paper.
    """
    sheet = plan.sheet
    width, height = design.size()
    fig = Figure(figsize=(sheet.width / MM_PER_INCH, sheet.height / MM_PER_INCH))
    fig.patch.set_facecolor("white")
    window = plan.window(row, column)

    ax = fig.add_axes(
        (
            plan.bands.margin / sheet.width,
            (plan.bands.margin + plan.bands.footer) / sheet.height,
            plan.window_w / sheet.width,
            plan.window_h / sheet.height,
        )
    )
    ax.set_xlim(window[0], window[2])
    ax.set_ylim(window[1], window[3])
    ax.set_aspect("equal")
    ax.axis("off")
    _draw_geometry(ax, design)
    _draw_registration(ax, window)

    sheet_no = row * plan.columns + column + 1
    subtitle = (
        f"{width:.0f} x {height:.0f} mm in {design.thickness:g} mm "
        f"{design.material}   |   1:1 template, sheet {sheet_no} of {plan.total}"
    )
    if plan.total > 1:
        subtitle += f"  (row {row + 1}, column {column + 1})"
    compact = plan.bands.header < 10.0
    top = 1 - plan.bands.margin / sheet.height
    if compact:
        fig.text(
            plan.bands.margin / sheet.width,
            top - 1.0 / sheet.height,
            f"{design.name}  |  {subtitle}",
            fontsize=6.5,
            va="top",
            color="#333333",
        )
    else:
        fig.text(
            plan.bands.margin / sheet.width,
            top - 4.0 / sheet.height,
            design.name,
            fontsize=10,
            fontweight="bold",
            va="top",
        )
        fig.text(
            plan.bands.margin / sheet.width,
            top - 9.5 / sheet.height,
            subtitle,
            fontsize=7,
            va="top",
            color="#444444",
        )
        if plan.total > 1:
            fig.text(
                1 - plan.bands.margin / sheet.width,
                top - 9.5 / sheet.height,
                f"overlap {plan.overlap_mm:g} mm - align on the corner crosses",
                fontsize=7,
                va="top",
                ha="right",
                color="#444444",
            )
    _draw_scale_bar(fig, sheet, plan.bands)
    return fig


def write_pdf(
    design: Design,
    path: str | Path,
    paper: PaperSize = A4,
    margin_mm: float | None = None,
    overlap_mm: float = OVERLAP_MM,
    config: ValidationConfig | None = None,
    validate: bool = True,
) -> tuple[Path, Report]:
    """Write a 1:1 printable template, tiled across sheets if it does not fit.

    Args:
        design: The design to print; it is normalised to the origin first.
        path: Destination PDF path; parent directories are created.
        paper: Sheet size.  Landscape is chosen automatically when it needs
            fewer sheets.
        margin_mm: Margin override in mm; ``None`` lets the layout choose
            between a roomy title block and a compact one, preferring the
            roomy one unless the compact one saves a sheet.
        overlap_mm: How much adjacent sheets share, for alignment and glue.
        config: Validation limits; defaults to the design's own.
        validate: Set ``False`` only to inspect a known-bad design.

    Returns:
        ``(written_path, report)``.

    Raises:
        ValidationError: If the design fails validation and ``validate`` is
            ``True``.  No file is written in that case.
        ValueError: If the margins leave no printable area.
    """
    placed = design.normalized()
    report = validate_design(placed, config)
    if validate:
        report.raise_for_status()

    plan = plan_pages(placed, paper, margin_mm, overlap_mm)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(out) as pdf:
        for row in range(plan.rows):
            for column in range(plan.columns):
                pdf.savefig(build_page(placed, plan, row, column))
        info = pdf.infodict()
        info["Title"] = f"{placed.name} - 1:1 template"
        info["Subject"] = placed.description
        info["Creator"] = "dxfgen"
        # matplotlib stamps the current time here, which is the only thing
        # that would make two runs of the same design differ.  A generated
        # template has no meaningful creation moment - it is a function of its
        # parameters - so it gets a fixed one and the file stays comparable.
        info["CreationDate"] = EPOCH
    return out, report
