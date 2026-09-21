"""Tests for the 1:1 template PDF.

The assertion that matters is that the drawing is exactly life size on paper,
so it is tested directly on the figure geometry rather than through a PDF
parser: the axes width in inches, converted to millimetres, must equal the
millimetre span of the axes limits.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from dxfgen.core import geometry as geo
from dxfgen.core.design import Design, Drill, Part, Pocket
from dxfgen.core.export_pdf import (
    A3,
    A4,
    COMPACT_BANDS,
    LETTER,
    MM_PER_INCH,
    OVERLAP_MM,
    PAPER_SIZES,
    STANDARD_BANDS,
    PaperSize,
    build_page,
    plan_pages,
    write_pdf,
)
from dxfgen.core.validate import ValidationError

POINTS_PER_MM = 72.0 / 25.4


def design_of(width: float, height: float, **kwargs) -> Design:
    part = Part(
        "tray",
        geo.rounded_rect_ring(width, height, min(20.0, min(width, height) / 4)),
        pockets=[
            Pocket(
                geo.rounded_rect_ring(width - 80, height - 80, 15, 40, 40), 8.0
            )
        ],
        drills=[Drill((20, 20), 6.0)],
    )
    kwargs.setdefault("thickness", 19.0)
    kwargs.setdefault("sheet", (1220.0, 2440.0))
    return Design("demo", "Demo Tray", "trays", "a demo", [part], **kwargs)


# --------------------------------------------------------------------------- #
# the 1:1 guarantee
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("size", [(200, 150), (360, 260), (520, 390)])
@pytest.mark.parametrize("paper", [A4, A3, LETTER])
def test_the_drawing_is_exactly_life_size(size, paper: PaperSize) -> None:
    design = design_of(*size)
    plan = plan_pages(design, paper)
    figure = build_page(design, plan, 0, 0)
    ax = figure.axes[0]
    position = ax.get_position()
    fig_w, fig_h = figure.get_size_inches()
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    assert position.width * fig_w * MM_PER_INCH == pytest.approx(x1 - x0, abs=1e-9)
    assert position.height * fig_h * MM_PER_INCH == pytest.approx(y1 - y0, abs=1e-9)


def test_the_page_is_the_paper_size(tmp_path: Path) -> None:
    design = design_of(200, 150)
    path, _ = write_pdf(design, tmp_path / "t.pdf", paper=A4)
    boxes = re.findall(rb"/MediaBox \[([^\]]+)\]", path.read_bytes())
    assert boxes
    values = [float(v) for v in boxes[0].split()]
    plan = plan_pages(design, A4)
    assert values[2] / POINTS_PER_MM == pytest.approx(plan.sheet.width, abs=0.01)
    assert values[3] / POINTS_PER_MM == pytest.approx(plan.sheet.height, abs=0.01)


# --------------------------------------------------------------------------- #
# tiling
# --------------------------------------------------------------------------- #
def test_a_small_design_fits_one_sheet() -> None:
    assert plan_pages(design_of(240, 150), A4).total == 1


def test_a_large_design_tiles() -> None:
    plan = plan_pages(design_of(520, 390), A4)
    assert plan.total > 1
    assert plan.columns >= 2 and plan.rows >= 2


def test_bigger_paper_needs_fewer_sheets() -> None:
    design = design_of(520, 390)
    assert plan_pages(design, A3).total < plan_pages(design, A4).total


def test_the_layout_prefers_the_roomy_title_block_when_it_is_free() -> None:
    plan = plan_pages(design_of(240, 150), A4)
    assert plan.bands == STANDARD_BANDS


def test_the_layout_goes_compact_only_to_save_a_sheet() -> None:
    """A 260x180 design fits one sheet only if the title block gives way."""
    design = design_of(260, 180)
    plan = plan_pages(design, A4)
    assert plan.total == 1
    assert plan.bands == COMPACT_BANDS


def test_tiles_overlap_by_the_requested_amount() -> None:
    plan = plan_pages(design_of(520, 390), A4, overlap_mm=15.0)
    assert plan.window_w - plan.step_x == pytest.approx(15.0)
    assert plan.window_h - plan.step_y == pytest.approx(15.0)


def test_tiles_cover_the_whole_design() -> None:
    design = design_of(520, 390)
    plan = plan_pages(design, A4)
    width, height = design.size()
    assert plan.step_x * (plan.columns - 1) + plan.window_w >= width - 1e-9
    assert plan.step_y * (plan.rows - 1) + plan.window_h >= height - 1e-9


def test_windows_are_contiguous_across_columns() -> None:
    plan = plan_pages(design_of(520, 390), A4)
    first = plan.window(0, 0)
    second = plan.window(0, 1)
    assert second[0] < first[2]  # they overlap rather than leaving a gap


def test_an_absurd_margin_is_refused() -> None:
    with pytest.raises(ValueError, match="no printable area"):
        plan_pages(design_of(200, 150), A4, margin_mm=200.0)


def test_an_overlap_wider_than_the_sheet_is_refused() -> None:
    with pytest.raises(ValueError, match="no usable width"):
        plan_pages(design_of(900, 700), A4, overlap_mm=400.0)


def test_paper_sizes_are_selectable_by_name() -> None:
    assert PAPER_SIZES["a4"] == A4
    assert PAPER_SIZES["a3"].width == 297.0
    assert A4.landscape().width == A4.height


# --------------------------------------------------------------------------- #
# page content
# --------------------------------------------------------------------------- #
def test_every_page_carries_a_scale_bar() -> None:
    """It is the only defence against the printer scaling the page."""
    design = design_of(520, 390)
    plan = plan_pages(design, A4)
    for row in range(plan.rows):
        for column in range(plan.columns):
            figure = build_page(design, plan, row, column)
            texts = [t.get_text() for t in figure.texts]
            bar_axes = figure.axes[1]
            labels = [t.get_text() for t in bar_axes.texts]
            assert "100 mm" in labels
            assert any("100%" in t for t in labels)


def test_the_scale_bar_is_exactly_100_mm_on_paper() -> None:
    design = design_of(200, 150)
    plan = plan_pages(design, A4)
    figure = build_page(design, plan, 0, 0)
    bar = figure.axes[1]
    position = bar.get_position()
    fig_w, _ = figure.get_size_inches()
    span = bar.get_xlim()[1] - bar.get_xlim()[0]
    mm_per_unit = position.width * fig_w * MM_PER_INCH / span
    assert mm_per_unit == pytest.approx(1.0, abs=1e-9)


def test_pages_are_numbered_and_located() -> None:
    design = design_of(520, 390)
    plan = plan_pages(design, A4)
    figure = build_page(design, plan, 1, 1)
    joined = " ".join(t.get_text() for t in figure.texts)
    assert f"sheet {1 * plan.columns + 2} of {plan.total}" in joined
    assert "row 2, column 2" in joined
    assert "align on the corner crosses" in joined


def test_a_single_sheet_says_so_without_tiling_noise() -> None:
    design = design_of(240, 150)
    plan = plan_pages(design, A4)
    joined = " ".join(t.get_text() for t in build_page(design, plan, 0, 0).texts)
    assert "sheet 1 of 1" in joined
    assert "row" not in joined


def test_write_pdf_writes_one_page_per_tile(tmp_path: Path) -> None:
    design = design_of(520, 390)
    plan = plan_pages(design, A4)
    path, report = write_pdf(design, tmp_path / "t.pdf", paper=A4)
    assert report.ok
    pages = len(re.findall(rb"/Type\s*/Page[^s]", path.read_bytes()))
    assert pages == plan.total


def test_write_pdf_refuses_an_invalid_design(tmp_path: Path) -> None:
    bad = Part("bad", geo.rect_ring(300, 200),
               pockets=[Pocket(geo.rect_ring(200, 100, 50, 50), 8.0)])
    design = Design("d", "D", "trays", "x", [bad], thickness=19.0, sheet=(600, 400))
    target = tmp_path / "never.pdf"
    with pytest.raises(ValidationError):
        write_pdf(design, target)
    assert not target.exists()
