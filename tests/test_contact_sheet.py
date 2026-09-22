"""Tests for the contact sheet grid."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from dxfgen.bundles.contact_sheet import (
    MAX_COLUMNS,
    _MAX_CELL_ASPECT,
    _MIN_CELL_ASPECT,
    Tile,
    _cell_aspect,
    _wrap,
    grid_shape,
    render_contact_sheet,
)


def _png(path: Path, width: int, height: int, colour=(200, 60, 60)) -> Path:
    """Write a solid PNG of a given size, standing in for a preview."""
    array = np.zeros((height, width, 3), dtype=np.uint8)
    array[:, :] = colour
    Image.fromarray(array).save(path)
    return path


@pytest.fixture
def squares(tmp_path: Path) -> list[Tile]:
    return [
        Tile(_png(tmp_path / f"s{i}.png", 300, 300), f"Design {i}") for i in range(6)
    ]


@pytest.fixture
def wides(tmp_path: Path) -> list[Tile]:
    return [
        Tile(_png(tmp_path / f"w{i}.png", 600, 300), f"Design {i}") for i in range(6)
    ]


# --------------------------------------------------------------------------- #
# grid shape
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "count, expected",
    [(1, (1, 1)), (2, (2, 1)), (4, (2, 2)), (9, (3, 3)), (20, (5, 4)), (50, (8, 7))],
)
def test_grid_shape_is_roughly_square_and_slightly_wide(count, expected) -> None:
    assert grid_shape(count) == expected


def test_grid_always_has_room_for_every_tile() -> None:
    for count in range(1, 120):
        cols, rows = grid_shape(count)
        assert cols * rows >= count
        assert (rows - 1) * cols < count, "an entirely empty last row"


def test_grid_stops_widening_and_grows_downwards() -> None:
    cols, rows = grid_shape(400)
    assert cols == MAX_COLUMNS
    assert rows == math.ceil(400 / MAX_COLUMNS)


def test_columns_can_be_forced() -> None:
    assert grid_shape(10, columns=2) == (2, 5)


@pytest.mark.parametrize("bad", [dict(count=0), dict(count=5, columns=0)])
def test_a_nonsense_grid_is_refused(bad) -> None:
    with pytest.raises(ValueError):
        grid_shape(**bad)


# --------------------------------------------------------------------------- #
# caption fitting
# --------------------------------------------------------------------------- #
def test_short_text_is_left_alone() -> None:
    assert _wrap("Oak tray", 40) == "Oak tray"


def test_text_wraps_rather_than_truncating_when_it_fits_in_two_lines() -> None:
    name = "Rounded Serving Tray, 560 x 280 mm, 2 compartments"
    fitted = _wrap(name, 28)
    assert fitted.count("\n") == 1
    assert "..." not in fitted
    assert fitted.replace("\n", " ") == name, "wrapping must not drop words"


def test_text_too_long_for_its_lines_is_ellipsised() -> None:
    fitted = _wrap("word " * 40, 10, max_lines=2)
    assert fitted.count("\n") == 1
    assert fitted.endswith("...")


def test_every_wrapped_line_fits_the_width() -> None:
    fitted = _wrap("Rounded Serving Tray, 560 x 280 mm, 2 compartments", 22)
    assert all(len(line) <= 22 for line in fitted.split("\n"))


def test_empty_text_stays_empty() -> None:
    assert _wrap("", 20) == ""


# --------------------------------------------------------------------------- #
# cells follow their contents
# --------------------------------------------------------------------------- #
def test_cell_aspect_follows_the_tiles(squares, wides) -> None:
    assert _cell_aspect(squares, 300, {}) == pytest.approx(1.0)
    assert _cell_aspect(wides, 300, {}) == pytest.approx(2.0)


def test_cell_aspect_is_clamped_at_both_ends(tmp_path: Path) -> None:
    """One very long design must not turn the page into a letterbox."""
    strips = [Tile(_png(tmp_path / f"r{i}.png", 900, 100), "") for i in range(3)]
    poles = [Tile(_png(tmp_path / f"c{i}.png", 100, 900), "") for i in range(3)]
    assert _cell_aspect(strips, 300, {}) == pytest.approx(_MAX_CELL_ASPECT)
    assert _cell_aspect(poles, 300, {}) == pytest.approx(_MIN_CELL_ASPECT)


def test_cell_aspect_ignores_one_odd_tile(tmp_path: Path, wides) -> None:
    mixed = [*wides, Tile(_png(tmp_path / "tall.png", 100, 900), "Tall")]
    assert _cell_aspect(mixed, 300, {}) == pytest.approx(
        2.0
    ), "one outlier must not reshape the grid"


def test_wide_tiles_make_a_shorter_page(tmp_path: Path, squares, wides) -> None:
    square_sheet = render_contact_sheet(squares, tmp_path / "sq.png", columns=3)
    wide_sheet = render_contact_sheet(wides, tmp_path / "wd.png", columns=3)
    with Image.open(square_sheet) as a, Image.open(wide_sheet) as b:
        assert a.size[0] == b.size[0], "same columns, same width"
        assert b.size[1] < a.size[1] * 0.75, "wide tiles must not sit in square cells"


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #
def test_a_sheet_is_written(tmp_path: Path, squares) -> None:
    out = render_contact_sheet(squares, tmp_path / "deep" / "sheet.png", title="Trays")
    assert out.exists() and out.stat().st_size > 0


def test_more_rows_make_a_taller_page(tmp_path: Path, squares) -> None:
    short = render_contact_sheet(squares[:3], tmp_path / "a.png", columns=3)
    tall = render_contact_sheet(squares, tmp_path / "b.png", columns=3)
    with Image.open(short) as a, Image.open(tall) as b:
        assert b.size[1] > a.size[1]
        assert b.size[0] == a.size[0]


def test_a_title_adds_a_header_band(tmp_path: Path, squares) -> None:
    plain = render_contact_sheet(squares, tmp_path / "p.png", columns=3)
    titled = render_contact_sheet(squares, tmp_path / "t.png", columns=3, title="Trays")
    with Image.open(plain) as a, Image.open(titled) as b:
        assert b.size[1] > a.size[1]


def test_captions_reserve_space_only_when_there_are_captions(tmp_path: Path) -> None:
    images = [_png(tmp_path / f"n{i}.png", 300, 300) for i in range(4)]
    bare = render_contact_sheet([Tile(p) for p in images], tmp_path / "b.png", columns=2)
    named = render_contact_sheet(
        [Tile(p, "Name") for p in images], tmp_path / "n.png", columns=2
    )
    with Image.open(bare) as a, Image.open(named) as b:
        assert b.size[1] > a.size[1]


def test_the_sheet_scales_with_cell_size(tmp_path: Path, squares) -> None:
    small = render_contact_sheet(squares, tmp_path / "s.png", columns=3, cell_px=200)
    large = render_contact_sheet(squares, tmp_path / "l.png", columns=3, cell_px=400)
    with Image.open(small) as a, Image.open(large) as b:
        assert b.size[0] > a.size[0] * 1.8


def test_text_is_sized_in_points_not_pixels(tmp_path: Path, squares) -> None:
    """A caption must be a sane fraction of its cell at any dpi.

    Matplotlib font sizes are points; feeding them a pixel or millimetre
    measurement gives text many times too large.  Rendering the same grid at
    two dpi values and getting the same proportions is what rules that out.
    """
    low = render_contact_sheet(squares, tmp_path / "lo.png", columns=3, dpi=100)
    high = render_contact_sheet(squares, tmp_path / "hi.png", columns=3, dpi=200)
    with Image.open(low) as a, Image.open(high) as b:
        assert b.size[0] == pytest.approx(a.size[0], rel=0.02)
        assert b.size[1] == pytest.approx(a.size[1], rel=0.02)


def test_an_empty_sheet_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least one tile"):
        render_contact_sheet([], tmp_path / "x.png")


@pytest.mark.parametrize("kwargs", [dict(cell_px=4), dict(dpi=0)])
def test_nonsense_sizes_are_refused(tmp_path: Path, squares, kwargs) -> None:
    with pytest.raises(ValueError):
        render_contact_sheet(squares, tmp_path / "x.png", **kwargs)


def test_a_missing_tile_image_is_reported(tmp_path: Path, squares) -> None:
    with pytest.raises(FileNotFoundError, match="not found"):
        render_contact_sheet(
            [*squares, Tile(tmp_path / "gone.png", "Missing")], tmp_path / "x.png"
        )
