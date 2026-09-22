"""The one image that sells a bundle: every design in it, on one page.

A buyer deciding whether fifty tray files are worth the money does not open
fifty PNGs.  They look at one grid, and either the grid shows fifty designs
that are recognisably a family but plainly different from each other, or it
shows padding.

Tiles are the shipped previews with their built-in dimension caption turned
off, because that caption is burnt into the image and so shrinks by a
different amount in every cell: a wide tray thumbnails smaller than a tall
stand, leaving its text unreadable beside its neighbour's.  Text the sheet
draws itself is the same size in every cell.  The geometry is unchanged, so
the grid still shows exactly the designs in the folder.

Sizing here is all derived from one pixel geometry.  Matplotlib font sizes and
line widths are *points*, not pixels and certainly not millimetres, so every
text size in this module is converted through ``72 / dpi`` from a pixel
measurement.  Getting that wrong is how you end up with 60 pt captions.
"""

from __future__ import annotations

import math
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from PIL import Image  # noqa: E402

__all__ = ["Tile", "grid_shape", "render_contact_sheet", "MAX_COLUMNS"]

#: Beyond this the tiles get too small to tell designs apart on a screen, so a
#: big bundle grows downwards instead of sideways.
MAX_COLUMNS: int = 8

#: Slightly wider than square, because screens are.
_ASPECT_BIAS: float = 1.3

#: How far the cell shape may stray from square to suit its contents.
_MIN_CELL_ASPECT: float = 0.55
_MAX_CELL_ASPECT: float = 2.2

_TEXT = "#2f2f2f"
_MUTED = "#7a7a7a"
_RULE = "#d8d3c9"
_ELLIPSIS = "..."


@dataclass(frozen=True)
class Tile:
    """One cell of a contact sheet.

    Attributes:
        image: Path to the PNG to show.
        caption: Main line under the image, usually the design name.
        subcaption: Second, smaller line, usually the size and material.
    """

    image: Path
    caption: str = ""
    subcaption: str = ""


def grid_shape(count: int, columns: int | None = None) -> tuple[int, int]:
    """Choose a ``(columns, rows)`` grid for ``count`` tiles.

    Args:
        count: How many tiles there are.
        columns: Forced column count, or ``None`` to choose one.

    Returns:
        ``(columns, rows)``; the last row may be partly empty.

    Raises:
        ValueError: If ``count`` is not positive or ``columns`` is not positive.
    """
    if count < 1:
        raise ValueError(f"count must be >= 1, got {count}")
    if columns is not None:
        if columns < 1:
            raise ValueError(f"columns must be >= 1, got {columns}")
        chosen = columns
    else:
        chosen = max(1, min(MAX_COLUMNS, round(math.sqrt(count * _ASPECT_BIAS))))
    return chosen, math.ceil(count / chosen)


def _load(path: Path, cell_px: int, cache: dict[Path, np.ndarray]) -> np.ndarray:
    """Load a preview and shrink it to fit a cell, preserving its aspect.

    Downscaling happens here rather than in matplotlib so a fifty-tile sheet
    holds fifty small arrays instead of fifty full-resolution ones.  Images are
    read twice per sheet - once to measure, once to draw - hence the cache.

    Args:
        path: The PNG to read.
        cell_px: The square bound the image must fit inside.
        cache: Per-sheet store of already decoded arrays.

    Returns:
        An RGB array.

    Raises:
        FileNotFoundError: If the image is missing.
    """
    hit = cache.get(path)
    if hit is not None:
        return hit
    if not path.is_file():
        raise FileNotFoundError(f"contact sheet tile image not found: {path}")
    with Image.open(path) as handle:
        image = handle.convert("RGB")
        image.thumbnail((cell_px, cell_px), Image.Resampling.LANCZOS)
        array = np.asarray(image)
    cache[path] = array
    return array


def _wrap(text: str, max_chars: int, max_lines: int = 2) -> str:
    """Fit text into a cell by wrapping, then truncating what will not fit.

    Wrapping beats truncating because design names put the distinguishing part
    last: "Rounded Serving Tray, 560 x 280 mm, 2 compartments" cut to one line
    loses the compartment count, which is what a buyer is scanning for.

    Args:
        text: The text to fit.
        max_chars: Roughly how many characters fit on one line.
        max_lines: How many lines the cell has room for.

    Returns:
        The text with newlines inserted, ellipsised if it still overflows.
    """
    text = " ".join(text.split())
    if not text:
        return ""
    width = max(4, max_chars)
    lines = textwrap.wrap(text, width=width) or [""]
    if len(lines) <= max_lines:
        return "\n".join(lines)
    kept = lines[:max_lines]
    kept[-1] = kept[-1][: max(1, width - len(_ELLIPSIS))].rstrip() + _ELLIPSIS
    return "\n".join(kept)


def _cell_aspect(
    tiles: Sequence[Tile], cell_px: int, cache: dict[Path, np.ndarray]
) -> float:
    """The middle width-to-height ratio of a set of tiles.

    Cells are shaped to this rather than left square, because a grid of wide
    trays in square cells is mostly empty space and a buyer reads empty space
    as padding.  The median rather than the extreme, so one tall outlier does
    not stretch every cell; outliers letterbox sideways instead, which costs
    nothing.

    Args:
        tiles: The tiles that will fill the grid.
        cell_px: Cell width, used to load each image at working size.
        cache: Per-sheet image cache.

    Returns:
        A width/height ratio, clamped to a range that keeps the page sane.
    """
    ratios: list[float] = []
    for tile in tiles:
        height, width = _load(Path(tile.image), cell_px, cache).shape[:2]
        if height:
            ratios.append(width / height)
    if not ratios:
        return 1.0
    ratios.sort()
    middle = ratios[len(ratios) // 2]
    return min(_MAX_CELL_ASPECT, max(_MIN_CELL_ASPECT, middle))


def render_contact_sheet(
    tiles: Sequence[Tile],
    path: str | Path,
    title: str = "",
    subtitle: str = "",
    columns: int | None = None,
    cell_px: int = 400,
    dpi: int = 150,
    background: str = "#ffffff",
) -> Path:
    """Composite every tile's preview into one labelled grid.

    Args:
        tiles: The tiles, in the order they should appear.
        path: Destination PNG; parent directories are created.
        title: Heading drawn above the grid; omitted if empty.
        subtitle: Second heading line, drawn right-aligned beside the title.
        columns: Forced column count, or ``None`` to choose one.
        cell_px: Image area width per tile, in pixels; the height follows the
            tiles' own proportions.
        dpi: Output resolution; also sets how point sizes map to pixels.
        background: Page colour.

    Returns:
        The written path.

    Raises:
        ValueError: If ``tiles`` is empty, or a size argument is not positive.
        FileNotFoundError: If a tile's image is missing.
    """
    if not tiles:
        raise ValueError("a contact sheet needs at least one tile")
    if cell_px < 32:
        raise ValueError(f"cell_px must be >= 32, got {cell_px}")
    if dpi < 1:
        raise ValueError(f"dpi must be >= 1, got {dpi}")

    cache: dict[Path, np.ndarray] = {}
    cols, rows = grid_shape(len(tiles), columns)
    image_h = max(32, round(cell_px / _cell_aspect(tiles, cell_px, cache)))

    # Pixel geometry first; every size below is derived from it.
    gutter = max(6, round(cell_px * 0.05))
    line_px = max(13, round(cell_px * 0.055))
    caption_lines = 2 if any(t.caption for t in tiles) else 0
    sub_lines = 1 if any(t.subcaption for t in tiles) else 0
    caption_px = line_px * caption_lines
    sub_px = round(line_px * 0.88) * sub_lines
    label_px = caption_px + sub_px + (round(line_px * 0.45) if caption_lines else 0)
    margin = max(gutter, round(cell_px * 0.09))
    title_px = max(20, round(cell_px * 0.115)) if title else 0
    subtitle_px = max(15, round(cell_px * 0.075)) if subtitle else 0
    header_px = (title_px + round(margin * 0.8)) if title else 0

    cell_h = image_h + label_px
    total_w = 2 * margin + cols * cell_px + (cols - 1) * gutter
    total_h = 2 * margin + header_px + rows * cell_h + (rows - 1) * gutter

    # Points, not pixels: matplotlib text is sized in points.
    to_points = 72.0 / dpi
    caption_pt = line_px * 0.62 * to_points
    sub_pt = line_px * 0.55 * to_points
    title_pt = title_px * 0.62 * to_points
    subtitle_pt = subtitle_px * 0.64 * to_points
    # DejaVu Sans averages about 0.55 em per character; used only to decide
    # where a caption wraps so it cannot run into its neighbour.
    caption_chars = max(6, int(cell_px / max(1.0, line_px * 0.62 * 0.55)))
    sub_chars = max(6, int(cell_px / max(1.0, line_px * 0.55 * 0.55)))

    fig = Figure(figsize=(total_w / dpi, total_h / dpi), dpi=dpi)
    fig.patch.set_facecolor(background)

    def fx(px: float) -> float:
        """Pixels from the left edge, as a figure fraction."""
        return px / total_w

    def fy(px: float) -> float:
        """Pixels from the *top* edge, as a figure fraction from the bottom."""
        return 1.0 - px / total_h

    if title:
        baseline = margin + title_px * 0.78
        fig.text(
            fx(margin),
            fy(baseline),
            title,
            ha="left",
            va="baseline",
            fontsize=title_pt,
            color=_TEXT,
            weight="bold",
        )
        if subtitle:
            fig.text(
                fx(total_w - margin),
                fy(baseline),
                subtitle,
                ha="right",
                va="baseline",
                fontsize=subtitle_pt,
                color=_MUTED,
            )
        rule = margin + title_px + round(margin * 0.30)
        fig.add_artist(
            Line2D(
                [fx(margin), fx(total_w - margin)],
                [fy(rule), fy(rule)],
                color=_RULE,
                linewidth=0.8,
                transform=fig.transFigure,
            )
        )

    top0 = margin + header_px
    for index, tile in enumerate(tiles):
        row, col = divmod(index, cols)
        left = margin + col * (cell_px + gutter)
        top = top0 + row * (cell_h + gutter)

        ax = fig.add_axes(
            (fx(left), fy(top + image_h), cell_px / total_w, image_h / total_h)
        )
        ax.set_facecolor(background)
        ax.axis("off")
        ax.imshow(_load(Path(tile.image), cell_px, cache), interpolation="antialiased")
        ax.set_anchor("C")

        text_top = top + image_h + round(line_px * 0.45)
        if tile.caption:
            fig.text(
                fx(left + cell_px / 2),
                fy(text_top + line_px * 0.78),
                _wrap(tile.caption, caption_chars),
                ha="center",
                va="baseline",
                linespacing=1.25,
                fontsize=caption_pt,
                color=_TEXT,
            )
        if tile.subcaption:
            fig.text(
                fx(left + cell_px / 2),
                fy(text_top + caption_px + sub_px * 0.78),
                _wrap(tile.subcaption, sub_chars, max_lines=1),
                ha="center",
                va="baseline",
                fontsize=sub_pt,
                color=_MUTED,
            )

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, facecolor=background)
    return out
