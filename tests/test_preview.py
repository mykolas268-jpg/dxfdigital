"""Tests for the PNG renderers.

Images are hard to assert on, so these tests check the things that can be
checked: that the files are produced at the requested size, that the same
design always renders identically, that a hole actually shows the background
through it, and that the texture is driven by the material.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
from matplotlib.path import Path as MplPath

from dxfgen.core import geometry as geo
from dxfgen.core.design import Contour, Design, Drill, Label, Part, Pocket
from dxfgen.core.layers import ENGRAVE
from dxfgen.core.preview import (
    PREVIEW_COLORS,
    WOOD_PALETTES,
    compound_path,
    render_mockup,
    render_preview,
)


def sample_design(**kwargs) -> Design:
    part = Part(
        name="tray",
        outline=geo.rounded_rect_ring(300, 200, 20),
        holes=[geo.stadium_ring(95, 32, 30, 84)],
        pockets=[Pocket(geo.rounded_rect_ring(140, 150, 15, 145, 25), 8.0)],
        drills=[Drill((20, 30), 6.0)],
        engrave=[Contour(geo.circle_ring(20, 170, 6), ENGRAVE)],
        labels=[Label("OAK 19 mm", (150, 8), 6.0, align="center")],
    )
    kwargs.setdefault("parts", [part])
    kwargs.setdefault("thickness", 19.0)
    kwargs.setdefault("material", "oak")
    kwargs.setdefault("sheet", (600.0, 400.0))
    return Design("demo", "Demo Tray", "trays", "a demo", **kwargs)


def read_pixels(path: Path) -> np.ndarray:
    import matplotlib.image as mpimg

    return mpimg.imread(path)


# --------------------------------------------------------------------------- #
# compound paths
# --------------------------------------------------------------------------- #
def rasterise(path: MplPath, size: int = 100) -> np.ndarray:
    """Fill a path black on white and return the pixels.

    Rendering is the right oracle here.  ``Path.contains_point`` answers with a
    different fill rule than the Agg renderer uses, so it would pass or fail
    independently of what the image actually shows.
    """
    import io

    import matplotlib.image as mpimg
    from matplotlib.figure import Figure
    from matplotlib.patches import PathPatch

    figure = Figure(figsize=(2, 2), dpi=size / 2)
    figure.patch.set_facecolor("white")
    ax = figure.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    ax.add_patch(PathPatch(path, facecolor="black", edgecolor="none"))
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", facecolor="white")
    buffer.seek(0)
    return mpimg.imread(buffer)


def test_a_hole_renders_as_a_hole() -> None:
    """Agg fills by nonzero winding, so holes must wind the other way."""
    pixels = rasterise(
        compound_path(geo.rect_ring(100, 100), [geo.rect_ring(20, 20, 40, 40)])
    )
    assert pixels[50, 50][0] == pytest.approx(1.0)  # hole: background
    assert pixels[90, 10][0] == pytest.approx(0.0)  # material


def test_hole_winding_is_corrected_whichever_way_it_arrives() -> None:
    hole = geo.rect_ring(20, 20, 40, 40)
    for ring in (hole, hole[::-1]):
        pixels = rasterise(compound_path(geo.rect_ring(100, 100), [ring]))
        assert pixels[50, 50][0] == pytest.approx(1.0)


def test_compound_path_closes_every_ring() -> None:
    path = compound_path(geo.rect_ring(10, 10), [geo.rect_ring(2, 2, 4, 4)])
    assert list(path.codes).count(MplPath.CLOSEPOLY) == 2
    assert list(path.codes).count(MplPath.MOVETO) == 2


def test_compound_path_skips_degenerate_rings() -> None:
    path = compound_path(geo.rect_ring(10, 10), [[(0, 0), (1, 1)]])
    assert list(path.codes).count(MplPath.MOVETO) == 1


# --------------------------------------------------------------------------- #
# preview
# --------------------------------------------------------------------------- #
def test_preview_is_written_at_the_requested_width(tmp_path: Path) -> None:
    path = render_preview(sample_design(), tmp_path / "p.png", px_width=800, dpi=100)
    assert read_pixels(path).shape[1] == 800


def test_preview_creates_directories(tmp_path: Path) -> None:
    assert render_preview(sample_design(), tmp_path / "a" / "b" / "p.png").exists()


def test_preview_shows_the_background_through_a_handle_cutout(tmp_path: Path) -> None:
    """The cutout is a hole; if it renders as material the file lies."""
    path = render_preview(
        sample_design(), tmp_path / "p.png", px_width=600, dpi=100,
        margin_mm=0.0, caption=False,
    )
    pixels = read_pixels(path)
    rows, cols = pixels.shape[:2]
    # The handle spans x 30..125, y 84..116 of a 300x200 design.
    hole = pixels[int(rows * (1 - 100 / 200)), int(cols * (78 / 300))][:3]
    rim = pixels[int(rows * (1 - 10 / 200)), int(cols * (150 / 300))][:3]
    assert np.allclose(hole, (1.0, 1.0, 1.0), atol=0.02)  # background
    assert not np.allclose(rim, (1.0, 1.0, 1.0), atol=0.02)  # material


def test_preview_uses_the_designed_palette_not_raw_cad_colours(
    tmp_path: Path,
) -> None:
    from matplotlib.colors import to_rgb

    path = render_preview(
        sample_design(), tmp_path / "p.png", px_width=600, dpi=100,
        margin_mm=0.0, caption=False,
    )
    pixels = read_pixels(path)
    rows, cols = pixels.shape[:2]
    pocket = pixels[int(rows * (1 - 100 / 200)), int(cols * (215 / 300))][:3]
    assert not np.allclose(pocket, (0.0, 1.0, 0.0), atol=0.15)  # not ACI green
    shallow = np.asarray(to_rgb(PREVIEW_COLORS["pocket_shallow"]))
    deep = np.asarray(to_rgb(PREVIEW_COLORS["pocket_deep"]))
    assert np.all(pocket <= shallow + 0.02) and np.all(pocket >= deep - 0.02)


def test_preview_is_reproducible(tmp_path: Path) -> None:
    first = render_preview(sample_design(), tmp_path / "a.png", px_width=400, dpi=100)
    second = render_preview(sample_design(), tmp_path / "b.png", px_width=400, dpi=100)
    assert hashlib.sha256(first.read_bytes()).digest() == hashlib.sha256(
        second.read_bytes()
    ).digest()


def test_preview_caption_can_be_switched_off(tmp_path: Path) -> None:
    with_caption = read_pixels(
        render_preview(sample_design(), tmp_path / "a.png", px_width=400, dpi=100)
    )
    without = read_pixels(
        render_preview(
            sample_design(), tmp_path / "b.png", px_width=400, dpi=100, caption=False
        )
    )
    assert with_caption.shape[0] > without.shape[0]  # caption adds height


# --------------------------------------------------------------------------- #
# mockup
# --------------------------------------------------------------------------- #
def test_mockup_is_written_at_the_requested_width(tmp_path: Path) -> None:
    path = render_mockup(sample_design(), tmp_path / "m.png", px_width=700, dpi=100)
    assert read_pixels(path).shape[1] == 700


def test_mockup_is_reproducible_from_the_slug(tmp_path: Path) -> None:
    first = render_mockup(sample_design(), tmp_path / "a.png", px_width=400, dpi=100)
    second = render_mockup(sample_design(), tmp_path / "b.png", px_width=400, dpi=100)
    assert first.read_bytes() == second.read_bytes()


def test_different_designs_get_different_grain(tmp_path: Path) -> None:
    one = render_mockup(sample_design(), tmp_path / "a.png", px_width=400, dpi=100)
    other_design = sample_design()
    other_design.slug = "another-tray"
    other = render_mockup(other_design, tmp_path / "b.png", px_width=400, dpi=100)
    assert one.read_bytes() != other.read_bytes()


def test_material_drives_the_palette(tmp_path: Path) -> None:
    """Walnut must not render the same colour as maple."""
    walnut = read_pixels(
        render_mockup(
            sample_design(material="walnut"), tmp_path / "w.png", px_width=400, dpi=100
        )
    )
    maple = read_pixels(
        render_mockup(
            sample_design(material="maple"), tmp_path / "m.png", px_width=400, dpi=100
        )
    )
    assert walnut.mean() < maple.mean() - 0.1


def test_an_unknown_material_still_renders(tmp_path: Path) -> None:
    path = render_mockup(
        sample_design(material="unobtanium"), tmp_path / "u.png", px_width=400, dpi=100
    )
    assert path.exists()


def test_every_palette_entry_is_a_light_dark_pair() -> None:
    for name, (light, dark) in WOOD_PALETTES.items():
        assert sum(light) > sum(dark), name
        assert all(0.0 <= channel <= 1.0 for channel in (*light, *dark)), name
