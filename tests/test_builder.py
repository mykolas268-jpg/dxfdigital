"""Tests for assembling one design's output folder."""

from __future__ import annotations

from pathlib import Path

import pytest

from dxfgen.bundles.builder import LICENSE_SUMMARY, readme_text, write_design
from dxfgen.core import geometry as geo
from dxfgen.core.design import Design, Machine, Mode, Part, Pocket
from dxfgen.core.export_pdf import A3, A4
from dxfgen.core.validate import ValidationError
from dxfgen.niches import get_generator


@pytest.fixture(scope="module")
def design() -> Design:
    return get_generator("trays").make(
        length=460, width=290, layout="halves", material="walnut", thickness=19
    )


@pytest.fixture(scope="module")
def written(tmp_path_factory: pytest.TempPathFactory, design: Design):
    root = tmp_path_factory.mktemp("out")
    return write_design(design, root)


# --------------------------------------------------------------------------- #
# the folder
# --------------------------------------------------------------------------- #
def test_every_file_is_written(written) -> None:
    assert set(written.files) == {"dxf", "svg", "pdf", "preview", "mockup", "readme"}
    for kind, path in written.files.items():
        assert path.exists() and path.stat().st_size > 0, kind


def test_files_land_in_a_niche_and_slug_folder(written, design: Design) -> None:
    assert written.directory.name == design.slug
    assert written.directory.parent.name == design.niche


def test_files_are_named_after_the_slug(written, design: Design) -> None:
    assert written.files["dxf"].name == f"{design.slug}.dxf"
    assert written.files["svg"].name == f"{design.slug}.svg"
    assert written.files["pdf"].name == f"{design.slug}_template.pdf"
    assert written.files["preview"].name == f"{design.slug}_preview.png"
    assert written.files["mockup"].name == f"{design.slug}_mockup.png"
    assert written.files["readme"].name == "README.txt"


def test_the_export_carries_the_validation_report(written) -> None:
    assert written.report is not None and written.report.ok


def test_a_subset_of_formats_can_be_requested(tmp_path: Path, design: Design) -> None:
    out = write_design(design, tmp_path, formats=("dxf", "readme"))
    assert set(out.files) == {"dxf", "readme"}
    assert not (out.directory / f"{design.slug}.svg").exists()


def test_an_unknown_format_is_refused(tmp_path: Path, design: Design) -> None:
    with pytest.raises(ValueError, match="unknown output format"):
        write_design(design, tmp_path, formats=("dxf", "dwg"))


def test_an_invalid_design_writes_no_geometry(tmp_path: Path) -> None:
    bad = Part(
        "bad", geo.rect_ring(300, 200),
        pockets=[Pocket(geo.rect_ring(200, 100, 50, 50), 8.0)],
    )
    design = Design("bad", "Bad", "trays", "x", [bad], thickness=19.0, sheet=(600, 400))
    with pytest.raises(ValidationError):
        write_design(design, tmp_path)
    assert not (tmp_path / "trays" / "bad" / "bad.dxf").exists()


# --------------------------------------------------------------------------- #
# the README
# --------------------------------------------------------------------------- #
def test_readme_states_the_material_and_thickness(design: Design) -> None:
    text = readme_text(design)
    assert "walnut, 19 mm thick" in text
    assert "460.0 x 290.0 mm" in text


def test_readme_states_the_cutter(design: Design) -> None:
    assert "6.35 mm diameter" in readme_text(design)


def test_readme_lists_every_layer_with_its_depth(design: Design) -> None:
    text = readme_text(design)
    for layer in design.layers_used():
        assert layer in text
    assert "pocket, 8 mm deep" in text


def test_readme_gives_the_cutting_order_in_order(design: Design) -> None:
    text = readme_text(design)
    body = text.split("CUTTING ORDER")[1]
    assert body.index("POCKET_8") < body.index("CUT_INSIDE") < body.index("CUT_OUTSIDE")


def test_readme_states_the_remaining_floor(design: Design) -> None:
    assert "11.0 mm under the deepest recess" in readme_text(design)


def test_readme_explains_how_to_print_the_template(design: Design) -> None:
    text = readme_text(design)
    assert "100% scale" in text
    assert "100 mm bar" in text


def test_readme_mentions_tiling_only_when_tiled(design: Design) -> None:
    assert "tape them together" in readme_text(design, paper=A4)
    small = get_generator("trays").make(length=240, width=150, handle="none")
    assert "tape them together" not in readme_text(small, paper=A3)


def test_readme_carries_the_licence_summary(design: Design) -> None:
    text = readme_text(design)
    assert LICENSE_SUMMARY in text
    assert "MAY NOT resell" in text
    assert "sell the physical items" in text


def test_readme_describes_laser_designs_in_laser_terms() -> None:
    design = get_generator("trays").make(
        mode="laser", thickness=6.0, pocket_depth=2.0, pocket_floor=2.0,
        length=300, width=200, handle="none", material="6 mm birch ply",
    )
    text = readme_text(design)
    assert "Kerf" in text and "already compensated" in text
    assert "Cutter" not in text


def test_readme_names_every_file_it_ships_with(written, design: Design) -> None:
    text = written.files["readme"].read_text()
    for path in written.files.values():
        if path.name != "README.txt":
            assert path.name in text
