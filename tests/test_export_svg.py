"""Tests for SVG export."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from dxfgen.core import geometry as geo
from dxfgen.core.design import Contour, Design, Drill, Label, Part, Pocket
from dxfgen.core.export_svg import aci_to_hex, build_svg, write_svg
from dxfgen.core.layers import CUT_INSIDE, CUT_OUTSIDE, ENGRAVE, INFO, STANDARD_LAYERS
from dxfgen.core.validate import ValidationError

SVG = "{http://www.w3.org/2000/svg}"


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
    kwargs.setdefault("sheet", (600.0, 400.0))
    return Design("demo", "Demo Tray", "trays", "a demo", **kwargs)


@pytest.fixture
def root() -> ET.Element:
    return ET.fromstring(build_svg(sample_design()))


def test_document_is_sized_in_millimetres(root: ET.Element) -> None:
    assert root.get("width") == "300mm"
    assert root.get("height") == "200mm"
    assert root.get("viewBox") == "0 0 300 200"


def test_one_user_unit_is_one_millimetre(root: ET.Element) -> None:
    """The viewBox spans the same numbers as the mm size, so nothing scales."""
    width = float(root.get("width").removesuffix("mm"))
    view = [float(v) for v in root.get("viewBox").split()]
    assert view[2] == width


def test_each_layer_becomes_a_named_group(root: ET.Element) -> None:
    ids = [g.get("id") for g in root.findall(f"{SVG}g")]
    assert ids == [CUT_INSIDE, CUT_OUTSIDE, "POCKET_8", "DRILL", ENGRAVE, INFO]


def test_layer_colours_match_the_dxf(root: ET.Element) -> None:
    groups = {g.get("id"): g for g in root.findall(f"{SVG}g")}
    assert groups[CUT_OUTSIDE].get("stroke") == aci_to_hex(
        STANDARD_LAYERS[CUT_OUTSIDE].color
    )
    assert groups[CUT_INSIDE].get("stroke") == aci_to_hex(
        STANDARD_LAYERS[CUT_INSIDE].color
    )


def test_cut_paths_are_closed(root: ET.Element) -> None:
    for group in root.findall(f"{SVG}g"):
        if group.get("id") in (INFO, "DRILL"):
            continue
        for path in group.findall(f"{SVG}path"):
            assert path.get("d").endswith("Z")


def test_cut_layers_are_not_filled(root: ET.Element) -> None:
    groups = {g.get("id"): g for g in root.findall(f"{SVG}g")}
    assert groups[CUT_OUTSIDE].get("fill") == "none"
    assert groups[CUT_INSIDE].get("fill") == "none"


def test_pockets_are_tinted_so_recesses_read(root: ET.Element) -> None:
    pocket = {g.get("id"): g for g in root.findall(f"{SVG}g")}["POCKET_8"]
    assert pocket.get("fill") != "none"
    assert float(pocket.get("fill-opacity")) < 0.5


def test_pocket_fill_can_be_switched_off() -> None:
    root = ET.fromstring(build_svg(sample_design(), pocket_fill=False))
    pocket = {g.get("id"): g for g in root.findall(f"{SVG}g")}["POCKET_8"]
    assert pocket.get("fill") == "none"


def test_drills_are_real_circles(root: ET.Element) -> None:
    drill = {g.get("id"): g for g in root.findall(f"{SVG}g")}["DRILL"]
    circles = drill.findall(f"{SVG}circle")
    assert len(circles) == 1
    assert float(circles[0].get("r")) == pytest.approx(3.0)


def test_y_is_flipped_for_svg_coordinates(root: ET.Element) -> None:
    """SVG counts down the page; the design counts up."""
    drill = {g.get("id"): g for g in root.findall(f"{SVG}g")}["DRILL"]
    circle = drill.findall(f"{SVG}circle")[0]
    assert float(circle.get("cx")) == pytest.approx(20.0)
    assert float(circle.get("cy")) == pytest.approx(200.0 - 30.0)


def test_text_is_placed_and_anchored(root: ET.Element) -> None:
    info = {g.get("id"): g for g in root.findall(f"{SVG}g")}[INFO]
    text = info.findall(f"{SVG}text")[0]
    assert text.text == "OAK 19 mm"
    assert text.get("text-anchor") == "middle"
    assert float(text.get("y")) == pytest.approx(200.0 - 8.0)


def test_text_is_escaped() -> None:
    design = sample_design()
    design.parts[0].labels[0].text = "5 < 6 & 7 > 2"
    svg = build_svg(design)
    assert "5 &lt; 6 &amp; 7 &gt; 2" in svg
    ET.fromstring(svg)  # still well formed


def test_rotated_text_gets_a_negated_transform() -> None:
    design = sample_design()
    design.parts[0].labels[0].rotation = 90.0
    root = ET.fromstring(build_svg(design))
    info = {g.get("id"): g for g in root.findall(f"{SVG}g")}[INFO]
    assert info.findall(f"{SVG}text")[0].get("transform").startswith("rotate(-90")


def test_title_and_description_carry_the_listing_text(root: ET.Element) -> None:
    assert root.find(f"{SVG}title").text == "Demo Tray"
    assert root.find(f"{SVG}desc").text == "a demo"


def test_write_svg_creates_directories(tmp_path: Path) -> None:
    path, report = write_svg(sample_design(), tmp_path / "a" / "b.svg")
    assert path.exists() and report.ok
    ET.fromstring(path.read_text())


def test_write_svg_refuses_an_invalid_design(tmp_path: Path) -> None:
    bad = Part("bad", geo.rect_ring(300, 200),
               pockets=[Pocket(geo.rect_ring(200, 100, 50, 50), 8.0)])
    target = tmp_path / "never.svg"
    with pytest.raises(ValidationError):
        write_svg(sample_design(parts=[bad]), target)
    assert not target.exists()


def test_aci_to_hex_round_trips_known_colours() -> None:
    assert aci_to_hex(1) == "#ff0000"
    assert aci_to_hex(5) == "#0000ff"
