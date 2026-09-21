"""Tests for DXF export and file-level validation."""

from __future__ import annotations

from pathlib import Path

import ezdxf
import pytest

from dxfgen.core import geometry as g
from dxfgen.core.design import Design, Drill, Label, Machine, Part, Pocket
from dxfgen.core.export_dxf import DXF_VERSION, audit_file, build_document, write_dxf
from dxfgen.core.layers import CUT_INSIDE, CUT_OUTSIDE, INFO, STANDARD_LAYERS
from dxfgen.core.validate import ValidationError, validate_dxf_file


def sample_design(**kwargs) -> Design:
    part = Part(
        name="tray",
        outline=g.rounded_rect_ring(300, 200, 20),
        holes=[g.stadium_ring(95, 32, 30, 84)],
        pockets=[Pocket(g.rounded_rect_ring(140, 150, 15, 145, 25), 8.0)],
        drills=[Drill((20, 30), 6.0)],
        labels=[Label("OAK 19 mm", (150, 8), 6.0, align="center")],
    )
    kwargs.setdefault("parts", [part])
    kwargs.setdefault("thickness", 19.0)
    kwargs.setdefault("sheet", (600.0, 400.0))
    kwargs.setdefault("material", "oak")
    return Design("tray-demo", "Tray Demo", "trays", "A demo tray", **kwargs)


@pytest.fixture
def written(tmp_path: Path) -> Path:
    path, report = write_dxf(sample_design(), tmp_path / "tray.dxf")
    assert report.ok, report.format()
    return path


# --------------------------------------------------------------------------- #
# output profile
# --------------------------------------------------------------------------- #
def test_file_is_r2010_in_millimetres(written: Path) -> None:
    doc = ezdxf.readfile(written)
    assert doc.acad_release == DXF_VERSION
    assert doc.dxfversion == "AC1024"
    assert doc.header["$INSUNITS"] == 4
    assert doc.header["$MEASUREMENT"] == 1
    assert doc.units == 4


def test_file_passes_ezdxf_audit(written: Path) -> None:
    errors, fixes = audit_file(written)
    assert errors == []
    assert fixes == []


def test_extents_are_recorded_so_cad_opens_zoomed_in(written: Path) -> None:
    doc = ezdxf.readfile(written)
    assert tuple(doc.header["$EXTMIN"])[:2] == pytest.approx((0.0, 0.0))
    assert tuple(doc.header["$EXTMAX"])[:2] == pytest.approx((300.0, 200.0))


def test_every_contour_is_a_closed_lwpolyline(written: Path) -> None:
    doc = ezdxf.readfile(written)
    polylines = list(doc.modelspace().query("LWPOLYLINE"))
    assert len(polylines) == 3
    assert all(p.closed for p in polylines)
    assert all(len(p) >= 3 for p in polylines)


def test_no_unsupported_entity_types_are_emitted(written: Path) -> None:
    doc = ezdxf.readfile(written)
    kinds = {e.dxftype() for e in doc.modelspace()}
    assert kinds == {"LWPOLYLINE", "CIRCLE", "TEXT"}
    assert not kinds & {"SPLINE", "ELLIPSE", "HATCH", "POLYLINE", "INSERT", "ARC"}


def test_drills_are_real_circles_not_polygons(written: Path) -> None:
    doc = ezdxf.readfile(written)
    circles = list(doc.modelspace().query("CIRCLE"))
    assert len(circles) == 1
    assert circles[0].dxf.radius == pytest.approx(3.0)
    assert circles[0].dxf.layer == "DRILL"


def test_layers_carry_the_standard_colours(written: Path) -> None:
    doc = ezdxf.readfile(written)
    for name in (CUT_OUTSIDE, CUT_INSIDE, INFO):
        assert doc.layers.get(name).color == STANDARD_LAYERS[name].color
    assert doc.layers.get("POCKET_8").color == 3


def test_info_layer_is_marked_not_to_plot(written: Path) -> None:
    doc = ezdxf.readfile(written)
    assert doc.layers.get(INFO).dxf.plot == 0
    assert doc.layers.get(CUT_OUTSIDE).dxf.plot == 1


def test_geometry_round_trips_to_the_millimetre(written: Path) -> None:
    doc = ezdxf.readfile(written)
    outline = next(
        p for p in doc.modelspace().query("LWPOLYLINE") if p.dxf.layer == CUT_OUTSIDE
    )
    pts = [(p[0], p[1]) for p in outline.get_points("xy")]
    assert g.bbox(pts) == pytest.approx((0, 0, 300, 200))
    assert g.signed_area(pts) == pytest.approx(
        g.signed_area(sample_design().parts[0].outline)
    )


def test_text_is_placed_and_carries_its_height(written: Path) -> None:
    doc = ezdxf.readfile(written)
    text = next(iter(doc.modelspace().query("TEXT")))
    assert text.dxf.text == "OAK 19 mm"
    assert text.dxf.height == pytest.approx(6.0)
    assert text.dxf.layer == INFO


# --------------------------------------------------------------------------- #
# behaviour
# --------------------------------------------------------------------------- #
def test_export_normalises_the_design_to_the_origin(tmp_path: Path) -> None:
    design = sample_design()
    design.parts[0].origin = (-40.0, 25.0)
    path, report = write_dxf(design, tmp_path / "moved.dxf")
    assert report.ok
    doc = ezdxf.readfile(path)
    assert tuple(doc.header["$EXTMIN"])[:2] == pytest.approx((0.0, 0.0))


def test_an_invalid_design_raises_and_writes_nothing(tmp_path: Path) -> None:
    bad = Part("bad", g.rect_ring(300, 200), pockets=[Pocket(g.rect_ring(200, 100, 50, 50), 8.0)])
    design = sample_design(parts=[bad])
    target = tmp_path / "never.dxf"
    with pytest.raises(ValidationError) as exc:
        write_dxf(design, target)
    assert "E_TOOL_UNREACHABLE" in str(exc.value)
    assert not target.exists()


def test_validation_can_be_bypassed_deliberately(tmp_path: Path) -> None:
    bad = Part("bad", g.rect_ring(300, 200), pockets=[Pocket(g.rect_ring(200, 100, 50, 50), 8.0)])
    path, report = write_dxf(
        sample_design(parts=[bad]), tmp_path / "forced.dxf", validate=False
    )
    assert path.exists()
    assert not report.ok


def test_export_creates_missing_directories(tmp_path: Path) -> None:
    path, _ = write_dxf(sample_design(), tmp_path / "a" / "b" / "c.dxf")
    assert path.exists()


def test_build_document_does_not_touch_the_filesystem() -> None:
    doc = build_document(sample_design())
    assert doc.modelspace()


# --------------------------------------------------------------------------- #
# validating files
# --------------------------------------------------------------------------- #
def test_our_own_output_passes_file_validation(written: Path) -> None:
    report = validate_dxf_file(written)
    assert report.ok, report.format()
    assert report.warnings == []


def test_file_validation_reports_a_missing_file(tmp_path: Path) -> None:
    report = validate_dxf_file(tmp_path / "nope.dxf")
    assert not report.ok
    assert report.errors[0].code == "E_UNREADABLE"


def test_file_validation_catches_third_party_problems(tmp_path: Path) -> None:
    """A deliberately awful DXF, of the kind bought files often are."""
    doc = ezdxf.new("R2010", setup=True)
    doc.header["$INSUNITS"] = 1  # inches
    msp = doc.modelspace()
    doc.layers.add(CUT_OUTSIDE)
    doc.layers.add("MY_CUSTOM_LAYER")
    msp.add_lwpolyline([(0, 0), (10, 0), (10, 10)], close=False,
                       dxfattribs={"layer": CUT_OUTSIDE})
    square = [(50, 50), (80, 50), (80, 80), (50, 80)]
    msp.add_lwpolyline(square, close=True, dxfattribs={"layer": CUT_OUTSIDE})
    msp.add_lwpolyline(square, close=True, dxfattribs={"layer": "MY_CUSTOM_LAYER"})
    msp.add_lwpolyline([(0, 0), (30, 30), (30, 0), (0, 30)], close=True,
                       dxfattribs={"layer": CUT_OUTSIDE})
    msp.add_spline([(0, 0), (10, 5), (20, 0)], dxfattribs={"layer": CUT_OUTSIDE})
    msp.add_line((0, 0), (5, 5), dxfattribs={"layer": CUT_OUTSIDE})
    msp.add_text("do not cut", dxfattribs={"layer": CUT_OUTSIDE, "height": 5})
    path = tmp_path / "awful.dxf"
    doc.saveas(path)

    report = validate_dxf_file(path)
    found = {i.code for i in report.issues}
    assert not report.ok
    assert {
        "E_UNITS",
        "E_OPEN_CONTOUR",
        "E_DUPLICATE_CONTOUR",
        "E_SELF_INTERSECT",
        "E_SPLINE",
        "W_OPEN_PRIMITIVE",
        "E_TEXT_ON_CUT_LAYER",
        "W_NONSTANDARD_LAYER",
    } <= found


def test_file_validation_warns_when_geometry_is_off_origin(tmp_path: Path) -> None:
    doc = ezdxf.new("R2010", setup=True)
    doc.header["$INSUNITS"] = 4
    doc.layers.add(CUT_OUTSIDE)
    doc.modelspace().add_lwpolyline(
        g.rect_ring(50, 50, 400, 400), close=True, dxfattribs={"layer": CUT_OUTSIDE}
    )
    path = tmp_path / "offset.dxf"
    doc.saveas(path)
    assert "W_ORIGIN" in {i.code for i in validate_dxf_file(path).issues}


def test_file_validation_notices_a_file_with_no_closed_contours(tmp_path: Path) -> None:
    doc = ezdxf.new("R2010", setup=True)
    doc.header["$INSUNITS"] = 4
    doc.modelspace().add_line((0, 0), (10, 10))
    path = tmp_path / "lines.dxf"
    doc.saveas(path)
    assert "W_NO_CLOSED_CONTOURS" in {i.code for i in validate_dxf_file(path).issues}
