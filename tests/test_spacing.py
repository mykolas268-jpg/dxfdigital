"""Tests for the spacing between parts on the sheet.

A router cuts a channel a full cutter diameter wide outside every profile, so
two parts that do not overlap can still ruin each other: laid out 5 mm apart,
a 6.35 mm cutter takes 1.35 mm off the neighbour.  These tests hold the
parameters, the design validator and the file validator to that rule.
"""

from __future__ import annotations

import ezdxf
import pytest
from pydantic import ValidationError as ParamsError

from dxfgen.core import geometry as g
from dxfgen.core.design import Design, Machine, Mode, Part
from dxfgen.core.export_dxf import write_dxf
from dxfgen.core.layers import CUT_INSIDE, CUT_OUTSIDE
from dxfgen.core.validate import Report, validate_design, validate_dxf_file
from dxfgen.niches import get_generator
from dxfgen.niches.base import SCRAP_WEB
from dxfgen.niches.boxes import BoxParams
from dxfgen.niches.coasters import CoasterParams
from dxfgen.niches.furniture import FurnitureParams
from dxfgen.niches.ornaments import OrnamentParams
from dxfgen.niches.seasonal import SeasonalParams
from dxfgen.niches.stands import StandParams

SPACED = [
    (BoxParams, "gap", 8.0),
    (CoasterParams, "gap", 12.0),
    (FurnitureParams, "sheet_gap", 14.0),
    (OrnamentParams, "gap", 6.0),
    (SeasonalParams, "gap", 12.0),
    (StandParams, "gap", 12.0),
]


def codes(report: Report) -> set[str]:
    return {issue.code for issue in report.issues}


def pair(gap: float, machine: Machine | None = None) -> Design:
    """Two 100 mm squares side by side, ``gap`` apart."""
    a = Part("a", g.rect_ring(100, 100))
    b = Part("b", g.rect_ring(100, 100), origin=(100 + gap, 0))
    return Design(
        "d", "D", "trays", "x", [a, b],
        machine=machine or Machine(), sheet=(600, 400), thickness=19,
    )


# --------------------------------------------------------------------------- #
# the machine
# --------------------------------------------------------------------------- #
def test_a_router_needs_a_cutter_diameter_between_parts() -> None:
    machine = Machine(tool_diameter=6.35)
    assert machine.min_part_gap == pytest.approx(6.35)
    # Outlines are chords of their curves: each may sit ARC_TOLERANCE inside
    # the true one, so a layout judged on chords needs both in hand.
    assert machine.min_layout_gap == pytest.approx(6.35 + 2 * g.ARC_TOLERANCE)


def test_a_laser_only_needs_parts_not_to_overlap() -> None:
    machine = Machine(mode=Mode.LASER)
    assert machine.min_part_gap == 0.0
    assert machine.min_layout_gap == 0.0


# --------------------------------------------------------------------------- #
# the design validator
# --------------------------------------------------------------------------- #
def test_parts_closer_than_the_cutter_are_rejected() -> None:
    report = validate_design(pair(5.0))
    assert "E_PART_GAP" in codes(report)
    assert "1.35 mm into the other" in report.format()


def test_parts_exactly_a_cutter_apart_leave_no_allowance_for_curves() -> None:
    report = validate_design(pair(6.35))
    assert "E_PART_GAP" in codes(report)
    assert "allowance" in report.format()


def test_parts_at_the_layout_minimum_pass() -> None:
    assert validate_design(pair(Machine().min_layout_gap)).ok


def test_the_rule_follows_the_cutter() -> None:
    assert validate_design(pair(10.0)).ok
    report = validate_design(pair(10.0, Machine(tool_diameter=12.7)))
    assert "E_PART_GAP" in codes(report)


def test_laser_parts_may_sit_close() -> None:
    report = validate_design(pair(1.0, Machine(mode=Mode.LASER)))
    assert "E_PART_GAP" not in codes(report)


def test_the_gap_is_measured_between_outlines_not_boxes() -> None:
    # Two discs whose bounding boxes are 2 mm apart diagonally-offset are far
    # more than a cutter apart: the check must not trust the boxes alone.
    a = Part("a", g.circle_ring(50, 50, 50))
    b = Part("b", g.circle_ring(50, 50, 50), origin=(102, 102))
    design = Design("d", "D", "trays", "x", [a, b], sheet=(600, 400), thickness=19)
    assert "E_PART_GAP" not in codes(validate_design(design))


# --------------------------------------------------------------------------- #
# the file validator
# --------------------------------------------------------------------------- #
def test_a_file_with_parts_closer_than_the_cutter_is_rejected(tmp_path) -> None:
    path, _ = write_dxf(pair(5.0), tmp_path / "close.dxf", validate=False)
    report = validate_dxf_file(path)
    assert "E_PART_GAP" in codes(report)
    assert "1.35 mm into the other" in report.format()


def test_a_file_is_judged_against_the_cutter_given(tmp_path) -> None:
    path, _ = write_dxf(pair(10.0), tmp_path / "ten.dxf")
    assert "E_PART_GAP" not in codes(validate_dxf_file(path))
    report = validate_dxf_file(path, machine=Machine(tool_diameter=12.7))
    assert "E_PART_GAP" in codes(report)


def _bare_file(path, rings) -> None:
    """Write a DXF with no provenance, as another CAD package would."""
    doc = ezdxf.new("R2010", setup=False)
    doc.header["$INSUNITS"] = 4
    for name in {layer for _, layer in rings}:
        doc.layers.add(name)
    msp = doc.modelspace()
    for ring, layer in rings:
        msp.add_lwpolyline(ring, close=True, dxfattribs={"layer": layer})
    doc.saveas(path)


def test_a_file_of_unknown_origin_is_not_spacing_checked(tmp_path) -> None:
    path = tmp_path / "foreign.dxf"
    _bare_file(
        path,
        [(g.rect_ring(100, 100), CUT_OUTSIDE), (g.rect_ring(100, 100, 105, 0), CUT_OUTSIDE)],
    )
    report = validate_dxf_file(path)
    assert "INFO_NO_MACHINE" in codes(report)
    assert "E_PART_GAP" not in codes(report)
    # ...until the operator says what they will cut it with.
    assert "E_PART_GAP" in codes(validate_dxf_file(path, machine=Machine()))


def test_a_part_cut_from_anothers_waste_is_left_alone(tmp_path) -> None:
    path = tmp_path / "island.dxf"
    _bare_file(
        path,
        [
            (g.rect_ring(200, 200), CUT_OUTSIDE),
            (g.rect_ring(150, 150, 25, 25), CUT_INSIDE),
            (g.rect_ring(100, 100, 50, 50), CUT_OUTSIDE),
        ],
    )
    report = validate_dxf_file(path, machine=Machine())
    assert "E_PART_GAP" not in codes(report)
    assert "E_PART_OVERLAP" not in codes(report)


def test_overlapping_outlines_in_a_file_are_rejected(tmp_path) -> None:
    path = tmp_path / "overlap.dxf"
    _bare_file(
        path,
        [(g.rect_ring(100, 100), CUT_OUTSIDE), (g.rect_ring(100, 100, 50, 0), CUT_OUTSIDE)],
    )
    assert "E_PART_OVERLAP" in codes(validate_dxf_file(path, machine=Machine()))


def test_a_laser_file_may_have_parts_close_together(tmp_path) -> None:
    path, _ = write_dxf(pair(1.0, Machine(mode=Mode.LASER)), tmp_path / "laser.dxf")
    assert "E_PART_GAP" not in codes(validate_dxf_file(path))


# --------------------------------------------------------------------------- #
# the parameters
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("model, field, nominal", SPACED)
def test_unset_spacing_is_the_nominal_one_for_a_small_cutter(model, field, nominal) -> None:
    params = model(mode="router", tool_diameter=3.0)
    assert getattr(params, field) is None
    assert params.part_gap() == pytest.approx(max(nominal, 3.0 + SCRAP_WEB))


@pytest.mark.parametrize("model, field, nominal", SPACED)
def test_unset_spacing_widens_for_a_large_cutter(model, field, nominal) -> None:
    params = model(mode="router", tool_diameter=12.7)
    assert params.part_gap() == pytest.approx(max(nominal, 12.7 + SCRAP_WEB))
    assert params.part_gap() >= params.machine().min_layout_gap


@pytest.mark.parametrize("model, field, nominal", SPACED)
def test_unset_spacing_on_a_laser_is_the_nominal_one(model, field, nominal) -> None:
    assert model(mode="laser").part_gap() == pytest.approx(nominal)


@pytest.mark.parametrize("model, field, nominal", SPACED)
def test_an_explicit_spacing_the_cutter_would_cut_through_is_refused(
    model, field, nominal
) -> None:
    with pytest.raises(ParamsError, match="too narrow for a 12.7 mm cutter"):
        model(mode="router", tool_diameter=12.7, **{field: 12.0})


@pytest.mark.parametrize("model, field, nominal", SPACED)
def test_an_explicit_spacing_that_clears_the_cutter_is_used_as_given(
    model, field, nominal
) -> None:
    params = model(mode="router", tool_diameter=6.35, **{field: 7.0})
    assert params.part_gap() == pytest.approx(7.0)


def test_the_spacing_follows_a_change_of_machine() -> None:
    params = CoasterParams(mode="laser")
    assert params.part_gap() == pytest.approx(12.0)
    params.mode = Mode.ROUTER
    params.tool_diameter = 12.7
    assert params.part_gap() == pytest.approx(17.7)


@pytest.mark.parametrize("niche", ["coasters", "furniture", "stands"])
def test_a_large_cutter_gets_a_layout_it_can_cut(niche) -> None:
    generator = get_generator(niche)
    params = generator.params_model(mode="router", tool_diameter=12.7)
    design = generator.generate(params).normalized()
    report = validate_design(design)
    assert report.ok, report.format()
    assert "E_PART_GAP" not in codes(report)
