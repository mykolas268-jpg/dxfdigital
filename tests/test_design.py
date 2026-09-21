"""Tests for the design container."""

from __future__ import annotations

import math

import pytest

from dxfgen.core import geometry as g
from dxfgen.core.design import (
    LASER_SHEET,
    ROUTER_SHEET,
    Contour,
    Design,
    Drill,
    Label,
    Machine,
    Mode,
    Part,
    Pocket,
)
from dxfgen.core.layers import CUT_INSIDE, CUT_OUTSIDE, ENGRAVE, INFO


def make_part(**kwargs) -> Part:
    kwargs.setdefault("name", "panel")
    kwargs.setdefault("outline", g.rounded_rect_ring(200, 120, 15))
    return Part(**kwargs)


def make_design(**kwargs) -> Design:
    kwargs.setdefault("slug", "demo")
    kwargs.setdefault("name", "Demo")
    kwargs.setdefault("niche", "trays")
    kwargs.setdefault("description", "a demo")
    kwargs.setdefault("parts", [make_part()])
    return Design(**kwargs)


# --------------------------------------------------------------------------- #
# Machine
# --------------------------------------------------------------------------- #
def test_machine_derives_radius_and_sheet() -> None:
    m = Machine()
    assert m.tool_radius == pytest.approx(3.175)
    assert m.default_sheet() == ROUTER_SHEET
    assert not m.is_laser
    laser = Machine(mode=Mode.LASER, kerf=0.15)
    assert laser.is_laser
    assert laser.default_sheet() == LASER_SHEET


def test_machine_slot_width_is_the_nominal_fit_on_both_modes() -> None:
    router = Machine(mode=Mode.ROUTER, clearance=0.2)
    laser = Machine(mode=Mode.LASER, clearance=0.2, kerf=0.15)
    assert router.slot_width(18.0) == pytest.approx(18.2)
    assert laser.slot_width(3.0) == pytest.approx(3.2)


def test_kerf_compensation_grows_the_outline_and_shrinks_the_cutouts() -> None:
    kerf = 0.15
    part = Part("p", g.rect_ring(100, 60), holes=[g.rect_ring(20, 20, 40, 20)])
    fixed = part.kerf_compensated(kerf)
    assert g.size_of(fixed.outline)[0] == pytest.approx(100 + kerf)
    assert g.size_of(fixed.holes[0])[0] == pytest.approx(20 - kerf)


def test_kerf_compensation_is_a_no_op_at_zero() -> None:
    part = Part("p", g.rect_ring(100, 60))
    assert part.kerf_compensated(0.0) is part


def test_kerf_compensation_refuses_a_cutout_narrower_than_the_beam() -> None:
    part = Part("p", g.rect_ring(100, 60), holes=[g.rect_ring(0.1, 20, 40, 20)])
    with pytest.raises(ValueError, match="narrower than the"):
        part.kerf_compensated(0.5)


@pytest.mark.parametrize(
    "kwargs", [{"tool_diameter": 0}, {"kerf": -1}, {"clearance": -0.1}]
)
def test_machine_rejects_bad_values(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        Machine(**kwargs)


# --------------------------------------------------------------------------- #
# Pocket / Drill / Label
# --------------------------------------------------------------------------- #
def test_pocket_layer_and_region() -> None:
    pocket = Pocket(g.rect_ring(50, 50), 8.0, islands=[g.rect_ring(10, 10, 20, 20)])
    assert pocket.layer == "POCKET_8"
    assert pocket.region().area == pytest.approx(2500 - 100)


def test_pocket_rejects_non_positive_depth() -> None:
    with pytest.raises(ValueError):
        Pocket(g.rect_ring(10, 10), 0)


def test_drill_radius_and_validation() -> None:
    assert Drill((0, 0), 6.0).radius == 3.0
    with pytest.raises(ValueError):
        Drill((0, 0), 0)


def test_label_defaults_to_info_layer() -> None:
    assert Label("x", (0, 0)).layer == INFO
    with pytest.raises(ValueError):
        Label("x", (0, 0), height=0)


def test_contour_rejects_unknown_layer() -> None:
    Contour([(0, 0), (1, 1)], ENGRAVE, closed=False)
    with pytest.raises(KeyError):
        Contour([(0, 0), (1, 1)], "NOT_A_LAYER")


# --------------------------------------------------------------------------- #
# Part
# --------------------------------------------------------------------------- #
def test_part_rejects_degenerate_outline() -> None:
    with pytest.raises(ValueError):
        Part("bad", [(0, 0), (1, 0)])


def test_part_rejects_zero_quantity() -> None:
    with pytest.raises(ValueError):
        make_part(quantity=0)


def test_part_cut_rings_cover_every_layer() -> None:
    part = make_part(
        holes=[g.circle_ring(100, 60, 10)],
        pockets=[Pocket(g.rect_ring(40, 40, 20, 20), 6.0, islands=[g.rect_ring(5, 5, 30, 30)])],
    )
    layers = [layer for _, layer in part.cut_rings()]
    assert layers == [CUT_OUTSIDE, CUT_INSIDE, "POCKET_6", "POCKET_6"]


def test_part_placed_applies_translation() -> None:
    part = make_part(origin=(10, 5), drills=[Drill((0, 0), 6)])
    placed = part.placed()
    assert placed.bbox() == pytest.approx((10, 5, 210, 125))
    assert placed.drills[0].center == pytest.approx((10, 5))


def test_part_placed_applies_rotation_before_translation() -> None:
    part = Part("p", g.rect_ring(40, 10), rotation=90, origin=(100, 0))
    placed = part.placed()
    assert placed.size() == pytest.approx((10, 40))
    assert placed.bbox() == pytest.approx((90, 0, 100, 40))


def test_part_placed_rotates_labels_and_pockets() -> None:
    part = Part(
        "p",
        g.rect_ring(40, 20),
        pockets=[Pocket(g.rect_ring(10, 10, 5, 5), 4.0)],
        labels=[Label("t", (5, 5), rotation=0)],
        rotation=90,
    )
    placed = part.placed()
    assert placed.labels[0].rotation == pytest.approx(90)
    assert placed.labels[0].position == pytest.approx((-5, 5))
    assert g.bbox(placed.pockets[0].ring) == pytest.approx((-15, 5, -5, 15))


def test_part_placed_is_identity_without_transform() -> None:
    part = make_part()
    assert part.placed() is part


def test_part_translated_accumulates() -> None:
    part = make_part(origin=(5, 5)).translated(10, 0)
    assert part.origin == (15, 5)


def test_part_area_excludes_holes() -> None:
    part = Part("p", g.rect_ring(100, 100), holes=[g.rect_ring(10, 10, 20, 20)])
    assert part.area() == pytest.approx(10000 - 100)
    assert part.region().is_valid


# --------------------------------------------------------------------------- #
# Design
# --------------------------------------------------------------------------- #
def test_design_rejects_empty_parts_and_bad_thickness() -> None:
    with pytest.raises(ValueError):
        make_design(parts=[])
    with pytest.raises(ValueError):
        make_design(thickness=0)


def test_design_normalized_moves_bbox_to_origin() -> None:
    design = make_design(parts=[make_part(origin=(-30, 17))])
    assert design.bbox()[:2] == pytest.approx((-30, 17))
    fixed = design.normalized()
    assert fixed.bbox() == pytest.approx((0, 0, 200, 120))
    assert design.bbox()[:2] == pytest.approx((-30, 17))  # original untouched


def test_design_normalized_is_identity_when_already_at_origin() -> None:
    design = make_design()
    assert design.normalized() is design


def test_design_layers_used_orders_cut_layers_first() -> None:
    part = make_part(
        holes=[g.circle_ring(100, 60, 10)],
        pockets=[Pocket(g.rect_ring(40, 40, 140, 20), 6.0)],
        drills=[Drill((20, 20), 6)],
        labels=[Label("x", (10, 10))],
        engrave=[Contour(g.circle_ring(60, 60, 5), ENGRAVE)],
    )
    used = make_design(parts=[part]).layers_used()
    assert used == ["CUT_INSIDE", "CUT_OUTSIDE", "POCKET_6", "DRILL", "ENGRAVE", "INFO"]


def test_design_pocket_depths_are_sorted_and_unique() -> None:
    part = make_part(
        pockets=[
            Pocket(g.rect_ring(30, 30, 10, 10), 8.0),
            Pocket(g.rect_ring(30, 30, 60, 10), 3.0),
            Pocket(g.rect_ring(30, 30, 110, 10), 8.0),
        ]
    )
    assert make_design(parts=[part]).pocket_depths() == [3.0, 8.0]


def test_design_sheet_falls_back_to_machine_default() -> None:
    assert make_design().sheet_size() == ROUTER_SHEET
    assert make_design(machine=Machine(mode=Mode.LASER)).sheet_size() == LASER_SHEET
    assert make_design(sheet=(500, 300)).sheet_size() == (500, 300)


def test_design_total_cut_length_counts_every_contour() -> None:
    part = Part("p", g.rect_ring(100, 50), drills=[Drill((50, 25), 6.0)])
    expected = 300 + 2 * math.pi * 3.0
    assert make_design(parts=[part]).total_cut_length() == pytest.approx(expected)


def test_design_region_unions_parts() -> None:
    parts = [
        Part("a", g.rect_ring(50, 50)),
        Part("b", g.rect_ring(50, 50), origin=(60, 0)),
    ]
    assert make_design(parts=parts).region().area == pytest.approx(5000)


def test_design_summary_mentions_size_material_and_mode() -> None:
    text = make_design(material="walnut", thickness=19).summary()
    assert "200.0x120.0" in text and "walnut" in text and "router" in text
