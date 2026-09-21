"""Tests for the finger-jointed box generator.

The corner joint is the product, so the central test assembles the box in
plan view at every finger and checks that the two panels neither overlap nor
leave the corner unfilled.
"""

from __future__ import annotations

import pytest
from shapely.geometry import box as shapely_box
from shapely.ops import unary_union

from dxfgen.core import geometry as geo
from dxfgen.niches.boxes import BoxGenerator, BoxParams, LidStyle, _panel, _tab_spans


@pytest.fixture(scope="module")
def gen() -> BoxGenerator:
    return BoxGenerator()


# --------------------------------------------------------------------------- #
# the corner joint
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("thickness", [3.0, 4.0, 6.0])
@pytest.mark.parametrize("fingers", [5, 7, 9])
def test_corners_interlock_without_overlapping(thickness: float, fingers: int) -> None:
    """Assemble the box in plan view, one finger at a time.

    At every height exactly one of the two panels must own the corner cube:
    both owning it is material in the same place twice, neither owning it is a
    hole in the corner.
    """
    width, depth = 200.0, 150.0
    corner = shapely_box(0.0, 0.0, thickness, thickness)
    for index in range(fingers):
        front_keeps = index % 2 == 0
        front = shapely_box(
            0.0 if front_keeps else thickness,
            0.0,
            width - (0.0 if front_keeps else thickness),
            thickness,
        )
        side = shapely_box(
            0.0,
            thickness if front_keeps else 0.0,
            thickness,
            depth - (thickness if front_keeps else 0.0),
        )
        assert front.intersection(side).area == pytest.approx(0.0, abs=1e-9)
        covered = unary_union([front, side]).intersection(corner).area
        assert covered == pytest.approx(thickness * thickness, abs=1e-9)


def test_panels_alternate_complementarily() -> None:
    """The two sides of a corner are one call apart, so they cannot drift."""
    thickness, height, fingers = 4.0, 88.0, 11
    front = _panel(200.0, height, thickness, True, True, fingers)
    side = _panel(150.0, height, thickness, False, False, fingers)
    from shapely.geometry import Point, Polygon

    front_poly, side_poly = Polygon(front), Polygon(side)
    cell = height / fingers
    for index in range(fingers):
        # Sample the strip each panel would occupy at the shared corner.
        probe = Point(thickness / 2.0, (index + 0.5) * cell)
        assert front_poly.contains(probe) != side_poly.contains(probe), index


def test_a_panel_is_a_closed_simple_ring() -> None:
    from shapely.geometry import LinearRing, Polygon

    ring = _panel(200.0, 90.0, 4.0, True, True, 9)
    assert LinearRing(ring + [ring[0]]).is_simple
    assert Polygon(ring).is_valid


def test_finger_counts_are_odd(gen: BoxGenerator) -> None:
    for height in (40.0, 70.0, 120.0):
        params = BoxParams(height=height, thickness=3.0)
        assert params.fingers_for(height, None) % 2 == 1
        assert params.fingers_for(height, 8) % 2 == 1


# --------------------------------------------------------------------------- #
# the floor joint
# --------------------------------------------------------------------------- #
def test_floor_tabs_and_mortises_are_derived_from_one_place() -> None:
    """Computed separately they end up a material thickness apart."""
    params = BoxParams(width=200.0, depth=150.0, thickness=4.0, min_wall=6.0, kerf=0.0)
    spans = _tab_spans(params, params.width, params.floor_tabs)
    design = BoxGenerator().generate(params)
    front = next(p for p in design.parts if p.name == "front")
    floor = next(p for p in design.parts if p.name == "floor")

    mortise_lefts = sorted(geo.bbox(h)[0] for h in front.holes)
    expected = [start - params.clearance / 2.0 for start, _ in spans]
    assert mortise_lefts == pytest.approx(expected, abs=0.01)

    # Floor tabs live in floor coordinates, one thickness in from the box edge.
    tab_x = sorted({round(x + params.thickness, 3) for x, y in floor.outline if y < -0.001})
    assert tab_x[0] == pytest.approx(spans[0][0], abs=0.01)


def test_the_mortise_gives_the_tab_its_clearance() -> None:
    params = BoxParams(width=200.0, depth=150.0, thickness=4.0, min_wall=6.0, kerf=0.0,
                       clearance=0.3)
    design = BoxGenerator().generate(params)
    front = next(p for p in design.parts if p.name == "front")
    length, height = geo.size_of(front.holes[0])
    assert length == pytest.approx(params.floor_tab_width + 0.3)
    assert height == pytest.approx(params.thickness + 0.3)


def test_the_floor_sits_on_a_plinth(gen: BoxGenerator) -> None:
    params = BoxParams(thickness=3.0, floor_offset=9.0, kerf=0.0)
    design = gen.generate(params)
    front = next(p for p in design.parts if p.name == "front")
    assert min(geo.bbox(h)[1] for h in front.holes) == pytest.approx(
        9.0 - params.clearance / 2.0
    )


def test_floor_tabs_that_do_not_fit_are_refused() -> None:
    with pytest.raises(ValueError, match="floor tabs"):
        BoxGenerator().make(width=90.0, depth=70.0, floor_tabs=4, floor_tab_width=40.0)


# --------------------------------------------------------------------------- #
# kerf
# --------------------------------------------------------------------------- #
def test_both_halves_of_the_corner_finish_on_size(gen: BoxGenerator) -> None:
    kerf = 0.2
    plain = gen.generate(BoxParams(width=200.0, depth=150.0, thickness=4.0, kerf=0.0))
    burnt = gen.generate(BoxParams(width=200.0, depth=150.0, thickness=4.0, kerf=kerf))
    a = next(p for p in plain.parts if p.name == "front")
    b = next(p for p in burnt.parts if p.name == "front")
    # Drawn larger by half a kerf all round, so it burns down to nominal.
    assert geo.size_of(b.outline)[0] == pytest.approx(geo.size_of(a.outline)[0] + kerf)
    assert geo.size_of(b.holes[0])[0] == pytest.approx(
        geo.size_of(a.holes[0])[0] - kerf
    )


def test_a_router_box_is_not_kerf_compensated(gen: BoxGenerator) -> None:
    router = gen.generate(
        BoxParams(mode="router", thickness=6.0, tool_diameter=3.0, min_wall=9.0, kerf=0.2)
    )
    laser = gen.generate(BoxParams(thickness=6.0, min_wall=9.0, kerf=0.0))
    a = next(p for p in router.parts if p.name == "front")
    b = next(p for p in laser.parts if p.name == "front")
    assert geo.size_of(a.outline)[0] == pytest.approx(geo.size_of(b.outline)[0])


# --------------------------------------------------------------------------- #
# the box as a product
# --------------------------------------------------------------------------- #
def test_the_default_box_validates(gen: BoxGenerator) -> None:
    design = gen.make()
    assert gen.check(design).ok, gen.check(design).format()
    assert [p.name for p in design.parts] == ["front", "back", "left", "right", "floor"]


def test_a_lid_adds_two_parts(gen: BoxGenerator) -> None:
    design = gen.make(lid=LidStyle.CAP)
    names = [p.name for p in design.parts]
    assert "lid-top" in names and "lid-locator" in names
    assert len(design.parts) == 7


def test_the_lid_locator_drops_inside_with_clearance(gen: BoxGenerator) -> None:
    params = BoxParams(lid=LidStyle.CAP, lid_clearance=0.5, kerf=0.0)
    design = gen.generate(params)
    locator = next(p for p in design.parts if p.name == "lid-locator")
    inside_w = params.width - 2 * params.thickness
    assert geo.size_of(locator.outline)[0] == pytest.approx(inside_w - 1.0)


def test_inside_dimensions_are_reported_correctly(gen: BoxGenerator) -> None:
    params = BoxParams(width=200.0, depth=150.0, height=90.0, thickness=4.0,
                       floor_offset=6.0)
    assert params.inside() == pytest.approx((192.0, 142.0, 80.0))
    assert "192 x 142 x 80 mm" in gen.generate(params).description


def test_a_box_with_no_room_inside_is_refused() -> None:
    with pytest.raises(ValueError, match="leaves only"):
        BoxParams(width=50.0, depth=50.0, thickness=12.0)


def test_parts_do_not_overlap_on_the_sheet(gen: BoxGenerator) -> None:
    from shapely.geometry import Polygon

    placed = [Polygon(p.outline) for p in gen.make(lid=LidStyle.CAP).placed_parts()]
    for index, first in enumerate(placed):
        for second in placed[index + 1 :]:
            assert not first.intersects(second)
