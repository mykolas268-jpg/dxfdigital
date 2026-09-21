"""Tests for the slot-together dock generator.

The joint is the product here, so most of these tests are about fit: that a
mortise finishes at the size the material actually is, on both machines, and
that the corners a cutter cannot reach have been relieved.
"""

from __future__ import annotations

import pytest
from shapely.geometry import Polygon

from dxfgen.core import geometry as geo
from dxfgen.niches.stands import StandGenerator, StandParams, StandSize


@pytest.fixture(scope="module")
def gen() -> StandGenerator:
    return StandGenerator()


def router(**kwargs) -> dict:
    base = {"thickness": 18.0, "base_depth": 115.0}
    base.update(kwargs)
    return base


def laser(**kwargs) -> dict:
    base = {
        "mode": "laser", "thickness": 3.0, "device_gap": 11.0,
        "min_wall": 5.0, "pocket_floor": 2.0,
    }
    base.update(kwargs)
    return base


# --------------------------------------------------------------------------- #
# the joint
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("thickness", [12.0, 18.0])
def test_router_mortises_carry_the_fit_clearance(thickness: float) -> None:
    params = StandParams(thickness=thickness, clearance=0.2, base_depth=120)
    assert params.mortise_width() == pytest.approx(thickness + 0.2)
    assert params.mortise_length() == pytest.approx(params.tab_width + 0.2)


@pytest.mark.parametrize("thickness", [3.0, 6.0])
def test_laser_mortises_are_drawn_one_kerf_narrower(thickness: float) -> None:
    """Drawn small, burnt out to size: the finished slot is what matters."""
    params = StandParams(**laser(thickness=thickness, kerf=0.15, clearance=0.2))
    drawn = params.mortise_width()
    assert drawn == pytest.approx(thickness + 0.2 - 0.15)
    assert drawn + 0.15 == pytest.approx(thickness + 0.2)


def test_the_finished_fit_is_the_same_on_both_machines() -> None:
    cut = StandParams(thickness=18.0, base_depth=120).mortise_width()
    burn = StandParams(**laser(thickness=3.0))
    assert cut - 18.0 == pytest.approx(burn.mortise_width() + burn.kerf - 3.0)


def test_tabs_protrude_by_exactly_one_material_thickness(gen: StandGenerator) -> None:
    """So they come through the base and finish flush with its underside."""
    design = gen.make(**router())
    back = next(p for p in design.parts if p.name == "back")
    ys = sorted({round(y, 3) for _, y in back.outline})
    assert ys[0] == pytest.approx(0.0)
    # The shoulder sits one thickness up from the tab ends.
    assert any(abs(y - 18.0) < 0.6 for y in ys)


def test_mortises_match_the_tabs_they_take(gen: StandGenerator) -> None:
    design = gen.make(**router())
    params = StandParams(**design.params)
    base = next(p for p in design.parts if p.name == "base")
    assert len(base.holes) == 2 * params.tabs
    for hole in base.holes:
        width, length = geo.size_of(hole)
        # Relief reaches a little past the rectangle at each corner.
        assert width >= params.mortise_width() - 0.01
        assert length >= params.mortise_length() - 0.01
        assert width <= params.mortise_width() + params.tool_diameter
        assert length <= params.mortise_length() + params.tool_diameter


# --------------------------------------------------------------------------- #
# relief
# --------------------------------------------------------------------------- #
def test_router_mortises_are_relieved_so_a_square_tab_seats(gen: StandGenerator) -> None:
    design = gen.make(**router())
    radius = design.machine.tool_radius
    base = next(p for p in design.parts if p.name == "base")
    for hole in base.holes:
        assert geo.unreachable_zones(Polygon(hole), radius, 0.35) == []


def test_tab_roots_are_relieved_so_the_shoulder_seats(gen: StandGenerator) -> None:
    """A tab root is an inside corner on the profile; a cutter cannot cut one."""
    design = gen.make(**router())
    radius = design.machine.tool_radius
    for name in ("back", "lip"):
        part = next(p for p in design.parts if p.name == name)
        assert geo.excess_zones(Polygon(part.outline), radius, 0.35) == [], name


def test_a_laser_design_carries_no_relief(gen: StandGenerator) -> None:
    """A beam is a fraction of a millimetre; relief would only waste material."""
    design = gen.make(**laser())
    base = next(p for p in design.parts if p.name == "base")
    for hole in base.holes:
        assert len(geo.dedupe(hole)) == 4  # a plain rectangle


def test_turning_relief_off_on_a_router_fails_validation(gen: StandGenerator) -> None:
    """The parameter exists, but the geometry it produces cannot be cut."""
    design = gen.make(**router(relief=False))
    codes = {i.code for i in gen.check(design).issues}
    assert "E_TOOL_UNREACHABLE" in codes


# --------------------------------------------------------------------------- #
# layout
# --------------------------------------------------------------------------- #
def test_the_channel_sits_between_the_two_uprights(gen: StandGenerator) -> None:
    gap = 16.0
    design = gen.make(**router(device_gap=gap))
    base = next(p for p in design.parts if p.name == "base")
    xs = sorted({round(geo.centroid(h)[0], 2) for h in base.holes})
    assert len(xs) == 2
    assert xs[1] - xs[0] == pytest.approx(18.0 + gap, abs=0.2)


def test_a_base_too_shallow_for_the_channel_is_refused(gen: StandGenerator) -> None:
    with pytest.raises(ValueError, match="cannot hold a"):
        gen.make(**router(base_depth=60.0, device_gap=20.0))


def test_tabs_that_do_not_fit_across_the_width_are_refused(gen: StandGenerator) -> None:
    with pytest.raises(ValueError, match="no material between them"):
        gen.make(**router(width=60.0, tabs=3, tab_width=30.0))


def test_features_keep_full_wall_near_a_rounded_corner(gen: StandGenerator) -> None:
    """A rounded corner cuts the corner off, so a feature must sit further in."""
    for radius in (0.0, 6.0, 12.0):
        design = gen.make(**laser(corner_radius=radius))
        assert gen.check(design).ok, f"radius {radius}: {gen.check(design).format()}"


def test_a_narrow_channel_for_thick_material_is_refused() -> None:
    with pytest.raises(ValueError, match="too narrow to be worth cutting"):
        StandParams(thickness=18.0, device_gap=9.0)


def test_the_cable_notch_is_in_the_lip(gen: StandGenerator) -> None:
    with_notch = gen.make(**router(cable_slot=True))
    without = gen.make(**router(cable_slot=False))
    lip_a = next(p for p in with_notch.parts if p.name == "lip")
    lip_b = next(p for p in without.parts if p.name == "lip")
    assert geo.signed_area(lip_a.outline) < geo.signed_area(lip_b.outline)


def test_three_parts_and_no_fasteners(gen: StandGenerator) -> None:
    design = gen.make(**router())
    assert [p.name for p in design.parts] == ["base", "back", "lip"]
    assert "no glue" in design.description.lower()


def test_tablet_docks_are_bigger_than_phone_docks(gen: StandGenerator) -> None:
    phone = gen.make(**router(size=StandSize.PHONE))
    tablet = gen.make(**router(size=StandSize.TABLET, base_depth=150))
    assert tablet.size()[0] * tablet.size()[1] > phone.size()[0] * phone.size()[1]


def test_the_notes_state_the_mortise_size_and_channel(gen: StandGenerator) -> None:
    design = gen.make(**router(device_gap=15.0))
    joined = " ".join(design.notes)
    assert "18.20" in joined
    assert "15 mm channel" in joined
