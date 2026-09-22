"""Tests for the slot-together dock generator.

The joint is the product here, so most of these tests are about fit: that a
mortise finishes at the size the material actually is, on both machines, and
that the corners a cutter cannot reach have been relieved.
"""

from __future__ import annotations

import pytest
from shapely.geometry import Polygon

from dxfgen.core import geometry as geo
from dxfgen.niches.stands import (
    BackStyle,
    StandGenerator,
    StandParams,
    StandSize,
)


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
def test_both_halves_of_a_laser_joint_finish_on_size(
    gen: StandGenerator, thickness: float
) -> None:
    """The tab matters as much as the mortise.

    Compensating only the mortise leaves the joint a full kerf loose, because
    the beam takes the same kerf off the tab it goes into.
    """
    kerf, clearance = 0.15, 0.2
    kwargs = laser(thickness=thickness, kerf=kerf, clearance=clearance)
    design = gen.make(**kwargs)
    params = StandParams(**kwargs)
    base = next(p for p in design.parts if p.name == "base")
    back = next(p for p in design.parts if p.name == "back")

    finished_mortise = geo.size_of(base.holes[0])[0] + kerf
    lowest = min(y for _, y in back.outline)
    tab_edge = sorted(x for x, y in back.outline if abs(y - lowest) < 1e-6)
    finished_tab = (tab_edge[1] - tab_edge[0]) - kerf

    assert finished_mortise == pytest.approx(thickness + clearance, abs=0.01)
    assert finished_tab == pytest.approx(params.tab_width, abs=0.01)


def test_the_finished_fit_is_the_same_on_both_machines(gen: StandGenerator) -> None:
    cut = StandParams(thickness=18.0, base_depth=120)
    burn = StandParams(**laser(thickness=3.0))
    assert cut.mortise_width() - 18.0 == pytest.approx(burn.mortise_width() - 3.0)


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


# --------------------------------------------------------------------------- #
# the shape of the back panel
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("style", list(BackStyle))
def test_every_back_style_builds_and_validates(
    gen: StandGenerator, style: BackStyle
) -> None:
    design = gen.make(size="tablet", back_style=style)
    assert gen.check(design).ok, gen.check(design).format()


@pytest.mark.parametrize("style", list(BackStyle))
def test_a_shaped_back_keeps_its_width_and_height(
    gen: StandGenerator, style: BackStyle
) -> None:
    """Shaping the top must not change what the dock is, only how it looks."""
    square = gen.make(size="tablet", back_style=BackStyle.SQUARE)
    shaped = gen.make(size="tablet", back_style=style)
    back_of = lambda d: next(p for p in d.parts if p.name == "back")
    assert back_of(shaped).size() == pytest.approx(back_of(square).size())


@pytest.mark.parametrize("style", [BackStyle.ARCH, BackStyle.TAPER, BackStyle.PEAK])
def test_a_shaped_back_removes_material(
    gen: StandGenerator, style: BackStyle
) -> None:
    square = gen.make(size="tablet", back_style=BackStyle.SQUARE)
    shaped = gen.make(size="tablet", back_style=style, back_shape_rise=34)
    back_of = lambda d: next(p for p in d.parts if p.name == "back")
    assert back_of(shaped).area() < back_of(square).area()


@pytest.mark.parametrize("style", list(BackStyle))
def test_only_the_back_is_shaped(gen: StandGenerator, style: BackStyle) -> None:
    """The lip carries the cable notch; a profile on it is geometry to dodge."""
    square = gen.make(size="tablet", back_style=BackStyle.SQUARE)
    shaped = gen.make(size="tablet", back_style=style)
    lip_of = lambda d: next(p for p in d.parts if p.name == "lip")
    assert lip_of(shaped).area() == pytest.approx(lip_of(square).area())


def test_the_back_style_is_in_the_slug(gen: StandGenerator) -> None:
    for style in BackStyle:
        assert style.value in gen.make(size="phone", back_style=style).slug


def test_a_bigger_rise_takes_more_off(gen: StandGenerator) -> None:
    small = gen.make(size="tablet", back_style=BackStyle.PEAK, back_shape_rise=12)
    large = gen.make(size="tablet", back_style=BackStyle.PEAK, back_shape_rise=48)
    back_of = lambda d: next(p for p in d.parts if p.name == "back")
    assert back_of(large).area() < back_of(small).area()


def test_the_rise_is_capped_by_the_panel(gen: StandGenerator) -> None:
    """A rise taller than the panel would cut the back off its own tabs."""
    design = gen.make(size="phone", back_style=BackStyle.PEAK, back_shape_rise=120)
    back = next(p for p in design.parts if p.name == "back")
    assert back.area() > 0
    assert gen.check(design).ok


# --------------------------------------------------------------------------- #
# tabs are sized to the panel they go in
# --------------------------------------------------------------------------- #
def test_the_widest_tab_is_the_widest_that_actually_fits() -> None:
    params = StandParams(size="tablet", width=140, tabs=3)
    widest = params.widest_tab()
    assert params.model_copy(update={"tab_width": widest - 0.1}).tab_positions()
    with pytest.raises(ValueError, match="leaving no material"):
        params.model_copy(update={"tab_width": widest + 0.5}).tab_positions()


def test_a_narrower_panel_allows_a_narrower_tab() -> None:
    wide = StandParams(size="tablet", width=180, tabs=2).widest_tab()
    narrow = StandParams(size="tablet", width=110, tabs=2).widest_tab()
    assert narrow < wide


def test_more_tabs_allow_a_narrower_tab() -> None:
    two = StandParams(size="tablet", width=160, tabs=2).widest_tab()
    three = StandParams(size="tablet", width=160, tabs=3).widest_tab()
    assert three < two


def test_sampled_docks_rarely_fail_to_build(gen: StandGenerator) -> None:
    """Drawing a tab width independently of the panel wasted a quarter of them."""
    attempts = failures = 0
    for seed in (7, 31, 42, 2026):
        run = gen.sample_variants(12, seed=seed)
        attempts += run.attempts
        failures += len(run.skips)
    assert failures / attempts < 0.1, f"{failures}/{attempts} rejected"
