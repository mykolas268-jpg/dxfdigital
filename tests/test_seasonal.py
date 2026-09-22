"""Tests for the seasonal niche."""

from __future__ import annotations

import math

import pytest
from shapely.geometry import LineString, Polygon

from dxfgen.core import geometry as geo
from dxfgen.core.layers import ENGRAVE
from dxfgen.niches.seasonal import SeasonalForm, SeasonalGenerator, SeasonalParams
from dxfgen.niches.shapes import shape_names, shape_ring


@pytest.fixture(scope="module")
def gen() -> SeasonalGenerator:
    return SeasonalGenerator()


ROUTER_SHAPES = [n for n in shape_names() if n != "snowflake"]

#: Shapes a recess can be sunk into.  A snowflake has no rim to sink one into,
#: and a paw's toes dissolve when the outline is offset inwards.
TRAY_SHAPES = [n for n in ROUTER_SHAPES if n != "paw"]


@pytest.mark.parametrize("shape", ROUTER_SHAPES)
def test_every_shape_makes_a_valid_plaque(gen: SeasonalGenerator, shape: str) -> None:
    design = gen.make(shape=shape, form=SeasonalForm.PLAQUE, width=260)
    assert gen.check(design).ok, gen.check(design).format()


@pytest.mark.parametrize("shape", TRAY_SHAPES)
def test_every_trayable_shape_makes_a_valid_tray(
    gen: SeasonalGenerator, shape: str
) -> None:
    design = gen.make(shape=shape, form=SeasonalForm.TRAY, width=380)
    assert gen.check(design).ok, gen.check(design).format()


def test_a_snowflake_tray_is_refused(gen: SeasonalGenerator) -> None:
    """There is no rim on a snowflake to sink a recess into."""
    with pytest.raises(ValueError, match="no single recess"):
        gen.make(
            shape="snowflake", form=SeasonalForm.TRAY, width=380,
            mode="laser", thickness=3.0, min_wall=4.0, pocket_floor=1.0,
        )


def test_a_small_router_snowflake_is_refused(gen: SeasonalGenerator) -> None:
    """Its arms are finer than the cutter until the piece gets large."""
    with pytest.raises(ValueError, match="cut it on a laser"):
        gen.make(shape="snowflake", form=SeasonalForm.PLAQUE, width=100)
    # Big enough, and the same cutter follows it perfectly well.
    assert gen.make(shape="snowflake", form=SeasonalForm.PLAQUE, width=260)


def test_a_laser_snowflake_plaque_works(gen: SeasonalGenerator) -> None:
    design = gen.make(
        shape="snowflake", form=SeasonalForm.PLAQUE, width=260,
        mode="laser", thickness=3.0, min_wall=4.0, pocket_floor=1.0,
    )
    assert gen.check(design).ok


def test_the_tray_recess_follows_the_shape(gen: SeasonalGenerator) -> None:
    params = SeasonalParams(shape="heart", form=SeasonalForm.TRAY, width=400, pocket_inset=25)
    design = gen.generate(params)
    part = design.parts[0]
    assert len(part.pockets) == 1
    outline = Polygon(part.outline)
    recess = Polygon(part.pockets[0].ring)
    assert outline.contains(recess)
    assert outline.exterior.distance(recess) == pytest.approx(25.0, abs=0.6)


def test_the_recess_corners_are_reachable(gen: SeasonalGenerator) -> None:
    design = gen.make(shape="star", form=SeasonalForm.TRAY, width=420)
    radius = design.machine.tool_radius
    assert geo.unreachable_zones(design.parts[0].pockets[0].region(), radius, 0.35) == []


def test_a_rim_that_leaves_no_recess_is_refused(gen: SeasonalGenerator) -> None:
    with pytest.raises(ValueError, match="recess"):
        gen.make(shape="heart", form=SeasonalForm.TRAY, width=200, pocket_inset=70)


def test_coaster_sets_produce_one_part_each(gen: SeasonalGenerator) -> None:
    design = gen.make(shape="leaf", form=SeasonalForm.COASTER, width=110, count=6)
    assert len(design.parts) == 6
    areas = {round(geo.signed_area(p.outline), 3) for p in design.parts}
    assert len(areas) == 1  # identical


def test_only_coasters_come_in_sets() -> None:
    with pytest.raises(ValueError, match="only the coaster form"):
        SeasonalParams(shape="heart", form=SeasonalForm.PLAQUE, count=4)


def test_engraved_detail_lands_on_the_engrave_layer(gen: SeasonalGenerator) -> None:
    design = gen.make(shape="pumpkin", form=SeasonalForm.PLAQUE, width=300)
    part = design.parts[0]
    assert part.engrave
    assert all(c.layer == ENGRAVE for c in part.engrave)
    region = Polygon(part.outline)
    for contour in part.engrave:
        points = list(contour.points) + ([contour.points[0]] if contour.closed else [])
        assert region.covers(LineString(points))


def test_tray_engraving_stays_on_the_rim(gen: SeasonalGenerator) -> None:
    """Decoration across the recess wall would be cut through by the pocket."""
    design = gen.make(shape="pumpkin", form=SeasonalForm.TRAY, width=420, pocket_inset=30)
    part = design.parts[0]
    rim = Polygon(part.outline).difference(part.pockets[0].region())
    for contour in part.engrave:
        points = list(contour.points) + ([contour.points[0]] if contour.closed else [])
        assert rim.covers(LineString(points))


def test_engraving_can_be_switched_off(gen: SeasonalGenerator) -> None:
    assert gen.make(shape="star", engrave_detail=False).parts[0].engrave == []


def test_a_hanging_hole_is_placed_in_material(gen: SeasonalGenerator) -> None:
    design = gen.make(shape="tree", hang_hole=True, hang_hole_diameter=10, min_wall=10)
    part = design.parts[0]
    assert len(part.holes) == 1
    outline = Polygon(part.outline)
    assert outline.contains(Polygon(part.holes[0]))
    assert outline.exterior.distance(Polygon(part.holes[0])) >= 10.0 - 0.05


def test_an_unknown_shape_is_refused_with_the_list() -> None:
    with pytest.raises(ValueError, match="available: "):
        SeasonalParams(shape="unicorn")


def test_the_name_and_slug_carry_the_occasion(gen: SeasonalGenerator) -> None:
    design = gen.make(shape="pumpkin", form=SeasonalForm.PLAQUE, width=260)
    assert design.slug.startswith("halloween-pumpkin")
    assert "Pumpkin Plaque" in design.name
    assert "halloween" in design.description


def test_the_notes_state_the_work_is_original(gen: SeasonalGenerator) -> None:
    joined = " ".join(gen.make(shape="heart").notes)
    assert "equations" in joined and "original" in joined


# --------------------------------------------------------------------------- #
# a recess has to follow the outline it is set into
# --------------------------------------------------------------------------- #
def test_the_shape_factor_rises_with_wiggliness() -> None:
    """A circle is the least convoluted shape there is; a star is not."""
    from dxfgen.niches.seasonal import _shape_factor

    circle = _shape_factor(geo.circle_ring(0.0, 0.0, 50.0))
    square = _shape_factor(geo.rect_ring(100.0, 100.0))
    star = _shape_factor(shape_ring("star", 100.0))
    assert circle == pytest.approx(4 * math.pi, rel=0.01), "4 pi for a circle"
    assert circle < square < star


def test_the_shape_factor_ignores_size() -> None:
    from dxfgen.niches.seasonal import _shape_factor

    small = _shape_factor(shape_ring("heart", 80.0))
    large = _shape_factor(shape_ring("heart", 640.0))
    assert small == pytest.approx(large, rel=0.02)


def test_a_paw_tray_is_refused_because_its_toes_dissolve(
    gen: SeasonalGenerator,
) -> None:
    """Offsetting inward eats small lobes; the result reads as a puddle."""
    with pytest.raises(ValueError, match="does not follow"):
        gen.make(shape="paw", form="tray", width=400, mode="router", thickness=25)


@pytest.mark.parametrize("width", [300.0, 400.0, 480.0])
def test_a_paw_tray_is_refused_at_every_size(
    gen: SeasonalGenerator, width: float
) -> None:
    with pytest.raises(ValueError):
        gen.make(shape="paw", form="tray", width=width, mode="router", thickness=25)


def test_a_paw_still_works_as_a_plaque(gen: SeasonalGenerator) -> None:
    """The refusal is about the recess, not about the shape."""
    design = gen.make(shape="paw", form="plaque", width=350)
    assert gen.check(design).ok
