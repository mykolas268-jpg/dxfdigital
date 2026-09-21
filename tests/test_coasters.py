"""Tests for the coaster generator."""

from __future__ import annotations

import pytest
from shapely.geometry import LineString, Polygon

from dxfgen.core import geometry as geo
from dxfgen.core.layers import ENGRAVE
from dxfgen.niches.coasters import CoasterGenerator, CoasterParams, CoasterShape, Pattern


@pytest.fixture(scope="module")
def gen() -> CoasterGenerator:
    return CoasterGenerator()


@pytest.mark.parametrize("shape", list(CoasterShape))
@pytest.mark.parametrize("pattern", list(Pattern))
def test_every_shape_and_pattern_validates(
    gen: CoasterGenerator, shape: CoasterShape, pattern: Pattern
) -> None:
    design = gen.make(shape=shape, pattern=pattern, size=100, count=4)
    assert gen.check(design).ok, gen.check(design).format()


def test_the_set_has_one_part_per_coaster_plus_the_holder(gen: CoasterGenerator) -> None:
    design = gen.make(count=6, holder=True)
    assert len(design.parts) == 7
    assert design.parts[-1].name == "holder"
    assert [p.name for p in design.parts[:6]] == [f"coaster-{i}" for i in range(1, 7)]


def test_the_holder_can_be_left_out(gen: CoasterGenerator) -> None:
    design = gen.make(count=4, holder=False)
    assert len(design.parts) == 4
    assert all("holder" not in p.name for p in design.parts)


def test_parts_do_not_overlap_on_the_sheet(gen: CoasterGenerator) -> None:
    design = gen.make(count=6, holder=True)
    placed = [Polygon(p.outline) for p in design.placed_parts()]
    for index, first in enumerate(placed):
        for second in placed[index + 1 :]:
            assert not first.intersects(second)


def test_the_holder_recess_accepts_a_coaster_with_clearance(gen: CoasterGenerator) -> None:
    clearance = 1.2
    design = gen.make(shape=CoasterShape.ROUND, size=100, holder_clearance=clearance)
    coaster = design.parts[0]
    holder = design.parts[-1]
    recess = holder.pockets[0].ring
    assert geo.size_of(recess)[0] == pytest.approx(
        geo.size_of(coaster.outline)[0] + 2 * clearance, abs=0.1
    )


def test_the_push_hole_is_a_counterbore_in_the_recess_floor(gen: CoasterGenerator) -> None:
    design = gen.make(holder_push_hole=32.0)
    holder = design.parts[-1]
    assert len(holder.holes) == 1
    recess = Polygon(holder.pockets[0].ring)
    assert recess.contains(Polygon(holder.holes[0]))
    assert gen.check(design).ok  # nested, not a collision


def test_the_push_hole_can_be_left_out(gen: CoasterGenerator) -> None:
    assert gen.make(holder_push_hole=0.0).parts[-1].holes == []


def test_a_push_hole_that_crowds_the_recess_floor_is_refused(gen: CoasterGenerator) -> None:
    with pytest.raises(ValueError, match="push hole"):
        gen.make(size=90, holder_push_hole=80.0)


def test_ring_patterns_follow_the_coaster_shape(gen: CoasterGenerator) -> None:
    """A hexagonal coaster gets hexagonal rings, not a circle stamped on it."""
    design = gen.make(shape=CoasterShape.HEX, pattern=Pattern.RINGS, pattern_lines=3)
    coaster = design.parts[0]
    assert len(coaster.engrave) == 3
    for contour in coaster.engrave:
        assert contour.layer == ENGRAVE and contour.closed
        # A hexagon rounded at the corners still has far fewer vertices than a
        # tessellated circle of the same size.
        assert len(contour.points) < 80


def test_line_patterns_are_clipped_to_the_coaster(gen: CoasterGenerator) -> None:
    design = gen.make(shape=CoasterShape.ROUND, pattern=Pattern.RAYS, pattern_lines=6, size=100)
    coaster = design.parts[0]
    assert coaster.engrave
    face = Polygon(coaster.outline)
    for contour in coaster.engrave:
        assert not contour.closed
        assert face.covers(LineString(contour.points))


def test_grid_patterns_run_both_ways(gen: CoasterGenerator) -> None:
    design = gen.make(shape=CoasterShape.SQUARE, pattern=Pattern.GRID, pattern_lines=3)
    contours = design.parts[0].engrave
    horizontal = sum(1 for c in contours if abs(c.points[0][1] - c.points[-1][1]) < 0.01)
    vertical = sum(1 for c in contours if abs(c.points[0][0] - c.points[-1][0]) < 0.01)
    assert horizontal >= 3 and vertical >= 3


def test_no_pattern_means_no_engraving(gen: CoasterGenerator) -> None:
    assert gen.make(pattern=Pattern.NONE).parts[0].engrave == []


def test_a_pattern_that_does_not_fit_is_refused(gen: CoasterGenerator) -> None:
    with pytest.raises(ValueError, match="do not fit|leaves nothing"):
        gen.make(size=80, pattern=Pattern.RINGS, pattern_lines=24, pattern_inset=30)


def test_engraving_stays_out_of_the_recess_wall(gen: CoasterGenerator) -> None:
    design = gen.make(
        recess=True, recess_inset=12, recess_depth=2, pattern=Pattern.RINGS,
        pattern_lines=3, pattern_inset=16, thickness=19,
    )
    coaster = design.parts[0]
    region = coaster.pockets[0].region()
    for contour in coaster.engrave:
        assert region.covers(LineString(list(contour.points) + [contour.points[0]]))


def test_a_recess_deeper_than_the_material_is_refused() -> None:
    with pytest.raises(ValueError, match="too little floor"):
        CoasterParams(recess=True, recess_depth=9.0, thickness=12.0, pocket_floor=5.0)


def test_polygon_coaster_corners_are_rounded_for_the_cutter(gen: CoasterGenerator) -> None:
    design = gen.make(shape=CoasterShape.HEX, size=100)
    radius = design.machine.tool_radius
    assert geo.excess_zones(Polygon(design.parts[0].outline), radius, 0.35) == []


def test_all_the_coasters_are_identical(gen: CoasterGenerator) -> None:
    design = gen.make(count=4, holder=False)
    first = geo.signed_area(design.parts[0].outline)
    for part in design.parts[1:]:
        assert geo.signed_area(part.outline) == pytest.approx(first)
