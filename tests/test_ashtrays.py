"""Tests for the cigar and whiskey tray generator."""

from __future__ import annotations

import pytest
from shapely.geometry import Polygon

from dxfgen.core import geometry as geo
from dxfgen.niches.ashtrays import (
    MIN_REST_LENGTH,
    AshtrayGenerator,
    AshtrayParams,
    TrayShape,
)


@pytest.fixture(scope="module")
def gen() -> AshtrayGenerator:
    return AshtrayGenerator()


@pytest.mark.parametrize("shape", list(TrayShape))
@pytest.mark.parametrize("rests", [1, 2, 3])
def test_every_shape_and_rest_count_validates(
    gen: AshtrayGenerator, shape: TrayShape, rests: int
) -> None:
    design = gen.make(shape=shape, rests=rests, length=420, width=280)
    assert gen.check(design).ok, gen.check(design).format()


def test_the_glass_recess_is_where_it_says_it_is(gen: AshtrayGenerator) -> None:
    diameter = 92.0
    design = gen.make(glass_diameter=diameter, length=420, width=260)
    glass = design.parts[0].pockets[0]
    width, height = geo.size_of(glass.ring)
    assert width == pytest.approx(diameter, abs=0.1)
    assert height == pytest.approx(diameter, abs=0.1)


def test_the_ash_well_is_a_separate_deeper_pocket(gen: AshtrayGenerator) -> None:
    design = gen.make(ash_well=True, length=460, width=280, glass_depth=6, well_depth=14)
    depths = sorted({p.depth for p in design.parts[0].pockets})
    assert 6.0 in depths and 14.0 in depths
    assert "POCKET_14" in design.layers_used()


def test_an_ash_well_no_deeper_than_the_glass_is_refused(gen: AshtrayGenerator) -> None:
    with pytest.raises(ValueError, match="no deeper than"):
        gen.make(ash_well=True, length=460, width=280, glass_depth=10, well_depth=8)


def test_the_rests_are_cigar_length_not_tray_length(gen: AshtrayGenerator) -> None:
    """A groove running the whole tray is a pen rest, not a cigar rest."""
    design = gen.make(length=560, width=300, rest_length=90)
    # Rests are the elongated pockets; the glass recess and ash well are round.
    rests = [
        p
        for p in design.parts[0].pockets
        if geo.size_of(p.ring)[0] > geo.size_of(p.ring)[1] * 1.5
    ]
    assert rests
    for rest in rests:
        assert geo.size_of(rest.ring)[0] == pytest.approx(90.0, abs=0.5)


def test_the_rests_are_spread_across_the_width(gen: AshtrayGenerator) -> None:
    design = gen.make(rests=3, length=440, width=300)
    centres = sorted(
        geo.centroid(p.ring)[1]
        for p in design.parts[0].pockets
        if geo.size_of(p.ring)[0] > geo.size_of(p.ring)[1] * 1.5
    )
    assert len(centres) == 3
    gaps = [b - a for a, b in zip(centres, centres[1:])]
    assert gaps[0] == pytest.approx(gaps[1], rel=0.05)


def test_a_tray_too_short_for_a_cigar_is_refused(gen: AshtrayGenerator) -> None:
    with pytest.raises(ValueError, match=f"{MIN_REST_LENGTH:g} mm a cigar needs"):
        gen.make(length=200, width=140, glass_diameter=140)


def test_rests_that_do_not_fit_across_the_width_are_refused(gen: AshtrayGenerator) -> None:
    with pytest.raises(ValueError, match="mm across"):
        gen.make(length=420, width=130, rests=4, rest_width=38)


def test_features_keep_full_wall_between_them(gen: AshtrayGenerator) -> None:
    design = gen.make(ash_well=True, length=480, width=300, min_wall=12)
    regions = [Polygon(p.ring) for p in design.parts[0].pockets]
    for index, first in enumerate(regions):
        for second in regions[index + 1 :]:
            assert first.distance(second) >= 12.0 - 0.05


def test_every_pocket_is_reachable_by_the_cutter(gen: AshtrayGenerator) -> None:
    design = gen.make(ash_well=True, length=460, width=280)
    radius = design.machine.tool_radius
    for pocket in design.parts[0].pockets:
        assert geo.unreachable_zones(pocket.region(), radius, 0.35) == []


def test_the_deepest_cut_leaves_a_floor(gen: AshtrayGenerator) -> None:
    design = gen.make(
        ash_well=True, thickness=25.0, well_depth=14.0, length=460, width=280
    )
    assert max(design.pocket_depths()) <= 25.0 - design.limits.pocket_floor


def test_the_notes_state_the_glass_size_and_floor(gen: AshtrayGenerator) -> None:
    design = gen.make(glass_diameter=90, thickness=25)
    joined = " ".join(design.notes)
    assert "88 mm across" in joined
    assert "floor" in joined
