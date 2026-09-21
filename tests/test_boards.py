"""Tests for the board generator."""

from __future__ import annotations

import pytest
from shapely.geometry import Polygon

from dxfgen.core import geometry as geo
from dxfgen.niches.boards import BoardGenerator, BoardParams, BoardStyle, HangHole


@pytest.fixture(scope="module")
def gen() -> BoardGenerator:
    return BoardGenerator()


@pytest.mark.parametrize("style", list(BoardStyle))
def test_every_style_validates(gen: BoardGenerator, style: BoardStyle) -> None:
    kwargs = {"style": style, "length": 420}
    if style is BoardStyle.ROUND:
        kwargs["juice_groove"] = False
    design = gen.make(**kwargs)
    assert gen.check(design).ok, gen.check(design).format()


def test_the_juice_groove_is_a_channel_with_a_standing_island(gen: BoardGenerator) -> None:
    design = gen.make(juice_groove=True, groove_width=10, groove_inset=25)
    pocket = design.parts[0].pockets[0]
    assert len(pocket.islands) == 1
    outer, inner = Polygon(pocket.ring), Polygon(pocket.islands[0])
    assert outer.contains(inner)
    assert pocket.region().area == pytest.approx(outer.area - inner.area)


def test_the_groove_keeps_its_width_all_the_way_round(gen: BoardGenerator) -> None:
    """Rounding the corners must not pinch the channel."""
    width = 11.0
    design = gen.make(juice_groove=True, groove_width=width, groove_inset=24)
    pocket = design.parts[0].pockets[0]
    outer, inner = Polygon(pocket.ring), Polygon(pocket.islands[0])
    assert outer.exterior.distance(inner.exterior) == pytest.approx(width, abs=0.05)


def test_a_groove_narrower_than_the_cutter_is_refused(gen: BoardGenerator) -> None:
    with pytest.raises(ValueError, match="too narrow for a"):
        gen.make(juice_groove=True, groove_width=5.5, tool_diameter=6.35)


def test_the_groove_corners_are_reachable(gen: BoardGenerator) -> None:
    design = gen.make(juice_groove=True, groove_width=9, groove_inset=22)
    radius = design.machine.tool_radius
    region = design.parts[0].pockets[0].region()
    assert geo.unreachable_zones(region, radius, 0.35) == []


def test_a_groove_that_leaves_no_working_surface_is_refused(gen: BoardGenerator) -> None:
    with pytest.raises(ValueError, match="working surface"):
        gen.make(length=200, width=160, juice_groove=True, groove_inset=50)


def test_the_groove_follows_the_body_not_the_tongue(gen: BoardGenerator) -> None:
    """A groove running down the handle would be a groove down the handle."""
    design = gen.make(
        style=BoardStyle.PADDLE, juice_groove=True, length=520, width=280,
        tongue_length=120, tongue_width=95,
    )
    params = BoardParams(**design.params)
    groove = Polygon(design.parts[0].pockets[0].ring)
    assert geo.bbox(design.parts[0].pockets[0].ring)[2] < params.body_length()


def test_a_hang_hole_in_the_tongue_needs_the_paddle_style() -> None:
    with pytest.raises(ValueError, match="tongue hang hole needs"):
        BoardParams(style=BoardStyle.ROUNDED, hang_hole=HangHole.TONGUE)


def test_an_end_hang_hole_is_refused_when_a_groove_takes_the_rim(
    gen: BoardGenerator,
) -> None:
    with pytest.raises(ValueError, match="use the paddle style"):
        gen.make(juice_groove=True, hang_hole=HangHole.END, hang_hole_diameter=22)


def test_an_end_hang_hole_works_without_a_groove(gen: BoardGenerator) -> None:
    design = gen.make(juice_groove=False, hang_hole=HangHole.END, hang_hole_diameter=22)
    assert len(design.parts[0].holes) == 1
    assert gen.check(design).ok


def test_the_hang_hole_keeps_full_wall_to_the_edge(gen: BoardGenerator) -> None:
    design = gen.make(
        juice_groove=False, hang_hole=HangHole.END, hang_hole_diameter=22, min_wall=10
    )
    part = design.parts[0]
    boundary = Polygon(part.outline).exterior
    assert boundary.distance(Polygon(part.holes[0])) >= 10.0 - 0.05


def test_the_tongue_junction_is_machinable(gen: BoardGenerator) -> None:
    design = gen.make(style=BoardStyle.PADDLE, length=500, width=270)
    radius = design.machine.tool_radius
    assert geo.excess_zones(Polygon(design.parts[0].outline), radius, 0.35) == []


def test_a_tongue_wider_than_the_board_is_refused(gen: BoardGenerator) -> None:
    with pytest.raises(ValueError, match="too wide for a"):
        gen.make(style=BoardStyle.PADDLE, length=500, width=200, tongue_width=190)


def test_a_round_board_must_be_square_in_plan() -> None:
    with pytest.raises(ValueError, match="width equal to length"):
        BoardParams(style=BoardStyle.ROUND, length=300, width=200)


def test_the_readme_notes_explain_the_groove_profile(gen: BoardGenerator) -> None:
    design = gen.make(juice_groove=True)
    joined = " ".join(design.notes)
    assert "round-nose" in joined
    assert "food-safe" in joined


def test_the_name_reflects_what_the_board_is(gen: BoardGenerator) -> None:
    assert "Cutting Board" in gen.make(juice_groove=True).name
    assert "Charcuterie" in gen.make(juice_groove=False).name
    assert "Paddle" in gen.make(style=BoardStyle.PADDLE).name
    assert "Cheese" in gen.make(style=BoardStyle.ROUND, juice_groove=False).name
