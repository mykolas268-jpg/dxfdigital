"""Tests for the ornament niche."""

from __future__ import annotations

import pytest
from shapely.geometry import Polygon

from dxfgen.core import geometry as geo
from dxfgen.core.layers import ENGRAVE
from dxfgen.niches.ornaments import (
    ORNAMENT_SHEET,
    OrnamentGenerator,
    OrnamentParams,
    OrnamentUse,
)


@pytest.fixture(scope="module")
def gen() -> OrnamentGenerator:
    return OrnamentGenerator()


def test_the_default_set_validates(gen: OrnamentGenerator) -> None:
    design = gen.make()
    assert gen.check(design).ok, gen.check(design).format()
    assert design.machine.is_laser


def test_one_part_per_shape_per_copy(gen: OrnamentGenerator) -> None:
    design = gen.make(shapes=["star", "tree", "heart"], copies=3, width=60)
    assert len(design.parts) == 9


def test_every_ornament_can_be_hung(gen: OrnamentGenerator) -> None:
    design = gen.make(shapes=["star", "tree", "heart", "ghost"], width=75)
    for part in design.parts:
        assert len(part.holes) == 1, part.name
        outline = Polygon(part.outline)
        assert outline.contains(Polygon(part.holes[0]))


def test_the_hole_keeps_its_wall(gen: OrnamentGenerator) -> None:
    params = OrnamentParams(shapes=["heart"], width=80, min_wall=4.0, hole_diameter=5.0)
    design = gen.generate(params)
    part = design.parts[0]
    outline = Polygon(part.outline)
    # Kerf compensation moves both edges by half a kerf in opposite directions.
    assert outline.exterior.distance(Polygon(part.holes[0])) >= 4.0 - params.kerf


def test_a_keychain_gets_a_bigger_hole(gen: OrnamentGenerator) -> None:
    thread = OrnamentParams(use=OrnamentUse.THREAD).resolved_hole()
    ring = OrnamentParams(use=OrnamentUse.RING).resolved_hole()
    assert ring > thread
    assert "split ring" in gen.make(
        use=OrnamentUse.RING, shapes=["heart", "paw"], width=55
    ).description


def test_a_shape_too_small_to_hang_is_refused(gen: OrnamentGenerator) -> None:
    with pytest.raises(ValueError, match="does not fit in this shape"):
        gen.make(shapes=["snowflake"], width=40)


def test_parts_are_kerf_compensated(gen: OrnamentGenerator) -> None:
    # A smooth outline grows by exactly one kerf across.
    plain = gen.generate(OrnamentParams(shapes=["pumpkin"], width=70, kerf=0.0))
    burnt = gen.generate(OrnamentParams(shapes=["pumpkin"], width=70, kerf=0.2))
    assert geo.size_of(burnt.parts[0].outline)[0] == pytest.approx(
        geo.size_of(plain.parts[0].outline)[0] + 0.2, abs=0.01
    )
    # The hanging hole is a cutout, so it goes the other way.
    assert geo.size_of(burnt.parts[0].holes[0])[0] < geo.size_of(plain.parts[0].holes[0])[0]


def test_a_pointed_outline_grows_within_the_mitre_limit() -> None:
    """A sharp point cannot grow by exactly half a kerf on every face.

    Offsetting a 36-degree star tip outward would run the point off to
    infinity, so the mitre is clipped; the extra is bounded, not unbounded.
    """
    gen = OrnamentGenerator()
    plain = gen.generate(OrnamentParams(shapes=["star"], width=70, kerf=0.0))
    burnt = gen.generate(OrnamentParams(shapes=["star"], width=70, kerf=0.2))
    grew = geo.size_of(burnt.parts[0].outline)[0] - geo.size_of(plain.parts[0].outline)[0]
    assert 0.2 <= grew <= 0.2 * 2 + 0.01


def test_parts_do_not_overlap_and_stay_on_the_sheet(gen: OrnamentGenerator) -> None:
    design = gen.make(shapes=["star", "tree", "heart", "moon"], copies=2, width=70)
    placed = [p.placed() for p in design.parts]
    polys = [Polygon(p.outline) for p in placed]
    for index, first in enumerate(polys):
        for second in polys[index + 1 :]:
            assert not first.intersects(second)
    x0, y0, x1, y1 = geo.bbox_of(p.outline for p in placed)
    assert x1 - x0 <= ORNAMENT_SHEET[0]
    assert y1 - y0 <= ORNAMENT_SHEET[1]


def test_a_set_too_big_for_one_sheet_is_refused(gen: OrnamentGenerator) -> None:
    with pytest.raises(ValueError, match="sheets of"):
        gen.make(shapes=["star", "tree", "heart"], copies=12, width=150)


def test_engraving_is_clipped_around_the_hole_not_dropped(gen: OrnamentGenerator) -> None:
    """Dropping any contour that grazes the hole leaves the piece bare."""
    design = gen.make(shapes=["heart", "pumpkin", "ghost"], width=75)
    for part in design.parts:
        assert part.engrave, part.name
        assert all(c.layer == ENGRAVE for c in part.engrave)


def test_engraving_can_be_switched_off(gen: OrnamentGenerator) -> None:
    assert gen.make(shapes=["heart"], width=70, engrave_detail=False).parts[0].engrave == []


def test_every_part_is_labelled(gen: OrnamentGenerator) -> None:
    design = gen.make(shapes=["star", "tree"], copies=2, width=70)
    for part in design.parts:
        assert any(label.text == part.name for label in part.labels)


def test_a_repeated_shape_is_refused() -> None:
    with pytest.raises(ValueError, match="repeats a shape"):
        OrnamentParams(shapes=["star", "star"])


def test_an_unknown_shape_is_refused_with_the_list() -> None:
    with pytest.raises(ValueError, match="available: "):
        OrnamentParams(shapes=["unicorn"])


def test_the_name_reflects_a_single_season(gen: OrnamentGenerator) -> None:
    assert "Christmas" in gen.make(shapes=["star", "tree"], width=70).name
    assert "Mixed" in gen.make(shapes=["star", "pumpkin"], width=70).name


def test_the_notes_state_the_work_is_original(gen: OrnamentGenerator) -> None:
    joined = " ".join(gen.make(shapes=["star"], width=70).notes)
    assert "equations" in joined and "original" in joined
