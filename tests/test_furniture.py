"""Tests for flat-pack furniture and the sheet nester."""

from __future__ import annotations

import pytest
from shapely.geometry import Polygon

from dxfgen.core import geometry as geo
from dxfgen.core.design import Part
from dxfgen.niches.base import label_parts, nest_parts
from dxfgen.niches.furniture import (
    SHEET,
    FurnitureForm,
    FurnitureGenerator,
    FurnitureParams,
    _tab_spans,
    tenon_reach,
)


@pytest.fixture(scope="module")
def gen() -> FurnitureGenerator:
    return FurnitureGenerator()


# --------------------------------------------------------------------------- #
# the nester
# --------------------------------------------------------------------------- #
def rects(*sizes) -> list[Part]:
    return [
        Part(f"p{index}", geo.rect_ring(w, h), quantity=q)
        for index, (w, h, q) in enumerate(sizes)
    ]


def test_nesting_places_every_copy() -> None:
    sheets = nest_parts(rects((800, 400, 2), (600, 300, 1), (300, 900, 3)), SHEET)
    assert sum(len(sheet) for sheet in sheets) == 6


def test_nested_parts_never_overlap() -> None:
    sheets = nest_parts(rects((800, 400, 2), (600, 300, 2), (300, 900, 3)), SHEET)
    for sheet in sheets:
        placed = [Polygon(p.placed().outline) for p in sheet]
        for index, first in enumerate(placed):
            for second in placed[index + 1 :]:
                assert not first.intersects(second)


def test_nested_parts_stay_on_the_sheet() -> None:
    margin = 10.0
    sheets = nest_parts(rects((800, 400, 3), (300, 900, 2)), SHEET, margin=margin)
    for sheet in sheets:
        for part in sheet:
            x0, y0, x1, y1 = part.placed().bbox()
            assert x0 >= margin - 1e-6 and y0 >= margin - 1e-6
            assert x1 <= SHEET[0] - margin + 1e-6
            assert y1 <= SHEET[1] - margin + 1e-6


def test_copies_are_independent() -> None:
    """dataclasses.replace is shallow, so clones would share their lists."""
    sheets = nest_parts(rects((400, 300, 3)), SHEET)
    parts = [p for sheet in sheets for p in sheet]
    label_parts(parts, height=20.0)
    assert [len(p.labels) for p in parts] == [1, 1, 1]
    assert sorted(p.labels[0].text for p in parts) == ["p0-1", "p0-2", "p0-3"]
    parts[0].holes.append(geo.rect_ring(10, 10, 20, 20))
    assert [len(p.holes) for p in parts] == [1, 0, 0]


def test_copies_are_named_apart() -> None:
    sheets = nest_parts(rects((400, 300, 2), (200, 200, 1)), SHEET)
    names = sorted(p.name for sheet in sheets for p in sheet)
    assert names == ["p0-1", "p0-2", "p1"]


def test_a_part_too_big_for_the_sheet_is_refused() -> None:
    with pytest.raises(ValueError, match="does not fit"):
        nest_parts(rects((1400, 2600, 1)), SHEET)


def test_a_long_part_is_turned_to_fit() -> None:
    sheets = nest_parts(rects((2000, 400, 1)), SHEET)
    part = sheets[0][0]
    assert part.rotation == 90.0
    assert part.placed().size() == pytest.approx((400.0, 2000.0))


def test_nesting_overflows_to_more_sheets() -> None:
    assert len(nest_parts(rects((1100, 700, 8)), SHEET)) > 1


def test_labels_skip_parts_too_small_to_carry_them() -> None:
    parts = rects((400, 300, 1), (10, 10, 1))
    for part in parts:
        part.origin = (0.0, 0.0)
    label_parts(parts, height=20.0)
    assert len(parts[0].labels) == 1 and parts[1].labels == []


# --------------------------------------------------------------------------- #
# the shelf unit
# --------------------------------------------------------------------------- #
def test_the_default_shelf_validates(gen: FurnitureGenerator) -> None:
    design = gen.make()
    assert gen.check(design).ok, gen.check(design).format()
    assert {p.name.split("-")[0] for p in design.parts} == {"upright", "shelf"}


def test_shelf_tabs_line_up_with_the_upright_mortises(gen: FurnitureGenerator) -> None:
    params = FurnitureParams(width=760.0, depth=320.0, height=1200.0, shelves=3)
    design = gen.generate(params)
    upright = next(p for p in design.parts if p.name.startswith("upright"))
    shelf = next(p for p in design.parts if p.name.startswith("shelf"))
    spans = _tab_spans(params, params.depth)

    mortise_y = sorted({round(geo.bbox(h)[1], 2) for h in upright.holes})
    assert len(mortise_y) == params.shelves

    # Shelf tabs stand proud on the left edge; their spans must match.
    tab_y = sorted({round(y, 2) for x, y in shelf.outline if x < -0.001})
    assert tab_y[0] == pytest.approx(spans[0][0], abs=0.01)
    assert tab_y[-1] == pytest.approx(spans[-1][1], abs=0.01)


def test_the_mortise_gives_the_tab_its_clearance(gen: FurnitureGenerator) -> None:
    params = FurnitureParams(clearance=0.3, tab_width=70.0, thickness=18.0)
    design = gen.generate(params)
    upright = next(p for p in design.parts if p.name.startswith("upright"))
    length, width = geo.size_of(upright.holes[0])
    assert length >= 70.0 + 0.3 - 0.01
    assert width >= 18.0 + 0.3 - 0.01


def test_shelf_tabs_protrude_by_one_thickness(gen: FurnitureGenerator) -> None:
    params = FurnitureParams(thickness=18.0)
    design = gen.generate(params)
    shelf = next(p for p in design.parts if p.name.startswith("shelf"))
    assert geo.size_of(shelf.outline)[0] == pytest.approx(params.width, abs=0.5)


def test_the_bottom_shelf_clears_the_foot_arch(gen: FurnitureGenerator) -> None:
    params = FurnitureParams(foot_arch=80.0, height=1400.0, shelves=3)
    design = gen.generate(params)
    upright = next(p for p in design.parts if p.name.startswith("upright"))
    lowest = min(geo.bbox(h)[1] for h in upright.holes)
    assert lowest >= 80.0 + params.min_wall - 0.5


def test_too_many_shelves_for_the_height_is_refused(gen: FurnitureGenerator) -> None:
    # 8 shelves in 350 mm leaves 43 mm of headroom, below one thickness plus
    # four walls; the same count in 400 mm is a spice rack and is allowed.
    with pytest.raises(ValueError, match="leaves only"):
        gen.make(height=350.0, depth=300.0, shelves=8, foot_arch=0.0)
    assert gen.make(height=400.0, depth=300.0, shelves=8, foot_arch=0.0)


def test_a_shelf_deeper_than_it_is_tall_is_refused() -> None:
    with pytest.raises(ValueError, match="sideboard"):
        FurnitureParams(height=300.0, depth=500.0)


# --------------------------------------------------------------------------- #
# the table
# --------------------------------------------------------------------------- #
def test_the_table_validates(gen: FurnitureGenerator) -> None:
    design = gen.make(form=FurnitureForm.TABLE, width=900, depth=500, height=450)
    assert gen.check(design).ok, gen.check(design).format()
    assert {p.name for p in design.parts} == {"top", "leg-long", "leg-short"}


def test_the_legs_cross_lap_from_opposite_edges(gen: FurnitureGenerator) -> None:
    params = FurnitureParams(form=FurnitureForm.TABLE, width=900, depth=500, height=450)
    design = gen.generate(params)
    long_leg = next(p for p in design.parts if p.name == "leg-long")
    short_leg = next(p for p in design.parts if p.name == "leg-short")
    # One slot is open at the top, the other at the bottom, so they slide
    # together; both open at the same edge would not assemble.
    long_top = max(y for _, y in long_leg.outline)
    short_bottom = min(y for _, y in short_leg.outline)
    assert geo.signed_area(long_leg.outline) > 0
    assert geo.signed_area(short_leg.outline) > 0
    assert long_top > short_bottom


def test_leg_tabs_are_square_not_rounded(gen: FurnitureGenerator) -> None:
    """Rounding the panel after adding tabs rounds the tabs too."""
    params = FurnitureParams(
        form=FurnitureForm.TABLE, width=900, depth=500, height=450, corner_radius=16.0
    )
    design = gen.generate(params)
    leg = next(p for p in design.parts if p.name == "leg-long")
    top = max(y for _, y in leg.outline)
    flat = [x for x, y in leg.outline if abs(y - top) < 1e-6]
    # A square tab contributes two vertices at its full height; a rounded one
    # contributes a run of arc points.
    assert len(flat) == 2 * params.tabs


def test_every_inside_corner_is_machinable(gen: FurnitureGenerator) -> None:
    for kwargs in (
        {},
        {"foot_arch": 0.0},
        {"form": FurnitureForm.TABLE, "width": 900, "depth": 500, "height": 450},
    ):
        design = gen.make(**kwargs)
        radius = design.machine.tool_radius
        for part in design.parts:
            assert geo.excess_zones(Polygon(part.outline), radius, 0.35) == [], (
                kwargs, part.name
            )


# --------------------------------------------------------------------------- #
# sheets
# --------------------------------------------------------------------------- #
def test_every_part_is_labelled(gen: FurnitureGenerator) -> None:
    design = gen.make()
    for part in design.parts:
        assert any(label.text == part.name for label in part.labels)


def test_sheet_markers_sit_inside_the_drawing_extents(gen: FurnitureGenerator) -> None:
    design = gen.make(width=760, depth=320, height=1200, shelves=4)
    x0, y0, x1, y1 = design.bbox()
    markers = [
        label
        for part in design.placed_parts()
        for label in part.labels
        if label.text.startswith("SHEET")
    ]
    assert markers
    for marker in markers:
        assert x0 <= marker.position[0] <= x1
        assert y0 <= marker.position[1] <= y1


def test_a_design_needing_two_sheets_says_so(gen: FurnitureGenerator) -> None:
    design = gen.make(width=760, depth=320, height=1200, shelves=4)
    assert "2 sheets" in design.description
    assert any("side by side" in note for note in design.notes)


# --------------------------------------------------------------------------- #
# wedged through-tenons
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def wedged(gen: FurnitureGenerator):
    return gen.make(
        form="shelf", width=760, depth=320, height=1000, shelves=3, wedges=True
    )


def test_a_friction_tenon_finishes_flush(gen: FurnitureGenerator) -> None:
    params = gen.parse({"form": "shelf"})
    assert tenon_reach(params) == params.thickness


def test_a_wedged_tenon_stands_proud(gen: FurnitureGenerator) -> None:
    """It has to carry its slot plus material beyond it, so it protrudes."""
    params = gen.parse({"form": "shelf", "wedges": True})
    assert tenon_reach(params) > params.thickness + params.wedge_slot


def test_a_wedged_shelf_carries_a_slot_per_tenon(
    gen: FurnitureGenerator, wedged
) -> None:
    params = gen.parse(wedged.params)
    shelf = next(p for p in wedged.parts if p.name.startswith("shelf"))
    assert len(shelf.holes) == 2 * params.tabs, "both ends, every tab"


def test_an_unwedged_shelf_has_no_slots(gen: FurnitureGenerator) -> None:
    design = gen.make(form="shelf", width=760, depth=320, height=1000, shelves=3)
    shelf = next(p for p in design.parts if p.name.startswith("shelf"))
    assert shelf.holes == []


def test_the_slot_straddles_the_upright_face(gen: FurnitureGenerator, wedged) -> None:
    """The whole mechanism: a slot flush with the upright has no pull in it.

    The slot must reach back inside the upright's outer face by the bite, so
    driving the wedge can still move the tenon outwards.
    """
    params = gen.parse(wedged.params)
    shelf = next(p for p in wedged.parts if p.name.startswith("shelf"))
    reach = tenon_reach(params)
    # In shelf coordinates the right-hand upright's outer face sits one
    # thickness out from the shoulder.
    inner_width = params.width - 2 * params.thickness
    face = inner_width + params.thickness
    right = [h for h in shelf.holes if geo.centroid(h)[0] > inner_width / 2]
    assert right
    for hole in right:
        x0, _, x1, _ = geo.bbox(hole)
        assert x0 < face, "the slot must start inside the upright's outer face"
        # The bounding box includes the dogbone relief, which overshoots the
        # corner on purpose, so the bite is a floor rather than an equality.
        bite = face - x0
        assert params.wedge_bite - 0.05 <= bite <= (
            params.wedge_bite + params.relief_margin() + 0.05
        ), f"slot reaches {bite:.2f} mm inside the face"
        assert x1 < inner_width + reach, "and finish inside the tenon"


def test_a_wedge_is_cut_for_every_tenon(gen: FurnitureGenerator, wedged) -> None:
    params = gen.parse(wedged.params)
    wedges = [p for p in wedged.parts if p.name.startswith("wedge")]
    assert len(wedges) == params.shelves * 2 * params.tabs


def test_a_wedge_tapers(gen: FurnitureGenerator, wedged) -> None:
    """A parallel key is only ever as tight as it was cut."""
    wedge = next(p for p in wedged.parts if p.name.startswith("wedge"))
    points = sorted(wedge.outline, key=lambda p: p[1])
    low = max(x for x, y in points if y < 2.0)
    high = max(x for x, y in points if y > geo.bbox(wedge.outline)[3] * 0.6)
    assert high > low, "the wedge must be wider where it is driven from"


def test_a_wedge_fits_its_slot(gen: FurnitureGenerator, wedged) -> None:
    params = gen.parse(wedged.params)
    wedge = next(p for p in wedged.parts if p.name.startswith("wedge"))
    shelf = next(p for p in wedged.parts if p.name.startswith("shelf"))
    slot_length = geo.bbox(shelf.holes[0])[2] - geo.bbox(shelf.holes[0])[0]
    widest = geo.bbox(wedge.outline)[2] - geo.bbox(wedge.outline)[0]
    assert widest > slot_length, "the collar must not pass through"
    driven = max(x for x, y in wedge.outline if y < geo.bbox(wedge.outline)[3] * 0.8)
    assert driven <= slot_length + 0.6, "but the working length must"


def test_a_wedged_unit_validates(gen: FurnitureGenerator, wedged) -> None:
    report = gen.check(wedged)
    assert report.ok, report.format()


def test_the_readme_says_the_tenons_stand_proud(wedged) -> None:
    joined = " ".join(wedged.notes)
    assert "carcass" in joined and "proud" in joined
    assert "Tap them again" in joined


def test_wedging_is_in_the_slug(gen: FurnitureGenerator, wedged) -> None:
    assert "wedged" in wedged.slug
    plain = gen.make(form="shelf", width=760, depth=320, height=1000, shelves=3)
    assert "friction" in plain.slug


def test_a_table_is_never_wedged(gen: FurnitureGenerator) -> None:
    """Cross-lapped legs are held by their own geometry; nothing to pull."""
    for seed in (1, 7, 31):
        for design in gen.variants(10, seed=seed):
            if design.params["form"] == "table":
                assert design.params["wedges"] is False
