"""Tests for the tray generator.

The interesting assertions are the ones about *fitness for sale*: that every
combination of options actually validates, that a handle a hand can use is
never silently shrunk below that, that wall thicknesses come out where the
design rules say they should, and that the quoted dimensions match the file.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from shapely.geometry import Polygon

from dxfgen.core import geometry as geo
from dxfgen.core.layers import CUT_INSIDE, CUT_OUTSIDE, ENGRAVE, INFO
from dxfgen.niches.trays import (
    MAX_CELL_ASPECT,
    MIN_CELL,
    MIN_RECESS_FRACTION,
    HandleStyle,
    Layout,
    TrayGenerator,
    TrayParams,
    TrayStyle,
)

EXPECTED_CELLS = {
    Layout.SINGLE: 1,
    Layout.HALVES: 2,
    Layout.THIRDS: 3,
    Layout.GOLDEN: 2,
    Layout.GRID: 4,
    Layout.ONE_PLUS_TWO: 3,
}


@pytest.fixture(scope="module")
def gen() -> TrayGenerator:
    return TrayGenerator()


# --------------------------------------------------------------------------- #
# every option combination must produce a sellable file
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("style", list(TrayStyle))
@pytest.mark.parametrize("layout", list(Layout))
@pytest.mark.parametrize("handle", list(HandleStyle))
def test_every_combination_validates(
    gen: TrayGenerator, style: TrayStyle, layout: Layout, handle: HandleStyle
) -> None:
    if style is TrayStyle.PILL and handle is HandleStyle.SCALLOP:
        pytest.skip(
            "a pill outline has no flat end for a scallop to bite into; "
            "refusing that pair is tested separately"
        )
    design = gen.make(
        style=style, layout=layout, handle=handle, length=520, width=330,
        scallop_width=44,
    )
    report = gen.check(design)
    assert report.ok, report.format()
    assert len(design.parts[0].pockets) == EXPECTED_CELLS[layout]


def test_the_default_tray_is_valid(gen: TrayGenerator) -> None:
    design = gen.make()
    assert gen.check(design).ok
    assert design.layers_used() == [CUT_INSIDE, CUT_OUTSIDE, "POCKET_8", INFO]


# --------------------------------------------------------------------------- #
# proportion rules
# --------------------------------------------------------------------------- #
def test_derived_width_follows_the_golden_ratio_and_is_whole(gen: TrayGenerator) -> None:
    design = gen.make(length=450)
    assert design.size() == pytest.approx((450.0, 278.0))


def test_quoted_dimensions_match_the_geometry(gen: TrayGenerator) -> None:
    """Scallops shorten the outline, so the name must follow the geometry."""
    # A soft outline touches its bounding box over a very short run, which the
    # scallop bites straight through, so the overall length really does shrink.
    # On a rounded rectangle the long straight edge holds the box out.
    design = gen.make(
        length=400, width=260, style=TrayStyle.SOFT, handle=HandleStyle.SCALLOP,
        scallop_depth=12, scallop_width=40,
    )
    real = f"{design.size()[0]:.0f} x {design.size()[1]:.0f} mm"
    assert real in design.name
    assert real in design.description
    assert design.size()[0] < 400.0  # the scallops really did bite


def test_a_rim_that_would_dominate_the_tray_is_refused() -> None:
    params = TrayParams(length=400, width=200, handle=HandleStyle.CUTOUT, handle_width=40)
    with pytest.raises(ValueError, match="of the 200 mm width"):
        params.resolved_border()


def test_the_recess_fraction_rule_is_the_stated_one() -> None:
    width = 400.0
    params = TrayParams(
        length=700, width=width, handle=HandleStyle.NONE,
        border=width * (1 - MIN_RECESS_FRACTION) / 2 - 0.5,
    )
    assert params.resolved_border() > 0
    too_wide = TrayParams(
        length=700, width=width, handle=HandleStyle.NONE,
        border=width * (1 - MIN_RECESS_FRACTION) / 2 + 1.0,
    )
    with pytest.raises(ValueError, match="look like a"):
        too_wide.resolved_border()


def test_corner_radius_scales_with_the_tray() -> None:
    small = TrayParams(length=240, width=150).resolved_corner_radius()
    large = TrayParams(length=600, width=380).resolved_corner_radius()
    assert small < large
    assert small == pytest.approx(15.0)  # 10% of width


def test_pill_style_is_fully_rounded() -> None:
    params = TrayParams(length=400, width=200, style=TrayStyle.PILL)
    assert params.resolved_corner_radius() == pytest.approx(100.0)


def test_divider_cannot_go_below_min_wall() -> None:
    with pytest.raises(ValueError, match="below min_wall"):
        TrayParams(divider=7.0, min_wall=10.0).resolved_divider()


def test_compartment_corner_radius_never_drops_below_the_cutter() -> None:
    params = TrayParams(tool_diameter=12.7, pocket_radius=2.0)
    assert params.resolved_pocket_radius() >= params.tool_diameter * 0.8


# --------------------------------------------------------------------------- #
# handles
# --------------------------------------------------------------------------- #
def test_handle_cutout_dimensions_are_ergonomically_bounded() -> None:
    """A hand needs 90 x 30 mm; the schema will not accept less."""
    with pytest.raises(ValidationError):
        TrayParams(handle_length=80)
    with pytest.raises(ValidationError):
        TrayParams(handle_width=25)


def test_the_rim_is_widened_to_carry_the_handle() -> None:
    params = TrayParams(length=560, width=360, handle=HandleStyle.CUTOUT, handle_width=34, min_wall=8)
    assert params.resolved_border() == pytest.approx(34 + 2 * 8)


def test_handle_walls_are_uniform_on_a_curved_rim(gen: TrayGenerator) -> None:
    """The slot follows the rim, so the wall is the same at every point."""
    design = gen.make(
        length=520, width=330, style=TrayStyle.PILL, handle=HandleStyle.CUTOUT,
        handle_width=32, min_wall=8,
    )
    part = design.parts[0]
    outline = Polygon(part.outline).exterior
    assert len(part.holes) == 2
    for hole in part.holes:
        slot = Polygon(hole)
        assert outline.distance(slot) == pytest.approx(8.0, abs=0.05)
        for pocket in part.pockets:
            assert Polygon(pocket.ring).distance(slot) >= 8.0 - 0.05


def test_handle_slot_width_is_what_was_asked_for(gen: TrayGenerator) -> None:
    design = gen.make(
        length=520, width=330, handle=HandleStyle.CUTOUT, handle_length=120, handle_width=36
    )
    hole = design.parts[0].holes[0]
    width, height = geo.size_of(hole)
    assert min(width, height) == pytest.approx(36.0, abs=0.1)


def test_a_tray_too_narrow_for_a_handle_says_so(gen: TrayGenerator) -> None:
    with pytest.raises(ValueError):
        gen.make(length=300, width=150, handle=HandleStyle.CUTOUT)


def test_scallops_stay_machinable(gen: TrayGenerator) -> None:
    design = gen.make(
        length=420, width=260, handle=HandleStyle.SCALLOP, scallop_depth=18, scallop_width=90
    )
    report = gen.check(design)
    assert "E_PROFILE_TOO_TIGHT" not in {i.code for i in report.issues}


def test_a_scallop_needs_a_flat_end_to_bite_into(gen: TrayGenerator) -> None:
    """Otherwise it eats the whole end and the tray reads as a dog bone."""
    with pytest.raises(ValueError, match="straight end to bite into"):
        gen.make(
            length=400, width=260, style=TrayStyle.PILL, handle=HandleStyle.SCALLOP,
            scallop_width=40,
        )
    with pytest.raises(ValueError, match="straight end to bite into"):
        gen.make(
            length=400, width=260, style=TrayStyle.SOFT, handle=HandleStyle.SCALLOP,
            scallop_width=70,
        )
    # The rounded outline runs straight for most of its end, so it is fine.
    assert gen.make(
        length=400, width=260, style=TrayStyle.ROUNDED, handle=HandleStyle.SCALLOP,
        scallop_width=70,
    )


def test_the_straight_run_measurement_separates_the_outline_styles() -> None:
    from dxfgen.niches.trays import _end_straight_run, _outline

    runs = {
        style: _end_straight_run(
            _outline(TrayParams(length=400, width=260, style=style))
        )
        for style in TrayStyle
    }
    assert runs[TrayStyle.ROUNDED] > 200.0
    assert 20.0 < runs[TrayStyle.SOFT] < 80.0
    assert runs[TrayStyle.PILL] < 20.0


def test_a_scallop_too_tight_for_the_cutter_is_refused(gen: TrayGenerator) -> None:
    with pytest.raises(ValueError, match="too tight for a"):
        gen.make(
            length=420, width=260, handle=HandleStyle.SCALLOP,
            scallop_depth=40, scallop_width=40, tool_diameter=25.4,
        )


# --------------------------------------------------------------------------- #
# compartments
# --------------------------------------------------------------------------- #
def test_compartments_keep_the_divider_width_apart(gen: TrayGenerator) -> None:
    divider = 16.0
    design = gen.make(length=520, width=330, layout=Layout.HALVES, divider=divider)
    a, b = (Polygon(p.ring) for p in design.parts[0].pockets)
    assert a.distance(b) == pytest.approx(divider, abs=0.05)


def test_compartments_keep_the_rim_width_from_the_edge(gen: TrayGenerator) -> None:
    border = 40.0
    design = gen.make(
        length=520, width=330, border=border, handle=HandleStyle.NONE, layout=Layout.GRID
    )
    outline = Polygon(design.parts[0].outline).exterior
    for pocket in design.parts[0].pockets:
        assert outline.distance(Polygon(pocket.ring)) == pytest.approx(border, abs=0.1)


def test_depth_step_creates_a_second_pocket_layer(gen: TrayGenerator) -> None:
    design = gen.make(
        length=520, width=330, layout=Layout.ONE_PLUS_TWO, pocket_depth=10, depth_step=3
    )
    assert design.pocket_depths() == [7.0, 10.0]
    assert "POCKET_7" in design.layers_used()
    assert "POCKET_10" in design.layers_used()


def test_depth_step_cannot_exceed_the_depth() -> None:
    with pytest.raises(ValidationError):
        TrayParams(pocket_depth=6.0, depth_step=6.0)


def test_too_many_compartments_for_the_tray_is_refused(gen: TrayGenerator) -> None:
    with pytest.raises(ValueError, match=f"{MIN_CELL:g} mm minimum|do not fit"):
        gen.make(length=160, width=120, layout=Layout.GRID, handle=HandleStyle.NONE)


def test_a_grid_that_does_fit_is_allowed(gen: TrayGenerator) -> None:
    design = gen.make(length=240, width=160, layout=Layout.GRID, handle=HandleStyle.NONE)
    assert len(design.parts[0].pockets) == 4
    assert gen.check(design).ok


def test_slot_shaped_compartments_are_refused(gen: TrayGenerator) -> None:
    with pytest.raises(ValueError, match="reads as a slot"):
        gen.make(
            length=900, width=200, layout=Layout.SINGLE, handle=HandleStyle.NONE
        )


def test_compartment_corners_are_cutter_friendly(gen: TrayGenerator) -> None:
    design = gen.make(length=520, width=330, layout=Layout.GRID, handle=HandleStyle.NONE)
    radius = design.machine.tool_radius
    for pocket in design.parts[0].pockets:
        assert geo.unreachable_zones(Polygon(pocket.ring), radius, 0.35) == []


# --------------------------------------------------------------------------- #
# extras and metadata
# --------------------------------------------------------------------------- #
def test_engraved_border_lands_on_the_engrave_layer(gen: TrayGenerator) -> None:
    design = gen.make(length=460, width=290, engrave_border=True, engrave_inset=8)
    contours = design.parts[0].engrave
    assert contours and all(c.layer == ENGRAVE for c in contours)
    outline = Polygon(design.parts[0].outline).exterior
    assert outline.distance(Polygon(contours[0].points).exterior) == pytest.approx(8.0, abs=0.1)


def test_an_engraved_border_needs_a_rim_to_sit_on(gen: TrayGenerator) -> None:
    with pytest.raises(ValueError, match="no room for an engraved border"):
        gen.make(
            length=300, width=200, handle=HandleStyle.NONE, border=14,
            engrave_border=True, engrave_inset=20, tool_diameter=12.7,
        )


def test_the_info_label_reports_the_real_size_and_material(gen: TrayGenerator) -> None:
    design = gen.make(length=460, width=290, material="walnut", thickness=19)
    label = design.parts[0].labels[0]
    assert label.layer == INFO
    assert "460 x 290 x 19 mm" in label.text and "walnut" in label.text


def test_the_label_can_be_switched_off(gen: TrayGenerator) -> None:
    assert gen.make(label=False).parts[0].labels == []


def test_cutting_order_and_notes_are_populated(gen: TrayGenerator) -> None:
    design = gen.make(length=460, width=290)
    assert design.cutting_order[0].startswith("Pocket POCKET_8")
    assert CUT_OUTSIDE in design.cutting_order[-2]
    assert any("floor" in note for note in design.notes)
    assert any("Rim width" in note for note in design.notes)


def test_params_are_recorded_for_regeneration(gen: TrayGenerator) -> None:
    design = gen.make(length=460, width=290, layout=Layout.HALVES)
    assert design.params["length"] == 460.0
    assert design.params["layout"] == "halves"
    rebuilt = gen.make(**design.params)
    assert rebuilt.slug == design.slug
    assert rebuilt.size() == pytest.approx(design.size())


def test_stock_size_is_the_smallest_board_that_fits(gen: TrayGenerator) -> None:
    assert gen.make(length=260, width=180, handle=HandleStyle.NONE).sheet == (300.0, 200.0)
    assert gen.make(length=460, width=290).sheet == (600.0, 400.0)


def test_laser_mode_trays_validate(gen: TrayGenerator) -> None:
    design = gen.make(
        mode="laser", thickness=6.0, pocket_depth=2.0, pocket_floor=2.0,
        length=300, width=200, handle=HandleStyle.NONE, material="6 mm birch ply",
    )
    report = gen.check(design)
    assert report.ok, report.format()


# --------------------------------------------------------------------------- #
# variants
# --------------------------------------------------------------------------- #
def test_a_design_carries_the_limits_it_was_built_to(gen: TrayGenerator) -> None:
    """Otherwise the exporter would judge every design by the router defaults."""
    design = gen.make(
        mode="laser", thickness=6.0, pocket_depth=2.0, pocket_floor=2.0,
        min_wall=5.0, length=300, width=200, handle=HandleStyle.NONE,
    )
    assert design.limits.pocket_floor == 2.0
    assert design.limits.min_wall == 5.0


def test_variants_are_reproducible(gen: TrayGenerator) -> None:
    assert [d.slug for d in gen.variants(6, seed=99)] == [
        d.slug for d in gen.variants(6, seed=99)
    ]


def test_variants_are_all_valid_and_distinct(gen: TrayGenerator) -> None:
    designs = gen.variants(20, seed=4)
    assert len(designs) == 20
    assert len({d.slug for d in designs}) == 20
    for design in designs:
        assert gen.check(design).ok, design.slug


def test_variants_only_offer_hand_slots_where_they_fit(gen: TrayGenerator) -> None:
    for design in gen.variants(25, seed=8):
        if design.params["handle"] != HandleStyle.CUTOUT.value:
            continue
        params = TrayParams(**design.params)
        width = params.resolved_width()
        recess = (width - 2 * params.resolved_border()) / width
        assert recess >= MIN_RECESS_FRACTION
        assert params.handle_width >= 30.0
