"""Tests for the manufacturability checks.

Each test drives one check to failure with the smallest design that can do so,
and one test proves a well formed design passes everything.
"""

from __future__ import annotations

import pytest

from dxfgen.core import geometry as g
from dxfgen.core.design import (
    Contour,
    Design,
    Drill,
    Label,
    Machine,
    Mode,
    Part,
    Pocket,
)
from dxfgen.core.layers import CUT_OUTSIDE, ENGRAVE
from dxfgen.core.validate import (
    Report,
    Severity,
    ValidationConfig,
    ValidationError,
    validate_design,
)


def design_of(part: Part, **kwargs) -> Design:
    kwargs.setdefault("machine", Machine())
    kwargs.setdefault("thickness", 19.0)
    kwargs.setdefault("sheet", (600.0, 400.0))
    return Design("d", "D", "trays", "desc", [part], **kwargs)


def codes(report: Report) -> set[str]:
    return {issue.code for issue in report.issues}


def good_part() -> Part:
    """A tray that satisfies every rule, used as the baseline."""
    return Part(
        name="tray",
        outline=g.rounded_rect_ring(300, 200, 20),
        holes=[g.stadium_ring(95, 32, 30, 84)],
        pockets=[Pocket(g.rounded_rect_ring(140, 150, 15, 145, 25), 8.0)],
        drills=[Drill((20, 30), 6.0)],
        labels=[Label("OAK 19 mm", (150, 8), 6.0, align="center")],
    )


# --------------------------------------------------------------------------- #
# the baseline must pass
# --------------------------------------------------------------------------- #
def test_a_well_formed_design_passes_every_check() -> None:
    report = validate_design(design_of(good_part()))
    assert report.ok, report.format()
    assert report.warnings == []
    assert report.reason() == "ok"


# --------------------------------------------------------------------------- #
# contour hygiene
# --------------------------------------------------------------------------- #
def test_closing_duplicate_point_is_rejected() -> None:
    ring = g.rect_ring(200, 100)
    part = Part("p", ring + [ring[0]])
    assert "E_CLOSING_DUPLICATE" in codes(validate_design(design_of(part)))


def test_zero_length_segment_is_rejected() -> None:
    part = Part("p", [(0, 0), (0, 0.001), (200, 0), (200, 100), (0, 100)])
    assert "E_ZERO_SEGMENT" in codes(validate_design(design_of(part)))


def test_self_intersecting_outline_is_rejected() -> None:
    part = Part("p", [(0, 0), (100, 100), (100, 0), (0, 100)])
    assert "E_SELF_INTERSECT" in codes(validate_design(design_of(part)))


def test_duplicate_contour_is_rejected() -> None:
    hole = g.circle_ring(150, 100, 20)
    part = Part("p", g.rect_ring(300, 200), holes=[hole, list(hole)])
    assert "E_DUPLICATE_CONTOUR" in codes(validate_design(design_of(part)))


def test_duplicate_detection_ignores_start_vertex_and_direction() -> None:
    hole = g.circle_ring(150, 100, 20)
    rotated = hole[7:] + hole[:7]
    part = Part("p", g.rect_ring(300, 200), holes=[hole, rotated[::-1]])
    assert "E_DUPLICATE_CONTOUR" in codes(validate_design(design_of(part)))


def test_tiny_feature_warns() -> None:
    part = Part("p", g.rect_ring(300, 200), holes=[g.circle_ring(150, 100, 0.8)])
    report = validate_design(design_of(part))
    assert "W_TINY_FEATURE" in codes(report)


def test_dense_contour_warns() -> None:
    part = Part("p", g.circle_ring(150, 100, 140, tolerance=0.001))
    report = validate_design(
        design_of(part), ValidationConfig(max_points_per_contour=50)
    )
    assert "W_DENSE_CONTOUR" in codes(report)


def test_degenerate_engrave_contour_is_rejected() -> None:
    part = Part(
        "p",
        g.rect_ring(300, 200),
        engrave=[Contour([(10.0, 10.0)], ENGRAVE, closed=False)],
    )
    assert "E_DEGENERATE" in codes(validate_design(design_of(part)))


def test_text_on_a_cut_layer_is_rejected() -> None:
    part = Part(
        "p", g.rect_ring(300, 200), labels=[Label("cut me", (50, 50), layer=CUT_OUTSIDE)]
    )
    assert "E_TEXT_ON_CUT_LAYER" in codes(validate_design(design_of(part)))


# --------------------------------------------------------------------------- #
# layout
# --------------------------------------------------------------------------- #
def test_origin_offset_is_rejected() -> None:
    part = Part("p", g.rect_ring(200, 100), origin=(25, 0))
    assert "E_ORIGIN" in codes(validate_design(design_of(part)))


def test_validate_design_does_not_silently_normalise() -> None:
    """The exporter normalises; the validator reports what it is given."""
    part = Part("p", g.rect_ring(200, 100), origin=(25, 0))
    report = validate_design(design_of(part))
    assert not report.ok
    assert validate_design(design_of(part).normalized()).ok


def test_sheet_overflow_is_rejected_in_both_orientations() -> None:
    part = Part("p", g.rect_ring(700, 500))
    assert "E_SHEET_OVERFLOW" in codes(validate_design(design_of(part)))


def test_a_rotatable_layout_fits_the_sheet() -> None:
    part = Part("p", g.rect_ring(380, 550))  # fits 600x400 rotated
    assert "E_SHEET_OVERFLOW" not in codes(validate_design(design_of(part)))


def test_overlapping_parts_are_rejected() -> None:
    a = Part("a", g.rect_ring(100, 100))
    b = Part("b", g.rect_ring(100, 100), origin=(50, 0))
    design = Design("d", "D", "trays", "x", [a, b], sheet=(600, 400), thickness=19)
    assert "E_PART_OVERLAP" in codes(validate_design(design))


def test_feature_outside_the_part_is_rejected() -> None:
    part = Part("p", g.rect_ring(200, 100), holes=[g.circle_ring(250, 50, 10)])
    assert "E_FEATURE_OUTSIDE" in codes(validate_design(design_of(part)))


def test_drill_outside_the_part_is_rejected() -> None:
    part = Part("p", g.rect_ring(200, 100), drills=[Drill((199, 50), 6.0)])
    assert "E_FEATURE_OUTSIDE" in codes(validate_design(design_of(part)))


def test_thin_wall_to_the_edge_is_rejected() -> None:
    part = Part("p", g.rect_ring(200, 100), holes=[g.circle_ring(100, 50, 46)])
    report = validate_design(design_of(part))
    assert "E_WALL_THIN" in codes(report)
    assert "wall to the part edge" in report.errors[0].message


def test_thin_wall_between_two_pockets_is_rejected() -> None:
    part = Part(
        "p",
        g.rect_ring(300, 200),
        pockets=[
            Pocket(g.rounded_rect_ring(100, 100, 15, 20, 50), 8.0),
            Pocket(g.rounded_rect_ring(100, 100, 15, 125, 50), 8.0),
        ],
    )
    report = validate_design(design_of(part))
    assert "E_WALL_THIN" in codes(report)
    assert "apart" in " ".join(i.message for i in report.errors)


def test_wall_exactly_at_the_minimum_passes() -> None:
    part = Part(
        "p",
        g.rect_ring(300, 200),
        pockets=[
            Pocket(g.rounded_rect_ring(100, 100, 15, 20, 50), 8.0),
            Pocket(g.rounded_rect_ring(100, 100, 15, 128, 50), 8.0),
        ],
    )
    assert "E_WALL_THIN" not in codes(validate_design(design_of(part)))


def test_overlapping_features_are_rejected() -> None:
    part = Part(
        "p",
        g.rect_ring(300, 200),
        pockets=[
            Pocket(g.rounded_rect_ring(120, 100, 15, 20, 50), 8.0),
            Pocket(g.rounded_rect_ring(120, 100, 15, 100, 50), 8.0),
        ],
    )
    assert "E_FEATURE_OVERLAP" in codes(validate_design(design_of(part)))


def test_small_drill_warns() -> None:
    part = Part("p", g.rect_ring(200, 100), drills=[Drill((100, 50), 1.0)])
    assert "W_DRILL_SMALL" in codes(validate_design(design_of(part)))


# --------------------------------------------------------------------------- #
# tooling
# --------------------------------------------------------------------------- #
def test_pocket_deeper_than_the_floor_allows_is_rejected() -> None:
    part = Part(
        "p",
        g.rect_ring(300, 200),
        pockets=[Pocket(g.rounded_rect_ring(200, 100, 15, 50, 50), 15.0)],
    )
    report = validate_design(design_of(part, thickness=19.0))
    assert "E_POCKET_DEPTH" in codes(report)


def test_pocket_at_the_depth_limit_passes() -> None:
    part = Part(
        "p",
        g.rect_ring(300, 200),
        pockets=[Pocket(g.rounded_rect_ring(200, 100, 15, 50, 50), 14.0)],
    )
    assert "E_POCKET_DEPTH" not in codes(validate_design(design_of(part, thickness=19.0)))


def test_square_pocket_corners_are_rejected_for_a_router() -> None:
    part = Part(
        "p", g.rect_ring(300, 200), pockets=[Pocket(g.rect_ring(200, 100, 50, 50), 8.0)]
    )
    report = validate_design(design_of(part))
    assert "E_TOOL_UNREACHABLE" in codes(report)


def test_dogboned_pocket_corners_pass() -> None:
    relieved = g.apply_relief(g.rect_ring(200, 100, 50, 50), 3.175)
    part = Part("p", g.rect_ring(300, 200), pockets=[Pocket(relieved, 8.0)])
    assert "E_TOOL_UNREACHABLE" not in codes(validate_design(design_of(part)))


def test_prefilleted_pocket_corners_pass() -> None:
    filleted = g.rounded_rect_ring(200, 100, 4.0, 50, 50)
    part = Part("p", g.rect_ring(300, 200), pockets=[Pocket(filleted, 8.0)])
    assert "E_TOOL_UNREACHABLE" not in codes(validate_design(design_of(part)))


def test_feature_narrower_than_the_cutter_is_rejected() -> None:
    part = Part(
        "p", g.rect_ring(300, 200), holes=[g.stadium_ring(100, 6.0, 100, 100)]
    )
    report = validate_design(design_of(part))
    assert "E_FEATURE_NARROW" in codes(report)
    assert "chip clearance" in report.errors[0].message


def test_feature_just_above_the_cutter_width_still_warns() -> None:
    part = Part(
        "p", g.rect_ring(300, 200), holes=[g.stadium_ring(100, 7.05, 100, 100)]
    )
    report = validate_design(design_of(part))
    assert "E_FEATURE_NARROW" not in codes(report)
    assert "W_FEATURE_TIGHT" in codes(report)


def test_reachability_check_can_be_disabled() -> None:
    part = Part(
        "p", g.rect_ring(300, 200), pockets=[Pocket(g.rect_ring(200, 100, 50, 50), 8.0)]
    )
    report = validate_design(
        design_of(part), ValidationConfig(check_reachability=False)
    )
    assert "E_TOOL_UNREACHABLE" not in codes(report)


def test_laser_mode_skips_the_cutter_checks() -> None:
    """A 6 mm slot is impossible for a 6.35 mm router but fine for a laser."""
    part = Part(
        "p", g.rect_ring(300, 200), holes=[g.slot_ring(100, 6.0, 100, 100)]
    )
    laser = design_of(part, machine=Machine(mode=Mode.LASER), thickness=6.0)
    report = validate_design(laser)
    assert "E_FEATURE_NARROW" not in codes(report)
    assert "E_TOOL_UNREACHABLE" not in codes(report)


def test_laser_rejects_a_feature_thinner_than_two_kerfs() -> None:
    part = Part(
        "p", g.rect_ring(300, 200), holes=[g.slot_ring(100, 0.2, 100, 100)]
    )
    laser = design_of(part, machine=Machine(mode=Mode.LASER, kerf=0.15), thickness=6.0)
    assert "E_FEATURE_NARROW" in codes(validate_design(laser))


# --------------------------------------------------------------------------- #
# Report plumbing
# --------------------------------------------------------------------------- #
def test_report_separates_errors_from_warnings() -> None:
    report = Report()
    report.add("E_X", Severity.ERROR, "bad")
    report.add("W_Y", Severity.WARNING, "meh")
    report.add("INFO_Z", Severity.INFO, "fyi")
    assert len(report.errors) == 1 and len(report.warnings) == 1
    assert not report.ok
    assert "E_X" in report.format() and "INFO_Z" not in report.format()
    assert "INFO_Z" in report.format(include_info=True)


def test_report_reason_summarises_the_first_error() -> None:
    report = Report()
    assert report.reason() == "ok"
    report.add("E_A", Severity.ERROR, "first")
    report.add("E_B", Severity.ERROR, "second")
    assert report.reason() == "E_A: first (+1 more)"


def test_raise_for_status_raises_only_on_errors() -> None:
    ok = Report()
    ok.add("W", Severity.WARNING, "fine")
    ok.raise_for_status()
    bad = Report()
    bad.add("E", Severity.ERROR, "nope")
    with pytest.raises(ValidationError) as exc:
        bad.raise_for_status()
    assert exc.value.report is bad


def test_issue_str_includes_part_and_location() -> None:
    report = Report()
    report.add("E_X", Severity.ERROR, "bad", part="lid", location=(12.345, 6.0))
    assert str(report.issues[0]) == "ERROR E_X [lid]: bad at (12.3, 6.0)"


def test_a_notch_the_cutter_cannot_enter_is_rejected() -> None:
    outline = [
        (0, 0), (200, 0), (200, 120), (102, 120),
        (102, 60), (98, 60), (98, 120), (0, 120),
    ]
    report = validate_design(design_of(Part("p", outline)))
    assert "E_PROFILE_TOO_TIGHT" in codes(report)
    assert "cannot enter" in report.errors[0].message


def test_square_inside_corners_on_the_outer_profile_are_rejected() -> None:
    outline = [
        (0, 0), (200, 0), (200, 120), (130, 120),
        (130, 60), (70, 60), (70, 120), (0, 120),
    ]
    assert "E_PROFILE_TOO_TIGHT" in codes(validate_design(design_of(Part("p", outline))))


def test_filleted_inside_corners_on_the_outer_profile_pass() -> None:
    outline = g.fillet_ring(
        [
            (0, 0), (200, 0), (200, 120), (130, 120),
            (130, 60), (70, 60), (70, 120), (0, 120),
        ],
        5.0,
        corners="concave",
    )
    assert "E_PROFILE_TOO_TIGHT" not in codes(validate_design(design_of(Part("p", outline))))


def test_laser_mode_does_not_apply_the_profile_check() -> None:
    outline = [
        (0, 0), (200, 0), (200, 120), (102, 120),
        (102, 60), (98, 60), (98, 120), (0, 120),
    ]
    laser = design_of(Part("p", outline), machine=Machine(mode=Mode.LASER), thickness=6.0)
    assert "E_PROFILE_TOO_TIGHT" not in codes(validate_design(laser))
