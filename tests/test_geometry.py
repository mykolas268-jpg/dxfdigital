"""Tests for the geometry primitives.

Where a closed-form answer exists it is asserted against, rather than against
a recorded output, so the tests catch real geometric regressions instead of
pinning whatever the code happened to do.
"""

from __future__ import annotations

import math

import pytest
from shapely.geometry import LinearRing, Polygon

from dxfgen.core import geometry as g


def tessellation_area_bound(arc_length: float, tol: float) -> float:
    """Largest area error a chord tessellation may introduce.

    Each chord cuts off a near-parabolic segment of area about
    ``2/3 * chord * height`` with ``height <= tol``, so summed along an arc of
    ``arc_length`` the error is bounded by ``2/3 * arc_length * tol``.  The
    1.5x factor absorbs rounding the segment count up to a whole number.
    Asserting against this bound tests that tessellation honours its promised
    tolerance, rather than pinning whatever number the code happens to emit.
    """
    return (2.0 / 3.0) * arc_length * tol * 1.5


# --------------------------------------------------------------------------- #
# measurement
# --------------------------------------------------------------------------- #
def test_signed_area_sign_follows_winding() -> None:
    ccw = [(0, 0), (10, 0), (10, 5), (0, 5)]
    assert g.signed_area(ccw) == pytest.approx(50.0)
    assert g.signed_area(ccw[::-1]) == pytest.approx(-50.0)
    assert g.is_ccw(ccw)
    assert not g.is_ccw(ccw[::-1])


def test_signed_area_of_degenerate_ring_is_zero() -> None:
    assert g.signed_area([(0, 0), (1, 1)]) == 0.0


def test_ensure_winding() -> None:
    cw = [(0, 0), (0, 5), (10, 5), (10, 0)]
    assert g.is_ccw(g.ensure_ccw(cw))
    assert not g.is_ccw(g.ensure_cw(g.ensure_ccw(cw)))


def test_bbox_and_size() -> None:
    ring = [(2, 3), (12, 3), (12, 9)]
    assert g.bbox(ring) == (2, 3, 12, 9)
    assert g.size_of(ring) == (10, 6)
    assert g.bbox_of([[(0, 0)], [(5, -2)]]) == (0, -2, 5, 0)


def test_bbox_of_empty_raises() -> None:
    with pytest.raises(ValueError):
        g.bbox([])


def test_centroid_of_rectangle_is_its_middle() -> None:
    cx, cy = g.centroid(g.rect_ring(10, 4))
    assert (cx, cy) == pytest.approx((5.0, 2.0))


def test_centroid_falls_back_for_zero_area() -> None:
    assert g.centroid([(0, 0), (2, 0), (4, 0)]) == pytest.approx((2.0, 0.0))


def test_perimeter_open_vs_closed() -> None:
    ring = g.rect_ring(10, 5)
    assert g.perimeter(ring) == pytest.approx(30.0)
    assert g.perimeter(ring, closed=False) == pytest.approx(25.0)


def test_dedupe_removes_zero_length_and_closing_point() -> None:
    ring = [(0, 0), (0, 0.0001), (10, 0), (10, 5), (0, 5), (0, 0)]
    out = g.dedupe(ring)
    assert out == [(0.0, 0.0), (10.0, 0.0), (10.0, 5.0), (0.0, 5.0)]


def test_dedupe_open_keeps_endpoints() -> None:
    out = g.dedupe([(0, 0), (5, 0), (0, 0)], closed=False)
    assert out == [(0.0, 0.0), (5.0, 0.0), (0.0, 0.0)]


# --------------------------------------------------------------------------- #
# transforms
# --------------------------------------------------------------------------- #
def test_translate() -> None:
    assert g.translate([(1, 2)], 3, -4) == [(4, -2)]


def test_rotate_90_degrees() -> None:
    (x, y), = g.rotate([(1, 0)], 90)
    assert (x, y) == pytest.approx((0.0, 1.0), abs=1e-12)


def test_rotate_about_origin_point() -> None:
    (x, y), = g.rotate([(2, 1)], 180, origin=(1, 1))
    assert (x, y) == pytest.approx((0.0, 1.0), abs=1e-12)


def test_scale_uniform_and_anisotropic() -> None:
    assert g.scale([(2, 3)], 2) == [(4, 6)]
    assert g.scale([(2, 3)], 2, 1) == [(4, 3)]
    assert g.scale([(2, 3)], 2, origin=(2, 3)) == [(2, 3)]


def test_mirror_reverses_winding() -> None:
    ring = g.rect_ring(10, 5)
    assert g.is_ccw(ring)
    assert g.is_ccw(g.mirror_x(ring))  # mirror flips winding, so reversal restores it
    assert g.bbox(g.mirror_x(ring, axis=5)) == pytest.approx((0, 0, 10, 5))
    assert g.bbox(g.mirror_y(ring, axis=2.5)) == pytest.approx((0, 0, 10, 5))


# --------------------------------------------------------------------------- #
# curve tessellation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("radius", [1.0, 3.175, 20.0, 150.0])
@pytest.mark.parametrize("tol", [0.01, 0.05, 0.2])
def test_tessellated_circle_respects_chord_tolerance(radius: float, tol: float) -> None:
    ring = g.circle_ring(0, 0, radius, tol)
    n = len(ring)
    step = 2 * math.pi / n
    deviation = radius * (1 - math.cos(step / 2))
    assert deviation <= tol + 1e-12
    assert all(
        math.hypot(x, y) == pytest.approx(radius) for x, y in ring
    )


def test_arc_segment_count_grows_with_radius_and_sweep() -> None:
    assert g.arc_segment_count(10, math.pi) > g.arc_segment_count(10, math.pi / 4)
    assert g.arc_segment_count(100, math.pi) > g.arc_segment_count(10, math.pi)
    assert g.arc_segment_count(0.0, math.pi) == 1
    assert g.arc_segment_count(10, 0.0) == 1


def test_arc_points_hits_both_ends() -> None:
    pts = g.arc_points(0, 0, 5, 0, math.pi / 2)
    assert pts[0] == pytest.approx((5.0, 0.0))
    assert pts[-1] == pytest.approx((0.0, 5.0), abs=1e-12)
    assert g.arc_points(0, 0, 5, 0, math.pi / 2, include_last=False)[-1] != pytest.approx(
        (0.0, 5.0)
    )


def test_circle_ring_area_is_inscribed_within_the_arc_tolerance() -> None:
    tol, radius = 0.01, 25.0
    ring = g.circle_ring(0, 0, radius, tol)
    exact = math.pi * radius**2
    deficit = exact - g.signed_area(ring)
    assert 0 < deficit <= tessellation_area_bound(2 * math.pi * radius, tol)
    assert g.is_ccw(ring)


def test_circle_ring_rejects_bad_radius() -> None:
    with pytest.raises(ValueError):
        g.circle_ring(0, 0, 0)


def test_bezier_endpoints_and_straight_line() -> None:
    pts = g.bezier_points((0, 0), (0, 0), (10, 0), (10, 0), samples=8)
    assert len(pts) == 9
    assert pts[0] == pytest.approx((0, 0))
    assert pts[-1] == pytest.approx((10, 0))
    assert all(y == pytest.approx(0.0) for _, y in pts)


def test_bezier_rejects_zero_samples() -> None:
    with pytest.raises(ValueError):
        g.bezier_points((0, 0), (1, 1), (2, 2), (3, 3), samples=0)


# --------------------------------------------------------------------------- #
# rectangles
# --------------------------------------------------------------------------- #
def test_rect_ring_is_ccw_with_exact_area() -> None:
    ring = g.rect_ring(30, 12, 5, 7)
    assert g.is_ccw(ring)
    assert g.signed_area(ring) == pytest.approx(360.0)
    assert g.bbox(ring) == (5, 7, 35, 19)


@pytest.mark.parametrize("w,h", [(0, 10), (10, 0), (-5, 5)])
def test_rect_ring_rejects_bad_size(w: float, h: float) -> None:
    with pytest.raises(ValueError):
        g.rect_ring(w, h)


def test_rounded_rect_bbox_is_exact_and_area_matches_formula() -> None:
    w, h, r = 120.0, 80.0, 15.0
    ring = g.rounded_rect_ring(w, h, r, tolerance=0.005)
    assert g.bbox(ring) == pytest.approx((0, 0, w, h))
    exact = w * h - (4 - math.pi) * r * r
    assert g.signed_area(ring) == pytest.approx(exact, rel=1e-4)
    assert g.is_ccw(ring)
    assert Polygon(ring).is_valid


def test_rounded_rect_zero_radius_equals_rect() -> None:
    assert g.rounded_rect_ring(10, 5, 0) == g.rect_ring(10, 5)


def test_rounded_rect_per_corner_radii() -> None:
    ring = g.rounded_rect_ring(100, 60, (0, 10, 20, 0), tolerance=0.005)
    assert g.bbox(ring) == pytest.approx((0, 0, 100, 60))
    assert (0.0, 0.0) in ring  # square bottom-left corner survives
    assert (0.0, 60.0) in ring  # square top-left corner survives
    exact = 100 * 60 - (4 - math.pi) / 4 * (10**2 + 20**2)
    assert g.signed_area(ring) == pytest.approx(exact, rel=1e-4)


def test_rounded_rect_rejects_oversized_radius() -> None:
    with pytest.raises(ValueError, match="does not fit"):
        g.rounded_rect_ring(50, 20, 11)


def test_rounded_rect_rejects_negative_radius_and_bad_sequence() -> None:
    with pytest.raises(ValueError):
        g.rounded_rect_ring(50, 20, -1)
    with pytest.raises(ValueError):
        g.rounded_rect_ring(50, 20, (1, 2, 3))


def test_stadium_area_matches_rect_plus_circle() -> None:
    length, width = 95.0, 32.0
    tol = 0.005
    ring = g.stadium_ring(length, width, tolerance=tol)
    exact = (length - width) * width + math.pi * (width / 2) ** 2
    deficit = exact - g.signed_area(ring)
    assert 0 < deficit <= tessellation_area_bound(math.pi * width, tol)
    assert g.bbox(ring) == pytest.approx((0, 0, length, width))


def test_stadium_rejects_length_below_width() -> None:
    with pytest.raises(ValueError):
        g.stadium_ring(20, 30)


def test_superellipse_exponent_2_is_an_ellipse() -> None:
    ring = g.superellipse_ring(100, 60, 2.0, samples=720)
    assert g.signed_area(ring) == pytest.approx(math.pi * 50 * 30, rel=1e-4)


def test_superellipse_high_exponent_approaches_rectangle() -> None:
    ring = g.superellipse_ring(100, 60, 30.0, samples=720)
    assert g.signed_area(ring) == pytest.approx(100 * 60, rel=0.02)
    assert g.bbox(ring) == pytest.approx((-50, -30, 50, 30))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"width": 0, "height": 10},
        {"width": 10, "height": -1},
        {"width": 10, "height": 10, "exponent": 0},
        {"width": 10, "height": 10, "samples": 8},
    ],
)
def test_superellipse_rejects_bad_params(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        g.superellipse_ring(**kwargs)


# --------------------------------------------------------------------------- #
# shapely bridge and offsets
# --------------------------------------------------------------------------- #
def test_polygon_from_ring_with_holes() -> None:
    poly = g.polygon_from_ring(g.rect_ring(100, 100), [g.rect_ring(10, 10, 20, 20)])
    assert poly.is_valid
    assert poly.area == pytest.approx(100 * 100 - 100)


def test_polygon_from_ring_rejects_degenerate() -> None:
    with pytest.raises(ValueError):
        g.polygon_from_ring([(0, 0), (1, 0)])


def test_rings_of_returns_exterior_ccw_and_holes_cw() -> None:
    poly = g.polygon_from_ring(g.rect_ring(100, 100), [g.rect_ring(10, 10, 20, 20)])
    rings = g.rings_of(poly)
    assert len(rings) == 2
    assert g.is_ccw(rings[0])
    assert not g.is_ccw(rings[1])


def test_rings_of_empty_geometry() -> None:
    assert g.rings_of(Polygon()) == []


def test_offset_ring_grows_and_shrinks() -> None:
    ring = g.rect_ring(50, 50)
    grown = g.offset_ring(ring, 5, join="mitre")
    assert len(grown) == 1
    assert g.bbox(grown[0]) == pytest.approx((-5, -5, 55, 55))
    shrunk = g.offset_ring(ring, -5, join="mitre")
    assert g.bbox(shrunk[0]) == pytest.approx((5, 5, 45, 45))


def test_offset_ring_can_erase_a_region() -> None:
    assert g.offset_ring(g.rect_ring(4, 4), -3) == []


def test_kerf_compensation_grows_outline_and_shrinks_cutout() -> None:
    kerf = 0.15
    outer = g.kerf_compensate_ring(g.rect_ring(100, 50), kerf, outward=True)[0]
    x0, y0, x1, y1 = g.bbox(outer)
    assert (x1 - x0) == pytest.approx(100 + kerf)
    assert (y1 - y0) == pytest.approx(50 + kerf)
    inner = g.kerf_compensate_ring(g.rect_ring(20, 20), kerf, outward=False)[0]
    ix0, iy0, ix1, iy1 = g.bbox(inner)
    assert (ix1 - ix0) == pytest.approx(20 - kerf)


def test_kerf_compensation_zero_kerf_is_identity() -> None:
    ring = g.rect_ring(10, 10)
    assert g.kerf_compensate_ring(ring, 0.0, outward=True) == [ring]


def test_kerf_compensation_rejects_negative() -> None:
    with pytest.raises(ValueError):
        g.kerf_compensate_ring(g.rect_ring(10, 10), -0.1, outward=True)


# --------------------------------------------------------------------------- #
# corners
# --------------------------------------------------------------------------- #
def test_classify_convex_on_rectangle_and_l_shape() -> None:
    assert g.classify_convex(g.rect_ring(10, 10)) == [True] * 4
    l_shape = [(0, 0), (60, 0), (60, 60), (30, 60), (30, 30), (0, 30)]
    flags = g.classify_convex(l_shape)
    assert flags.count(False) == 1
    assert flags[4] is False  # the (30, 30) notch corner


def test_classify_convex_rejects_degenerate() -> None:
    with pytest.raises(ValueError):
        g.classify_convex([(0, 0), (1, 1)])


def test_fillet_ring_area_matches_formula() -> None:
    ring = g.fillet_ring(g.rect_ring(80, 50), 8, tolerance=0.005)
    exact = 80 * 50 - (4 - math.pi) * 64
    assert g.signed_area(ring) == pytest.approx(exact, rel=1e-4)
    assert g.bbox(ring) == pytest.approx((0, 0, 80, 50))
    assert Polygon(ring).is_valid


def test_fillet_reflex_corner_adds_material() -> None:
    l_shape = [(0, 0), (60, 0), (60, 60), (30, 60), (30, 30), (0, 30)]
    base = g.signed_area(l_shape)
    radius, tol = 6.0, 0.005
    filleted = g.fillet_ring(l_shape, radius, corners="concave", tolerance=tol)
    added = g.signed_area(filleted) - base
    exact = (1 - math.pi / 4) * radius**2
    quarter_arc = math.pi * radius / 2
    assert abs(added - exact) <= tessellation_area_bound(quarter_arc, tol)
    assert Polygon(filleted).is_valid


def test_fillet_convex_only_leaves_reflex_corner_sharp() -> None:
    l_shape = [(0, 0), (60, 0), (60, 60), (30, 60), (30, 30), (0, 30)]
    filleted = g.fillet_ring(l_shape, 6, corners="convex")
    assert (30.0, 30.0) in filleted


def test_fillet_clamps_radius_to_available_edge() -> None:
    ring = g.fillet_ring(g.rect_ring(20, 6), 50, clamp=True)
    assert Polygon(ring).is_valid
    assert g.bbox(ring) == pytest.approx((0, 0, 20, 6))


def test_fillet_without_clamp_skips_impossible_corners() -> None:
    ring = g.fillet_ring(g.rect_ring(20, 6), 50, clamp=False)
    assert ring == g.rect_ring(20, 6)


def test_fillet_rejects_non_positive_radius() -> None:
    with pytest.raises(ValueError):
        g.fillet_ring(g.rect_ring(10, 10), 0)


def test_relief_positions_finds_four_corners_of_a_slot() -> None:
    positions = g.relief_positions(g.rect_ring(60, 18), 3.175)
    assert len(positions) == 4
    assert all(r == 3.175 for _, r in positions)


def test_relief_positions_ignores_smooth_curves() -> None:
    assert g.relief_positions(g.circle_ring(0, 0, 20), 3.175) == []


def test_relief_positions_rejects_bad_tool() -> None:
    with pytest.raises(ValueError):
        g.relief_positions(g.rect_ring(10, 10), 0)


@pytest.mark.parametrize("style", ["dogbone", "tbone"])
def test_relief_makes_a_square_corner_reachable(style: str) -> None:
    tool_r = 3.175
    slot = g.slot_ring(60, 18)
    sharp = Polygon(slot)
    assert g.unreachable_zones(sharp, tool_r, min_thickness=0.35)

    relieved = g.apply_relief(slot, tool_r, style=style)
    poly = Polygon(relieved)
    assert poly.is_valid
    assert LinearRing(relieved + [relieved[0]]).is_simple
    assert g.unreachable_zones(poly, tool_r, min_thickness=0.35) == []
    assert poly.area > sharp.area  # relief removes extra material


def test_relief_on_a_part_removes_material() -> None:
    l_shape = [(0, 0), (60, 0), (60, 60), (30, 60), (30, 30), (0, 30)]
    relieved = g.apply_relief(l_shape, 3.0, region="part")
    assert Polygon(relieved).area < g.signed_area(l_shape)


def test_relief_raises_when_it_would_sever_the_part() -> None:
    # Two blocks joined by a 5 mm neck: a 10 mm cutter's corner relief eats
    # the neck from both sides, so the part falls in half.  Returning a
    # MultiPolygon silently would hand the exporter unusable geometry.
    dumbbell = [
        (0, 0), (30, 0), (30, 20), (50, 20), (50, 0), (80, 0),
        (80, 40), (50, 40), (50, 25), (30, 25), (30, 40), (0, 40),
    ]
    assert g.apply_relief(dumbbell, 6.0, region="part")  # fits, no complaint
    with pytest.raises(ValueError, match="split the contour"):
        g.apply_relief(dumbbell, 10.0, region="part")


def test_relief_skips_corners_whose_edges_are_shorter_than_the_tool() -> None:
    # A 4 mm square cannot host a 6 mm cutter's relief, so no corner qualifies
    # and the contour is returned untouched for the validator to reject.
    assert g.apply_relief(g.rect_ring(4, 4), 6.0) == g.rect_ring(4, 4)


# --------------------------------------------------------------------------- #
# joints
# --------------------------------------------------------------------------- #
def test_slot_ring_orientation_and_radius() -> None:
    assert g.size_of(g.slot_ring(40, 6)) == pytest.approx((40, 6))
    assert g.size_of(g.slot_ring(40, 6, vertical=True)) == pytest.approx((6, 40))
    rounded = g.slot_ring(40, 6, corner_radius=3)
    assert g.signed_area(rounded) < 240


def test_router_slot_width_carries_clearance_only() -> None:
    assert g.joint_slot_width(18.0, "router", clearance=0.2) == pytest.approx(18.2)
    assert g.joint_slot_width(18.0, "router", clearance=0.0) == pytest.approx(18.0)


def test_laser_slot_width_is_reduced_by_one_kerf() -> None:
    # Drawn 2.85 mm, the beam removes 0.15 mm, so the finished slot is 3.00 mm
    # and 3 mm material press-fits.
    assert g.joint_slot_width(3.0, "laser", clearance=0.0, kerf=0.15) == pytest.approx(
        2.85
    )
    assert g.joint_slot_width(6.0, "laser", clearance=0.1, kerf=0.2) == pytest.approx(
        5.9
    )


def test_laser_slot_width_finished_size_hits_nominal() -> None:
    thickness, kerf, clearance = 3.0, 0.15, 0.0
    drawn = g.joint_slot_width(thickness, "laser", clearance, kerf)
    finished = drawn + kerf
    assert finished == pytest.approx(thickness + clearance)


def test_slot_width_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        g.joint_slot_width(18, "plasma")
    with pytest.raises(ValueError):
        g.joint_slot_width(0, "router")
    with pytest.raises(ValueError):
        g.joint_slot_width(0.1, "laser", clearance=0.0, kerf=0.5)


@pytest.mark.parametrize("length", [60.0, 120.0, 300.0, 47.5])
@pytest.mark.parametrize("thickness", [3.0, 6.0, 18.0])
def test_suggest_finger_count_is_odd_and_well_proportioned(
    length: float, thickness: float
) -> None:
    if length < 3 * 1.5 * thickness:
        with pytest.raises(ValueError):
            g.suggest_finger_count(length, thickness)
        return
    n = g.suggest_finger_count(length, thickness)
    assert n % 2 == 1 and n >= 3
    cell = length / n
    assert 1.5 * thickness <= cell <= 3.0 * thickness + 1e-9


def test_suggest_finger_count_rejects_short_edges() -> None:
    with pytest.raises(ValueError):
        g.suggest_finger_count(10, 18)


def test_finger_joint_male_and_female_are_complementary() -> None:
    thickness, fingers = 6.0, 5
    male = g.finger_joint_edge((0, 0), (60, 0), thickness, fingers)
    female = g.finger_joint_edge((0, 0), (60, 0), thickness, fingers, male=False)
    assert male[0] == (0.0, 0.0) and male[-1] == (60.0, 0.0)
    assert len(male) == len(female)
    for (mx, my), (fx, fy) in zip(male, female):
        assert mx == pytest.approx(fx)
        assert my == pytest.approx(-fy)


def test_finger_joint_tabs_have_the_right_count_size_and_direction() -> None:
    thickness, fingers, length = 6.0, 5, 60.0
    male = g.finger_joint_edge((0, 0), (length, 0), thickness, fingers)
    # Outward normal of a left-to-right bottom edge on a CCW ring points -Y.
    tab_pts = [p for p in male if p[1] == pytest.approx(-thickness)]
    assert len(tab_pts) == 6  # three tabs, two corners each
    cell = length / fingers
    assert sorted({round(x, 6) for x, _ in tab_pts}) == [
        pytest.approx(0.0),
        pytest.approx(cell),
        pytest.approx(2 * cell),
        pytest.approx(3 * cell),
        pytest.approx(4 * cell),
        pytest.approx(5 * cell),
    ]


def test_finger_joint_spliced_into_a_panel_is_valid() -> None:
    thickness, fingers, length = 6.0, 5, 60.0
    bottom = g.finger_joint_edge((0, 0), (length, 0), thickness, fingers)
    ring = g.dedupe(bottom + [(length, 40.0), (0.0, 40.0)])
    poly = Polygon(ring)
    assert poly.is_valid
    assert poly.area == pytest.approx(length * 40 + 3 * (length / fingers) * thickness)


def test_finger_joint_clearance_shrinks_tabs_and_grows_notches() -> None:
    clearance = 0.4
    male = g.finger_joint_edge((0, 0), (60, 0), 6.0, 5, clearance=clearance)
    xs = sorted({round(x, 6) for x, y in male if y < -1})
    assert xs[0] == pytest.approx(clearance / 2)
    female = g.finger_joint_edge((0, 0), (60, 0), 6.0, 5, male=False, clearance=clearance)
    fxs = sorted({round(x, 6) for x, y in female if y > 1})
    assert fxs[0] == pytest.approx(0.0)  # clamped at the edge start
    assert fxs[1] == pytest.approx(12 + clearance / 2)


def test_finger_joint_tab_first_shifts_the_phase() -> None:
    a = g.finger_joint_edge((0, 0), (60, 0), 6.0, 5, tab_first=True)
    b = g.finger_joint_edge((0, 0), (60, 0), 6.0, 5, tab_first=False)
    assert len([p for p in a if p[1] < 0]) == 6
    assert len([p for p in b if p[1] < 0]) == 4  # two interior notches only


@pytest.mark.parametrize(
    "kwargs",
    [
        {"thickness": 0},
        {"fingers": 2},
    ],
)
def test_finger_joint_rejects_bad_params(kwargs: dict) -> None:
    base = {"start": (0, 0), "end": (60, 0), "thickness": 6.0, "fingers": 5}
    base.update(kwargs)
    with pytest.raises(ValueError):
        g.finger_joint_edge(**base)


def test_finger_joint_rejects_zero_length_edge() -> None:
    with pytest.raises(ValueError):
        g.finger_joint_edge((0, 0), (0, 0), 6.0, 5)


# --------------------------------------------------------------------------- #
# manufacturability geometry
# --------------------------------------------------------------------------- #
def test_opening_of_a_wide_region_keeps_almost_everything() -> None:
    poly = Polygon(g.rect_ring(100, 100))
    assert g.opening(poly, 3.175).area == pytest.approx(poly.area, rel=1e-3)


def test_opening_of_a_narrow_region_is_empty() -> None:
    assert g.opening(Polygon(g.rect_ring(100, 4)), 3.175).is_empty


def test_opening_rejects_bad_radius_or_slack() -> None:
    poly = Polygon(g.rect_ring(10, 10))
    with pytest.raises(ValueError):
        g.opening(poly, 0)
    with pytest.raises(ValueError):
        g.opening(poly, 1.0, slack=1.0)


def test_sharp_corner_residual_matches_the_analytic_value() -> None:
    """A square inside corner leaves a sliver of known thickness.

    The largest circle that fits between the two walls and the cutter's
    corner arc has radius ``r*(sqrt2-1)/(sqrt2+1)``, so the leftover material
    is twice that thick.
    """
    tool_r = 3.175
    zones = g.unreachable_zones(Polygon(g.rect_ring(60, 40)), tool_r)
    assert len(zones) == 4
    exact = 2 * tool_r * (math.sqrt(2) - 1) / (math.sqrt(2) + 1)
    assert zones[0][2] == pytest.approx(exact, rel=0.05)


@pytest.mark.parametrize("style", ["dogbone", "tbone"])
def test_relief_leaves_an_order_of_magnitude_less_material(style: str) -> None:
    tool_r = 3.175
    slot = g.slot_ring(60, 18)
    sharp = g.unreachable_zones(Polygon(slot), tool_r)[0][2]
    zones = g.unreachable_zones(
        Polygon(g.apply_relief(slot, tool_r, style=style)), tool_r
    )
    relieved = zones[0][2] if zones else 0.0
    assert relieved < sharp / 10


def test_relief_positions_ignores_a_fine_tessellation() -> None:
    # A rounded rectangle is a long run of small turns; relieving those would
    # shred the contour instead of relieving a corner.
    assert g.relief_positions(g.rounded_rect_ring(100, 60, 15), 3.175) == []
    assert len(g.relief_positions(g.rect_ring(60, 18), 3.175)) == 4


def test_residual_thickness_of_empty_region_is_zero() -> None:
    assert g.residual_thickness(Polygon()) == 0.0


def test_unreachable_zones_min_thickness_filters_slivers() -> None:
    poly = Polygon(g.apply_relief(g.slot_ring(60, 18), 3.175))
    assert g.unreachable_zones(poly, 3.175) != []
    assert g.unreachable_zones(poly, 3.175, min_thickness=0.35) == []


def test_a_smooth_shape_has_no_meaningful_residual() -> None:
    # A circle far bigger than the cutter is fully reachable; only
    # tessellation-scale slivers remain.
    poly = Polygon(g.circle_ring(0, 0, 40))
    assert g.unreachable_zones(poly, 3.175, min_thickness=2 * g.ARC_TOLERANCE) == []


# --------------------------------------------------------------------------- #
# text
# --------------------------------------------------------------------------- #
def test_text_contours_produce_closed_rings_at_the_requested_height() -> None:
    rings = g.text_contours("HANDMADE", 12.0)
    assert rings
    _, y0, _, y1 = g.bbox_of(rings)
    assert (y1 - y0) == pytest.approx(12.0, rel=1e-6)
    for ring in rings:
        assert len(ring) >= 3
        assert Polygon(ring).is_valid


def test_text_letter_with_a_counter_yields_two_rings() -> None:
    polys = g.text_polygons("O", 20.0)
    assert len(polys) == 1
    assert len(polys[0].interiors) == 1
    assert len(g.text_contours("O", 20.0)) == 2


def test_text_alignment_positions_the_bounding_box() -> None:
    left = g.bbox_of(g.text_contours("AB", 10.0, origin=(0, 0), align="left"))
    centre = g.bbox_of(g.text_contours("AB", 10.0, origin=(0, 0), align="center"))
    right = g.bbox_of(g.text_contours("AB", 10.0, origin=(0, 0), align="right"))
    assert left[0] == pytest.approx(0.0)
    assert centre[0] == pytest.approx(-(left[2] - left[0]) / 2)
    assert right[2] == pytest.approx(0.0)


def test_text_rejects_blank_and_bad_height() -> None:
    with pytest.raises(ValueError):
        g.text_contours("   ", 10.0)
    with pytest.raises(ValueError):
        g.text_contours("A", 0.0)
