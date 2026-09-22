"""Tests for the parametric shape library.

The point of these shapes is that they are generated, not traced, so the
tests check the properties that follow from that: every shape is a single
simple closed ring, normalised to the requested size at the origin, and
relieved to whatever cutter it is asked about.
"""

from __future__ import annotations

import pytest
from shapely.geometry import LinearRing, LineString, Polygon

from dxfgen.core import geometry as geo
from dxfgen.niches import shapes

NAMES = shapes.shape_names()


@pytest.mark.parametrize("name", NAMES)
def test_every_shape_is_one_simple_closed_ring(name: str) -> None:
    ring = shapes.shape_ring(name, 100.0)
    assert len(ring) >= 3
    assert LinearRing(ring + [ring[0]]).is_simple
    poly = Polygon(ring)
    assert poly.is_valid
    assert poly.area > 0


@pytest.mark.parametrize("name", NAMES)
def test_every_shape_is_normalised_to_the_requested_width(name: str) -> None:
    for width in (40.0, 100.0, 380.0):
        x0, y0, x1, y1 = geo.bbox(shapes.shape_ring(name, width))
        assert (x0, y0) == pytest.approx((0.0, 0.0), abs=1e-6)
        assert x1 - x0 == pytest.approx(width, abs=1e-6)


@pytest.mark.parametrize("name", NAMES)
def test_every_shape_is_counter_clockwise(name: str) -> None:
    assert geo.is_ccw(shapes.shape_ring(name, 100.0))


@pytest.mark.parametrize("name", NAMES)
def test_shapes_keep_their_proportions_at_any_size(name: str) -> None:
    small = geo.size_of(shapes.shape_ring(name, 50.0))
    large = geo.size_of(shapes.shape_ring(name, 250.0))
    assert small[1] / small[0] == pytest.approx(large[1] / large[0], rel=1e-3)


@pytest.mark.parametrize("name", [n for n in NAMES if n != "snowflake"])
def test_shapes_come_back_machinable_when_asked(name: str) -> None:
    """Relieved for a cutter means a cutter can actually follow them."""
    radius = 3.175
    ring = shapes.shape_ring(name, 160.0, radius)
    assert geo.excess_zones(Polygon(ring), radius, 0.35) == []


def test_a_shape_finer_than_the_cutter_is_refused_with_advice() -> None:
    with pytest.raises(ValueError, match="cut it on a laser"):
        shapes.shape_ring("snowflake", 100.0, 3.175)


def test_unrelieved_shapes_keep_their_sharp_corners() -> None:
    """A laser wants the shape as drawn, points and all."""
    sharp = shapes.shape_ring("star", 100.0, 0.0)
    relieved = shapes.shape_ring("star", 100.0, 3.175)
    assert len(sharp) < len(relieved)
    assert geo.excess_zones(Polygon(sharp), 3.175, 0.35) != []


def test_unknown_shapes_are_reported_with_the_list() -> None:
    with pytest.raises(KeyError, match="available: "):
        shapes.shape_ring("unicorn", 100.0)


# --------------------------------------------------------------------------- #
# individual shapes
# --------------------------------------------------------------------------- #
def test_the_heart_is_wider_than_it_is_tall() -> None:
    width, height = geo.size_of(shapes.heart(100.0))
    assert height < width


def test_a_star_has_the_points_asked_for() -> None:
    for count in (5, 6, 8):
        ring = shapes.star(100.0, points_count=count)
        # Each point contributes one local maximum in radius from the centre.
        centre = geo.centroid(ring)
        radii = [geo._norm(geo._sub(p, centre)) for p in ring]
        peaks = sum(
            1
            for i in range(len(radii))
            if radii[i] >= radii[i - 1] and radii[i] >= radii[(i + 1) % len(radii)]
        )
        assert peaks == count


@pytest.mark.parametrize("count", [2, 4])
def test_star_and_ghost_reject_degenerate_counts(count: int) -> None:
    with pytest.raises(ValueError):
        shapes.star(100.0, points_count=2)
    with pytest.raises(ValueError):
        shapes.ghost(100.0, scallops=1)
    with pytest.raises(ValueError):
        shapes.tree(100.0, tiers=1)
    with pytest.raises(ValueError):
        shapes.paw(100.0, toes=2)


def test_the_crescent_is_thinner_than_a_disc() -> None:
    ring = shapes.moon(100.0)
    width, height = geo.size_of(ring)
    # A crescent fills far less of its bounding box than a disc's 79%.
    assert Polygon(ring).area / (width * height) < 0.45


def test_the_paw_toes_all_merge_into_the_pad() -> None:
    for toes in (3, 4, 5):
        ring = shapes.paw(100.0, toes=toes)
        assert Polygon(ring).is_valid  # one piece, not a scatter of toes


def test_a_snowflake_holds_together_at_the_centre() -> None:
    for arms in (5, 6, 8):
        assert Polygon(shapes.snowflake(100.0, arms=arms)).is_valid


def test_shapes_reject_impossible_options() -> None:
    with pytest.raises(ValueError):
        shapes.star(100.0, inner_ratio=1.5)
    with pytest.raises(ValueError):
        shapes.moon(100.0, waning=0.05)


# --------------------------------------------------------------------------- #
# detail and hanging holes
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", NAMES)
def test_detail_stays_inside_the_shape(name: str) -> None:
    region = Polygon(shapes.shape_ring(name, 120.0))
    for points, closed in shapes.detail_contours(name, 120.0, inset=8.0):
        line = LineString(list(points) + ([points[0]] if closed else []))
        assert region.covers(line), name


def test_shapes_with_their_own_detail_do_not_get_a_plain_border() -> None:
    assert len(shapes.detail_contours("pumpkin", 120.0, inset=0.0)) == 4
    assert len(shapes.detail_contours("ghost", 120.0, inset=0.0)) == 3
    assert shapes.detail_contours("star", 120.0, inset=0.0) == []


def test_a_border_is_offered_where_there_is_no_specific_detail() -> None:
    borders = shapes.detail_contours("star", 120.0, inset=9.0)
    assert len(borders) == 1 and borders[0][1] is True


@pytest.mark.parametrize("name", NAMES)
def test_a_hanging_hole_lands_in_material(name: str) -> None:
    ring = shapes.shape_ring(name, 90.0)
    hole = shapes.place_hang_hole(ring, 4.0, 3.0)
    region = Polygon(ring)
    disc = Polygon(hole)
    assert region.contains(disc)
    assert region.exterior.distance(disc) >= 3.0 - 1e-6


def test_the_hanging_hole_sits_in_the_upper_half() -> None:
    for name in ("heart", "tree", "star", "pumpkin"):
        ring = shapes.shape_ring(name, 90.0)
        _, y0, _, y1 = geo.bbox(ring)
        centre_y = geo.centroid(shapes.place_hang_hole(ring, 4.0, 3.0))[1]
        assert centre_y > y0 + (y1 - y0) * 0.5, name


def test_the_hole_search_leaves_the_centreline_when_it_has_to() -> None:
    """A crescent's thick part is not on its centreline."""
    ring = shapes.shape_ring("moon", 70.0)
    x0, _, x1, _ = geo.bbox(ring)
    centre_x = geo.centroid(shapes.place_hang_hole(ring, 4.0, 3.0))[0]
    assert abs(centre_x - (x0 + x1) / 2.0) > 1.0


def test_a_shape_too_small_to_hang_says_so() -> None:
    with pytest.raises(ValueError, match="does not fit in this shape"):
        shapes.place_hang_hole(shapes.shape_ring("snowflake", 40.0), 6.0, 4.0)


def test_every_shape_is_tagged_with_an_occasion() -> None:
    assert set(shapes.seasonal_tags) == set(NAMES)
    assert all(shapes.seasonal_tags[name] for name in NAMES)
