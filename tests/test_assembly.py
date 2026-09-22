"""Tests for the assembly model and its isometric drawing.

The drawing is a marketing image, so most of what is asserted here is about
the *model*: a placement that is wrong puts a shelf inside an upright, and no
amount of looking at a thumbnail reliably catches that. Sizes and containment
do.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
from PIL import Image

from dxfgen.core import geometry as geo
from dxfgen.core.assembly import (
    ISO_COS,
    ISO_SIN,
    Assembly,
    Placement,
    Plane,
    depth_of,
    panel_prism,
    plane_normal,
    project,
    to_world,
)
from dxfgen.core.design import Design, Machine, Part
from dxfgen.core.preview import assembled_size, render_assembly
from dxfgen.niches import get_generator


# --------------------------------------------------------------------------- #
# mapping a flat panel into the object
# --------------------------------------------------------------------------- #
def test_a_flat_panel_lies_down() -> None:
    assert to_world((10.0, 20.0), Plane.FLAT, (1.0, 2.0, 3.0)) == (11.0, 22.0, 3.0)


def test_a_front_panel_stands_facing_the_viewer() -> None:
    assert to_world((10.0, 20.0), Plane.FRONT, (1.0, 2.0, 3.0)) == (11.0, 2.0, 23.0)


def test_a_side_panel_stands_facing_sideways() -> None:
    assert to_world((10.0, 20.0), Plane.SIDE, (1.0, 2.0, 3.0)) == (1.0, 12.0, 23.0)


@pytest.mark.parametrize("plane", list(Plane))
def test_every_plane_has_a_unit_normal(plane: Plane) -> None:
    normal = plane_normal(plane)
    assert math.isclose(math.dist(normal, (0.0, 0.0, 0.0)), 1.0)


@pytest.mark.parametrize("plane", list(Plane))
def test_a_panel_is_exactly_its_thickness_deep(plane: Plane) -> None:
    ring = geo.rect_ring(100.0, 60.0)
    placement = Placement("p", plane, (0.0, 0.0, 0.0))
    near, far, _ = panel_prism(ring, placement, 18.0)
    axis = plane_normal(plane).index(1.0)
    spread = {round(point[axis], 6) for point in near + far}
    assert sorted(spread) == [0.0, 18.0]


# --------------------------------------------------------------------------- #
# the projection
# --------------------------------------------------------------------------- #
def test_the_origin_projects_to_the_origin() -> None:
    assert project((0.0, 0.0, 0.0)) == (0.0, 0.0)


def test_the_axes_go_where_isometric_says() -> None:
    assert project((1.0, 0.0, 0.0)) == pytest.approx((ISO_COS, ISO_SIN))
    assert project((0.0, 1.0, 0.0)) == pytest.approx((-ISO_COS, ISO_SIN))
    assert project((0.0, 0.0, 1.0)) == pytest.approx((0.0, 1.0))


def test_the_projection_is_affine() -> None:
    """Equal steps in the object are equal steps on the page: no perspective."""
    a, b = project((0.0, 0.0, 0.0)), project((100.0, 0.0, 0.0))
    c, d = project((900.0, 0.0, 0.0)), project((1000.0, 0.0, 0.0))
    assert math.dist(a, b) == pytest.approx(math.dist(c, d))


def test_the_collapsed_direction_really_collapses() -> None:
    """Points differing along (1, 1, -1) land on the same place, by design."""
    for step in (1.0, 37.0, -12.5):
        assert project((step, step, -step)) == pytest.approx((0.0, 0.0))


def test_higher_is_nearer_and_further_back_is_further() -> None:
    assert depth_of((0.0, 0.0, 100.0)) > depth_of((0.0, 0.0, 0.0))
    assert depth_of((0.0, 200.0, 0.0)) < depth_of((0.0, 0.0, 0.0))
    assert depth_of((200.0, 0.0, 0.0)) < depth_of((0.0, 0.0, 0.0))


def test_the_near_face_of_a_panel_is_the_nearer_one() -> None:
    ring = geo.rect_ring(100.0, 60.0)
    for plane in Plane:
        near, far, _ = panel_prism(ring, Placement("p", plane, (0.0, 0.0, 0.0)), 18.0)
        assert depth_of(near[0]) >= depth_of(far[0])


def test_a_prism_has_one_side_quad_per_edge() -> None:
    ring = geo.rect_ring(100.0, 60.0)
    _, _, quads = panel_prism(ring, Placement("p", Plane.FLAT), 18.0)
    assert len(quads) == len(ring)
    assert all(len(quad) == 4 for quad in quads)


def test_a_degenerate_panel_is_refused() -> None:
    with pytest.raises(ValueError, match="at least 3 points"):
        panel_prism([(0.0, 0.0), (1.0, 1.0)], Placement("p", Plane.FLAT), 3.0)


# --------------------------------------------------------------------------- #
# the model guards itself
# --------------------------------------------------------------------------- #
def test_an_empty_assembly_is_refused() -> None:
    with pytest.raises(ValueError, match="at least one placement"):
        Assembly(())


def test_an_assembly_reports_parts_the_design_does_not_cut() -> None:
    assembly = Assembly((Placement("lid", Plane.FLAT), Placement("base", Plane.FLAT)))
    assert assembly.missing_from(["base"]) == {"lid"}
    assert assembly.missing_from(["base", "lid"]) == set()


def test_a_design_refuses_an_assembly_of_parts_it_does_not_cut() -> None:
    part = Part("base", geo.rect_ring(100.0, 100.0))
    with pytest.raises(ValueError, match="assembles part"):
        Design(
            slug="x", name="X", niche="test", description="",
            parts=[part],
            assembly=Assembly((Placement("nonexistent", Plane.FLAT),)),
        )


# --------------------------------------------------------------------------- #
# the real generators
# --------------------------------------------------------------------------- #
def test_a_box_assembles_to_the_size_it_was_asked_for() -> None:
    design = get_generator("boxes").make(width=200, depth=140, height=90, kerf=0.0)
    width, depth, height = assembled_size(design)
    assert (width, depth, height) == pytest.approx((200.0, 140.0, 90.0), abs=0.6)


def test_a_shelf_unit_assembles_to_the_size_it_was_asked_for() -> None:
    design = get_generator("furniture").make(
        form="shelf", width=760, depth=320, height=1100, shelves=4
    )
    assert assembled_size(design) == pytest.approx((760.0, 320.0, 1100.0), abs=0.6)


def test_a_table_assembles_to_the_size_it_was_asked_for() -> None:
    design = get_generator("furniture").make(
        form="table", width=900, depth=500, height=450
    )
    assert assembled_size(design) == pytest.approx((900.0, 500.0, 450.0), abs=0.6)


def test_a_dock_is_taller_than_it_is_deep() -> None:
    design = get_generator("stands").make(size="tablet")
    width, depth, height = assembled_size(design)
    assert height > depth, "a dock that is not taller than deep is lying flat"


def test_shelves_sit_at_the_heights_their_mortises_were_cut_for() -> None:
    """The drawing and the joinery read the same list, so they cannot drift."""
    from dxfgen.niches.furniture import shelf_levels

    generator = get_generator("furniture")
    params = generator.parse({"form": "shelf", "shelves": 3, "height": 900})
    design = generator.make(form="shelf", shelves=3, height=900)
    placed = {
        placement.part: placement.origin[2]
        for placement in design.assembly.placements
        if placement.part.startswith("shelf")
    }
    assert sorted(placed.values()) == pytest.approx(sorted(shelf_levels(params)))


def test_box_walls_stand_on_all_four_sides() -> None:
    design = get_generator("boxes").make(width=200, depth=140, height=90)
    planes = {p.part: p.plane for p in design.assembly.placements}
    assert planes["front"] is Plane.FRONT and planes["back"] is Plane.FRONT
    assert planes["left"] is Plane.SIDE and planes["right"] is Plane.SIDE
    assert planes["floor"] is Plane.FLAT


# --------------------------------------------------------------------------- #
# the drawing
# --------------------------------------------------------------------------- #
def test_an_assembly_drawing_is_written(tmp_path: Path) -> None:
    design = get_generator("boxes").make()
    out = render_assembly(design, tmp_path / "deep" / "iso.png", px_width=500, dpi=100)
    assert out.exists()
    with Image.open(out) as image:
        assert image.size[0] == 500


def test_a_flat_product_has_nothing_to_draw(tmp_path: Path) -> None:
    design = get_generator("trays").make()
    assert design.assembly is None
    with pytest.raises(ValueError, match="no assembly"):
        render_assembly(design, tmp_path / "x.png")
    with pytest.raises(ValueError, match="no assembly"):
        assembled_size(design)


def test_a_taller_object_makes_a_taller_drawing(tmp_path: Path) -> None:
    short = get_generator("boxes").make(width=200, depth=140, height=50)
    tall = get_generator("boxes").make(width=200, depth=140, height=200)
    a = render_assembly(short, tmp_path / "a.png", px_width=400, dpi=100, caption=False)
    b = render_assembly(tall, tmp_path / "b.png", px_width=400, dpi=100, caption=False)
    with Image.open(a) as first, Image.open(b) as second:
        assert second.size[1] > first.size[1]
