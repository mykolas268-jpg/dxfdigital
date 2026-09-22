"""How the flat parts go back together, and how to draw that.

A cut sheet is what the machine needs and the worst possible picture of what
the buyer gets.  A nest of six rectangles says nothing about a box; eight
slot-together docks on a contact sheet all read as "three rectangles with
slots in them".  The geometry is right and the communication fails.

So a design may carry an :class:`Assembly`: where each part sits in the
finished object.  Everything here assumes an orthogonal assembly - panels
lying in one of three planes, at right angles to each other - because that is
what slot-together furniture, finger-jointed boxes and tab-and-mortise docks
are.  Anything mitred, curved or hinged is out of scope, and a generator that
cannot describe itself this way simply does not carry an assembly.

The projection is isometric: no perspective, no camera, no hidden-surface
solver.  Panels are sorted back to front by their centroid and painted in that
order, which is exact for panels that do not interpenetrate and is what these
assemblies are.  It is a drawing, not a render.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Sequence

from .geometry import Point, Ring

__all__ = [
    "Plane",
    "Placement",
    "Assembly",
    "Point3",
    "ISO_COS",
    "ISO_SIN",
    "to_world",
    "plane_normal",
    "project",
    "depth_of",
    "panel_prism",
]

#: A point in the assembled object, in mm.
Point3 = tuple[float, float, float]

#: Isometric axis foreshortening.  x runs right and up at 30 degrees, y left
#: and up at 30 degrees, z straight up, so the viewer sits above and in front
#: and sees the top and two sides of any box.
ISO_COS: float = math.cos(math.radians(30.0))
ISO_SIN: float = math.sin(math.radians(30.0))


class Plane(str, Enum):
    """Which way a flat panel faces in the assembled object.

    A panel is cut in its own 2D ``(u, v)`` space and is ``thickness`` deep.
    The plane says how that space maps into the object and which way the
    material goes.
    """

    FLAT = "flat"
    """Lying down: ``(u, v)`` to ``(x, y)``, material upwards. A shelf, a
    floor, a lid, a table top."""

    FRONT = "front"
    """Standing, facing the viewer: ``(u, v)`` to ``(x, z)``, material
    backwards. A back panel, the long wall of a box."""

    SIDE = "side"
    """Standing, facing sideways: ``(u, v)`` to ``(y, z)``, material
    sideways. A gable end, the short wall of a box."""


def plane_normal(plane: Plane) -> Point3:
    """The direction a panel's material is laid into, as a unit vector.

    Args:
        plane: The panel's plane.

    Returns:
        The unit normal.
    """
    return {
        Plane.FLAT: (0.0, 0.0, 1.0),
        Plane.FRONT: (0.0, 1.0, 0.0),
        Plane.SIDE: (1.0, 0.0, 0.0),
    }[plane]


@dataclass(frozen=True)
class Placement:
    """Where one part sits in the assembled object.

    Attributes:
        part: The :attr:`~dxfgen.core.design.Part.name` this places.  A name
            may be placed more than once, which is how one cut part serves as
            both the left and right side.
        plane: Which way the panel faces.
        origin: Where the part's own ``(0, 0)`` lands, in mm.
        label: Optional name for this instance, when one part is placed
            several times and the instances need telling apart.
    """

    part: str
    plane: Plane
    origin: Point3 = (0.0, 0.0, 0.0)
    label: str = ""

    def to_world(self, point: Point) -> Point3:
        """Map a point of the flat part into the assembled object.

        Args:
            point: A point in the part's own 2D space, in mm.

        Returns:
            The same point in the assembly, in mm.
        """
        return to_world(point, self.plane, self.origin)


def to_world(point: Point, plane: Plane, origin: Point3) -> Point3:
    """Map a 2D point on a panel into assembly coordinates.

    Args:
        point: ``(u, v)`` on the panel, in mm.
        plane: Which way the panel faces.
        origin: Where the panel's ``(0, 0)`` sits.

    Returns:
        ``(x, y, z)`` in mm.
    """
    u, v = point
    ox, oy, oz = origin
    if plane is Plane.FLAT:
        return (ox + u, oy + v, oz)
    if plane is Plane.FRONT:
        return (ox + u, oy, oz + v)
    return (ox, oy + u, oz + v)


def project(point: Point3) -> Point:
    """Project an assembly point to the drawing plane.

    Args:
        point: ``(x, y, z)`` in mm.

    Returns:
        ``(screen_x, screen_y)``, in the same units, y upwards.
    """
    x, y, z = point
    return ((x - y) * ISO_COS, (x + y) * ISO_SIN + z)


def depth_of(point: Point3) -> float:
    """How near the viewer a point is; larger is nearer.

    The projection collapses the direction ``(1, 1, -1)``, so the viewer sits
    along ``(-1, -1, 1)`` and nearness is that dot product.

    Args:
        point: ``(x, y, z)`` in mm.

    Returns:
        A depth score, comparable between points.
    """
    x, y, z = point
    return -x - y + z


@dataclass(frozen=True)
class Assembly:
    """Every part's place in the finished object.

    Attributes:
        placements: One entry per panel instance.
        caption: What the assembled thing is, for the drawing.
    """

    placements: tuple[Placement, ...] = ()
    caption: str = ""

    def __post_init__(self) -> None:
        if not self.placements:
            raise ValueError("an assembly needs at least one placement")

    def parts_used(self) -> set[str]:
        """Every part name this assembly places."""
        return {placement.part for placement in self.placements}

    def missing_from(self, names: Iterable[str]) -> set[str]:
        """Placed part names that are not among ``names``.

        A placement naming a part the design does not cut is a bug in the
        generator, not something to draw around.

        Args:
            names: The part names the design actually has.

        Returns:
            The unknown names, empty when the assembly is consistent.
        """
        return self.parts_used() - set(names)


def panel_prism(
    outline: Sequence[Point], placement: Placement, thickness: float
) -> tuple[list[Point3], list[Point3], list[tuple[Point3, Point3, Point3, Point3]]]:
    """Extrude a panel outline into the assembly.

    Args:
        outline: The panel's closed outline in its own 2D space.
        placement: Where it sits.
        thickness: Material thickness in mm.

    Returns:
        ``(near_face, far_face, side_quads)``.  The near face is the one
        towards the viewer, so a caller painting far, sides, near in that
        order gets a solid panel with its edge thickness showing.

    Raises:
        ValueError: If the outline has fewer than three points.
    """
    if len(outline) < 3:
        raise ValueError(f"a panel needs at least 3 points, got {len(outline)}")
    nx, ny, nz = plane_normal(placement.plane)
    base = [placement.to_world(point) for point in outline]
    lifted = [(x + nx * thickness, y + ny * thickness, z + nz * thickness)
              for x, y, z in base]

    # Whichever face has the greater depth score faces the viewer.
    if depth_of(lifted[0]) >= depth_of(base[0]):
        near, far = lifted, base
    else:
        near, far = base, lifted

    quads: list[tuple[Point3, Point3, Point3, Point3]] = []
    count = len(base)
    for index in range(count):
        following = (index + 1) % count
        quads.append((far[index], far[following], near[following], near[index]))
    return near, far, quads
