"""Manufacturing limits a design is built and judged against.

These live apart from the checks themselves so that a :class:`Design` can
carry the limits it was designed to, and the validator can then hold it to its
own standard rather than to a global default.  A 6 mm laser design with a 2 mm
pocket floor and an 18 mm router design with a 5 mm floor are both correct; the
only way to know which applies is to ask the design.
"""

from __future__ import annotations

from dataclasses import dataclass

from .geometry import MIN_SEGMENT

__all__ = ["ValidationConfig"]


@dataclass(frozen=True)
class ValidationConfig:
    """Tunable limits for the checks.

    Attributes:
        min_wall: Minimum material left between a cut feature and the part
            edge, and between two features, in mm.
        pocket_floor: Material that must remain under the deepest pocket, mm.
        feature_factor: A cut region must be at least this multiple of the
            cutter diameter wide; below it there is no chip clearance.
        tight_factor: A cut region narrower than this multiple of the cutter
            diameter forces the cutter to run at full engagement with no room
            for a separate finishing pass.  Cuttable, but it earns a warning.
        max_corner_residual: How much material a round cutter may leave in a
            corner, measured as the thickness of the leftover sliver in mm.
            A relieved corner leaves only tessellation-scale slivers of
            about 0.05 mm, while an un-relieved square corner leaves
            ``0.34 x`` the cutter radius - 1.07 mm for a 6.35 mm cutter - so
            this threshold separates the two by a wide margin.
        origin_tolerance: How far the bounding box corner may sit from the
            origin, mm.
        duplicate_tolerance: Coordinate rounding used to detect duplicate
            contours, mm.
        min_segment: Segments shorter than this count as zero-length, mm.
        min_drill_diameter: Smallest sensible drill, mm.
        min_feature_area: Cut regions smaller than this are reported as
            probable mistakes, mm^2.
        max_points_per_contour: Above this a contour is flagged as bloated.
        check_reachability: Whether to run the (relatively slow) cutter
            reachability test.
    """

    min_wall: float = 8.0
    pocket_floor: float = 5.0
    feature_factor: float = 1.1
    tight_factor: float = 2.0
    max_corner_residual: float = 0.35
    origin_tolerance: float = 0.01
    duplicate_tolerance: float = 0.05
    min_segment: float = MIN_SEGMENT
    min_drill_diameter: float = 2.0
    min_feature_area: float = 4.0
    max_points_per_contour: int = 20000
    check_reachability: bool = True
