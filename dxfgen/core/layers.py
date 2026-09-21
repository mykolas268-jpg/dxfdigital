"""Standard layer definitions shared by every exported file.

A single vocabulary of layers keeps downstream CAM setup mechanical: the
operator maps one layer to one toolpath type and never has to guess.

Layers
------
``CUT_OUTSIDE``
    Outer profile of a part.  CAM offsets the tool *outside* the line.
``CUT_INSIDE``
    Through cuts that remove material from inside the part (holes, slots,
    handle cutouts).  CAM offsets the tool *inside* the line.
``POCKET_<depth>``
    A pocket/clearing operation at one specific depth in millimetres.  One
    layer per distinct depth, e.g. ``POCKET_8`` or ``POCKET_1.5``.
``ENGRAVE``
    Single-line or V-carve decoration and text.  Cut on the line.
``DRILL``
    Circles marking drill points.  Diameter is the circle diameter.
``INFO``
    Dimensions, part labels, material notes.  Never machined.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "LayerDef",
    "CUT_OUTSIDE",
    "CUT_INSIDE",
    "ENGRAVE",
    "DRILL",
    "INFO",
    "POCKET_PREFIX",
    "STANDARD_LAYERS",
    "POCKET_COLORS",
    "pocket_layer_name",
    "parse_pocket_depth",
    "is_pocket_layer",
    "is_cut_layer",
    "layer_def",
    "format_depth",
]

CUT_OUTSIDE = "CUT_OUTSIDE"
CUT_INSIDE = "CUT_INSIDE"
ENGRAVE = "ENGRAVE"
DRILL = "DRILL"
INFO = "INFO"
POCKET_PREFIX = "POCKET_"


@dataclass(frozen=True)
class LayerDef:
    """Definition of one DXF layer.

    Attributes:
        name: DXF layer name.
        color: AutoCAD Color Index (ACI) 1-255.
        description: Human readable purpose, used in README generation.
        lineweight: Lineweight in 1/100 mm as DXF expects it.
        machined: ``False`` for annotation-only layers that must never be cut.
        depth: Pocket depth in mm for ``POCKET_*`` layers, else ``None``.
    """

    name: str
    color: int
    description: str
    lineweight: int = 25
    machined: bool = True
    depth: float | None = None


#: ACI colours chosen to stay legible on both black and white CAD backgrounds.
STANDARD_LAYERS: dict[str, LayerDef] = {
    CUT_OUTSIDE: LayerDef(
        CUT_OUTSIDE, 1, "Outer profile - cut through, tool outside the line", 35
    ),
    CUT_INSIDE: LayerDef(
        CUT_INSIDE, 5, "Interior through cuts - tool inside the line", 35
    ),
    ENGRAVE: LayerDef(ENGRAVE, 6, "Engrave / V-carve on the line", 13),
    DRILL: LayerDef(DRILL, 2, "Drill points - circle diameter = hole diameter", 13),
    INFO: LayerDef(
        INFO, 8, "Annotation only - never machine this layer", 9, machined=False
    ),
}

#: Deterministic palette for pocket layers, cool colours to contrast with cuts.
POCKET_COLORS: tuple[int, ...] = (3, 4, 74, 94, 114, 134, 154, 174)


def format_depth(depth_mm: float) -> str:
    """Format a depth for use inside a layer name.

    Integral depths render without a decimal point so an 8 mm pocket becomes
    ``POCKET_8`` rather than ``POCKET_8.0``.

    Args:
        depth_mm: Pocket depth in millimetres, must be > 0.

    Returns:
        The depth as a compact string, e.g. ``"8"`` or ``"1.5"``.

    Raises:
        ValueError: If ``depth_mm`` is not positive.
    """
    if depth_mm <= 0:
        raise ValueError(f"pocket depth must be > 0, got {depth_mm}")
    rounded = round(depth_mm, 2)
    if abs(rounded - round(rounded)) < 1e-9:
        return str(int(round(rounded)))
    return f"{rounded:g}"


def pocket_layer_name(depth_mm: float) -> str:
    """Return the canonical layer name for a pocket at ``depth_mm``.

    Args:
        depth_mm: Pocket depth in millimetres.

    Returns:
        Layer name such as ``"POCKET_8"``.
    """
    return f"{POCKET_PREFIX}{format_depth(depth_mm)}"


def is_pocket_layer(name: str) -> bool:
    """Return ``True`` when ``name`` is a well formed pocket layer name."""
    return parse_pocket_depth(name) is not None


def parse_pocket_depth(name: str) -> float | None:
    """Extract the depth in mm encoded in a pocket layer name.

    Args:
        name: Layer name to inspect.

    Returns:
        The depth in millimetres, or ``None`` if ``name`` is not a pocket
        layer.
    """
    if not name.startswith(POCKET_PREFIX):
        return None
    tail = name[len(POCKET_PREFIX) :]
    try:
        depth = float(tail)
    except ValueError:
        return None
    return depth if depth > 0 else None


def is_cut_layer(name: str) -> bool:
    """Return ``True`` for layers whose geometry must be a closed contour."""
    return name in (CUT_OUTSIDE, CUT_INSIDE) or is_pocket_layer(name)


def pocket_color(depth_mm: float) -> int:
    """Pick a deterministic ACI colour for a pocket depth.

    Colours only need to be *distinct per depth within one file*; the mapping
    is stable across runs so regenerated files diff cleanly.

    Args:
        depth_mm: Pocket depth in millimetres.

    Returns:
        An ACI colour index.
    """
    key = int(round(depth_mm * 2)) % len(POCKET_COLORS)
    return POCKET_COLORS[key]


def layer_def(name: str) -> LayerDef:
    """Resolve any layer name to a :class:`LayerDef`.

    Pocket layers are synthesised on demand since their names carry data.

    Args:
        name: Layer name.

    Returns:
        The matching :class:`LayerDef`.

    Raises:
        KeyError: If ``name`` is not part of the dxfgen layer standard.
    """
    if name in STANDARD_LAYERS:
        return STANDARD_LAYERS[name]
    depth = parse_pocket_depth(name)
    if depth is None:
        raise KeyError(
            f"{name!r} is not a dxfgen standard layer; "
            f"expected one of {sorted(STANDARD_LAYERS)} or {POCKET_PREFIX}<depth>"
        )
    return LayerDef(
        name,
        pocket_color(depth),
        f"Pocket cleared to {format_depth(depth)} mm deep",
        25,
        machined=True,
        depth=depth,
    )
