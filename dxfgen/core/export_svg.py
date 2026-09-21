"""SVG export.

SVG is the format laser software wants: LightBurn, LaserGRBL and the browser
all read it, and unlike DXF it carries colour that survives the import.

The output is deliberately literal:

* ``width`` and ``height`` are given in millimetres and the ``viewBox`` uses
  the same numbers, so one user unit is one millimetre and nothing downstream
  has to guess a DPI.
* Y is flipped on the way out, because SVG counts downwards and every other
  part of this project counts upwards.  Coordinates are flipped individually
  rather than by a group transform, so text comes out the right way up.
* One ``<g>`` per layer, named and coloured to match the DXF.
* Closed contours are single ``<path>`` elements ending in ``Z``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable
from xml.sax.saxutils import escape

from ezdxf import colors as ezcolors

from .design import Design
from .geometry import Point
from .layers import layer_def
from .limits import ValidationConfig
from .validate import Report, validate_design

__all__ = ["DEFAULT_STROKE_MM", "aci_to_hex", "build_svg", "write_svg"]

#: Stroke width in mm.  Thin enough to show the true cut line, thick enough to
#: see on screen.
DEFAULT_STROKE_MM = 0.25

_ANCHOR = {"left": "start", "center": "middle", "right": "end"}
_BASELINE = {"left": "auto", "center": "middle", "right": "auto"}


def aci_to_hex(aci: int) -> str:
    """Convert an AutoCAD Color Index to an SVG hex colour.

    Args:
        aci: Colour index 1-255.

    Returns:
        A ``#rrggbb`` string.
    """
    r, g, b = ezcolors.aci2rgb(aci)
    return f"#{r:02x}{g:02x}{b:02x}"


def _num(value: float) -> str:
    """Format a coordinate compactly, to micron precision."""
    text = f"{value:.3f}".rstrip("0").rstrip(".")
    return text if text not in ("-0", "") else "0"


def _path_data(points: Iterable[Point], height: float, closed: bool) -> str:
    """Build an SVG path string, flipping Y as it goes."""
    parts: list[str] = []
    for index, (x, y) in enumerate(points):
        command = "M" if index == 0 else "L"
        parts.append(f"{command}{_num(x)},{_num(height - y)}")
    if closed:
        parts.append("Z")
    return "".join(parts)


def build_svg(
    design: Design, stroke_mm: float = DEFAULT_STROKE_MM, pocket_fill: bool = True
) -> str:
    """Render a design as an SVG document.

    Args:
        design: The design to render; use :meth:`Design.normalized` first.
        stroke_mm: Stroke width in mm.
        pocket_fill: Tint pocket areas so recesses read at a glance.  The fill
            is cosmetic and ignored by every CAM package, which cuts the
            outline.

    Returns:
        The SVG document as a string.
    """
    x0, y0, x1, y1 = design.bbox()
    width, height = x1 - x0, y1 - y0
    parts = design.placed_parts()

    by_layer: dict[str, list[str]] = {}

    def add(layer: str, element: str) -> None:
        by_layer.setdefault(layer, []).append(element)

    for part in parts:
        for ring, layer in part.cut_rings():
            add(layer, f'<path d="{_path_data(ring, height, True)}"/>')
        for drill in part.drills:
            cx, cy = drill.center
            add(
                "DRILL",
                f'<circle cx="{_num(cx)}" cy="{_num(height - cy)}" '
                f'r="{_num(drill.radius)}"/>',
            )
        for contour in part.engrave:
            add(
                contour.layer,
                f'<path d="{_path_data(contour.points, height, contour.closed)}"/>',
            )
        for label in part.labels:
            lx, ly = label.position
            sx, sy = _num(lx), _num(height - ly)
            rotate = (
                f' transform="rotate({_num(-label.rotation)} {sx} {sy})"'
                if abs(label.rotation) > 1e-9
                else ""
            )
            add(
                label.layer,
                f'<text x="{sx}" y="{sy}" font-size="{_num(label.height)}" '
                f'font-family="sans-serif" text-anchor="{_ANCHOR[label.align]}" '
                f'dominant-baseline="{_BASELINE[label.align]}"{rotate}>'
                f"{escape(label.text)}</text>",
            )

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" version="1.1" '
        f'width="{_num(width)}mm" height="{_num(height)}mm" '
        f'viewBox="0 0 {_num(width)} {_num(height)}">',
        f"  <title>{escape(design.name)}</title>",
        f"  <desc>{escape(design.description)}</desc>",
    ]
    for layer in design.layers_used():
        elements = by_layer.get(layer)
        if not elements:
            continue
        spec = layer_def(layer)
        colour = aci_to_hex(spec.color)
        if spec.depth is not None and pocket_fill:
            style = f'fill="{colour}" fill-opacity="0.15" stroke="{colour}"'
        elif layer == "INFO":
            style = f'fill="{colour}" stroke="none"'
        else:
            style = f'fill="none" stroke="{colour}"'
        lines.append(
            f'  <g id="{escape(layer)}" {style} stroke-width="{_num(stroke_mm)}" '
            f'stroke-linejoin="round">'
        )
        lines.extend(f"    {element}" for element in elements)
        lines.append("  </g>")
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def write_svg(
    design: Design,
    path: str | Path,
    config: ValidationConfig | None = None,
    validate: bool = True,
    stroke_mm: float = DEFAULT_STROKE_MM,
) -> tuple[Path, Report]:
    """Validate a design and write it to an SVG file.

    Args:
        design: The design to export; it is normalised to the origin first.
        path: Destination file path; parent directories are created.
        config: Validation limits; defaults to the design's own.
        validate: Set ``False`` only to inspect a known-bad design.
        stroke_mm: Stroke width in mm.

    Returns:
        ``(written_path, report)``.

    Raises:
        ValidationError: If the design fails validation and ``validate`` is
            ``True``.  No file is written in that case.
    """
    placed = design.normalized()
    report = validate_design(placed, config)
    if validate:
        report.raise_for_status()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_svg(placed, stroke_mm), encoding="utf-8")
    return out, report
