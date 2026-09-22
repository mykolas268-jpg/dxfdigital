"""DXF export.

The output profile is deliberately narrow, because every CAM package agrees
on this subset and disagrees about everything else:

* **DXF R2010** (AC1024), the newest format every current CAM release reads.
* **Millimetres**, with ``$INSUNITS = 4`` and ``$MEASUREMENT = 1`` so nothing
  downstream has to guess the scale.
* **Closed ``LWPOLYLINE``** for every contour, **``CIRCLE``** for drills,
  **``TEXT``** for annotation only.  No splines, no ellipses, no blocks, no
  hatches, no dimensions.
* **One layer per operation**, following :mod:`dxfgen.core.layers`.
* **Bottom-left of the bounding box at the origin.**

A design is validated before anything is written.  An invalid design raises
:class:`~dxfgen.core.validate.ValidationError` instead of producing a file.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import ezdxf
from ezdxf import zoom
from ezdxf.document import Drawing
from ezdxf.enums import TextEntityAlignment

from .design import Design, Machine, Mode
from .layers import layer_def
from .validate import Report, ValidationConfig, validate_design

__all__ = [
    "reproducible_output",
    "PROVENANCE_MODE",
    "PROVENANCE_TOOL",
    "PROVENANCE_KERF",
    "PROVENANCE_THICKNESS",
    "PROVENANCE_SLUG",
    "machine_from_document",
    "DXF_VERSION",
    "build_document",
    "write_dxf",
    "audit_file",
]

#: Target DXF release.  R2010 is the sweet spot for CAM compatibility.
DXF_VERSION = "R2010"

_ALIGNMENT = {
    "left": TextEntityAlignment.LEFT,
    "center": TextEntityAlignment.MIDDLE_CENTER,
    "right": TextEntityAlignment.RIGHT,
}


def _prepare_header(doc: Drawing) -> None:
    """Set units so downstream software never has to guess the scale."""
    doc.units = ezdxf.units.MM
    header = doc.header
    header["$INSUNITS"] = 4  # millimetres
    header["$MEASUREMENT"] = 1  # metric
    header["$LUNITS"] = 2  # decimal
    header["$AUNITS"] = 0  # decimal degrees
    header["$LUPREC"] = 3
    header["$LWDISPLAY"] = 1


def _set_extents(doc: Drawing, design: Design) -> None:
    """Record the drawing extents and frame the view on the geometry.

    ezdxf overwrites the ``$EXT*`` and ``$LIM*`` header variables from the
    modelspace layout when the file is written, so the layout is where the
    values have to be set.  Getting this right is what makes the file open
    zoomed to the part instead of somewhere in empty space.
    """
    x0, y0, x1, y1 = design.bbox()
    msp = doc.modelspace()
    msp.dxf.extmin = (x0, y0, 0.0)
    msp.dxf.extmax = (x1, y1, 0.0)
    msp.dxf.limmin = (x0, y0)
    msp.dxf.limmax = (x1, y1)
    doc.header["$EXTMIN"] = (x0, y0, 0.0)
    doc.header["$EXTMAX"] = (x1, y1, 0.0)
    doc.header["$LIMMIN"] = (x0, y0)
    doc.header["$LIMMAX"] = (x1, y1)
    zoom.window(msp, (x0, y0), (x1, y1))


def _prepare_layers(doc: Drawing, design: Design) -> None:
    """Create every layer the design uses, with its standard colour."""
    for name in design.layers_used():
        spec = layer_def(name)
        if name in doc.layers:
            layer = doc.layers.get(name)
        else:
            layer = doc.layers.add(name)
        layer.color = spec.color
        layer.dxf.lineweight = spec.lineweight
        layer.description = spec.description
        # INFO is annotation: switching plotting off keeps it out of any
        # toolpath a CAM package derives from plottable layers.
        layer.dxf.plot = 0 if not spec.machined else 1


#: Header custom variables that record which machine a file was cut for.  A
#: DXF has nowhere standard to say "this is laser work", so a checker reading
#: the file back would otherwise have to assume, and assuming router rules on
#: a laser file flags every finger joint.  Any CAD package ignores unknown
#: custom variables, so this costs nothing to carry.
PROVENANCE_MODE: str = "DXFGEN_MODE"
PROVENANCE_TOOL: str = "DXFGEN_TOOL_DIAMETER"
PROVENANCE_KERF: str = "DXFGEN_KERF"
PROVENANCE_THICKNESS: str = "DXFGEN_THICKNESS"
PROVENANCE_SLUG: str = "DXFGEN_DESIGN"


@contextmanager
def reproducible_output() -> "Iterator[None]":
    """Write DXF files that depend only on their contents.

    ezdxf stamps every document with the current time and a fresh version
    GUID at save.  That is right for a drawing somebody is editing and wrong
    for a generated one: it means the same seed produces files that differ in
    twelve lines of metadata, so a seller who regenerates a bundle cannot tell
    a real change from a re-run and no checksum is stable.

    ezdxf exposes exactly this as ``write_fixed_meta_data_for_testing``.  The
    name says testing, but what it does - a fixed date and a constant GUID -
    is what a generated file wants, and a DXF fingerprint GUID means nothing
    for a file that is a pure function of its parameters.  It is a global
    option, so it is set only around our own writing and put back afterwards,
    rather than changed for anyone who imports this package.

    The document has to be *built* inside this as well as saved inside it:
    ezdxf stamps a file once when it is created and again when it is written,
    and wrapping only the save leaves the creation stamp ticking.

    Yields:
        Nothing; the option is restored on the way out, exceptions included.
    """
    previous = ezdxf.options.write_fixed_meta_data_for_testing
    ezdxf.options.write_fixed_meta_data_for_testing = True
    try:
        yield
    finally:
        ezdxf.options.write_fixed_meta_data_for_testing = previous


def _write_provenance(doc: Drawing, design: Design) -> None:
    """Record the machine settings the design was built for.

    Args:
        doc: The document being built.
        design: The design being written.
    """
    machine = design.machine
    for tag, value in (
        (PROVENANCE_MODE, machine.mode.value),
        (PROVENANCE_TOOL, f"{machine.tool_diameter:g}"),
        (PROVENANCE_KERF, f"{machine.kerf:g}"),
        (PROVENANCE_THICKNESS, f"{design.thickness:g}"),
        (PROVENANCE_SLUG, design.slug),
    ):
        doc.header.custom_vars.append(tag, value)


def build_document(design: Design) -> Drawing:
    """Build an in-memory DXF document from a design.

    The design is used exactly as given; call :meth:`Design.normalized` first
    if it has not already been placed at the origin.

    Args:
        design: The design to render.

    Returns:
        An ezdxf :class:`~ezdxf.document.Drawing`.
    """
    # setup=False keeps the tables to what this project actually writes: the
    # "Standard" text style and three linetypes.  The full setup adds 20 KB of
    # unused text and dimension styles to every file and gives a CAM importer
    # more table entries to choke on.
    doc = ezdxf.new(DXF_VERSION, setup=False)
    _prepare_header(doc)
    _write_provenance(doc, design)
    _prepare_layers(doc, design)
    msp = doc.modelspace()

    for part in design.placed_parts():
        for ring, layer in part.cut_rings():
            msp.add_lwpolyline(ring, close=True, dxfattribs={"layer": layer})
        for drill in part.drills:
            msp.add_circle(
                drill.center, drill.radius, dxfattribs={"layer": "DRILL"}
            )
        for contour in part.engrave:
            msp.add_lwpolyline(
                contour.points,
                close=contour.closed,
                dxfattribs={"layer": contour.layer},
            )
        for label in part.labels:
            text = msp.add_text(
                label.text,
                dxfattribs={
                    "layer": label.layer,
                    "height": label.height,
                    "rotation": label.rotation,
                    "style": "Standard",
                },
            )
            text.set_placement(
                label.position, align=_ALIGNMENT[label.align]
            )
    _set_extents(doc, design)
    return doc


def write_dxf(
    design: Design,
    path: str | Path,
    config: ValidationConfig | None = None,
    validate: bool = True,
) -> tuple[Path, Report]:
    """Validate a design and write it to a DXF file.

    Args:
        design: The design to export.  It is normalised to the origin first.
        path: Destination file path; parent directories are created.
        config: Validation limits.
        validate: Set ``False`` only to inspect a known-bad design; the
            normal path refuses to write invalid geometry.

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
    # Both the build and the save have to be inside: ezdxf stamps the document
    # once when it is created and again when it is written, and covering only
    # the save leaves the creation stamp ticking.
    with reproducible_output():
        doc = build_document(placed)
        doc.saveas(out)
    return out, report


def audit_file(path: str | Path) -> tuple[list[str], list[str]]:
    """Run ezdxf's own structural audit on a written file.

    This is the closest available stand-in for opening the file in CAD: it
    walks the entity database, checks handles, layer references, and DXF
    attribute validity.

    Args:
        path: DXF file to audit.

    Returns:
        ``(errors, fixes)`` as lists of human readable strings.
    """
    doc = ezdxf.readfile(str(path))
    auditor = doc.audit()
    return (
        [str(e) for e in auditor.errors],
        [str(f) for f in auditor.fixes],
    )


def machine_from_document(doc: Drawing) -> Machine | None:
    """Recover the machine a file was written for, if it says.

    Only files this tool wrote carry the information; anything else returns
    ``None`` so the caller can ask the operator rather than assume.

    Args:
        doc: A loaded DXF document.

    Returns:
        The machine, or ``None`` if the file does not record one or records
        something unreadable.
    """
    custom = doc.header.custom_vars
    mode_name = custom.get(PROVENANCE_MODE, "")
    if not mode_name:
        return None
    try:
        mode = Mode(mode_name.strip().lower())
        tool = float(custom.get(PROVENANCE_TOOL, "") or Machine().tool_diameter)
        kerf = float(custom.get(PROVENANCE_KERF, "") or Machine().kerf)
    except (ValueError, TypeError):
        return None
    return Machine(mode=mode, tool_diameter=tool, kerf=kerf)
