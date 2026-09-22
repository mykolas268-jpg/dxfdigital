"""The command line: list what can be made, make it, bundle it, check it.

Four commands, because there are four things a person actually does with this
tool.  ``list`` to find out what exists and what it takes, ``make`` for one
design with parameters you chose, ``bundle`` for a whole niche ready to sell,
and ``validate`` to ask whether a DXF - this tool's or anybody else's - will
import cleanly and cut.

Exit codes are meant for scripts: 0 succeeded, 1 the work failed on its own
terms (a design that will not validate, a DXF with errors), 2 the command was
wrong (unknown niche, bad parameter).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated, Any, Optional

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.table import Table

from .bundles.builder import DEFAULT_SELLER, Bundle, build_all, write_design
from .core.design import Machine, Mode
from .core.export_pdf import PAPER_SIZES, A4, PaperSize
from .core.validate import Report, Severity, ValidationError as GeometryInvalid
from .core.validate import validate_dxf_file
from .niches import all_generators, get_generator, niche_names

__all__ = ["app", "main"]

ALL_FORMATS = ("dxf", "svg", "pdf", "preview", "mockup", "readme")

#: Etsy, Gumroad and most other download marketplaces cap a single file at
#: 20 MB.  A bundle over that is not an error, but it cannot be uploaded as
#: one file, so say so rather than let the seller find out at upload time.
MARKETPLACE_FILE_LIMIT_MB: float = 20.0

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Generate machine-ready DXF files for CNC routers and lasers.",
)
out = Console()
err = Console(stderr=True)

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2


# --------------------------------------------------------------------------- #
# shared plumbing
# --------------------------------------------------------------------------- #
def _fail(message: str, code: int = EXIT_USAGE) -> "typer.Exit":
    """Print an error and build the exception that ends the command."""
    err.print(f"[bold red]error[/bold red] {message}")
    return typer.Exit(code)


def _parse_overrides(pairs: list[str]) -> dict[str, str]:
    """Turn ``key=value`` strings into a mapping.

    Args:
        pairs: Raw ``-p`` arguments.

    Returns:
        The overrides, later coerced to their real types by pydantic.

    Raises:
        typer.Exit: If a pair has no ``=`` or an empty key.
    """
    values: dict[str, str] = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        key = key.strip()
        if not sep or not key:
            raise _fail(f"expected key=value, got {pair!r}")
        values[key] = value.strip()
    return values


def _load_config(path: Path) -> dict[str, Any]:
    """Read a YAML preset.

    Args:
        path: The file to read.

    Returns:
        Its top-level mapping.

    Raises:
        typer.Exit: If it is missing or is not a mapping.
    """
    import yaml

    if not path.is_file():
        raise _fail(f"no such config file: {path}")
    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        raise _fail(f"{path} must contain a mapping, got {type(loaded).__name__}")
    return loaded


def _resolve_preset(preset: str) -> Path:
    """Find a named preset shipped with the package or in the working tree.

    Args:
        preset: A bare name, with or without the ``.yaml`` suffix.

    Returns:
        The file's path.

    Raises:
        typer.Exit: If no such preset exists, listing the ones that do.
    """
    stem = preset[:-5] if preset.endswith(".yaml") else preset
    shipped = Path(__file__).parent / "configs" / f"{stem}.yaml"
    local = Path.cwd() / f"{stem}.yaml"
    for candidate in (Path(preset), local, shipped):
        if candidate.is_file():
            return candidate
    available = sorted(p.stem for p in (Path(__file__).parent / "configs").glob("*.yaml"))
    raise _fail(f"unknown preset {preset!r}; available: {', '.join(available) or 'none'}")


def _check_formats(formats: list[str]) -> tuple[str, ...]:
    """Validate the ``--format`` list.

    Args:
        formats: Requested output kinds.

    Returns:
        The kinds as a tuple, or every kind if none were named.

    Raises:
        typer.Exit: If a kind is unknown.
    """
    if not formats:
        return ALL_FORMATS
    unknown = sorted(set(formats) - set(ALL_FORMATS))
    if unknown:
        raise _fail(
            f"unknown format(s): {', '.join(unknown)}; "
            f"choose from {', '.join(ALL_FORMATS)}"
        )
    return tuple(formats)


def _paper(name: str) -> PaperSize:
    """Look up a paper size by name.

    Args:
        name: Paper name, case insensitive.

    Returns:
        The paper size.

    Raises:
        typer.Exit: If the name is unknown.
    """
    try:
        return PAPER_SIZES[name.lower()]
    except KeyError:
        raise _fail(
            f"unknown paper {name!r}; choose from {', '.join(sorted(PAPER_SIZES))}"
        ) from None


def _print_report(report: Report, label: str) -> None:
    """Print a validation report as a table, worst first."""
    if not report.issues:
        out.print(f"[green]ok[/green] {label}: no issues")
        return
    table = Table(title=label, title_justify="left", header_style="bold")
    table.add_column("severity")
    table.add_column("code")
    table.add_column("part")
    table.add_column("detail", overflow="fold")
    order = {Severity.ERROR: 0, Severity.WARNING: 1, Severity.INFO: 2}
    for issue in sorted(report.issues, key=lambda i: order.get(i.severity, 3)):
        colour = {"error": "red", "warning": "yellow"}.get(issue.severity.value, "dim")
        table.add_row(
            f"[{colour}]{issue.severity.value}[/{colour}]",
            issue.code,
            issue.part or "-",
            issue.message,
        )
    out.print(table)


# --------------------------------------------------------------------------- #
# list
# --------------------------------------------------------------------------- #
@app.command("list")
def list_command(
    niche: Annotated[
        Optional[str],
        typer.Argument(help="Show this niche's parameters instead of the niche list."),
    ] = None,
) -> None:
    """List the niches, or one niche's parameters."""
    if niche is None:
        table = Table(header_style="bold")
        table.add_column("niche")
        table.add_column("title")
        table.add_column("what it makes", overflow="fold")
        for generator in sorted(all_generators(), key=lambda g: g.niche):
            table.add_row(generator.niche, generator.title, generator.summary)
        out.print(table)
        out.print(f"\n{len(niche_names())} niches. "
                  "Run [bold]dxfgen list <niche>[/bold] for its parameters.")
        return

    try:
        generator = get_generator(niche)
    except KeyError as exc:
        raise _fail(str(exc).strip("'")) from None

    out.print(f"[bold]{generator.title}[/bold] - {generator.summary}\n")
    table = Table(header_style="bold")
    table.add_column("parameter")
    table.add_column("type")
    table.add_column("default")
    table.add_column("limits")
    table.add_column("meaning", overflow="fold")
    for doc in generator.parameter_docs():
        default = doc.default
        shown = getattr(default, "value", default)
        table.add_row(
            doc.name,
            doc.type_name,
            "-" if shown is None else f"{shown}",
            doc.constraints or "-",
            doc.description,
        )
    out.print(table)
    out.print(
        f"\nExample: [bold]dxfgen make {generator.niche} "
        "-p key=value -p key=value[/bold]"
    )


# --------------------------------------------------------------------------- #
# make
# --------------------------------------------------------------------------- #
@app.command()
def make(
    niche: Annotated[str, typer.Argument(help="Which niche to generate.")],
    param: Annotated[
        list[str],
        typer.Option("--param", "-p", help="Parameter override, as key=value."),
    ] = [],
    mode: Annotated[
        Optional[Mode], typer.Option("--mode", help="Machine mode.", case_sensitive=False)
    ] = None,
    output: Annotated[
        Path, typer.Option("--out", "-o", help="Output root directory.")
    ] = Path("output"),
    formats: Annotated[
        list[str], typer.Option("--format", "-f", help="Output kinds; repeatable.")
    ] = [],
    paper: Annotated[
        str, typer.Option("--paper", help="Paper size for the template PDF.")
    ] = "A4",
    preset: Annotated[
        Optional[str],
        typer.Option("--preset", help="Named YAML preset, or a path to one."),
    ] = None,
) -> None:
    """Generate one design from parameters you choose."""
    try:
        generator = get_generator(niche)
    except KeyError as exc:
        raise _fail(str(exc).strip("'")) from None

    values: dict[str, Any] = {}
    if preset is not None:
        loaded = _load_config(_resolve_preset(preset))
        named = loaded.get("niche")
        if named is not None and named != niche:
            raise _fail(f"preset {preset!r} is for niche {named!r}, not {niche!r}")
        values.update(loaded.get("params") or {})
    values.update(_parse_overrides(param))
    if mode is not None:
        values["mode"] = mode.value

    try:
        design = generator.make(**values)
    except ValidationError as exc:
        raise _fail(f"bad parameters for {niche}:\n{exc}") from None
    except ValueError as exc:
        raise _fail(f"{niche} cannot be built with these parameters: {exc}") from None

    try:
        written = write_design(
            design,
            root=output,
            formats=_check_formats(formats),
            paper=_paper(paper),
        )
    except GeometryInvalid as exc:
        err.print(f"[bold red]refused[/bold red] {design.slug} did not validate")
        _print_report(exc.report, design.slug)
        raise typer.Exit(EXIT_FAILED) from None

    width, height = design.size()
    out.print(f"[green]made[/green] {design.name}")
    out.print(f"  {width:.0f} x {height:.0f} mm, {design.machine.mode.value}")
    out.print(f"  {written.directory}")
    for kind in sorted(written.files):
        out.print(f"    {written.files[kind].name}")
    if written.report and written.report.warnings:
        _print_report(written.report, f"{design.slug}: warnings")


# --------------------------------------------------------------------------- #
# bundle
# --------------------------------------------------------------------------- #
def _report_bundle(bundle: Bundle) -> None:
    """Print what one bundle produced."""
    out.print(f"[green]{bundle.niche}[/green]  {len(bundle.designs)}/{bundle.requested} designs")
    if bundle.short:
        for stage, count in sorted(bundle.skip_counts().items()):
            out.print(f"    [yellow]{count} rejected[/yellow] at the {stage} stage")
        first = next(iter(bundle.skips), None)
        if first is not None:
            out.print(f"    first reason: {first.reason}")
    if bundle.stale:
        out.print(
            f"    [yellow]{len(bundle.stale)} older folder(s)[/yellow] in "
            f"{bundle.directory} were left out of the zip: "
            f"{', '.join(bundle.stale[:4])}"
            + (" ..." if len(bundle.stale) > 4 else "")
        )
    if bundle.contact_sheet:
        out.print(f"    {bundle.contact_sheet}")
    if bundle.archive:
        size = bundle.archive.stat().st_size / 1e6
        out.print(f"    {bundle.archive}  ({size:.1f} MB)")
        if size > MARKETPLACE_FILE_LIMIT_MB:
            out.print(
                f"    [yellow]over the {MARKETPLACE_FILE_LIMIT_MB:g} MB per-file "
                "limit most marketplaces impose[/yellow] (Etsy among them). "
                "The mockups are the bulk:"
            )
            out.print(
                "      drop them with [bold]-f dxf -f svg -f pdf -f preview "
                "-f readme[/bold], or split the count across two bundles."
            )


@app.command()
def bundle(
    niche: Annotated[
        Optional[str], typer.Option("--niche", "-n", help="Which niche to bundle.")
    ] = None,
    every: Annotated[
        bool, typer.Option("--all", help="Bundle every niche instead of one.")
    ] = False,
    count: Annotated[
        int, typer.Option("--count", "-c", min=1, help="Designs wanted per niche.")
    ] = 20,
    seed: Annotated[
        int, typer.Option("--seed", "-s", help="Master seed; repeats reproduce the set.")
    ] = 0,
    output: Annotated[
        Path, typer.Option("--out", "-o", help="Output root directory.")
    ] = Path("output"),
    formats: Annotated[
        list[str], typer.Option("--format", "-f", help="Output kinds; repeatable.")
    ] = [],
    paper: Annotated[
        str, typer.Option("--paper", help="Paper size for the template PDFs.")
    ] = "A4",
    seller: Annotated[
        str, typer.Option("--seller", help="Name used throughout LICENSE.txt.")
    ] = DEFAULT_SELLER,
    contact_sheet: Annotated[
        bool, typer.Option("--sheet/--no-sheet", help="Render the contact sheet.")
    ] = True,
    archive: Annotated[
        bool, typer.Option("--zip/--no-zip", help="Build the zip.")
    ] = True,
    quiet: Annotated[
        bool, typer.Option("--quiet", "-q", help="Only print the final summary.")
    ] = False,
) -> None:
    """Generate, validate and package a whole niche ready to sell."""
    if every and niche:
        raise _fail("--all and --niche are mutually exclusive")
    if not every and not niche:
        raise _fail("name a niche with --niche, or pass --all")

    chosen = niche_names() if every else [niche]  # type: ignore[list-item]
    for name in chosen:
        try:
            get_generator(name)
        except KeyError as exc:
            raise _fail(str(exc).strip("'")) from None

    status = None if quiet else (lambda line: out.print(f"  [dim]{line}[/dim]"))
    bundles = build_all(
        count=count,
        seed=seed,
        root=output,
        niches=chosen,
        formats=_check_formats(formats),
        paper=_paper(paper),
        seller=seller,
        contact_sheet=contact_sheet,
        archive=archive,
        progress=status,
    )
    for built in bundles:
        _report_bundle(built)

    made = sum(len(b.designs) for b in bundles)
    wanted = sum(b.requested for b in bundles)
    out.print(f"\n[bold]{made}/{wanted}[/bold] designs in {len(bundles)} bundle(s) "
              f"under {output}")
    if made < wanted:
        out.print(
            "[yellow]Short bundles are on purpose[/yellow]: the rejected variants "
            "would not have cut. See each INDEX.txt."
        )
    if made == 0:
        raise typer.Exit(EXIT_FAILED)


# --------------------------------------------------------------------------- #
# validate
# --------------------------------------------------------------------------- #
@app.command()
def validate(
    files: Annotated[
        list[Path], typer.Argument(help="DXF files to check.", exists=True)
    ],
    tool_diameter: Annotated[
        float, typer.Option("--tool", min=0.01, help="Cutter diameter to judge against.")
    ] = 6.35,
    mode: Annotated[
        Mode, typer.Option("--mode", help="Machine mode.", case_sensitive=False)
    ] = Mode.ROUTER,
    kerf: Annotated[float, typer.Option("--kerf", min=0.0, help="Laser kerf.")] = 0.15,
) -> None:
    """Check DXF files, including ones this tool did not make."""
    machine = Machine(mode=mode, tool_diameter=tool_diameter, kerf=kerf)
    worst = EXIT_OK
    for path in files:
        report = validate_dxf_file(path, machine=machine)
        _print_report(report, str(path))
        if report.errors:
            worst = EXIT_FAILED
    raise typer.Exit(worst)


def main() -> None:
    """Console entry point."""
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    app()


if __name__ == "__main__":
    main()
