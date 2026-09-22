"""Tests for the command line.

These run the commands through typer's runner rather than a subprocess, so a
failure points at the code and not at the shell.  Exit codes are asserted
everywhere, because a tool that prints an error and exits 0 is worse than one
that crashes: a script will not notice.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest
from typer.testing import CliRunner

from dxfgen.cli import EXIT_FAILED, EXIT_OK, EXIT_USAGE, app
from dxfgen.niches import niche_names

runner = CliRunner()

#: Skip the slow outputs; the packaging tests cover the rest.
FAST = ["-f", "dxf", "-f", "readme"]


def run(*args: str):
    """Invoke the CLI with a wide terminal so tables do not wrap mid-word."""
    return runner.invoke(app, list(args), env={"COLUMNS": "200"})


# --------------------------------------------------------------------------- #
# list
# --------------------------------------------------------------------------- #
def test_list_names_every_niche() -> None:
    result = run("list")
    assert result.exit_code == EXIT_OK
    for name in niche_names():
        assert name in result.stdout


def test_list_one_niche_shows_its_parameters() -> None:
    result = run("list", "trays")
    assert result.exit_code == EXIT_OK
    for expected in ("length", "width", "thickness", "tool_diameter"):
        assert expected in result.stdout


def test_list_shows_defaults_and_limits() -> None:
    result = run("list", "boxes")
    assert result.exit_code == EXIT_OK
    assert "laser" in result.stdout, "the default mode"
    assert ">=" in result.stdout, "the limits column"


def test_list_of_an_unknown_niche_is_a_usage_error() -> None:
    result = run("list", "sideboards")
    assert result.exit_code == EXIT_USAGE
    assert "unknown niche" in result.output


# --------------------------------------------------------------------------- #
# make
# --------------------------------------------------------------------------- #
def test_make_writes_a_design(tmp_path: Path) -> None:
    result = run("make", "trays", "-p", "length=460", "-p", "width=300", "-o", str(tmp_path), *FAST)
    assert result.exit_code == EXIT_OK, result.output
    written = list(tmp_path.glob("trays/*/*.dxf"))
    assert len(written) == 1
    assert "460 x 300 mm" in result.stdout


def test_make_honours_the_mode_flag(tmp_path: Path) -> None:
    result = run("make", "ornaments", "--mode", "laser", "-o", str(tmp_path), *FAST)
    assert result.exit_code == EXIT_OK, result.output
    assert "laser" in result.stdout


def test_make_reads_a_shipped_preset(tmp_path: Path) -> None:
    result = run("make", "boxes", "--preset", "laser-3mm-ply", "-o", str(tmp_path), *FAST)
    assert result.exit_code == EXIT_OK, result.output
    readme = next(tmp_path.glob("boxes/*/README.txt")).read_text()
    assert "3 mm birch ply" in readme


def test_an_override_beats_the_preset(tmp_path: Path) -> None:
    result = run(
        "make", "boxes", "--preset", "laser-3mm-ply", "-p", "material=acrylic",
        "-o", str(tmp_path), *FAST,
    )
    assert result.exit_code == EXIT_OK, result.output
    assert "acrylic" in next(tmp_path.glob("boxes/*/README.txt")).read_text()


def test_a_preset_for_another_niche_is_refused(tmp_path: Path) -> None:
    preset = tmp_path / "wrong.yaml"
    preset.write_text("niche: boxes\nparams:\n  thickness: 3.0\n")
    result = run("make", "trays", "--preset", str(preset), "-o", str(tmp_path))
    assert result.exit_code == EXIT_USAGE
    assert "is for niche" in result.output


def test_an_unknown_preset_lists_the_known_ones(tmp_path: Path) -> None:
    result = run("make", "trays", "--preset", "nope", "-o", str(tmp_path))
    assert result.exit_code == EXIT_USAGE
    assert "laser-3mm-ply" in result.output


def test_make_of_an_unknown_niche_is_a_usage_error(tmp_path: Path) -> None:
    result = run("make", "sideboards", "-o", str(tmp_path))
    assert result.exit_code == EXIT_USAGE
    assert "unknown niche" in result.output


def test_a_parameter_out_of_range_is_a_usage_error(tmp_path: Path) -> None:
    result = run("make", "trays", "-p", "length=99999", "-o", str(tmp_path))
    assert result.exit_code == EXIT_USAGE
    assert "bad parameters" in result.output


def test_a_parameter_without_a_value_is_a_usage_error(tmp_path: Path) -> None:
    result = run("make", "trays", "-p", "length", "-o", str(tmp_path))
    assert result.exit_code == EXIT_USAGE
    assert "key=value" in result.output


def test_an_unknown_parameter_is_a_usage_error(tmp_path: Path) -> None:
    result = run("make", "trays", "-p", "lenght=400", "-o", str(tmp_path))
    assert result.exit_code == EXIT_USAGE
    assert "bad parameters" in result.output


def test_geometry_that_cannot_be_built_is_a_usage_error(tmp_path: Path) -> None:
    """A 60 mm cutter in a 70 mm tray is a parameter mistake, not a crash."""
    result = run(
        "make", "trays", "-p", "length=200", "-p", "width=140",
        "-p", "tool_diameter=25", "-p", "min_wall=45", "-o", str(tmp_path),
    )
    assert result.exit_code == EXIT_USAGE
    assert "cannot be built" in result.output or "bad parameters" in result.output


def test_an_unknown_format_is_a_usage_error(tmp_path: Path) -> None:
    result = run("make", "trays", "-f", "gcode", "-o", str(tmp_path))
    assert result.exit_code == EXIT_USAGE
    assert "unknown format" in result.output


def test_an_unknown_paper_size_is_a_usage_error(tmp_path: Path) -> None:
    result = run("make", "trays", "--paper", "A0", "-o", str(tmp_path))
    assert result.exit_code == EXIT_USAGE
    assert "unknown paper" in result.output


def test_paper_size_is_case_insensitive(tmp_path: Path) -> None:
    result = run("make", "trays", "--paper", "a3", "-o", str(tmp_path), "-f", "pdf")
    assert result.exit_code == EXIT_OK, result.output


# --------------------------------------------------------------------------- #
# bundle
# --------------------------------------------------------------------------- #
def test_bundle_writes_a_niche(tmp_path: Path) -> None:
    result = run("bundle", "-n", "coasters", "-c", "3", "-s", "5", "-o", str(tmp_path), *FAST)
    assert result.exit_code == EXIT_OK, result.output
    assert (tmp_path / "coasters" / "LICENSE.txt").is_file()
    assert (tmp_path / "coasters_bundle.zip").is_file()
    assert "3/3" in result.stdout


def test_bundle_names_the_seller_in_the_licence(tmp_path: Path) -> None:
    run("bundle", "-n", "coasters", "-c", "2", "-o", str(tmp_path),
        "--seller", "Northwood Digital", *FAST)
    assert "Northwood Digital" in (tmp_path / "coasters" / "LICENSE.txt").read_text()


def test_bundle_all_covers_every_niche(tmp_path: Path) -> None:
    result = run("bundle", "--all", "-c", "1", "-s", "3", "-o", str(tmp_path),
                 "--no-zip", "--no-sheet", *FAST)
    assert result.exit_code == EXIT_OK, result.output
    for name in niche_names():
        assert (tmp_path / name / "INDEX.txt").is_file(), name


def test_the_zip_can_be_skipped(tmp_path: Path) -> None:
    run("bundle", "-n", "coasters", "-c", "2", "-o", str(tmp_path), "--no-zip", *FAST)
    assert not list(tmp_path.glob("*.zip"))


def test_the_contact_sheet_can_be_skipped(tmp_path: Path) -> None:
    run("bundle", "-n", "coasters", "-c", "2", "-o", str(tmp_path), "--no-sheet",
        "-f", "dxf", "-f", "preview")
    assert not (tmp_path / "coasters" / "CONTACT_SHEET.png").exists()


def test_the_zip_holds_what_the_bundle_made(tmp_path: Path) -> None:
    run("bundle", "-n", "coasters", "-c", "3", "-s", "5", "-o", str(tmp_path), *FAST)
    with zipfile.ZipFile(tmp_path / "coasters_bundle.zip") as archive:
        names = archive.namelist()
    assert sum(1 for n in names if n.endswith(".dxf")) == 3
    assert "coasters/LICENSE.txt" in names


def test_bundle_needs_a_target(tmp_path: Path) -> None:
    result = run("bundle", "-c", "2", "-o", str(tmp_path))
    assert result.exit_code == EXIT_USAGE
    assert "--all" in result.output


def test_bundle_refuses_all_and_a_niche_together(tmp_path: Path) -> None:
    result = run("bundle", "--all", "-n", "trays", "-c", "2", "-o", str(tmp_path))
    assert result.exit_code == EXIT_USAGE
    assert "mutually exclusive" in result.output


def test_bundle_of_an_unknown_niche_fails_before_writing(tmp_path: Path) -> None:
    result = run("bundle", "-n", "sideboards", "-c", "2", "-o", str(tmp_path))
    assert result.exit_code == EXIT_USAGE
    assert not list(tmp_path.iterdir()), "nothing must be written on a usage error"


def test_a_zero_count_is_refused(tmp_path: Path) -> None:
    result = run("bundle", "-n", "trays", "-c", "0", "-o", str(tmp_path))
    assert result.exit_code != EXIT_OK


def test_quiet_drops_the_per_design_lines(tmp_path: Path) -> None:
    loud = run("bundle", "-n", "coasters", "-c", "2", "-o", str(tmp_path / "a"), *FAST)
    quiet = run("bundle", "-n", "coasters", "-c", "2", "-o", str(tmp_path / "b"),
                "-q", *FAST)
    assert len(quiet.stdout.splitlines()) < len(loud.stdout.splitlines())


# --------------------------------------------------------------------------- #
# validate
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def cut_file(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("dxf")
    result = runner.invoke(
        app, ["make", "trays", "-p", "length=460", "-o", str(root), "-f", "dxf"]
    )
    assert result.exit_code == EXIT_OK, result.output
    return next(root.glob("trays/*/*.dxf"))


def test_a_file_we_wrote_passes(cut_file: Path) -> None:
    result = run("validate", str(cut_file))
    assert result.exit_code == EXIT_OK, result.output


def test_validate_reports_the_extents(cut_file: Path) -> None:
    result = run("validate", str(cut_file))
    assert "INFO_EXTENTS" in result.stdout


def test_validate_judges_against_the_cutter_it_is_given(cut_file: Path) -> None:
    """A 20 mm cutter cannot reach corners cut for a 6.35 mm one."""
    fine = run("validate", str(cut_file), "--tool", "6.35")
    coarse = run("validate", str(cut_file), "--tool", "20")
    assert fine.exit_code == EXIT_OK
    assert coarse.output != fine.output


def test_a_broken_file_fails(tmp_path: Path) -> None:
    broken = tmp_path / "broken.dxf"
    broken.write_text("this is not a DXF file at all\n")
    result = run("validate", str(broken))
    assert result.exit_code == EXIT_FAILED
    assert "E_UNREADABLE" in result.stdout or "error" in result.stdout.lower()


def test_a_missing_file_is_a_usage_error(tmp_path: Path) -> None:
    result = run("validate", str(tmp_path / "nothing.dxf"))
    assert result.exit_code == EXIT_USAGE


def test_several_files_are_all_checked(cut_file: Path) -> None:
    result = run("validate", str(cut_file), str(cut_file))
    assert result.exit_code == EXIT_OK
    assert result.stdout.count("INFO_EXTENTS") == 2
