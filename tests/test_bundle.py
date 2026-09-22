"""Tests for batch mode: a whole niche generated, packaged and archived."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from dxfgen.bundles.builder import (
    CONTACT_SHEET_NAME,
    INDEX_NAME,
    LICENSE_NAME,
    Bundle,
    build_all,
    build_bundle,
    index_text,
    license_text,
    zip_bundle,
)
from dxfgen.niches import niche_names
from dxfgen.niches.base import Skip, VariantRun

#: Enough to exercise the packaging without rendering a template PDF or a
#: wood mockup for every design; those are covered by the per-design tests.
FAST = ("dxf", "svg", "preview", "readme")


@pytest.fixture(scope="module")
def bundle(tmp_path_factory: pytest.TempPathFactory) -> Bundle:
    root = tmp_path_factory.mktemp("bundle")
    return build_bundle("coasters", count=4, seed=11, root=root, formats=FAST)


# --------------------------------------------------------------------------- #
# what a bundle contains
# --------------------------------------------------------------------------- #
def test_a_bundle_holds_the_requested_number_of_designs(bundle: Bundle) -> None:
    assert len(bundle.designs) == 4
    assert bundle.requested == 4
    assert bundle.short == 0


def test_every_design_gets_its_own_folder(bundle: Bundle) -> None:
    folders = {output.directory for output in bundle.designs}
    assert len(folders) == 4
    for output in bundle.designs:
        assert output.directory.parent == bundle.directory
        assert output.directory.name == output.design.slug


def test_the_bundle_lands_under_its_niche(bundle: Bundle) -> None:
    assert bundle.directory.name == "coasters"


def test_the_shared_files_are_written(bundle: Bundle) -> None:
    for name in (CONTACT_SHEET_NAME, LICENSE_NAME, INDEX_NAME):
        path = bundle.directory / name
        assert path.exists() and path.stat().st_size > 0, name


def test_the_contact_sheet_is_recorded_on_the_bundle(bundle: Bundle) -> None:
    assert bundle.contact_sheet is not None
    assert bundle.contact_sheet.name == CONTACT_SHEET_NAME


def test_no_variant_is_written_twice(bundle: Bundle) -> None:
    slugs = [output.design.slug for output in bundle.designs]
    assert len(set(slugs)) == len(slugs)


# --------------------------------------------------------------------------- #
# reproducibility
# --------------------------------------------------------------------------- #
def test_the_same_seed_gives_the_same_designs(tmp_path: Path) -> None:
    a = build_bundle("coasters", 3, seed=5, root=tmp_path / "a", formats=("dxf",))
    b = build_bundle("coasters", 3, seed=5, root=tmp_path / "b", formats=("dxf",))
    assert [o.design.slug for o in a.designs] == [o.design.slug for o in b.designs]


def test_a_different_seed_gives_different_designs(tmp_path: Path) -> None:
    a = build_bundle("coasters", 4, seed=5, root=tmp_path / "a", formats=("dxf",))
    b = build_bundle("coasters", 4, seed=6, root=tmp_path / "b", formats=("dxf",))
    assert [o.design.slug for o in a.designs] != [o.design.slug for o in b.designs]


def test_asking_for_more_does_not_disturb_the_first_ones(tmp_path: Path) -> None:
    """Raising the count must extend the set, not reshuffle it."""
    few = build_bundle("trays", 3, seed=9, root=tmp_path / "a", formats=("dxf",))
    many = build_bundle("trays", 6, seed=9, root=tmp_path / "b", formats=("dxf",))
    first = [o.design.slug for o in few.designs]
    assert [o.design.slug for o in many.designs][: len(first)] == first


# --------------------------------------------------------------------------- #
# the archive
# --------------------------------------------------------------------------- #
def test_the_archive_is_written_beside_the_folder(bundle: Bundle) -> None:
    assert bundle.archive is not None
    assert bundle.archive.name == "coasters_bundle.zip"
    assert bundle.archive.parent == bundle.directory.parent


def test_the_archive_holds_every_file_of_the_bundle(bundle: Bundle) -> None:
    with zipfile.ZipFile(bundle.archive) as archive:
        names = set(archive.namelist())
    assert len(names) == len(bundle.files())
    for name in (CONTACT_SHEET_NAME, LICENSE_NAME, INDEX_NAME):
        assert f"coasters/{name}" in names


def test_every_archive_entry_sits_under_one_top_level_folder(bundle: Bundle) -> None:
    """So unzipping into a Downloads folder makes one folder, not forty."""
    with zipfile.ZipFile(bundle.archive) as archive:
        tops = {name.split("/")[0] for name in archive.namelist()}
    assert tops == {"coasters"}


def test_the_archive_is_readable(bundle: Bundle) -> None:
    with zipfile.ZipFile(bundle.archive) as archive:
        assert archive.testzip() is None


def test_leftovers_from_an_earlier_run_stay_out_of_the_archive(tmp_path: Path) -> None:
    """The directory outlives the run, so it is never the source of truth.

    A previous bundle with another seed leaves design folders behind.  Zipping
    the directory would ship them; zipping the manifest does not.
    """
    old = build_bundle("coasters", 3, seed=1, root=tmp_path, formats=("dxf",))
    old_slugs = {o.design.slug for o in old.designs}
    new = build_bundle("coasters", 3, seed=2, root=tmp_path, formats=("dxf",))
    new_slugs = {o.design.slug for o in new.designs}
    orphans = old_slugs - new_slugs
    assert orphans, "seeds 1 and 2 must differ for this test to mean anything"

    assert set(new.stale) == orphans
    with zipfile.ZipFile(new.archive) as archive:
        packed = {name.split("/")[1] for name in archive.namelist()}
    assert not (packed & orphans)
    for slug in orphans:
        assert (tmp_path / "coasters" / slug).is_dir(), "leftovers are reported, not deleted"


def test_an_empty_bundle_cannot_be_archived(tmp_path: Path) -> None:
    empty = Bundle(niche="trays", title="Trays", directory=tmp_path)
    with pytest.raises(ValueError, match="empty"):
        zip_bundle(empty)


def test_the_archive_path_can_be_chosen(bundle: Bundle, tmp_path: Path) -> None:
    written = zip_bundle(bundle, tmp_path / "deep" / "custom.zip")
    assert [p.name for p in written] == ["custom.zip"]
    assert written[0].exists()


# --------------------------------------------------------------------------- #
# the licence
# --------------------------------------------------------------------------- #
def test_the_licence_names_the_seller() -> None:
    text = license_text("Northwood Digital", year=2031)
    assert "Northwood Digital" in text
    assert "2031" in text


def test_the_licence_allows_selling_the_physical_product() -> None:
    text = license_text().lower()
    assert "sell the physical items" in text
    assert "commercially" in text


def test_the_licence_forbids_passing_on_the_files() -> None:
    text = license_text().lower()
    for phrase in ("resell", "share", "sub-license", "redistribut"):
        assert phrase in text or phrase[:-1] in text, phrase


def test_the_licence_states_the_designs_are_original() -> None:
    text = license_text().lower()
    assert "traced" in text and "trademark" in text


def test_the_written_licence_matches(bundle: Bundle) -> None:
    assert (bundle.directory / LICENSE_NAME).read_text() == license_text()


# --------------------------------------------------------------------------- #
# the index
# --------------------------------------------------------------------------- #
def test_the_index_lists_every_design(bundle: Bundle) -> None:
    text = index_text(bundle)
    for output in bundle.designs:
        assert output.design.slug in text
        assert output.design.name in text


def test_the_index_records_the_seed(bundle: Bundle) -> None:
    assert f"seed {bundle.seed}" in index_text(bundle)


def test_the_index_names_the_shared_files(bundle: Bundle) -> None:
    text = index_text(bundle)
    assert CONTACT_SHEET_NAME in text and LICENSE_NAME in text


def test_the_index_explains_a_short_bundle() -> None:
    """A bundle that could not fill its count says so, with reasons."""
    run = VariantRun(
        designs=[],
        skips=[
            Skip(0, None, "handle too small", "build"),
            Skip(1, "tray-x", "E_MIN_WALL: 4.0 mm wall", "validate"),
            Skip(2, "tray-x", "duplicate of an earlier variant", "duplicate"),
        ],
        requested=9,
        attempts=54,
        seed=3,
    )
    text = index_text(Bundle("trays", "Trays", Path("x"), run=run))
    assert "9 designs were requested and 0 were produced" in text
    assert "failed the manufacturing checks" in text
    assert "not shipped" in text


# --------------------------------------------------------------------------- #
# every niche at once
# --------------------------------------------------------------------------- #
def test_build_all_covers_the_niches_asked_for(tmp_path: Path) -> None:
    bundles = build_all(
        2, seed=4, root=tmp_path, niches=["trays", "coasters"], formats=("dxf",)
    )
    assert [b.niche for b in bundles] == ["trays", "coasters"]
    assert all(len(b.designs) == 2 for b in bundles)


def test_build_all_defaults_to_every_registered_niche(tmp_path: Path) -> None:
    bundles = build_all(
        1, seed=4, root=tmp_path, formats=("dxf",), contact_sheet=False, archive=False
    )
    assert [b.niche for b in bundles] == niche_names()


def test_an_unknown_niche_is_reported(tmp_path: Path) -> None:
    with pytest.raises(KeyError, match="unknown niche"):
        build_bundle("sideboards", 2, root=tmp_path)


def test_progress_is_reported_per_design(tmp_path: Path) -> None:
    seen: list[str] = []
    build_bundle(
        "coasters", 3, seed=8, root=tmp_path, formats=("dxf",), progress=seen.append
    )
    assert sum(1 for line in seen if "coasters " in line) == 3
    assert any("archiving" in line for line in seen)


def test_a_bundle_summarises_itself(bundle: Bundle) -> None:
    line = bundle.summary()
    assert "coasters" in line and "4/4" in line


# --------------------------------------------------------------------------- #
# splitting for marketplaces that cap a single file
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def big(tmp_path_factory: pytest.TempPathFactory) -> Bundle:
    root = tmp_path_factory.mktemp("split")
    return build_bundle(
        "coasters", count=6, seed=3, root=root, formats=("dxf", "preview", "readme"),
        archive=False,
    )


def _split_limit_mb(bundle: Bundle, designs_per_part: float = 2.0) -> float:
    """A limit that forces a split while still fitting every design.

    The shared files ride in every part, so the budget left for designs is
    the limit minus their size; a limit picked as a fraction of the whole
    archive forgets that and asks for the impossible.
    """
    shared = sum(
        path.stat().st_size
        for path in (bundle.contact_sheet, bundle.license_path, bundle.index_path)
        if path is not None
    )
    sizes = [
        sum(f.stat().st_size for f in output.directory.rglob("*") if f.is_file())
        for output in bundle.designs
    ]
    return (shared + max(sizes) * designs_per_part) / 1e6


def test_an_unsplit_bundle_is_one_file(big: Bundle, tmp_path: Path) -> None:
    written = zip_bundle(big, tmp_path / "one.zip", max_mb=None)
    assert len(written) == 1
    assert written[0].name == "one.zip"


def test_a_bundle_over_the_limit_is_split(big: Bundle, tmp_path: Path) -> None:
    limit = _split_limit_mb(big)
    parts = zip_bundle(big, tmp_path / "part.zip", max_mb=limit)
    assert len(parts) > 1
    assert [p.name for p in parts] == [
        f"part_{i}of{len(parts)}.zip" for i in range(1, len(parts) + 1)
    ]


def test_every_part_stays_under_the_limit(big: Bundle, tmp_path: Path) -> None:
    limit_mb = _split_limit_mb(big)
    parts = zip_bundle(big, tmp_path / "p.zip", max_mb=limit_mb)
    for path in parts:
        assert path.stat().st_size <= limit_mb * 1e6, path.name


def test_the_parts_hold_every_design_exactly_once(
    big: Bundle, tmp_path: Path
) -> None:
    parts = zip_bundle(big, tmp_path / "p.zip", max_mb=_split_limit_mb(big))
    found: list[str] = []
    for path in parts:
        with zipfile.ZipFile(path) as archive:
            found += [n for n in archive.namelist() if n.endswith(".dxf")]
    assert len(found) == len(set(found)) == len(big.designs)


def test_a_design_folder_is_never_split_across_parts(
    big: Bundle, tmp_path: Path
) -> None:
    """Half a design folder is no use to anybody."""
    parts = zip_bundle(big, tmp_path / "p.zip", max_mb=_split_limit_mb(big))
    owner: dict[str, str] = {}
    for path in parts:
        with zipfile.ZipFile(path) as archive:
            for name in archive.namelist():
                bits = name.split("/")
                if len(bits) < 3:
                    continue  # a shared file, which every part carries
                owner.setdefault(bits[1], path.name)
                assert owner[bits[1]] == path.name, f"{bits[1]} spans parts"


def test_every_part_carries_the_licence(big: Bundle, tmp_path: Path) -> None:
    """A buyer with one part is still bound by terms they must be able to read."""
    parts = zip_bundle(big, tmp_path / "p.zip", max_mb=_split_limit_mb(big))
    assert len(parts) > 1
    for path in parts:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
        assert f"coasters/{LICENSE_NAME}" in names
        assert f"coasters/{INDEX_NAME}" in names


def test_a_design_bigger_than_the_limit_is_reported(big: Bundle, tmp_path: Path) -> None:
    shared = _split_limit_mb(big, designs_per_part=0.0)
    with pytest.raises(ValueError, match="alone is"):
        zip_bundle(big, tmp_path / "tiny.zip", max_mb=shared + 0.001)


def test_a_limit_smaller_than_the_shared_files_is_reported(
    big: Bundle, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="over the"):
        zip_bundle(big, tmp_path / "x.zip", max_mb=0.0005)


def test_the_bundle_records_every_part(big: Bundle, tmp_path: Path) -> None:
    parts = zip_bundle(big, tmp_path / "p.zip", max_mb=_split_limit_mb(big))
    assert big.archives == parts
    assert big.archive == parts[0]
