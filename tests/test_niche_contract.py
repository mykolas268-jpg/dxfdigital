"""The contract every registered niche must satisfy.

Rather than repeating the same assertions in five files, these tests are
parametrised over the registry, so a niche added later is held to the same
standard automatically.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dxfgen.bundles.builder import write_design
from dxfgen.core.export_dxf import audit_file
from dxfgen.core.geometry import EPS
from dxfgen.core.validate import validate_dxf_file
from dxfgen.niches import Generator, all_generators, niche_names

GENERATORS = all_generators()
IDS = [g.niche for g in GENERATORS]


@pytest.mark.parametrize("gen", GENERATORS, ids=IDS)
def test_defaults_produce_a_valid_design(gen: Generator) -> None:
    design = gen.make()
    report = gen.check(design)
    assert report.ok, f"{gen.niche}: {report.format()}"


@pytest.mark.parametrize("gen", GENERATORS, ids=IDS)
def test_every_niche_is_registered_under_its_own_name(gen: Generator) -> None:
    assert gen.niche in niche_names()
    assert gen.title and gen.summary


@pytest.mark.parametrize("gen", GENERATORS, ids=IDS)
def test_designs_carry_the_metadata_a_listing_needs(gen: Generator) -> None:
    design = gen.make()
    assert design.slug and design.slug == design.slug.lower()
    assert " " not in design.slug
    assert design.name and design.description
    assert design.niche == gen.niche
    assert design.cutting_order
    assert design.params
    assert design.material and design.thickness > 0


@pytest.mark.parametrize("gen", GENERATORS, ids=IDS)
def test_designs_start_at_the_origin(gen: Generator) -> None:
    x0, y0, _, _ = gen.make().bbox()
    assert abs(x0) < EPS and abs(y0) < EPS


@pytest.mark.parametrize("gen", GENERATORS, ids=IDS)
def test_designs_fit_their_declared_stock(gen: Generator) -> None:
    design = gen.make()
    width, height = design.size()
    sheet_w, sheet_h = design.sheet_size()
    assert (width <= sheet_w and height <= sheet_h) or (
        width <= sheet_h and height <= sheet_w
    )


@pytest.mark.parametrize("gen", GENERATORS, ids=IDS)
def test_designs_carry_the_limits_they_were_built_to(gen: Generator) -> None:
    params = gen.parse({})
    design = gen.make()
    assert design.limits.min_wall == params.min_wall
    assert design.limits.pocket_floor == params.pocket_floor


@pytest.mark.parametrize("gen", GENERATORS, ids=IDS)
def test_parameters_can_be_round_tripped(gen: Generator) -> None:
    """The recorded parameters must regenerate the same design."""
    design = gen.make()
    rebuilt = gen.make(**design.params)
    assert rebuilt.slug == design.slug
    assert rebuilt.size() == pytest.approx(design.size())


@pytest.mark.parametrize("gen", GENERATORS, ids=IDS)
def test_parameter_docs_cover_every_field(gen: Generator) -> None:
    docs = gen.parameter_docs()
    assert {d.name for d in docs} == set(gen.params_model.model_fields)
    assert all(d.description for d in docs), gen.niche


@pytest.mark.parametrize("gen", GENERATORS, ids=IDS)
def test_variants_are_reproducible_and_valid(gen: Generator) -> None:
    first = gen.variants(6, seed=17)
    second = gen.variants(6, seed=17)
    assert [d.slug for d in first] == [d.slug for d in second]
    assert len({d.slug for d in first}) == len(first)
    for design in first:
        assert gen.check(design).ok, f"{gen.niche}/{design.slug}"


@pytest.mark.parametrize("gen", GENERATORS, ids=IDS)
def test_variants_fill_the_requested_count(gen: Generator) -> None:
    """A niche whose sampler keeps producing rejects is not much of a niche."""
    assert len(gen.variants(8, seed=23)) == 8


@pytest.mark.parametrize("gen", GENERATORS, ids=IDS)
def test_a_design_exports_to_a_clean_dxf(gen: Generator, tmp_path: Path) -> None:
    out = write_design(gen.make(), tmp_path, formats=("dxf",))
    errors, fixes = audit_file(out.files["dxf"])
    assert errors == [] and fixes == []
    report = validate_dxf_file(out.files["dxf"])
    assert report.ok, report.format()
    assert report.warnings == []


@pytest.mark.parametrize("gen", GENERATORS, ids=IDS)
def test_a_design_exports_every_format(gen: Generator, tmp_path: Path) -> None:
    """Every design writes every flat output, and nothing is empty."""
    design = gen.make()
    out = write_design(design, tmp_path)
    assert {"dxf", "svg", "pdf", "preview", "mockup", "readme"} <= set(out.files)
    for kind, path in out.files.items():
        assert path.stat().st_size > 0, kind


@pytest.mark.parametrize("gen", GENERATORS, ids=IDS)
def test_an_assembly_drawing_appears_exactly_when_there_is_one(
    gen: Generator, tmp_path: Path
) -> None:
    """A flat product has nothing to assemble; a box, dock or shelf does."""
    design = gen.make()
    out = write_design(design, tmp_path)
    assert ("assembly" in out.files) == (design.assembly is not None)


@pytest.mark.parametrize("gen", GENERATORS, ids=IDS)
def test_an_assembly_only_places_parts_the_design_cuts(gen: Generator) -> None:
    """A placement naming a part that is not cut is a generator bug."""
    for design in gen.variants(4, seed=17):
        if design.assembly is None:
            continue
        names = {part.name for part in design.parts}
        assert design.assembly.missing_from(names) == set()
        assert design.assembly.caption


@pytest.mark.parametrize("gen", GENERATORS, ids=IDS)
def test_an_assembly_is_no_smaller_than_the_parts_in_it(gen: Generator) -> None:
    """A wrong placement usually shows up as an assembly that collapses."""
    from dxfgen.core.preview import assembled_size

    for design in gen.variants(3, seed=17):
        if design.assembly is None:
            continue
        box = assembled_size(design)
        biggest = max(max(part.size()) for part in design.parts)
        assert max(box) >= biggest * 0.5, f"{design.slug} assembles to {box}"
        assert min(box) >= design.thickness


@pytest.mark.parametrize("gen", GENERATORS, ids=IDS)
def test_unknown_parameters_are_refused(gen: Generator) -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        gen.parse({"definitely_not_a_parameter": 1})


# --------------------------------------------------------------------------- #
# samplers must propose things their own geometry can build
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("gen", GENERATORS, ids=IDS)
def test_a_sampler_proposes_buildable_designs(gen: Generator) -> None:
    """A high rejection rate means parameters drawn independently of each other.

    Every instance found so far was the same shape: a tab wider than the panel
    it is cut into, a recess deeper than the material, a rest longer than the
    tray, a hole narrower than the cutter, a scallop wider than the flat it
    bites into.  Each is invisible design by design and obvious in aggregate,
    so it is measured rather than left to turn up again.
    """
    made = attempts = 0
    for seed in (1, 7, 31, 42, 2026, 11, 99):
        run = gen.sample_variants(10, seed=seed)
        made += len(run.designs)
        attempts += run.attempts
    assert made == 70, f"{gen.niche} produced {made} of 70 requested"
    rejected = (attempts - made) / attempts
    assert rejected <= 0.12, (
        f"{gen.niche} rejected {rejected:.0%} of what it proposed "
        f"({attempts - made} of {attempts})"
    )


def test_no_niche_wastes_much_of_what_it_draws() -> None:
    """The whole registry, so one niche cannot hide behind the others."""
    made = attempts = 0
    for gen in GENERATORS:
        for seed in (1, 31, 99):
            run = gen.sample_variants(10, seed=seed)
            made += len(run.designs)
            attempts += run.attempts
    assert (attempts - made) / attempts <= 0.06
