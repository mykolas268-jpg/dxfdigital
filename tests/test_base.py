"""Tests for the shared generator plumbing.

A throwaway generator is used rather than a real niche, so a failure here
points at the plumbing and not at somebody's tray geometry.
"""

from __future__ import annotations

import logging
import random

import pytest
from pydantic import Field, ValidationError

from dxfgen.core import geometry as geo
from dxfgen.core.design import Design, Mode, Part, Pocket
from dxfgen.core.layers import CUT_INSIDE, CUT_OUTSIDE, DRILL, ENGRAVE, INFO
from dxfgen.niches.base import (
    GOLDEN,
    STOCK_SIZES,
    Generator,
    GeneratorParams,
    cutting_order_for,
    slugify,
    smallest_stock,
)


class SquareParams(GeneratorParams):
    """Parameters for the test generator."""

    side: float = Field(150.0, ge=60.0, le=400.0, description="side length in mm")
    unbuildable: bool = Field(False, description="raise instead of building")
    bad_pocket: bool = Field(False, description="emit a pocket the cutter cannot cut")


class SquareGenerator(Generator):
    """A minimal generator used to exercise the base class."""

    niche = "squares"
    title = "Squares"
    summary = "test-only generator"
    params_model = SquareParams

    def generate(self, params: SquareParams) -> Design:  # type: ignore[override]
        if params.unbuildable or params.side > 380.0:
            raise ValueError(f"side {params.side:g} mm cannot be built")
        pockets = []
        if params.bad_pocket:
            pockets.append(Pocket(geo.rect_ring(80, 60, 30, 30), 6.0))
        part = Part(
            "square",
            geo.rounded_rect_ring(params.side, params.side, 12),
            pockets=pockets,
        )
        return Design(
            slug=slugify("square", params.side),
            name=f"Square {params.side:g} mm",
            niche=self.niche,
            description="a test square",
            parts=[part],
            machine=params.machine(),
            material=params.material,
            thickness=params.thickness,
            sheet=(600.0, 400.0),
        )

    def sample_params(self, rng: random.Random, index: int) -> SquareParams:
        return SquareParams(side=float(rng.randrange(60, 400, 10)))


@pytest.fixture
def gen() -> SquareGenerator:
    return SquareGenerator()


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def test_smallest_stock_picks_the_first_panel_that_fits() -> None:
    assert smallest_stock(280, 190) == (300.0, 200.0)
    assert smallest_stock(450, 280) == (600.0, 400.0)
    assert smallest_stock(190, 280) == (300.0, 200.0)  # rotated
    assert smallest_stock(9000, 9000) == STOCK_SIZES[-1]


def test_slugify_is_filesystem_safe() -> None:
    assert slugify("Tray", "Rounded", 450.0, "460x370") == "tray-rounded-450-460x370"
    assert slugify("a b/c", 1.5) == "a-b-c-1p5"
    assert slugify("", "x") == "x"


def test_golden_ratio_constant() -> None:
    assert GOLDEN == pytest.approx((1 + 5**0.5) / 2)


def test_cutting_order_puts_the_outer_profile_last(gen: SquareGenerator) -> None:
    design = gen.make(bad_pocket=True)
    design.parts[0].holes.append(geo.circle_ring(75, 75, 8))
    steps = cutting_order_for(design)
    assert "POCKET_6" in steps[0]
    assert CUT_INSIDE in steps[-3]
    assert CUT_OUTSIDE in steps[-2]
    assert INFO in steps[-1]


def test_cutting_order_sorts_pockets_shallowest_first() -> None:
    part = Part(
        "p",
        geo.rounded_rect_ring(300, 200, 15),
        pockets=[
            Pocket(geo.rounded_rect_ring(80, 80, 12, 20, 60), 10.0),
            Pocket(geo.rounded_rect_ring(80, 80, 12, 120, 60), 4.0),
        ],
    )
    design = Design("d", "D", "t", "x", [part], thickness=19, sheet=(600, 400))
    steps = cutting_order_for(design)
    assert "POCKET_4" in steps[0] and "POCKET_10" in steps[1]


# --------------------------------------------------------------------------- #
# parameters
# --------------------------------------------------------------------------- #
def test_params_build_a_matching_machine_and_validation_config() -> None:
    params = SquareParams(mode=Mode.LASER, kerf=0.2, min_wall=12.0, pocket_floor=3.0)
    machine = params.machine()
    assert machine.is_laser and machine.kerf == 0.2
    cfg = params.validation_config()
    assert cfg.min_wall == 12.0 and cfg.pocket_floor == 3.0


def test_parse_coerces_strings_like_a_command_line(gen: SquareGenerator) -> None:
    params = gen.parse({"side": "200", "mode": "laser", "thickness": "6"})
    assert params.side == 200.0
    assert params.mode is Mode.LASER
    assert params.thickness == 6.0


def test_parse_rejects_unknown_keys(gen: SquareGenerator) -> None:
    with pytest.raises(ValidationError, match="extra_forbidden|Extra inputs"):
        gen.parse({"sied": 200})


def test_parse_enforces_limits(gen: SquareGenerator) -> None:
    with pytest.raises(ValidationError):
        gen.parse({"side": 10})
    with pytest.raises(ValidationError):
        gen.parse({"thickness": -1})


def test_make_normalises_to_the_origin(gen: SquareGenerator) -> None:
    assert gen.make(side=200).bbox()[:2] == pytest.approx((0.0, 0.0))


def test_parameter_docs_expose_defaults_and_limits(gen: SquareGenerator) -> None:
    docs = {d.name: d for d in gen.parameter_docs()}
    assert docs["side"].default == 150.0
    assert docs["side"].constraints == ">= 60, <= 400"
    assert docs["side"].description == "side length in mm"
    assert docs["mode"].type_name == "router|laser"
    assert docs["thickness"].constraints == "> 0, <= 60"


# --------------------------------------------------------------------------- #
# variants
# --------------------------------------------------------------------------- #
def test_variants_are_reproducible_from_the_seed(gen: SquareGenerator) -> None:
    first = [d.slug for d in gen.variants(8, seed=11)]
    second = [d.slug for d in gen.variants(8, seed=11)]
    assert first == second


def test_asking_for_more_variants_keeps_the_earlier_ones(gen: SquareGenerator) -> None:
    few = [d.slug for d in gen.variants(4, seed=11)]
    many = [d.slug for d in gen.variants(12, seed=11)]
    assert many[: len(few)] == few


def test_different_seeds_give_different_variants(gen: SquareGenerator) -> None:
    assert [d.slug for d in gen.variants(6, seed=1)] != [
        d.slug for d in gen.variants(6, seed=2)
    ]


def test_variants_are_distinct(gen: SquareGenerator) -> None:
    slugs = [d.slug for d in gen.variants(15, seed=5)]
    assert len(slugs) == len(set(slugs))


def test_variants_are_all_valid(gen: SquareGenerator) -> None:
    for design in gen.variants(10, seed=3):
        assert gen.check(design).ok


def test_variants_skip_designs_that_cannot_be_built(
    gen: SquareGenerator, caplog: pytest.LogCaptureFixture
) -> None:
    class AlwaysBad(SquareGenerator):
        def sample_params(self, rng: random.Random, index: int) -> SquareParams:
            return SquareParams(side=150.0, unbuildable=index % 2 == 0)

    with caplog.at_level(logging.INFO, logger="dxfgen"):
        designs = AlwaysBad().variants(1, seed=0)
    assert len(designs) == 1
    assert "cannot be built" in caplog.text


def test_variants_skip_designs_that_fail_validation(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class AlwaysInvalid(SquareGenerator):
        def sample_params(self, rng: random.Random, index: int) -> SquareParams:
            return SquareParams(side=float(150 + index * 10), bad_pocket=True)

    with caplog.at_level(logging.INFO, logger="dxfgen"):
        designs = AlwaysInvalid().variants(3, seed=0)
    assert designs == []
    assert "E_TOOL_UNREACHABLE" in caplog.text


def test_variants_warns_when_it_cannot_meet_the_count(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class OneShape(SquareGenerator):
        def sample_params(self, rng: random.Random, index: int) -> SquareParams:
            return SquareParams(side=150.0)

    with caplog.at_level(logging.WARNING, logger="dxfgen"):
        designs = OneShape().variants(5, seed=0)
    assert len(designs) == 1  # all duplicates after the first
    assert "produced 1 of 5" in caplog.text


def test_variants_rejects_a_bad_count(gen: SquareGenerator) -> None:
    with pytest.raises(ValueError):
        gen.variants(0)


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #
def test_registry_exposes_the_built_in_niches() -> None:
    from dxfgen import niches

    assert "trays" in niches.niche_names()
    assert niches.get_generator("trays").title == "Trays"
    assert all(g.niche in niches.niche_names() for g in niches.all_generators())


def test_registry_reports_unknown_niches_helpfully() -> None:
    from dxfgen import niches

    with pytest.raises(KeyError, match="available: "):
        niches.get_generator("spaceships")


def test_registry_refuses_duplicate_niches() -> None:
    from dxfgen import niches

    with pytest.raises(ValueError, match="already registered"):
        niches.register(niches.get_generator("trays"))
