"""Tests for the layer standard."""

from __future__ import annotations

import pytest

from dxfgen.core import layers as L


def test_pocket_layer_name_strips_trailing_zeros() -> None:
    assert L.pocket_layer_name(8) == "POCKET_8"
    assert L.pocket_layer_name(8.0) == "POCKET_8"
    assert L.pocket_layer_name(1.5) == "POCKET_1.5"
    assert L.pocket_layer_name(12.25) == "POCKET_12.25"


def test_pocket_depth_round_trips() -> None:
    for depth in (0.5, 1.5, 3.0, 8.0, 12.25, 19.0):
        assert L.parse_pocket_depth(L.pocket_layer_name(depth)) == pytest.approx(depth)


@pytest.mark.parametrize("name", ["CUT_OUTSIDE", "POCKET_", "POCKET_abc", "POCKET_-3", "POCKET_0"])
def test_parse_pocket_depth_rejects_non_pockets(name: str) -> None:
    assert L.parse_pocket_depth(name) is None


def test_format_depth_rejects_non_positive() -> None:
    with pytest.raises(ValueError):
        L.format_depth(0)
    with pytest.raises(ValueError):
        L.format_depth(-2)


def test_standard_layers_have_distinct_colors() -> None:
    colors = [d.color for d in L.STANDARD_LAYERS.values()]
    assert len(colors) == len(set(colors))


def test_info_layer_is_never_machined() -> None:
    assert L.STANDARD_LAYERS[L.INFO].machined is False
    assert all(
        d.machined for name, d in L.STANDARD_LAYERS.items() if name != L.INFO
    )


def test_layer_def_synthesises_pocket_layers() -> None:
    d = L.layer_def("POCKET_8")
    assert d.depth == 8.0
    assert d.machined is True
    assert "8 mm" in d.description


def test_layer_def_rejects_unknown_layer() -> None:
    with pytest.raises(KeyError):
        L.layer_def("MY_RANDOM_LAYER")


def test_pocket_color_is_deterministic_and_in_palette() -> None:
    assert L.pocket_color(8) == L.pocket_color(8.0)
    for depth in (0.5, 3, 8, 12.5, 19):
        assert L.pocket_color(depth) in L.POCKET_COLORS


def test_is_cut_layer() -> None:
    assert L.is_cut_layer(L.CUT_OUTSIDE)
    assert L.is_cut_layer(L.CUT_INSIDE)
    assert L.is_cut_layer("POCKET_3")
    assert not L.is_cut_layer(L.INFO)
    assert not L.is_cut_layer(L.ENGRAVE)
    assert not L.is_cut_layer(L.DRILL)
