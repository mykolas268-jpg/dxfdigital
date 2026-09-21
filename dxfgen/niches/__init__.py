"""Per-niche design generators and the registry the CLI looks them up in."""

from __future__ import annotations

from .ashtrays import AshtrayGenerator
from .base import Generator, GeneratorParams, ParamDoc
from .boards import BoardGenerator
from .boxes import BoxGenerator
from .coasters import CoasterGenerator
from .furniture import FurnitureGenerator
from .stands import StandGenerator
from .trays import TrayGenerator

__all__ = [
    "Generator",
    "GeneratorParams",
    "ParamDoc",
    "REGISTRY",
    "register",
    "get_generator",
    "all_generators",
    "niche_names",
]

#: Every generator this build can produce, keyed by niche name.
REGISTRY: dict[str, Generator] = {}


def register(generator: Generator) -> Generator:
    """Add a generator to the registry.

    Args:
        generator: The instance to register.

    Returns:
        The same instance, so this can be used as a decorator.

    Raises:
        ValueError: If the niche name is already taken.
    """
    if generator.niche in REGISTRY:
        raise ValueError(f"niche {generator.niche!r} is already registered")
    REGISTRY[generator.niche] = generator
    return generator


def get_generator(niche: str) -> Generator:
    """Look up a generator by niche name.

    Args:
        niche: The niche key, e.g. ``"trays"``.

    Returns:
        The generator.

    Raises:
        KeyError: If no such niche exists, listing the ones that do.
    """
    try:
        return REGISTRY[niche]
    except KeyError:
        raise KeyError(
            f"unknown niche {niche!r}; available: {', '.join(niche_names())}"
        ) from None


def all_generators() -> list[Generator]:
    """Every registered generator, in registration order."""
    return list(REGISTRY.values())


def niche_names() -> list[str]:
    """Every registered niche name, sorted."""
    return sorted(REGISTRY)


for _generator in (
    TrayGenerator(),
    BoardGenerator(),
    CoasterGenerator(),
    AshtrayGenerator(),
    StandGenerator(),
    BoxGenerator(),
    FurnitureGenerator(),
):
    register(_generator)
del _generator
