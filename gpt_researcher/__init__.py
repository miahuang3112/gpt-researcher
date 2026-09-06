from __future__ import annotations

from typing import TYPE_CHECKING, Any


__all__ = [
    "GPTResearcher",
]


if TYPE_CHECKING:
    from .agent import GPTResearcher as GPTResearcher


def __getattr__(name: str) -> Any:
    """
    Lazily import the main GPTResearcher agent.

    This keeps lightweight subpackages such as
    gpt_researcher.academic importable without
    loading the full GPT-Researcher agent stack.
    """

    if name == "GPTResearcher":
        from .agent import GPTResearcher

        return GPTResearcher

    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}"
    )