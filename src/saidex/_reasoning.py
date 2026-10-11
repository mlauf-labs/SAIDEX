"""Reasoning-effort escalation: pick a ``reasoning_effort`` level per attempt."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any, Literal, TypeAlias

logger = logging.getLogger(__name__)

#: How a caller asks for reasoning-effort escalation: an explicit list of levels
#: (attempt *n* uses ``levels[min(n - 1, len(levels) - 1)]``) or ``True`` to use
#: the levels from the model's ``profile["reasoning_effort_levels"]``.
ReasoningEscalation: TypeAlias = Sequence[str] | Literal[True]


def _profile_levels(model: Any) -> tuple[str, ...]:
    """Return the model profile's ``reasoning_effort_levels``, or ``()``."""
    profile = getattr(model, "profile", None)
    if not isinstance(profile, Mapping):
        return ()
    levels = profile.get("reasoning_effort_levels")
    if not isinstance(levels, (list, tuple)):
        return ()
    return tuple(str(level) for level in levels)


def _resolve_reasoning_levels(
    model: Any, spec: ReasoningEscalation | None, *, param: str
) -> tuple[str, ...]:
    """Validate an escalation *spec* for *model* and return its ordered levels.

    Args:
        model: The chat model the levels will be bound to.
        spec: ``None`` (off), ``True`` (levels from the model profile) or a
            non-empty sequence of level names.
        param: Parameter name used in error and log messages.

    Returns:
        The levels to use, or ``()`` when escalation is off.

    Raises:
        ValueError: *spec* is a plain string, empty, contains non-strings, or
            names a level the model profile does not list.
    """
    if spec is None:
        return ()
    profile_levels = _profile_levels(model)
    if spec is True:
        if not profile_levels:
            logger.info(
                "%s=True, but the model profile lists no reasoning_effort levels; "
                "reasoning effort is not escalated.",
                param,
            )
        return profile_levels
    if isinstance(spec, str) or not isinstance(spec, Sequence):
        raise ValueError(
            f"{param} must be a list of reasoning-effort levels (e.g. ['low', 'high']) "
            f"or True; got {spec!r}"
        )
    levels = tuple(spec)
    if not levels:
        raise ValueError(f"{param} must not be empty")
    if not all(isinstance(level, str) and level for level in levels):
        raise ValueError(f"{param} must contain non-empty strings; got {spec!r}")
    if profile_levels:
        unknown = [level for level in levels if level not in profile_levels]
        if unknown:
            raise ValueError(
                f"{param}: unsupported reasoning_effort level(s) "
                f"{', '.join(repr(u) for u in unknown)}; the model supports "
                f"{', '.join(profile_levels)}"
            )
    return levels


def _level_for_attempt(levels: tuple[str, ...], attempt_index: int) -> str | None:
    """Return the level for the 0-based *attempt_index*; the last level repeats."""
    if not levels:
        return None
    return levels[min(attempt_index, len(levels) - 1)]


def _with_reasoning_effort(runnable: Any, level: str | None) -> Any:
    """Bind ``reasoning_effort=level`` to *runnable*, or return it unchanged."""
    if level is None:
        return runnable
    return runnable.bind(reasoning_effort=level)
