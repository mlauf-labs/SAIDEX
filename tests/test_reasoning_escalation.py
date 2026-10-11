"""Tests for reasoning-effort escalation across all extraction entry points."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from saidex._reasoning import (
    _level_for_attempt,
    _profile_levels,
    _resolve_reasoning_levels,
    _with_reasoning_effort,
)


def _model(levels: list[str] | None = None) -> Any:
    model = MagicMock()
    model.profile = None if levels is None else {"reasoning_effort_levels": levels}
    return model


def test_none_means_no_levels() -> None:
    assert _resolve_reasoning_levels(_model(["low", "high"]), None, param="p") == ()


def test_explicit_list_without_profile_is_passed_through() -> None:
    assert _resolve_reasoning_levels(_model(), ["low", "xhigh"], param="p") == ("low", "xhigh")


def test_true_uses_profile_levels() -> None:
    assert _resolve_reasoning_levels(_model(["low", "medium", "high"]), True, param="p") == (
        "low",
        "medium",
        "high",
    )


def test_true_without_profile_levels_disables_escalation(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level("INFO", logger="saidex._reasoning")
    assert _resolve_reasoning_levels(_model(), True, param="reasoning_escalation") == ()
    assert _resolve_reasoning_levels(_model([]), True, param="reasoning_escalation") == ()
    assert any("reasoning_escalation" in r.message for r in caplog.records)


def test_unknown_level_with_profile_raises() -> None:
    with pytest.raises(ValueError, match="'turbo'.*low.*high"):
        _resolve_reasoning_levels(_model(["low", "high"]), ["low", "turbo"], param="p")


def test_empty_list_raises() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        _resolve_reasoning_levels(_model(), [], param="p")


def test_plain_string_is_rejected() -> None:
    with pytest.raises(ValueError, match="list"):
        _resolve_reasoning_levels(_model(), "high", param="p")  # type: ignore[arg-type]


def test_non_string_level_is_rejected() -> None:
    with pytest.raises(ValueError):
        _resolve_reasoning_levels(_model(), ["low", 3], param="p")  # type: ignore[list-item]


def test_profile_levels_tolerates_missing_or_odd_profiles() -> None:
    assert _profile_levels(object()) == ()
    assert _profile_levels(_model()) == ()
    weird = MagicMock()
    weird.profile = {"reasoning_effort_levels": "high"}
    assert _profile_levels(weird) == ()


@pytest.mark.parametrize(
    ("index", "expected"), [(0, "low"), (1, "medium"), (2, "high"), (3, "high"), (9, "high")]
)
def test_level_for_attempt(index: int, expected: str) -> None:
    assert _level_for_attempt(("low", "medium", "high"), index) == expected


def test_level_for_attempt_without_levels() -> None:
    assert _level_for_attempt((), 0) is None


def test_with_reasoning_effort_binds_only_when_set() -> None:
    runnable = MagicMock()
    assert _with_reasoning_effort(runnable, None) is runnable
    runnable.bind.assert_not_called()
    _with_reasoning_effort(runnable, "high")
    runnable.bind.assert_called_once_with(reasoning_effort="high")
