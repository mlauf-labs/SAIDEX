"""Tests for the typed FailureReason enum and the primary/fallback combination rule."""

from __future__ import annotations

import json

import saidex
from saidex import ExtractorRunStats, FailureReason
from saidex.models import _combine_failure_reasons


def test_failure_reason_values_are_stable() -> None:
    assert [m.value for m in FailureReason] == [
        "validation_exhausted",
        "parse_error",
        "llm_error",
        "no_tool_call",
    ]


def test_failure_reason_behaves_like_its_string_value() -> None:
    reason = FailureReason.PARSE_ERROR
    assert reason == "parse_error"
    assert str(reason) == "parse_error"
    assert f"{reason}" == "parse_error"
    assert json.dumps({"r": reason}) == '{"r": "parse_error"}'
    assert FailureReason("llm_error") is FailureReason.LLM_ERROR


def test_failure_reason_is_exported() -> None:
    assert "FailureReason" in saidex.__all__
    assert saidex.FailureReason is FailureReason


def test_combine_prefers_fallback_reason() -> None:
    assert (
        _combine_failure_reasons(FailureReason.PARSE_ERROR, FailureReason.LLM_ERROR, success=False)
        is FailureReason.LLM_ERROR
    )


def test_combine_falls_back_to_primary_reason() -> None:
    assert (
        _combine_failure_reasons(FailureReason.PARSE_ERROR, None, success=False)
        is FailureReason.PARSE_ERROR
    )


def test_combine_success_has_no_reason() -> None:
    assert (
        _combine_failure_reasons(FailureReason.PARSE_ERROR, FailureReason.LLM_ERROR, success=True)
        is None
    )


def test_add_success_clears_failure_reason() -> None:
    failed = ExtractorRunStats(success=False, failure_reason=FailureReason.NO_TOOL_CALL)
    succeeded = ExtractorRunStats(success=True)
    combined = failed + succeeded
    assert combined.success is True
    assert combined.failure_reason is None


def test_add_keeps_fallback_reason_on_double_failure() -> None:
    first = ExtractorRunStats(success=False, failure_reason=FailureReason.NO_TOOL_CALL)
    second = ExtractorRunStats(success=False, failure_reason=FailureReason.LLM_ERROR)
    assert (first + second).failure_reason is FailureReason.LLM_ERROR
