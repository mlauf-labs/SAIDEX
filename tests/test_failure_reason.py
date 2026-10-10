"""Tests for the typed FailureReason enum and the primary/fallback combination rule."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

from langchain_core.messages import HumanMessage
from pydantic import BaseModel

import saidex
from saidex import ExtractorRunStats, FailureReason, extract_data, extract_data_with_tools
from saidex.models import _combine_failure_reasons
from saidex.retry import RetryConfig
from tests._mock_llm import make_llm, make_response

_NO_RETRY = RetryConfig(max_retries=0, retry_delays=[], retryable_exceptions=())


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


class Item(BaseModel):
    name: str
    value: int


def _call(args: dict[str, Any], name: str = "Item", call_id: str = "c1") -> dict[str, Any]:
    return {"name": name, "args": args, "id": call_id}


def _msgs() -> list[HumanMessage]:
    return [HumanMessage(content="extract")]


async def test_validation_exhausted_reason_is_enum() -> None:
    llm = make_llm([make_response([_call({"name": "a", "value": "x"})])] * 2)
    _, stats = await extract_data(llm, Item, _msgs(), max_primary_retries=2, retry_config=_NO_RETRY)
    assert stats.failure_reason is FailureReason.VALIDATION_EXHAUSTED


async def test_parse_error_reason_is_enum() -> None:
    llm = make_llm([make_response(invalid=True)] * 2)
    _, stats = await extract_data(llm, Item, _msgs(), max_primary_retries=2, retry_config=_NO_RETRY)
    assert stats.failure_reason is FailureReason.PARSE_ERROR


async def test_llm_error_reason_is_enum() -> None:
    llm = make_llm([RuntimeError("provider down")])
    _, stats = await extract_data(llm, Item, _msgs(), retry_config=_NO_RETRY)
    assert stats.failure_reason is FailureReason.LLM_ERROR


async def test_bind_tools_rejection_reason_is_enum() -> None:
    llm = MagicMock()
    llm.bind_tools = MagicMock(side_effect=TypeError("unexpected keyword 'strict'"))
    _, stats = await extract_data(llm, Item, _msgs(), retry_config=_NO_RETRY)
    assert stats.failure_reason is FailureReason.LLM_ERROR


async def test_fallback_without_attempts_keeps_primary_reason() -> None:
    primary = make_llm([make_response(invalid=True)])
    fallback = make_llm([])
    _, stats = await extract_data(
        primary,
        Item,
        _msgs(),
        fallback_llm_model=fallback,
        max_primary_retries=1,
        max_fallback_retries=0,
        retry_config=_NO_RETRY,
    )
    assert stats.fallback_used
    assert stats.failure_reason is FailureReason.PARSE_ERROR


async def test_agent_no_tool_call_reason_is_enum() -> None:
    llm = make_llm([make_response(tool_calls=[])] * 3)
    _, stats = await extract_data_with_tools(
        llm, Item, "text", tools=[], max_validation_retries=1, retry_config=_NO_RETRY
    )
    assert stats.failure_reason is FailureReason.NO_TOOL_CALL


async def test_agent_fallback_combination_uses_shared_rule() -> None:
    primary = make_llm([make_response(tool_calls=[])] * 3)
    fallback = make_llm([RuntimeError("fallback down")])
    _, stats = await extract_data_with_tools(
        primary,
        Item,
        "text",
        tools=[],
        fallback_llm_model=fallback,
        max_validation_retries=1,
        retry_config=_NO_RETRY,
    )
    assert stats.fallback_used
    assert stats.failure_reason is FailureReason.LLM_ERROR
