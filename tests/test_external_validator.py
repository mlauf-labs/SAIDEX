"""Tests for the external ``validator`` callable across all extraction entry points."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from saidex import (
    ExtractionMode,
    extract_data,
    extract_data_list,
    extract_data_sync,
    extract_data_with_tools,
)
from saidex.retry import RetryConfig

# ---------------------------------------------------------------------------
# Helpers & schemas
# ---------------------------------------------------------------------------

_NO_NETWORK_RETRY = RetryConfig(
    max_retries=0,
    retry_delays=[],
    retryable_exceptions=(),
    rate_limit_exceptions=(),
)


class SimpleSchema(BaseModel):
    name: str
    value: int


def _tool_response(args: dict[str, Any], name: str = "SimpleSchema") -> MagicMock:
    response = MagicMock()
    response.tool_calls = [{"args": args, "name": name, "id": "call_1"}]
    response.invalid_tool_calls = []
    response.content = ""
    return response


def _bound_llm(responses: list[MagicMock]) -> MagicMock:
    bound = MagicMock()
    bound.ainvoke = AsyncMock(side_effect=responses)
    llm = MagicMock()
    llm.bind_tools = MagicMock(return_value=bound)
    return llm


def _json_response(content: str) -> MagicMock:
    response = MagicMock()
    response.content = content
    response.tool_calls = []
    response.invalid_tool_calls = []
    return response


def _json_llm(contents: list[str]) -> MagicMock:
    llm = MagicMock()
    llm.ainvoke = AsyncMock(side_effect=[_json_response(c) for c in contents])
    llm.bind_tools = MagicMock(
        side_effect=AssertionError("bind_tools must not be called in JSON mode")
    )
    return llm


def _messages() -> list[HumanMessage]:
    return [HumanMessage(content="Extract data")]


def _reject_low(instance: SimpleSchema) -> None:
    """Sync validator that rejects values below 100 by raising."""
    if instance.value < 100:
        raise ValueError(f"value {instance.value} must be at least 100")


# ---------------------------------------------------------------------------
# extract_data — single shot, TOOL_CALLING
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_validator_rejects_once_then_passes() -> None:
    llm = _bound_llm(
        [
            _tool_response({"name": "Alice", "value": 5}),  # rejected by validator
            _tool_response({"name": "Alice", "value": 150}),  # accepted
        ]
    )

    result, stats = await extract_data(
        llm,
        SimpleSchema,
        _messages(),
        retry_config=_NO_NETWORK_RETRY,
        validator=_reject_low,
    )

    assert result is not None
    assert result.value == 150
    assert stats.success
    assert stats.primary_retries == 1
    # The rejection is recorded as a structured field issue.
    assert any(issue.error_type == "external_validator" for issue in stats.field_issues)


@pytest.mark.asyncio
async def test_validator_exhausts_retries_returns_none() -> None:
    llm = _bound_llm([_tool_response({"name": "Alice", "value": 1})] * 3)

    result, stats = await extract_data(
        llm,
        SimpleSchema,
        _messages(),
        max_primary_retries=3,
        retry_config=_NO_NETWORK_RETRY,
        validator=_reject_low,
    )

    assert result is None
    assert not stats.success
    assert stats.failure_reason == "validation_exhausted"
    assert stats.primary_retries == 3
    assert sum(i.error_type == "external_validator" for i in stats.field_issues) == 3


@pytest.mark.asyncio
async def test_validator_can_return_error_string() -> None:
    def reject_with_message(instance: SimpleSchema) -> str | None:
        if instance.value < 100:
            return "too small"
        return None

    llm = _bound_llm(
        [
            _tool_response({"name": "Alice", "value": 5}),
            _tool_response({"name": "Alice", "value": 200}),
        ]
    )

    result, stats = await extract_data(
        llm,
        SimpleSchema,
        _messages(),
        retry_config=_NO_NETWORK_RETRY,
        validator=reject_with_message,
    )

    assert result is not None
    assert result.value == 200
    assert stats.primary_retries == 1
    assert stats.field_issues[0].message == "too small"


@pytest.mark.asyncio
async def test_async_validator_supported() -> None:
    async def reject_low_async(instance: SimpleSchema) -> None:
        if instance.value < 100:
            raise ValueError("async rejection")

    llm = _bound_llm(
        [
            _tool_response({"name": "Alice", "value": 5}),
            _tool_response({"name": "Alice", "value": 150}),
        ]
    )

    result, stats = await extract_data(
        llm,
        SimpleSchema,
        _messages(),
        retry_config=_NO_NETWORK_RETRY,
        validator=reject_low_async,
    )

    assert result is not None
    assert result.value == 150
    assert stats.primary_retries == 1


@pytest.mark.asyncio
async def test_validator_not_invoked_on_pydantic_failure() -> None:
    seen: list[int] = []

    def record(instance: SimpleSchema) -> None:
        seen.append(instance.value)

    # First response fails Pydantic (value is not an int) — validator must not run.
    llm = _bound_llm(
        [
            _tool_response({"name": "Alice", "value": "not_an_int"}),
            _tool_response({"name": "Alice", "value": 42}),
        ]
    )

    result, stats = await extract_data(
        llm,
        SimpleSchema,
        _messages(),
        retry_config=_NO_NETWORK_RETRY,
        validator=record,
    )

    assert result is not None
    # Validator only ran for the structurally valid instance.
    assert seen == [42]
    assert stats.primary_retries == 1


@pytest.mark.asyncio
async def test_validator_triggers_fallback() -> None:
    primary = _bound_llm([_tool_response({"name": "Alice", "value": 1})] * 3)
    fallback = _bound_llm([_tool_response({"name": "Alice", "value": 999})])

    result, stats = await extract_data(
        primary,
        SimpleSchema,
        _messages(),
        fallback_llm_model=fallback,
        max_primary_retries=3,
        retry_config=_NO_NETWORK_RETRY,
        validator=_reject_low,
    )

    assert result is not None
    assert result.value == 999
    assert stats.fallback_used


# ---------------------------------------------------------------------------
# JSON mode
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_validator_json_mode() -> None:
    llm = _json_llm(
        [
            '{"name": "Alice", "value": 5}',
            '{"name": "Alice", "value": 150}',
        ]
    )

    result, stats = await extract_data(
        llm,
        SimpleSchema,
        _messages(),
        mode=ExtractionMode.JSON,
        retry_config=_NO_NETWORK_RETRY,
        validator=_reject_low,
    )

    assert result is not None
    assert result.value == 150
    assert stats.primary_retries == 1


# ---------------------------------------------------------------------------
# Batch — per-item validation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_validator_runs_per_item_in_batch() -> None:
    bad = _tool_response(
        {"items": [{"name": "A", "value": 150}, {"name": "B", "value": 5}]},
        name="SimpleSchemaList",
    )
    good = _tool_response(
        {"items": [{"name": "A", "value": 150}, {"name": "B", "value": 120}]},
        name="SimpleSchemaList",
    )
    llm = _bound_llm([bad, good])

    items, stats = await extract_data_list(
        llm,
        SimpleSchema,
        _messages(),
        retry_config=_NO_NETWORK_RETRY,
        validator=_reject_low,
    )

    assert items is not None
    assert [i.value for i in items] == [150, 120]
    assert stats.primary_retries == 1
    # The per-item message points at the offending index.
    assert any("items -> 1" in i.message for i in stats.field_issues)


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------


class FinalAnswer(BaseModel):
    result: str
    score: int = 0


def _agent_final(args: dict[str, Any], call_id: str = "c1") -> MagicMock:
    response = MagicMock()
    response.tool_calls = [{"name": "FinalAnswer", "args": args, "id": call_id}]
    response.invalid_tool_calls = []
    response.content = ""
    return response


def _agent_llm(responses: list[MagicMock]) -> MagicMock:
    bound = MagicMock()
    bound.ainvoke = AsyncMock(side_effect=responses)
    llm = MagicMock()
    llm.bind_tools = MagicMock(return_value=bound)
    llm.ainvoke = bound.ainvoke
    return llm


@pytest.mark.asyncio
async def test_validator_in_agent_loop_reject_then_pass() -> None:
    def reject_low_score(instance: FinalAnswer) -> None:
        if instance.score < 10:
            raise ValueError("score too low")

    llm = _agent_llm(
        [
            _agent_final({"result": "first", "score": 1}, "c1"),
            _agent_final({"result": "second", "score": 50}, "c2"),
        ]
    )

    result, stats = await extract_data_with_tools(
        llm,
        FinalAnswer,
        "do it",
        tools=[],
        retry_config=_NO_NETWORK_RETRY,
        validator=reject_low_score,
    )

    assert result is not None
    assert result.result == "second"
    assert stats.validation_retries == 1
    assert any(i.error_type == "external_validator" for i in stats.field_issues)


@pytest.mark.asyncio
async def test_validator_in_agent_loop_exhausts() -> None:
    def always_reject(instance: FinalAnswer) -> None:
        raise ValueError("never good enough")

    llm = _agent_llm([_agent_final({"result": "x", "score": 1}, f"c{i}") for i in range(10)])

    result, stats = await extract_data_with_tools(
        llm,
        FinalAnswer,
        "do it",
        tools=[],
        max_validation_retries=2,
        retry_config=_NO_NETWORK_RETRY,
        validator=always_reject,
    )

    assert result is None
    assert not stats.success


# ---------------------------------------------------------------------------
# Sync wrapper
# ---------------------------------------------------------------------------


def test_validator_sync_wrapper() -> None:
    llm = _bound_llm(
        [
            _tool_response({"name": "Alice", "value": 5}),
            _tool_response({"name": "Alice", "value": 150}),
        ]
    )

    result, stats = extract_data_sync(
        llm,
        SimpleSchema,
        _messages(),
        retry_config=_NO_NETWORK_RETRY,
        validator=_reject_low,
    )

    assert result is not None
    assert result.value == 150
    assert stats.primary_retries == 1
