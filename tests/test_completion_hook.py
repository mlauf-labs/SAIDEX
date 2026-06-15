"""Tests for the on_complete hook and capture_source_text option."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from saidex import (
    ExtractionEvent,
    extract_data,
    extract_data_from_text,
    extract_data_list,
    extract_data_with_tools,
)
from saidex.retry import RetryConfig

NO_RETRY = RetryConfig(
    max_retries=0,
    retry_delays=[],
    retryable_exceptions=(),
    rate_limit_exceptions=(),
)


class SimpleSchema(BaseModel):
    name: str
    value: int


def _response(args: dict[str, Any] | None = None) -> MagicMock:
    r = MagicMock()
    if args is None:
        r.tool_calls = []
        r.invalid_tool_calls = []
    else:
        r.tool_calls = [{"args": args, "name": "SimpleSchema", "id": "c1"}]
        r.invalid_tool_calls = []
    r.content = ""
    return r


def _bound_llm(responses: list[MagicMock]) -> MagicMock:
    bound = MagicMock()
    bound.ainvoke = AsyncMock(side_effect=responses)
    llm = MagicMock()
    llm.bind_tools = MagicMock(return_value=bound)
    return llm


@pytest.mark.asyncio
async def test_hook_fires_once_on_success() -> None:
    events: list[ExtractionEvent] = []

    async def hook(event: ExtractionEvent) -> None:
        events.append(event)

    llm = _bound_llm([_response({"name": "Alice", "value": 1})])
    result, stats = await extract_data(
        llm,
        SimpleSchema,
        [HumanMessage(content="x")],
        retry_config=NO_RETRY,
        on_complete=hook,
    )

    assert len(events) == 1
    event = events[0]
    assert event.schema_name == "SimpleSchema"
    assert event.result is result
    assert event.stats is stats
    assert event.stats.success is True
    assert event.source_text is None  # capture off by default


@pytest.mark.asyncio
async def test_hook_fires_on_failure() -> None:
    events: list[ExtractionEvent] = []

    async def hook(event: ExtractionEvent) -> None:
        events.append(event)

    bad = _response({"name": "Alice", "value": "not-int"})
    llm = _bound_llm([bad, bad, bad])
    result, _ = await extract_data(
        llm,
        SimpleSchema,
        [HumanMessage(content="x")],
        max_primary_retries=3,
        retry_config=NO_RETRY,
        on_complete=hook,
    )

    assert result is None
    assert len(events) == 1
    assert events[0].result is None
    assert events[0].stats.success is False
    assert events[0].stats.failure_reason == "validation_exhausted"


@pytest.mark.asyncio
async def test_capture_source_text_on() -> None:
    captured: list[ExtractionEvent] = []

    async def hook(event: ExtractionEvent) -> None:
        captured.append(event)

    llm = _bound_llm([_response({"name": "Alice", "value": 1})])
    _, stats = await extract_data_from_text(
        llm,
        SimpleSchema,
        "Alice is 1",
        retry_config=NO_RETRY,
        on_complete=hook,
        capture_source_text=True,
    )

    assert stats.source_text == "Alice is 1"
    assert captured[0].source_text == "Alice is 1"


@pytest.mark.asyncio
async def test_capture_source_text_off_by_default() -> None:
    llm = _bound_llm([_response({"name": "Alice", "value": 1})])
    _, stats = await extract_data_from_text(llm, SimpleSchema, "Alice is 1", retry_config=NO_RETRY)
    assert stats.source_text is None


@pytest.mark.asyncio
async def test_hook_exception_is_isolated() -> None:
    async def boom(event: ExtractionEvent) -> None:
        raise RuntimeError("hook failure must not propagate")

    llm = _bound_llm([_response({"name": "Alice", "value": 1})])
    # Should not raise despite the hook raising.
    result, stats = await extract_data(
        llm,
        SimpleSchema,
        [HumanMessage(content="x")],
        retry_config=NO_RETRY,
        on_complete=boom,
    )
    assert result is not None
    assert stats.success is True


@pytest.mark.asyncio
async def test_no_hook_is_a_noop() -> None:
    llm = _bound_llm([_response({"name": "Alice", "value": 1})])
    result, _ = await extract_data(
        llm, SimpleSchema, [HumanMessage(content="x")], retry_config=NO_RETRY
    )
    assert result is not None


@pytest.mark.asyncio
async def test_list_hook_reports_list_result_and_item_schema() -> None:
    events: list[ExtractionEvent] = []

    async def hook(event: ExtractionEvent) -> None:
        events.append(event)

    items = [{"name": "A", "value": 1}, {"name": "B", "value": 2}]
    llm = _bound_llm([_response({"items": items})])
    result, _ = await extract_data_list(
        llm,
        SimpleSchema,
        [HumanMessage(content="all")],
        retry_config=NO_RETRY,
        on_complete=hook,
    )

    assert len(events) == 1  # fired once, not also by the inner extract_data
    event = events[0]
    assert event.schema_name == "SimpleSchema"  # item schema, not the container
    assert isinstance(event.result, list)
    assert event.result == result
    assert event.stats.item_count == 2


@pytest.mark.asyncio
async def test_agent_hook_fires() -> None:
    events: list[ExtractionEvent] = []

    async def hook(event: ExtractionEvent) -> None:
        events.append(event)

    final = MagicMock()
    final.tool_calls = [{"name": "SimpleSchema", "args": {"name": "A", "value": 1}, "id": "f1"}]
    final.invalid_tool_calls = []
    final.content = ""
    llm = _bound_llm([final])

    result, _ = await extract_data_with_tools(
        llm,
        SimpleSchema,
        "do it",
        tools=[],
        retry_config=NO_RETRY,
        on_complete=hook,
        capture_source_text=True,
    )

    assert result is not None
    assert len(events) == 1
    assert events[0].schema_name == "SimpleSchema"
    assert events[0].source_text == "do it"
