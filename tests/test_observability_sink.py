"""Tests for collect_stats — the scoped extraction-stats sink."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from saidex import (
    StatsSink,
    collect_stats,
    extract_data,
    extract_data_from_text_sync,
    extract_data_list,
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
async def test_sink_collects_stats_in_order() -> None:
    llm = _bound_llm([_response({"name": "A", "value": 1}), _response({"name": "B", "value": 2})])
    async with collect_stats() as sink:
        _, s1 = await extract_data(
            llm, SimpleSchema, [HumanMessage(content="x")], retry_config=NO_RETRY
        )
        _, s2 = await extract_data(
            llm, SimpleSchema, [HumanMessage(content="y")], retry_config=NO_RETRY
        )

    assert isinstance(sink, StatsSink)
    assert sink.all() == [s1, s2]
    assert len(sink) == 2
    assert list(sink) == [s1, s2]


@pytest.mark.asyncio
async def test_sink_all_returns_a_copy() -> None:
    llm = _bound_llm([_response({"name": "A", "value": 1})])
    async with collect_stats() as sink:
        await extract_data(llm, SimpleSchema, [HumanMessage(content="x")], retry_config=NO_RETRY)

    snapshot = sink.all()
    snapshot.clear()
    assert len(sink) == 1  # mutating the returned list must not affect the sink


@pytest.mark.asyncio
async def test_sink_records_failures_too() -> None:
    bad = _response({"name": "A", "value": "not-int"})
    llm = _bound_llm([bad, bad, bad])
    async with collect_stats() as sink:
        result, _ = await extract_data(
            llm,
            SimpleSchema,
            [HumanMessage(content="x")],
            max_primary_retries=3,
            retry_config=NO_RETRY,
        )
    assert result is None
    assert len(sink) == 1
    assert sink.all()[0].success is False


@pytest.mark.asyncio
async def test_nested_sinks_inner_and_outer() -> None:
    llm = _bound_llm([_response({"name": "A", "value": 1}), _response({"name": "B", "value": 2})])
    async with collect_stats() as outer:
        _, s1 = await extract_data(
            llm, SimpleSchema, [HumanMessage(content="x")], retry_config=NO_RETRY
        )
        async with collect_stats() as inner:
            _, s2 = await extract_data(
                llm, SimpleSchema, [HumanMessage(content="y")], retry_config=NO_RETRY
            )
        assert inner.all() == [s2]  # inner sees only its own extraction
    assert outer.all() == [s1, s2]  # outer sees both


@pytest.mark.asyncio
async def test_no_state_leaks_after_block() -> None:
    llm = _bound_llm([_response({"name": "A", "value": 1})])
    async with collect_stats() as sink:
        await extract_data(llm, SimpleSchema, [HumanMessage(content="x")], retry_config=NO_RETRY)
    assert len(sink) == 1

    llm2 = _bound_llm([_response({"name": "B", "value": 2})])
    await extract_data(llm2, SimpleSchema, [HumanMessage(content="y")], retry_config=NO_RETRY)
    assert len(sink) == 1  # an extraction outside the block is not recorded


@pytest.mark.asyncio
async def test_concurrent_extractions_share_sink() -> None:
    llm = _bound_llm([_response({"name": "A", "value": 1}), _response({"name": "B", "value": 2})])
    async with collect_stats() as sink:
        await asyncio.gather(
            extract_data(llm, SimpleSchema, [HumanMessage(content="x")], retry_config=NO_RETRY),
            extract_data(llm, SimpleSchema, [HumanMessage(content="y")], retry_config=NO_RETRY),
        )
    assert len(sink) == 2


def test_sync_context_manager_around_sync_wrapper() -> None:
    llm = _bound_llm([_response({"name": "A", "value": 1})])
    with collect_stats() as sink:
        extract_data_from_text_sync(llm, SimpleSchema, "Alice is 1", retry_config=NO_RETRY)
    assert len(sink) == 1
    assert sink.all()[0].success is True


@pytest.mark.asyncio
async def test_list_extraction_records_once_with_item_schema() -> None:
    items = [{"name": "A", "value": 1}, {"name": "B", "value": 2}]
    llm = _bound_llm([_response({"items": items})])
    async with collect_stats() as sink:
        _, stats = await extract_data_list(
            llm, SimpleSchema, [HumanMessage(content="all")], retry_config=NO_RETRY
        )
    # Exactly one entry — the internal container extraction must be invisible.
    assert len(sink) == 1
    assert sink.all()[0] is stats
    assert sink.all()[0].schema_name == "SimpleSchema"  # item schema, not the container
    assert sink.all()[0].item_count == 2
