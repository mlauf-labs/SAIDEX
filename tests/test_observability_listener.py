"""Tests for on_extraction — the global extraction listener."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from saidex import (
    ExtractionEvent,
    collect_stats,
    extract_data,
    extract_data_list,
    observability,
    on_extraction,
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


@pytest.fixture(autouse=True)
def _clear_listeners() -> Iterator[None]:
    yield
    observability._global_listeners.clear()


def _response(args: dict[str, Any]) -> MagicMock:
    r = MagicMock()
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


async def _extract_once(args: dict[str, Any]) -> None:
    llm = _bound_llm([_response(args)])
    await extract_data(llm, SimpleSchema, [HumanMessage(content="x")], retry_config=NO_RETRY)


@pytest.mark.asyncio
async def test_sync_listener_receives_full_event() -> None:
    seen: list[ExtractionEvent] = []
    on_extraction(lambda e: seen.append(e))
    await _extract_once({"name": "A", "value": 1})

    assert len(seen) == 1
    assert seen[0].schema_name == "SimpleSchema"
    assert seen[0].stats.success is True
    assert seen[0].messages  # full event, not just stats


@pytest.mark.asyncio
async def test_async_listener_receives_event() -> None:
    seen: list[ExtractionEvent] = []

    async def listener(e: ExtractionEvent) -> None:
        seen.append(e)

    on_extraction(listener)
    await _extract_once({"name": "A", "value": 1})
    assert len(seen) == 1


@pytest.mark.asyncio
async def test_unsubscribe_stops_delivery() -> None:
    seen: list[ExtractionEvent] = []
    sub = on_extraction(lambda e: seen.append(e))
    sub.unsubscribe()
    await _extract_once({"name": "A", "value": 1})
    assert seen == []


@pytest.mark.asyncio
async def test_subscription_is_callable_and_idempotent() -> None:
    seen: list[ExtractionEvent] = []
    sub = on_extraction(lambda e: seen.append(e))
    sub()  # calling the handle unsubscribes
    sub()  # idempotent — must not raise
    await _extract_once({"name": "A", "value": 1})
    assert seen == []


@pytest.mark.asyncio
async def test_subscription_as_context_manager() -> None:
    seen: list[ExtractionEvent] = []
    with on_extraction(lambda e: seen.append(e)):
        await _extract_once({"name": "A", "value": 1})
    await _extract_once({"name": "B", "value": 2})  # listener removed on exit
    assert len(seen) == 1


@pytest.mark.asyncio
async def test_listener_exception_is_isolated() -> None:
    seen: list[ExtractionEvent] = []

    def boom(e: ExtractionEvent) -> None:
        raise RuntimeError("must not propagate")

    on_extraction(boom)
    on_extraction(lambda e: seen.append(e))
    # The raising listener must neither propagate nor prevent later listeners.
    await _extract_once({"name": "A", "value": 1})
    assert len(seen) == 1


@pytest.mark.asyncio
async def test_listener_and_sink_coexist() -> None:
    seen: list[ExtractionEvent] = []
    on_extraction(lambda e: seen.append(e))
    async with collect_stats() as sink:
        await _extract_once({"name": "A", "value": 1})
    assert len(seen) == 1
    assert len(sink) == 1


@pytest.mark.asyncio
async def test_list_extraction_fires_listener_once() -> None:
    seen: list[ExtractionEvent] = []
    on_extraction(lambda e: seen.append(e))
    items = [{"name": "A", "value": 1}, {"name": "B", "value": 2}]
    llm = _bound_llm([_response({"items": items})])
    await extract_data_list(llm, SimpleSchema, [HumanMessage(content="all")], retry_config=NO_RETRY)
    # Fired once for the list, not also for the internal container.
    assert len(seen) == 1
    assert seen[0].schema_name == "SimpleSchema"
    assert isinstance(seen[0].result, list)
