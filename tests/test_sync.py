"""Tests for the synchronous wrappers (extract_data_from_text_sync / extract_data_sync)."""

from __future__ import annotations

import threading
from contextvars import ContextVar
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel

from saidex import (
    Tool,
    extract_data_from_text_sync,
    extract_data_sync,
    extract_data_with_tools_sync,
    run_extractor_agent_sync,
)
from saidex.retry import RetryConfig
from saidex.sync import _run_blocking

# ---------------------------------------------------------------------------
# Helpers (mirror tests/test_extractor.py — no network, no API keys)
# ---------------------------------------------------------------------------


class SimpleSchema(BaseModel):
    name: str
    value: int


def _make_llm_response(args: dict[str, Any]) -> MagicMock:
    response = MagicMock()
    response.tool_calls = [{"args": args, "name": "SimpleSchema", "id": "call_1"}]
    response.invalid_tool_calls = []
    response.content = ""
    return response


def _make_bound_llm(responses: list[MagicMock]) -> MagicMock:
    bound = MagicMock()
    bound.ainvoke = AsyncMock(side_effect=responses)
    llm = MagicMock()
    llm.bind_tools = MagicMock(return_value=bound)
    return llm


NO_NETWORK_RETRY = RetryConfig(
    max_retries=0,
    retry_delays=[],
    retryable_exceptions=(),
    rate_limit_exceptions=(),
)


class _ToolArgs(BaseModel):
    query: str


async def _noop_handler(**kwargs: Any) -> dict[str, Any]:
    return {"status": "ok", **kwargs}


def _make_tool() -> Tool:
    return Tool(
        name="lookup",
        description="A test tool.",
        parameters=_ToolArgs,
        handler=_noop_handler,
    )


def _final_answer_response(args: dict[str, Any]) -> MagicMock:
    """A response where the LLM calls the schema as its final-answer tool."""
    response = MagicMock()
    response.tool_calls = [{"args": args, "name": "SimpleSchema", "id": "final_1"}]
    response.invalid_tool_calls = []
    response.content = ""
    return response


# ---------------------------------------------------------------------------
# Success — wrappers return the same result as the async functions
# ---------------------------------------------------------------------------


def test_extract_data_sync_success() -> None:
    from langchain_core.messages import HumanMessage

    llm = _make_bound_llm([_make_llm_response({"name": "Alice", "value": 42})])
    messages = [HumanMessage(content="Extract data")]

    result, stats = extract_data_sync(llm, SimpleSchema, messages, retry_config=NO_NETWORK_RETRY)

    assert result is not None
    assert result.name == "Alice"
    assert result.value == 42
    assert stats.primary_retries == 0


def test_extract_data_from_text_sync_success() -> None:
    llm = _make_bound_llm([_make_llm_response({"name": "Bob", "value": 7})])

    result, stats = extract_data_from_text_sync(
        llm, SimpleSchema, "Bob is 7", retry_config=NO_NETWORK_RETRY
    )

    assert result is not None
    assert result.name == "Bob"
    assert result.value == 7


# ---------------------------------------------------------------------------
# Running-loop guard — must raise a clear error, not deadlock
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_extract_data_sync_raises_inside_running_loop() -> None:
    from langchain_core.messages import HumanMessage

    llm = _make_bound_llm([_make_llm_response({"name": "Alice", "value": 1})])
    messages = [HumanMessage(content="Extract data")]

    with pytest.raises(RuntimeError, match="running event loop"):
        extract_data_sync(llm, SimpleSchema, messages, retry_config=NO_NETWORK_RETRY)


@pytest.mark.asyncio
async def test_extract_data_from_text_sync_raises_inside_running_loop() -> None:
    llm = _make_bound_llm([_make_llm_response({"name": "Bob", "value": 2})])

    with pytest.raises(RuntimeError) as excinfo:
        extract_data_from_text_sync(llm, SimpleSchema, "Bob is 2", retry_config=NO_NETWORK_RETRY)

    # Error names the sync function and points to the async counterpart.
    assert "extract_data_from_text_sync" in str(excinfo.value)
    assert "await extract_data_from_text" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Agent-loop wrappers — success
# ---------------------------------------------------------------------------


def test_extract_data_with_tools_sync_success() -> None:
    llm = _make_bound_llm([_final_answer_response({"name": "Carol", "value": 5})])

    result, stats = extract_data_with_tools_sync(
        llm,
        SimpleSchema,
        "Find Carol's record",
        tools=[_make_tool()],
        retry_config=NO_NETWORK_RETRY,
    )

    assert result is not None
    assert result.name == "Carol"
    assert result.value == 5
    assert stats.iterations == 1


def test_run_extractor_agent_sync_success() -> None:
    from langchain_core.messages import HumanMessage

    llm = _make_bound_llm([_final_answer_response({"name": "Dave", "value": 9})])
    messages = [HumanMessage(content="Find Dave's record")]

    result, stats = run_extractor_agent_sync(
        llm,
        SimpleSchema,
        messages,
        tools=[_make_tool()],
        retry_config=NO_NETWORK_RETRY,
    )

    assert result is not None
    assert result.name == "Dave"
    assert result.value == 9


# ---------------------------------------------------------------------------
# Agent-loop wrappers — running-loop guard
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_extract_data_with_tools_sync_raises_inside_running_loop() -> None:
    llm = _make_bound_llm([_final_answer_response({"name": "Carol", "value": 5})])

    with pytest.raises(RuntimeError) as excinfo:
        extract_data_with_tools_sync(
            llm,
            SimpleSchema,
            "Find Carol's record",
            tools=[_make_tool()],
            retry_config=NO_NETWORK_RETRY,
        )

    assert "extract_data_with_tools_sync" in str(excinfo.value)
    assert "await extract_data_with_tools" in str(excinfo.value)


@pytest.mark.asyncio
async def test_run_extractor_agent_sync_raises_inside_running_loop() -> None:
    from langchain_core.messages import HumanMessage

    llm = _make_bound_llm([_final_answer_response({"name": "Dave", "value": 9})])
    messages = [HumanMessage(content="Find Dave's record")]

    with pytest.raises(RuntimeError, match="running event loop"):
        run_extractor_agent_sync(
            llm, SimpleSchema, messages, tools=[_make_tool()], retry_config=NO_NETWORK_RETRY
        )


# ---------------------------------------------------------------------------
# _run_blocking — the bridge used by SaidexToolNode.invoke
# ---------------------------------------------------------------------------


def test_run_blocking_without_running_loop() -> None:
    async def double(x: int) -> int:
        return x * 2

    assert _run_blocking(double(21)) == 42


async def test_run_blocking_inside_running_loop_uses_worker_thread() -> None:
    async def which_thread() -> str:
        return threading.current_thread().name

    assert _run_blocking(which_thread()) == "saidex-sync-bridge"


async def test_run_blocking_reraises_inside_running_loop() -> None:
    async def boom() -> None:
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        _run_blocking(boom())


async def test_run_blocking_copies_context_into_worker() -> None:
    probe: ContextVar[str] = ContextVar("probe", default="unset")
    probe.set("caller")

    async def read() -> str:
        return probe.get()

    assert _run_blocking(read()) == "caller"
