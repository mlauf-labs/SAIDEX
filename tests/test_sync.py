"""Tests for the synchronous wrappers (extract_from_text_sync / get_structured_data_sync)."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel

from saidex import (
    Tool,
    extract_from_text_sync,
    extract_with_tools_sync,
    get_structured_data_sync,
    run_agent_loop_sync,
)
from saidex.retry import RetryConfig

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


def test_get_structured_data_sync_success() -> None:
    from langchain_core.messages import HumanMessage

    llm = _make_bound_llm([_make_llm_response({"name": "Alice", "value": 42})])
    messages = [HumanMessage(content="Extract data")]

    result, stats = get_structured_data_sync(
        llm, SimpleSchema, messages, retry_config=NO_NETWORK_RETRY
    )

    assert result is not None
    assert result.name == "Alice"
    assert result.value == 42
    assert stats.primary_retries == 0


def test_extract_from_text_sync_success() -> None:
    llm = _make_bound_llm([_make_llm_response({"name": "Bob", "value": 7})])

    result, stats = extract_from_text_sync(
        llm, SimpleSchema, "Bob is 7", retry_config=NO_NETWORK_RETRY
    )

    assert result is not None
    assert result.name == "Bob"
    assert result.value == 7


# ---------------------------------------------------------------------------
# Running-loop guard — must raise a clear error, not deadlock
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_structured_data_sync_raises_inside_running_loop() -> None:
    from langchain_core.messages import HumanMessage

    llm = _make_bound_llm([_make_llm_response({"name": "Alice", "value": 1})])
    messages = [HumanMessage(content="Extract data")]

    with pytest.raises(RuntimeError, match="running event loop"):
        get_structured_data_sync(llm, SimpleSchema, messages, retry_config=NO_NETWORK_RETRY)


@pytest.mark.asyncio
async def test_extract_from_text_sync_raises_inside_running_loop() -> None:
    llm = _make_bound_llm([_make_llm_response({"name": "Bob", "value": 2})])

    with pytest.raises(RuntimeError) as excinfo:
        extract_from_text_sync(llm, SimpleSchema, "Bob is 2", retry_config=NO_NETWORK_RETRY)

    # Error names the sync function and points to the async counterpart.
    assert "extract_from_text_sync" in str(excinfo.value)
    assert "await extract_from_text" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Agent-loop wrappers — success
# ---------------------------------------------------------------------------


def test_extract_with_tools_sync_success() -> None:
    llm = _make_bound_llm([_final_answer_response({"name": "Carol", "value": 5})])

    result, stats = extract_with_tools_sync(
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


def test_run_agent_loop_sync_success() -> None:
    from langchain_core.messages import HumanMessage

    llm = _make_bound_llm([_final_answer_response({"name": "Dave", "value": 9})])
    messages = [HumanMessage(content="Find Dave's record")]

    result, stats = run_agent_loop_sync(
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
async def test_extract_with_tools_sync_raises_inside_running_loop() -> None:
    llm = _make_bound_llm([_final_answer_response({"name": "Carol", "value": 5})])

    with pytest.raises(RuntimeError) as excinfo:
        extract_with_tools_sync(
            llm,
            SimpleSchema,
            "Find Carol's record",
            tools=[_make_tool()],
            retry_config=NO_NETWORK_RETRY,
        )

    assert "extract_with_tools_sync" in str(excinfo.value)
    assert "await extract_with_tools" in str(excinfo.value)


@pytest.mark.asyncio
async def test_run_agent_loop_sync_raises_inside_running_loop() -> None:
    from langchain_core.messages import HumanMessage

    llm = _make_bound_llm([_final_answer_response({"name": "Dave", "value": 9})])
    messages = [HumanMessage(content="Find Dave's record")]

    with pytest.raises(RuntimeError, match="running event loop"):
        run_agent_loop_sync(
            llm, SimpleSchema, messages, tools=[_make_tool()], retry_config=NO_NETWORK_RETRY
        )
