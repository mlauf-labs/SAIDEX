"""Tests for run_extractor_agent / extract_data_with_tools (using mocks)."""

from __future__ import annotations

import json
from typing import Any

import pytest
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from saidex import (
    ExtractionMode,
    ExtractorRunStats,
    Tool,
    extract_data_with_tools,
    run_extractor_agent,
)
from saidex.retry import RetryConfig
from tests._mock_llm import make_llm as _make_llm
from tests._mock_llm import make_response as _make_response

# ---------------------------------------------------------------------------
# Helpers & Schemas
# ---------------------------------------------------------------------------

_NO_RETRY = RetryConfig(max_retries=0, retry_delays=[])


class FinalAnswer(BaseModel):
    result: str
    confidence: float = 1.0


class ToolArgs(BaseModel):
    name: str
    parent_id: str | None = Field(default=None)


def _tc(name: str, args: dict[str, Any], call_id: str = "call_1") -> dict[str, Any]:
    """Build a tool_call dict as LangChain returns them."""
    return {"name": name, "args": args, "id": call_id}


async def _noop_handler(**kwargs: Any) -> dict[str, Any]:
    return {"status": "ok", **kwargs}


def _make_tool(name: str = "do_thing") -> Tool:
    return Tool(
        name=name,
        description="A test tool.",
        parameters=ToolArgs,
        handler=_noop_handler,
    )


# ---------------------------------------------------------------------------
# Tool.to_openai_tool
# ---------------------------------------------------------------------------


def test_tool_to_openai_tool_structure() -> None:
    tool = _make_tool("create_item")
    spec = tool.to_openai_tool()
    assert spec["type"] == "function"
    fn = spec["function"]
    assert fn["name"] == "create_item"
    assert fn["description"] == "A test tool."
    assert "properties" in fn["parameters"] or fn["parameters"]  # schema present
    # title should be stripped from parameters
    assert "title" not in fn["parameters"]


@pytest.mark.asyncio
async def test_tool_execute_success() -> None:
    tool = _make_tool()
    result = await tool.execute({"name": "foo", "parent_id": None})
    data = json.loads(result)
    assert data["name"] == "foo"
    assert data["status"] == "ok"


@pytest.mark.asyncio
async def test_tool_execute_invalid_args() -> None:
    tool = _make_tool()
    # 'name' is required — missing
    result = await tool.execute({"parent_id": "x"})
    assert "Invalid arguments" in result


@pytest.mark.asyncio
async def test_tool_execute_handler_exception() -> None:
    async def _failing(**kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("boom")

    tool = Tool(name="boom_tool", description=".", parameters=ToolArgs, handler=_failing)
    result = await tool.execute({"name": "test"})
    assert "boom_tool" in result
    assert "boom" in result


# ---------------------------------------------------------------------------
# extract_data_with_tools — TOOL_CALLING mode
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_direct_final_answer_no_helper_calls() -> None:
    """LLM immediately calls the FinalAnswer tool without any helper tools."""
    final_call = _tc("FinalAnswer", {"result": "done", "confidence": 0.9})
    llm = _make_llm([_make_response(tool_calls=[final_call])])

    result, stats = await extract_data_with_tools(
        llm,
        FinalAnswer,
        "test input",
        tools=[],
        retry_config=_NO_RETRY,
    )

    assert result is not None
    assert result.result == "done"
    assert stats.iterations == 1
    assert stats.tool_calls == 0
    assert stats.validation_retries == 0
    assert not stats.fallback_used


@pytest.mark.asyncio
async def test_helper_tool_call_then_final_answer() -> None:
    """LLM calls a helper tool once, then produces a final answer."""
    tool_call = _tc("do_thing", {"name": "root", "parent_id": None}, "call_1")
    final_call = _tc("FinalAnswer", {"result": "placed"}, "call_2")
    llm = _make_llm(
        [
            _make_response(tool_calls=[tool_call]),
            _make_response(tool_calls=[final_call]),
        ]
    )

    calls: list[str] = []

    async def _handler(name: str, parent_id: str | None = None) -> dict[str, Any]:
        calls.append(name)
        return {"folder_id": "abc", "path": name}

    tool = Tool(name="do_thing", description="d", parameters=ToolArgs, handler=_handler)

    result, stats = await extract_data_with_tools(
        llm,
        FinalAnswer,
        "input",
        tools=[tool],
        retry_config=_NO_RETRY,
    )

    assert result is not None
    assert result.result == "placed"
    assert stats.iterations == 2
    assert stats.tool_calls == 1
    assert calls == ["root"]


@pytest.mark.asyncio
async def test_multiple_helper_calls_before_final() -> None:
    """Three consecutive helper tool calls, then final answer."""
    helper_calls = [_tc("do_thing", {"name": f"f{i}"}, f"c{i}") for i in range(3)]
    final_call = _tc("FinalAnswer", {"result": "ok"}, "cf")

    responses = [
        _make_response(tool_calls=[helper_calls[0]]),
        _make_response(tool_calls=[helper_calls[1]]),
        _make_response(tool_calls=[helper_calls[2]]),
        _make_response(tool_calls=[final_call]),
    ]
    llm = _make_llm(responses)

    result, stats = await extract_data_with_tools(
        llm, FinalAnswer, "text", tools=[_make_tool()], retry_config=_NO_RETRY
    )

    assert result is not None
    assert stats.tool_calls == 3
    assert stats.iterations == 4


@pytest.mark.asyncio
async def test_final_answer_validation_retry() -> None:
    """Bad final answer on first attempt, corrected on second."""
    # Invalid payload: missing required field
    really_bad_call = _tc("FinalAnswer", {}, "c1")
    good_call = _tc("FinalAnswer", {"result": "fixed"}, "c2")
    llm = _make_llm(
        [
            _make_response(tool_calls=[really_bad_call]),
            _make_response(tool_calls=[good_call]),
        ]
    )

    result, stats = await extract_data_with_tools(
        llm, FinalAnswer, "text", tools=[], retry_config=_NO_RETRY
    )
    assert result is not None
    assert result.result == "fixed"
    assert stats.validation_retries == 1


@pytest.mark.asyncio
async def test_max_iterations_exhausted_returns_none() -> None:
    """Loop aborts when max_iterations is reached without a valid final answer."""
    tool_call = _tc("do_thing", {"name": "x"}, "c1")
    # Always return a helper tool call — never a final answer.
    llm = _make_llm([_make_response(tool_calls=[tool_call])] * 20)

    result, stats = await extract_data_with_tools(
        llm, FinalAnswer, "text", tools=[_make_tool()], max_iterations=3, retry_config=_NO_RETRY
    )
    assert result is None
    assert stats.iterations == 3


@pytest.mark.asyncio
async def test_invalid_tool_call_triggers_feedback() -> None:
    """An invalid (malformed) tool call should be met with feedback, then corrected."""
    good_call = _tc("FinalAnswer", {"result": "recovered"}, "c2")
    llm = _make_llm(
        [
            _make_response(invalid=True),
            _make_response(tool_calls=[good_call]),
        ]
    )

    result, stats = await extract_data_with_tools(
        llm, FinalAnswer, "text", tools=[], retry_config=_NO_RETRY
    )
    assert result is not None
    assert result.result == "recovered"


@pytest.mark.asyncio
async def test_unknown_tool_call_returns_error_message() -> None:
    """Calling a tool that doesn't exist returns an error ToolMessage."""
    unknown_call = _tc("nonexistent_tool", {"name": "x"}, "c1")
    final_call = _tc("FinalAnswer", {"result": "ok"}, "c2")
    llm = _make_llm(
        [
            _make_response(tool_calls=[unknown_call]),
            _make_response(tool_calls=[final_call]),
        ]
    )

    result, stats = await extract_data_with_tools(
        llm, FinalAnswer, "text", tools=[], retry_config=_NO_RETRY
    )
    # The LLM gets an "Unknown tool" message but can still recover
    assert result is not None
    # unknown tool attempts are still counted in tool_calls
    assert stats.tool_calls == 1


# ---------------------------------------------------------------------------
# extract_data_with_tools — JSON mode
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_json_mode_no_tool_calls_parses_content() -> None:
    """JSON mode: LLM emits text with no tool calls → parsed as final answer."""
    payload = json.dumps({"result": "json_answer", "confidence": 0.8})
    llm = _make_llm([_make_response(content=payload)])

    result, stats = await extract_data_with_tools(
        llm,
        FinalAnswer,
        "input",
        tools=[],
        final_answer_mode=ExtractionMode.JSON,
        retry_config=_NO_RETRY,
    )

    assert result is not None
    assert result.result == "json_answer"
    assert stats.tool_calls == 0


@pytest.mark.asyncio
async def test_json_mode_helper_tool_then_text() -> None:
    """JSON mode: helper tool called first, then LLM emits JSON text."""
    tool_call = _tc("do_thing", {"name": "root"}, "c1")
    payload = json.dumps({"result": "after_tool"})
    llm = _make_llm(
        [
            _make_response(tool_calls=[tool_call]),
            _make_response(content=payload),
        ]
    )

    result, stats = await extract_data_with_tools(
        llm,
        FinalAnswer,
        "input",
        tools=[_make_tool()],
        final_answer_mode=ExtractionMode.JSON,
        retry_config=_NO_RETRY,
    )

    assert result is not None
    assert result.result == "after_tool"
    assert stats.tool_calls == 1


# ---------------------------------------------------------------------------
# Fallback model
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fallback_model_used_when_primary_exhausted() -> None:
    """Fallback model is tried when primary exhausts max_iterations."""
    final_call = _tc("FinalAnswer", {"result": "fallback_result"}, "cf")
    primary_llm = _make_llm([_make_response(tool_calls=[])] * 3)  # no tool calls, no final
    fallback_llm = _make_llm([_make_response(tool_calls=[final_call])])

    result, stats = await extract_data_with_tools(
        primary_llm,
        FinalAnswer,
        "text",
        tools=[],
        fallback_llm_model=fallback_llm,
        max_iterations=2,
        max_validation_retries=2,
        retry_config=_NO_RETRY,
    )

    assert result is not None
    assert result.result == "fallback_result"
    assert stats.fallback_used


# ---------------------------------------------------------------------------
# run_extractor_agent — message list API
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_extractor_agent_with_custom_messages() -> None:
    """run_extractor_agent accepts a pre-built message list."""
    final_call = _tc("FinalAnswer", {"result": "from_loop"}, "c1")
    llm = _make_llm([_make_response(tool_calls=[final_call])])

    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="Do the thing."),
    ]

    result, stats = await run_extractor_agent(
        llm,
        FinalAnswer,
        messages,
        tools=[],
        retry_config=_NO_RETRY,
    )

    assert result is not None
    assert result.result == "from_loop"


# ---------------------------------------------------------------------------
# ExtractorRunStats
# ---------------------------------------------------------------------------


def test_agent_run_stats_add() -> None:
    a = ExtractorRunStats(iterations=3, tool_calls=2, validation_retries=1, fallback_used=False)
    b = ExtractorRunStats(iterations=2, tool_calls=1, validation_retries=0, fallback_used=True)
    c = a + b
    assert c.iterations == 5
    assert c.tool_calls == 3
    assert c.validation_retries == 1
    assert c.fallback_used is True


# ---------------------------------------------------------------------------
# Tool._invoke
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tool_invoke_success_has_no_error() -> None:
    tool = _make_tool()
    outcome = await tool._invoke({"name": "foo", "parent_id": None})
    assert outcome.handler_error is None
    data = json.loads(outcome.content)
    assert data["name"] == "foo"


@pytest.mark.asyncio
async def test_tool_invoke_invalid_args_is_not_a_handler_error() -> None:
    tool = _make_tool()
    outcome = await tool._invoke({"parent_id": "x"})  # 'name' missing
    assert outcome.handler_error is None
    assert "Invalid arguments" in outcome.content


@pytest.mark.asyncio
async def test_tool_invoke_handler_exception_is_captured() -> None:
    async def _failing(**kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("boom")

    tool = Tool(name="boom_tool", description=".", parameters=ToolArgs, handler=_failing)
    outcome = await tool._invoke({"name": "x"})
    assert isinstance(outcome.handler_error, RuntimeError)
    assert "boom" in outcome.content


@pytest.mark.asyncio
async def test_tool_invoke_non_mapping_args_returns_invalid() -> None:
    """Double-encoded JSON args arrive as a str — must not crash the loop."""
    tool = _make_tool()
    outcome = await tool._invoke("not a dict")  # type: ignore[arg-type]
    assert outcome.handler_error is None
    assert "Invalid arguments" in outcome.content
    assert "expected a JSON object" in outcome.content


@pytest.mark.asyncio
async def test_tool_invoke_reserved_key_collision_returns_invalid() -> None:
    """An arg named 'schema' collides with create_instance_safe's own parameter."""
    tool = _make_tool()
    outcome = await tool._invoke({"schema": "x", "name": "foo"})
    assert outcome.handler_error is None
    assert "Invalid arguments" in outcome.content
