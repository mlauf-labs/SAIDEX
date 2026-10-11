"""Tests for saidex._tool_calls, the framework-agnostic tool-call core (no langgraph needed)."""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from saidex._tool_calls import (
    CallContext,
    OnUnresolved,
    ToolCallPipeline,
    ToolValidator,
    _rewrite_additional_kwargs,
    _rewrite_content,
    build_correction_prompt,
    render_conversation,
)
from saidex.models import ExtractionMode, ToolCallValidatorContext
from tests._mock_llm import make_llm, make_response


def test_core_module_does_not_import_langgraph() -> None:
    code = "import sys, saidex._tool_calls; print('langgraph' in sys.modules)"
    completed = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert completed.stdout.strip() == "False"


def _history() -> list[Any]:
    return [
        SystemMessage(content="secret system prompt"),
        HumanMessage(content="What is 5 + 2?"),
        AIMessage(
            content="",
            tool_calls=[{"name": "add", "args": {"a": 1, "b": 1}, "id": "p1", "type": "tool_call"}],
        ),
        ToolMessage(content="2", name="add", tool_call_id="p1"),
        AIMessage(content="Let me add five and two."),
    ]


def test_render_conversation_selects_and_formats_messages() -> None:
    rendered = render_conversation(_history(), ai_index=4, limit=10)
    assert rendered is not None
    assert "secret system prompt" not in rendered
    assert rendered.splitlines() == [
        "[user] What is 5 + 2?",
        "[assistant] ",
        '  -> add({"a": 1, "b": 1})',
        "[tool result: add] 2",
        "[assistant] Let me add five and two.",
    ]


def test_render_conversation_keeps_only_the_last_n_messages() -> None:
    rendered = render_conversation(_history(), ai_index=4, limit=1)
    assert rendered == "[tool result: add] 2\n[assistant] Let me add five and two."


def test_render_conversation_disabled_with_zero() -> None:
    assert render_conversation(_history(), ai_index=4, limit=0) is None


def test_render_conversation_without_preceding_messages() -> None:
    assert render_conversation([AIMessage(content="")], ai_index=0, limit=10) is None


def test_render_conversation_truncates_long_messages() -> None:
    history = [HumanMessage(content="x" * 5000), AIMessage(content="")]
    rendered = render_conversation(history, ai_index=1, limit=10)
    assert rendered is not None
    assert len(rendered) == 2000
    assert rendered.endswith("…")


def test_render_conversation_uses_placeholders_for_non_text_blocks() -> None:
    history = [
        HumanMessage(content=[{"type": "text", "text": "Look:"}, {"type": "image_url"}]),
        AIMessage(content=""),
    ]
    assert render_conversation(history, ai_index=1, limit=10) == "[user] Look: [image_url]"


def test_build_correction_prompt_contains_every_part() -> None:
    prompt = build_correction_prompt(
        tool_name="add",
        description="Add two integers.",
        arguments='{"a": "five"}',
        error_text="a: not an int",
        conversation="[user] add five",
    )
    assert "invalid call to the tool 'add'" in prompt
    assert "Tool description:\nAdd two integers." in prompt
    assert 'Invalid arguments:\n{"a": "five"}' in prompt
    assert "Validation errors:\na: not an int" in prompt
    assert "Conversation so far (oldest first):\n[user] add five" in prompt
    assert "never invent missing values" in prompt


def test_build_correction_prompt_omits_empty_parts() -> None:
    prompt = build_correction_prompt(
        tool_name="add", description="", arguments="{}", error_text=None, conversation=None
    )
    assert "Tool description" not in prompt
    assert "Validation errors" not in prompt
    assert "Conversation so far" not in prompt


_FIXED = {"c1": {"name": "add", "args": {"a": 2, "b": 3}, "id": "c1", "type": "tool_call"}}


def test_rewrite_content_tool_use_block_drops_partial_json() -> None:
    content = [
        {"type": "text", "text": "Adding."},
        {"type": "tool_use", "id": "c1", "name": "add", "input": {}, "partial_json": '{"a": 2'},
    ]
    assert _rewrite_content(content, _FIXED) == [
        {"type": "text", "text": "Adding."},
        {"type": "tool_use", "id": "c1", "name": "add", "input": {"a": 2, "b": 3}},
    ]


def test_rewrite_content_tool_call_block() -> None:
    content = [{"type": "tool_call", "id": "c1", "name": "add", "args": {"a": "x"}}]
    assert _rewrite_content(content, _FIXED) == [
        {"type": "tool_call", "id": "c1", "name": "add", "args": {"a": 2, "b": 3}}
    ]


def test_rewrite_content_invalid_tool_call_becomes_tool_call() -> None:
    content = [
        {"type": "invalid_tool_call", "id": "c1", "name": "add", "args": "{bad", "error": "x"}
    ]
    assert _rewrite_content(content, _FIXED) == [
        {"type": "tool_call", "id": "c1", "name": "add", "args": {"a": 2, "b": 3}}
    ]


def test_rewrite_content_function_call_block_by_call_id() -> None:
    content = [{"type": "function_call", "id": "fc_9", "call_id": "c1", "arguments": "{bad"}]
    assert _rewrite_content(content, _FIXED) == [
        {"type": "function_call", "id": "fc_9", "call_id": "c1", "arguments": '{"a": 2, "b": 3}'}
    ]


def test_rewrite_content_function_call_block_with_unhashable_call_id_uses_id() -> None:
    content = [{"type": "function_call", "id": "c1", "call_id": ["odd"], "arguments": "{bad"}]
    assert _rewrite_content(content, _FIXED) == [
        {"type": "function_call", "id": "c1", "call_id": ["odd"], "arguments": '{"a": 2, "b": 3}'}
    ]


def test_rewrite_content_leaves_other_blocks_and_ids_alone() -> None:
    content = [
        {"type": "tool_use", "id": "other", "name": "add", "input": {"a": 0}},
        {"type": "image_url", "image_url": {"url": "x"}},
    ]
    assert _rewrite_content(content, _FIXED) is content
    assert _rewrite_content("plain text", _FIXED) == "plain text"


def test_rewrite_additional_kwargs_raw_openai_tool_calls() -> None:
    kwargs = {
        "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "add", "arguments": "{bad"}},
            {"id": "c2", "type": "function", "function": {"name": "add", "arguments": "{}"}},
        ],
        "refusal": None,
    }
    rewritten = _rewrite_additional_kwargs(kwargs, _FIXED)
    assert rewritten["tool_calls"][0]["function"]["arguments"] == '{"a": 2, "b": 3}'
    assert rewritten["tool_calls"][1] == kwargs["tool_calls"][1]
    assert rewritten["refusal"] is None
    assert _rewrite_additional_kwargs({"refusal": None}, _FIXED) == {"refusal": None}


def test_rewrite_handles_non_json_args() -> None:
    when = datetime(2026, 10, 11, 12, 0)
    fixed = {"c1": {"name": "book", "args": {"at": when}, "id": "c1", "type": "tool_call"}}
    content = [{"type": "function_call", "id": "fc_1", "call_id": "c1", "arguments": "{bad"}]
    kwargs = {
        "tool_calls": [
            {
                "id": "c1",
                "type": "function",
                "function": {"name": "book", "arguments": "{bad"},
            }
        ]
    }
    rewritten_content = _rewrite_content(content, fixed)
    rewritten_kwargs = _rewrite_additional_kwargs(kwargs, fixed)
    assert (
        rewritten_content[0]["arguments"]  # type: ignore[index]
        == '{"at": "2026-10-11 12:00:00"}'
    )
    assert (
        rewritten_kwargs["tool_calls"][0]["function"]["arguments"]
        == '{"at": "2026-10-11 12:00:00"}'
    )


# ---------------------------------------------------------------------------
# Passthrough exceptions (LangGraph's GraphBubbleUp in the adapter)
# ---------------------------------------------------------------------------


class _Bubble(Exception):  # noqa: N818 — named after GraphBubbleUp
    """Stands in for LangGraph's ``GraphBubbleUp``; the core never imports langgraph."""


@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


def _bubble_in(phase: str) -> ToolValidator:
    def validator(args: Any, ctx: ToolCallValidatorContext) -> None:
        if ctx.phase == phase:
            raise _Bubble(f"pause in {phase}")

    return validator


def _pipeline(
    validator: ToolValidator,
    *,
    passthrough: tuple[type[BaseException], ...] = (_Bubble,),
    correction_model: Any = None,
    on_unresolved: OnUnresolved = "feedback",
) -> ToolCallPipeline:
    return ToolCallPipeline(
        [add],
        correction_model=correction_model,
        max_correction_retries=1,
        correction_mode=ExtractionMode.TOOL_CALLING,
        strip_thinking=True,
        json_repair=True,
        on_unresolved=on_unresolved,
        validators={"add": validator},
        passthrough_exceptions=passthrough,
    )


def _planned(*args: dict[str, Any]) -> tuple[AIMessage, CallContext]:
    calls = [
        {"name": "add", "args": call_args, "id": f"c{index}", "type": "tool_call"}
        for index, call_args in enumerate(args, start=1)
    ]
    message = AIMessage(content="", tool_calls=calls)
    return message, CallContext(messages=(message,), ai_index=0, state={})


@pytest.mark.asyncio
async def test_passthrough_exception_escapes_initial_validation() -> None:
    message, ctx = _planned({"a": 1, "b": 2})
    with pytest.raises(_Bubble, match="pause in initial"):
        await _pipeline(_bubble_in("initial")).plan(message, ctx)


@pytest.mark.asyncio
async def test_passthrough_exception_escapes_async_validator() -> None:
    async def bubble(args: Any, ctx: ToolCallValidatorContext) -> None:
        raise _Bubble("async pause")

    message, ctx = _planned({"a": 1, "b": 2})
    with pytest.raises(_Bubble, match="async pause"):
        await _pipeline(bubble).plan(message, ctx)


@pytest.mark.asyncio
async def test_validator_exception_without_passthrough_is_a_rejection() -> None:
    message, ctx = _planned({"a": 1, "b": 2})
    plans = await _pipeline(_bubble_in("initial"), passthrough=()).plan(message, ctx)
    assert plans[0].rejected_by_validator
    assert "pause in initial" in (plans[0].error_text or "")


@pytest.mark.asyncio
@pytest.mark.parametrize("max_concurrency", [None, 1])
async def test_passthrough_exception_escapes_correction_cycle(max_concurrency: int | None) -> None:
    correction = make_llm(
        [make_response(tool_calls=[{"args": {"a": 1, "b": 2}, "name": "add", "id": "x"}])]
    )
    pipeline = _pipeline(_bubble_in("correction"), correction_model=correction)
    message, ctx = _planned({"a": "one", "b": 2})
    plans = await pipeline.plan(message, ctx)
    with pytest.raises(_Bubble, match="pause in correction"):
        await pipeline.correct(plans, ctx, callbacks=None, max_concurrency=max_concurrency)


@pytest.mark.asyncio
async def test_passthrough_exception_escapes_salvage_validation() -> None:
    pipeline = _pipeline(_bubble_in("salvage"), on_unresolved=lambda call: {"a": 1, "b": 2})
    message, ctx = _planned({"a": "one", "b": 2})
    plans = await pipeline.plan(message, ctx)
    with pytest.raises(_Bubble, match="pause in salvage"):
        await pipeline.resolve(plans, ctx)


@pytest.mark.asyncio
async def test_passthrough_exception_cancels_waiting_corrections() -> None:
    response = make_response(tool_calls=[{"args": {"a": 1, "b": 2}, "name": "add", "id": "x"}])
    correction = make_llm([response, response])
    pipeline = _pipeline(_bubble_in("correction"), correction_model=correction)
    message, ctx = _planned({"a": "one", "b": 2}, {"a": "two", "b": 2})
    plans = await pipeline.plan(message, ctx)
    with pytest.raises(_Bubble):
        await pipeline.correct(plans, ctx, callbacks=None, max_concurrency=1)
    assert correction.bind_tools.return_value.ainvoke.await_count == 1
