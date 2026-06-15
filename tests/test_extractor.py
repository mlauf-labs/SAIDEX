"""Tests for extract_data and extract_data_from_text (using mocks)."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from saidex import ExtractionMode, extract_data, extract_data_from_text
from saidex.retry import RetryConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class SimpleSchema(BaseModel):
    name: str
    value: int


def _make_llm_response(args: dict[str, Any] | None = None, invalid: bool = False) -> MagicMock:
    """Create a mock LLM response."""
    response = MagicMock()
    if args is not None:
        response.tool_calls = [{"args": args, "name": "SimpleSchema", "id": "call_1"}]
        response.invalid_tool_calls = []
    elif invalid:
        response.tool_calls = []
        response.invalid_tool_calls = [{"error": "bad json", "args": "{bad"}]
    else:
        response.tool_calls = []
        response.invalid_tool_calls = []
    response.content = ""
    return response


def _make_bound_llm(responses: list[MagicMock]) -> MagicMock:
    """Return a mock LLM whose .bind_tools() returns a model yielding *responses* in order."""
    bound = MagicMock()
    bound.ainvoke = AsyncMock(side_effect=responses)
    llm = MagicMock()
    llm.bind_tools = MagicMock(return_value=bound)
    return llm


def _make_json_response(content: str) -> MagicMock:
    """Create a mock LLM response that returns raw text content (JSON mode)."""
    response = MagicMock()
    response.content = content
    # No tool calls in JSON mode.
    response.tool_calls = []
    response.invalid_tool_calls = []
    return response


def _make_json_llm(contents: list[str]) -> MagicMock:
    """Return a mock LLM whose .ainvoke() yields *contents* as response content.

    The mock raises if ``bind_tools`` is ever called, asserting that JSON mode
    never binds tools.
    """
    responses = [_make_json_response(c) for c in contents]
    llm = MagicMock()
    llm.ainvoke = AsyncMock(side_effect=responses)
    llm.bind_tools = MagicMock(
        side_effect=AssertionError("bind_tools must not be called in JSON mode")
    )
    return llm


NO_NETWORK_RETRY = RetryConfig(
    max_retries=0,
    retry_delays=[],
    retryable_exceptions=(),
    rate_limit_exceptions=(),
)


# ---------------------------------------------------------------------------
# run_structured_output — success on first attempt
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_success_first_attempt() -> None:
    llm = _make_bound_llm([_make_llm_response({"name": "Alice", "value": 42})])
    messages = [HumanMessage(content="Extract data")]

    result, stats = await extract_data(llm, SimpleSchema, messages, retry_config=NO_NETWORK_RETRY)

    assert result is not None
    assert result.name == "Alice"
    assert result.value == 42
    assert stats.primary_retries == 0
    assert not stats.fallback_used


# ---------------------------------------------------------------------------
# run_structured_output — success after one validation retry
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_success_after_validation_retry() -> None:
    bad_response = _make_llm_response({"name": "Alice", "value": "not_an_int"})
    good_response = _make_llm_response({"name": "Alice", "value": 42})

    llm = _make_bound_llm([bad_response, good_response])
    messages = [HumanMessage(content="Extract data")]

    result, stats = await extract_data(llm, SimpleSchema, messages, retry_config=NO_NETWORK_RETRY)

    assert result is not None
    assert result.value == 42
    assert stats.primary_retries == 1


# ---------------------------------------------------------------------------
# run_structured_output — exhausts primary, uses fallback
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fallback_used_when_primary_exhausted() -> None:
    bad = _make_llm_response({"name": "Alice", "value": "bad"})
    primary_llm = _make_bound_llm([bad, bad, bad])

    good = _make_llm_response({"name": "Alice", "value": 99})
    fallback_llm = _make_bound_llm([good])

    messages = [HumanMessage(content="Extract")]

    result, stats = await extract_data(
        primary_llm,
        SimpleSchema,
        messages,
        fallback_llm_model=fallback_llm,
        max_primary_retries=3,
        retry_config=NO_NETWORK_RETRY,
    )

    assert result is not None
    assert result.value == 99
    assert stats.fallback_used is True
    assert stats.primary_retries == 3
    assert stats.fallback_retries == 0


# ---------------------------------------------------------------------------
# run_structured_output — all models fail
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_returns_none_when_all_fail() -> None:
    bad = _make_llm_response({"name": "Alice", "value": "bad"})
    primary_llm = _make_bound_llm([bad, bad, bad])
    fallback_llm = _make_bound_llm([bad, bad, bad])

    messages = [HumanMessage(content="Extract")]

    result, stats = await extract_data(
        primary_llm,
        SimpleSchema,
        messages,
        fallback_llm_model=fallback_llm,
        max_primary_retries=3,
        max_fallback_retries=3,
        retry_config=NO_NETWORK_RETRY,
    )

    assert result is None
    assert stats.fallback_used is True
    assert stats.total_retries == 6


# ---------------------------------------------------------------------------
# run_structured_output — no tool calls in response
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_tool_calls_triggers_retry() -> None:
    no_call = _make_llm_response()
    good = _make_llm_response({"name": "Bob", "value": 7})

    llm = _make_bound_llm([no_call, good])
    messages = [HumanMessage(content="Extract")]

    result, stats = await extract_data(llm, SimpleSchema, messages, retry_config=NO_NETWORK_RETRY)

    assert result is not None
    assert stats.primary_retries == 1


# ---------------------------------------------------------------------------
# extract_data_from_text — builds correct message list
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_extract_data_from_text_builds_messages() -> None:
    captured_messages: list[Any] = []

    async def fake_invoke(msgs: Any, **_kwargs: Any) -> MagicMock:
        captured_messages.extend(msgs)
        return _make_llm_response({"name": "Test", "value": 1})

    bound = MagicMock()
    bound.ainvoke = fake_invoke
    llm = MagicMock()
    llm.bind_tools = MagicMock(return_value=bound)

    result, _ = await extract_data_from_text(
        llm,
        SimpleSchema,
        "Some text",
        system_prompt="Be precise.",
        retry_config=NO_NETWORK_RETRY,
    )

    assert result is not None
    assert isinstance(captured_messages[0], SystemMessage)
    assert captured_messages[0].content == "Be precise."
    assert isinstance(captured_messages[1], HumanMessage)
    assert captured_messages[1].content == "Some text"


@pytest.mark.asyncio
async def test_extract_data_from_text_default_system_prompt() -> None:
    captured_messages: list[Any] = []

    async def fake_invoke(msgs: Any, **_kwargs: Any) -> MagicMock:
        captured_messages.extend(msgs)
        return _make_llm_response({"name": "Test", "value": 1})

    bound = MagicMock()
    bound.ainvoke = fake_invoke
    llm = MagicMock()
    llm.bind_tools = MagicMock(return_value=bound)

    await extract_data_from_text(llm, SimpleSchema, "text", retry_config=NO_NETWORK_RETRY)

    assert isinstance(captured_messages[0], SystemMessage)
    assert "SimpleSchema" in captured_messages[0].content


# ---------------------------------------------------------------------------
# JSON mode — success on first attempt (plain JSON)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_json_mode_success_plain() -> None:
    llm = _make_json_llm(['{"name": "Alice", "value": 42}'])
    messages = [HumanMessage(content="Extract data")]

    result, stats = await extract_data(
        llm,
        SimpleSchema,
        messages,
        mode=ExtractionMode.JSON,
        retry_config=NO_NETWORK_RETRY,
    )

    assert result is not None
    assert result.name == "Alice"
    assert result.value == 42
    assert stats.primary_retries == 0
    # bind_tools must never be called in JSON mode.
    llm.bind_tools.assert_not_called()


# ---------------------------------------------------------------------------
# JSON mode — strips markdown code fences
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_json_mode_strips_code_fences() -> None:
    fenced = '```json\n{"name": "Bob", "value": 7}\n```'
    llm = _make_json_llm([fenced])
    messages = [HumanMessage(content="Extract data")]

    result, stats = await extract_data(
        llm,
        SimpleSchema,
        messages,
        mode=ExtractionMode.JSON,
        retry_config=NO_NETWORK_RETRY,
    )

    assert result is not None
    assert result.name == "Bob"
    assert result.value == 7
    assert stats.primary_retries == 0


# ---------------------------------------------------------------------------
# JSON mode — isolates JSON surrounded by stray text
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_json_mode_isolates_object_from_text() -> None:
    noisy = 'Sure! Here is the data:\n{"name": "Carol", "value": 3}\nHope that helps.'
    llm = _make_json_llm([noisy])
    messages = [HumanMessage(content="Extract data")]

    result, _ = await extract_data(
        llm,
        SimpleSchema,
        messages,
        mode=ExtractionMode.JSON,
        retry_config=NO_NETWORK_RETRY,
    )

    assert result is not None
    assert result.name == "Carol"
    assert result.value == 3


# ---------------------------------------------------------------------------
# JSON mode — schema is injected into the prompt
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_json_mode_injects_schema_instructions() -> None:
    captured_messages: list[Any] = []

    async def fake_invoke(msgs: Any, **_kwargs: Any) -> MagicMock:
        captured_messages.clear()
        captured_messages.extend(msgs)
        return _make_json_response('{"name": "Dana", "value": 1}')

    llm = MagicMock()
    llm.ainvoke = fake_invoke
    llm.bind_tools = MagicMock(side_effect=AssertionError("must not bind tools"))

    messages = [HumanMessage(content="Extract data")]
    result, _ = await extract_data(
        llm,
        SimpleSchema,
        messages,
        mode=ExtractionMode.JSON,
        retry_config=NO_NETWORK_RETRY,
    )

    assert result is not None
    # The last message should be the injected JSON-schema instruction.
    last = captured_messages[-1]
    assert isinstance(last, HumanMessage)
    assert "JSON Schema" in last.content
    assert "SimpleSchema" in last.content


# ---------------------------------------------------------------------------
# JSON mode — invalid JSON triggers a retry, then succeeds
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_json_mode_retries_on_unparseable_then_succeeds() -> None:
    llm = _make_json_llm(["this is not json at all", '{"name": "Eve", "value": 5}'])
    messages = [HumanMessage(content="Extract data")]

    result, stats = await extract_data(
        llm,
        SimpleSchema,
        messages,
        mode=ExtractionMode.JSON,
        retry_config=NO_NETWORK_RETRY,
    )

    assert result is not None
    assert result.value == 5
    assert stats.primary_retries == 1


# ---------------------------------------------------------------------------
# JSON mode — validation error feeds back, then succeeds
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_json_mode_validation_retry() -> None:
    llm = _make_json_llm(
        ['{"name": "Frank", "value": "not_an_int"}', '{"name": "Frank", "value": 11}']
    )
    messages = [HumanMessage(content="Extract data")]

    result, stats = await extract_data(
        llm,
        SimpleSchema,
        messages,
        mode=ExtractionMode.JSON,
        retry_config=NO_NETWORK_RETRY,
    )

    assert result is not None
    assert result.value == 11
    assert stats.primary_retries == 1


# ---------------------------------------------------------------------------
# JSON mode — fallback model takes over
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_json_mode_fallback() -> None:
    primary = _make_json_llm(["nope", "still nope", "nope again"])
    fallback = _make_json_llm(['{"name": "Grace", "value": 8}'])

    messages = [HumanMessage(content="Extract")]
    result, stats = await extract_data(
        primary,
        SimpleSchema,
        messages,
        mode=ExtractionMode.JSON,
        fallback_llm_model=fallback,
        max_primary_retries=3,
        retry_config=NO_NETWORK_RETRY,
    )

    assert result is not None
    assert result.value == 8
    assert stats.fallback_used is True
    assert stats.primary_retries == 3


# ---------------------------------------------------------------------------
# JSON mode — extract_data_from_text end to end
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_json_mode_extract_data_from_text() -> None:
    llm = _make_json_llm(['{"name": "Heidi", "value": 21}'])

    result, _ = await extract_data_from_text(
        llm,
        SimpleSchema,
        "Heidi is 21",
        mode=ExtractionMode.JSON,
        retry_config=NO_NETWORK_RETRY,
    )

    assert result is not None
    assert result.name == "Heidi"
    assert result.value == 21
    llm.bind_tools.assert_not_called()
