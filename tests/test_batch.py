"""Tests for batch extraction (extract_data_list / extract_data_list_from_text)."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from saidex import (
    ExtractionMode,
    extract_data_list,
    extract_data_list_from_text,
)
from saidex.retry import RetryConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class InvoiceLine(BaseModel):
    description: str
    quantity: int
    price: float


def _make_tool_response(items: list[dict[str, Any]]) -> MagicMock:
    """Mock a tool-calling response that fills the dynamic ``items`` container."""
    response = MagicMock()
    response.tool_calls = [{"args": {"items": items}, "name": "InvoiceLineList", "id": "call_1"}]
    response.invalid_tool_calls = []
    response.content = ""
    return response


def _make_bound_llm(responses: list[MagicMock]) -> MagicMock:
    bound = MagicMock()
    bound.ainvoke = AsyncMock(side_effect=responses)
    llm = MagicMock()
    llm.bind_tools = MagicMock(return_value=bound)
    return llm


def _make_json_response(content: str) -> MagicMock:
    response = MagicMock()
    response.content = content
    response.tool_calls = []
    response.invalid_tool_calls = []
    return response


def _make_json_llm(contents: list[str]) -> MagicMock:
    llm = MagicMock()
    llm.ainvoke = AsyncMock(side_effect=[_make_json_response(c) for c in contents])
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

ITEMS = [
    {"description": "Widget", "quantity": 2, "price": 9.99},
    {"description": "Gadget", "quantity": 1, "price": 19.99},
]


# ---------------------------------------------------------------------------
# Tool-calling mode
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_returns_list_of_items() -> None:
    llm = _make_bound_llm([_make_tool_response(ITEMS)])
    messages = [HumanMessage(content="Extract the line items")]

    items, stats = await extract_data_list(
        llm, InvoiceLine, messages, retry_config=NO_NETWORK_RETRY
    )

    assert items is not None
    assert [i.description for i in items] == ["Widget", "Gadget"]
    assert all(isinstance(i, InvoiceLine) for i in items)
    assert stats.item_count == 2
    assert stats.primary_retries == 0


@pytest.mark.asyncio
async def test_empty_list_is_valid() -> None:
    llm = _make_bound_llm([_make_tool_response([])])
    messages = [HumanMessage(content="No items here")]

    items, stats = await extract_data_list(
        llm, InvoiceLine, messages, retry_config=NO_NETWORK_RETRY
    )

    assert items == []
    assert stats.item_count == 0


@pytest.mark.asyncio
async def test_per_item_validation_retry() -> None:
    bad = _make_tool_response(
        [
            {"description": "Widget", "quantity": "two", "price": 9.99},
            {"description": "Gadget", "quantity": 1, "price": 19.99},
        ]
    )
    good = _make_tool_response(ITEMS)
    llm = _make_bound_llm([bad, good])
    messages = [HumanMessage(content="Extract")]

    items, stats = await extract_data_list(
        llm, InvoiceLine, messages, retry_config=NO_NETWORK_RETRY
    )

    assert items is not None
    assert len(items) == 2
    assert stats.primary_retries == 1
    assert stats.item_count == 2


@pytest.mark.asyncio
async def test_per_item_error_feedback_points_to_offending_record() -> None:
    """A bad item should produce a field path that locates it (items -> 0 -> quantity)."""
    captured: list[Any] = []

    async def fake_invoke(msgs: Any, **_kwargs: Any) -> MagicMock:
        captured.clear()
        captured.extend(msgs)
        # First call is bad; subsequent calls succeed.
        if fake_invoke.calls == 0:  # type: ignore[attr-defined]
            fake_invoke.calls += 1  # type: ignore[attr-defined]
            return _make_tool_response(
                [{"description": "Widget", "quantity": "two", "price": 9.99}]
            )
        return _make_tool_response(ITEMS)

    fake_invoke.calls = 0  # type: ignore[attr-defined]
    bound = MagicMock()
    bound.ainvoke = fake_invoke
    llm = MagicMock()
    llm.bind_tools = MagicMock(return_value=bound)

    items, _ = await extract_data_list(
        llm, InvoiceLine, [HumanMessage(content="Extract")], retry_config=NO_NETWORK_RETRY
    )

    assert items is not None
    feedback = captured[-1]
    assert isinstance(feedback, HumanMessage)
    assert "items -> 0 -> quantity" in feedback.content


@pytest.mark.asyncio
async def test_fallback_used_when_primary_exhausted() -> None:
    bad = _make_tool_response([{"description": "X", "quantity": "bad", "price": 1.0}])
    primary = _make_bound_llm([bad, bad, bad])
    fallback = _make_bound_llm([_make_tool_response(ITEMS)])

    items, stats = await extract_data_list(
        primary,
        InvoiceLine,
        [HumanMessage(content="Extract")],
        fallback_llm_model=fallback,
        max_primary_retries=3,
        retry_config=NO_NETWORK_RETRY,
    )

    assert items is not None
    assert stats.fallback_used is True
    assert stats.item_count == 2


@pytest.mark.asyncio
async def test_returns_none_when_all_fail() -> None:
    bad = _make_tool_response([{"description": "X", "quantity": "bad", "price": 1.0}])
    primary = _make_bound_llm([bad, bad, bad])

    items, stats = await extract_data_list(
        primary,
        InvoiceLine,
        [HumanMessage(content="Extract")],
        max_primary_retries=3,
        retry_config=NO_NETWORK_RETRY,
    )

    assert items is None
    assert stats.item_count == 0
    assert stats.primary_retries == 3


# ---------------------------------------------------------------------------
# JSON mode
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_json_mode_returns_list() -> None:
    llm = _make_json_llm([json.dumps({"items": ITEMS})])

    items, stats = await extract_data_list(
        llm,
        InvoiceLine,
        [HumanMessage(content="Extract")],
        mode=ExtractionMode.JSON,
        retry_config=NO_NETWORK_RETRY,
    )

    assert items is not None
    assert len(items) == 2
    assert stats.item_count == 2
    llm.bind_tools.assert_not_called()


# ---------------------------------------------------------------------------
# extract_data_list_from_text
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_from_text_builds_messages_and_list_prompt() -> None:
    captured: list[Any] = []

    async def fake_invoke(msgs: Any, **_kwargs: Any) -> MagicMock:
        captured.clear()
        captured.extend(msgs)
        return _make_tool_response(ITEMS)

    bound = MagicMock()
    bound.ainvoke = fake_invoke
    llm = MagicMock()
    llm.bind_tools = MagicMock(return_value=bound)

    items, stats = await extract_data_list_from_text(
        llm, InvoiceLine, "2x Widget, 1x Gadget", retry_config=NO_NETWORK_RETRY
    )

    assert items is not None
    assert stats.item_count == 2
    assert isinstance(captured[0], SystemMessage)
    assert "every" in captured[0].content.lower()
    assert isinstance(captured[1], HumanMessage)
    assert captured[1].content == "2x Widget, 1x Gadget"


@pytest.mark.asyncio
async def test_from_text_custom_system_prompt() -> None:
    captured: list[Any] = []

    async def fake_invoke(msgs: Any, **_kwargs: Any) -> MagicMock:
        captured.clear()
        captured.extend(msgs)
        return _make_tool_response(ITEMS)

    bound = MagicMock()
    bound.ainvoke = fake_invoke
    llm = MagicMock()
    llm.bind_tools = MagicMock(return_value=bound)

    await extract_data_list_from_text(
        llm,
        InvoiceLine,
        "text",
        system_prompt="Custom instruction.",
        retry_config=NO_NETWORK_RETRY,
    )

    assert captured[0].content == "Custom instruction."
