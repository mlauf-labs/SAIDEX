"""Integration tests: source grounding across the extraction entry points.

Uses the mock-LLM pattern (no API keys, no network). The source text is the
human-turn message content; a hallucinated value must be rejected and corrected
via the existing retry loop.
"""

from __future__ import annotations

from typing import Annotated, Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from saidex import (
    ExtractionMode,
    Grounded,
    GroundedField,
    extract_data,
    extract_data_list,
    extract_data_sync,
    extract_data_with_tools,
)
from saidex.retry import RetryConfig

_NO_NETWORK_RETRY = RetryConfig(
    max_retries=0, retry_delays=[], retryable_exceptions=(), rate_limit_exceptions=()
)

SOURCE = "Invoice issued by ACME GmbH, total 1.234,50 EUR, dated 05.04.2024."


class Vendor(BaseModel):
    name: Annotated[str, Grounded()]


def _tool_response(args: dict[str, Any], name: str) -> MagicMock:
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


def _json_llm(contents: list[str]) -> MagicMock:
    llm = MagicMock()
    responses = []
    for content in contents:
        r = MagicMock()
        r.content = content
        r.tool_calls = []
        r.invalid_tool_calls = []
        responses.append(r)
    llm.ainvoke = AsyncMock(side_effect=responses)
    llm.bind_tools = MagicMock(side_effect=AssertionError("bind_tools must not be called in JSON"))
    return llm


def _src() -> list[HumanMessage]:
    return [HumanMessage(content=SOURCE)]


# ---------------------------------------------------------------------------
# extract_data — TOOL_CALLING
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_hallucinated_value_is_rejected_then_corrected() -> None:
    llm = _bound_llm(
        [
            _tool_response({"name": "Globex"}, "Vendor"),  # not in source
            _tool_response({"name": "ACME GmbH"}, "Vendor"),  # present
        ]
    )

    result, stats = await extract_data(llm, Vendor, _src(), retry_config=_NO_NETWORK_RETRY)

    assert result is not None
    assert result.name == "ACME GmbH"
    assert stats.success
    assert stats.primary_retries == 1
    assert any(i.category == "grounding" for i in stats.field_issues)


@pytest.mark.asyncio
async def test_grounding_exhausts_retries() -> None:
    llm = _bound_llm([_tool_response({"name": "Globex"}, "Vendor")] * 3)

    result, stats = await extract_data(
        llm, Vendor, _src(), max_primary_retries=3, retry_config=_NO_NETWORK_RETRY
    )

    assert result is None
    assert not stats.success
    assert sum(i.category == "grounding" for i in stats.field_issues) == 3


@pytest.mark.asyncio
async def test_present_value_passes_first_try() -> None:
    llm = _bound_llm([_tool_response({"name": "ACME GmbH"}, "Vendor")])

    result, stats = await extract_data(llm, Vendor, _src(), retry_config=_NO_NETWORK_RETRY)

    assert result is not None
    assert stats.primary_retries == 0
    assert not stats.field_issues


# ---------------------------------------------------------------------------
# extract_data — JSON mode
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_grounding_json_mode() -> None:
    llm = _json_llm(['{"name": "Globex"}', '{"name": "ACME GmbH"}'])

    result, stats = await extract_data(
        llm, Vendor, _src(), mode=ExtractionMode.JSON, retry_config=_NO_NETWORK_RETRY
    )

    assert result is not None
    assert result.name == "ACME GmbH"
    assert stats.primary_retries == 1


# ---------------------------------------------------------------------------
# locale-aware grounding via a sibling field
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_locale_field_grounds_localized_number() -> None:
    class Inv(BaseModel):
        country: str
        total: float = GroundedField(locale_field="country")

    llm = _bound_llm([_tool_response({"country": "DE", "total": 1234.5}, "Inv")])

    result, stats = await extract_data(
        llm,
        Inv,
        [HumanMessage(content="Rechnung, Summe 1.234,50 EUR.")],
        retry_config=_NO_NETWORK_RETRY,
    )

    assert result is not None
    assert stats.success
    assert stats.primary_retries == 0


# ---------------------------------------------------------------------------
# batch — per-item grounding
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_grounding_per_item_in_batch() -> None:
    class Line(BaseModel):
        name: Annotated[str, Grounded()]

    bad = _tool_response({"items": [{"name": "ACME GmbH"}, {"name": "Globex"}]}, "LineList")
    good = _tool_response({"items": [{"name": "ACME GmbH"}, {"name": "ACME GmbH"}]}, "LineList")
    llm = _bound_llm([bad, good])

    items, stats = await extract_data_list(llm, Line, _src(), retry_config=_NO_NETWORK_RETRY)

    assert items is not None
    assert [i.name for i in items] == ["ACME GmbH", "ACME GmbH"]
    assert stats.primary_retries == 1
    assert any(i.field_path == "items -> 1 -> name" for i in stats.field_issues)


# ---------------------------------------------------------------------------
# agent loop
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_grounding_in_agent_loop() -> None:
    def _agent_final(args: dict[str, Any], call_id: str) -> MagicMock:
        response = MagicMock()
        response.tool_calls = [{"name": "Vendor", "args": args, "id": call_id}]
        response.invalid_tool_calls = []
        response.content = ""
        return response

    bound = MagicMock()
    bound.ainvoke = AsyncMock(
        side_effect=[
            _agent_final({"name": "Globex"}, "c1"),
            _agent_final({"name": "ACME GmbH"}, "c2"),
        ]
    )
    llm = MagicMock()
    llm.bind_tools = MagicMock(return_value=bound)
    llm.ainvoke = bound.ainvoke

    result, stats = await extract_data_with_tools(
        llm, Vendor, SOURCE, tools=[], retry_config=_NO_NETWORK_RETRY
    )

    assert result is not None
    assert result.name == "ACME GmbH"
    assert stats.validation_retries == 1
    assert any(i.category == "grounding" for i in stats.field_issues)


# ---------------------------------------------------------------------------
# on_mismatch="flag" — keep the value, record the issue, no retry
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_flag_mode_keeps_value_without_retry() -> None:
    class Doc(BaseModel):
        name: Annotated[str, Grounded(on_mismatch="flag")]

    llm = _bound_llm([_tool_response({"name": "Globex"}, "Doc")])  # not in source

    result, stats = await extract_data(llm, Doc, _src(), retry_config=_NO_NETWORK_RETRY)

    assert result is not None
    assert result.name == "Globex"  # value kept
    assert stats.success
    assert stats.primary_retries == 0  # no retry consumed
    assert any(i.category == "grounding" for i in stats.field_issues)


# ---------------------------------------------------------------------------
# sync wrapper
# ---------------------------------------------------------------------------


def test_grounding_sync_wrapper() -> None:
    llm = _bound_llm(
        [
            _tool_response({"name": "Globex"}, "Vendor"),
            _tool_response({"name": "ACME GmbH"}, "Vendor"),
        ]
    )

    result, stats = extract_data_sync(llm, Vendor, _src(), retry_config=_NO_NETWORK_RETRY)

    assert result is not None
    assert result.name == "ACME GmbH"
    assert stats.primary_retries == 1
