"""JSON-mode replies holding several objects or an array-wrapped object (issue #101)."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from saidex import ExtractionMode, extract_data, extract_data_list, extract_data_with_tools
from saidex.extractor import _parse_json_response
from saidex.retry import RetryConfig
from tests._mock_llm import make_llm, make_response


class Person(BaseModel):
    name: str
    age: int


_RETRY = RetryConfig(
    max_retries=0, retry_delays=[], retryable_exceptions=(), rate_limit_exceptions=()
)

TWO_OBJECTS = '{"name": "A", "age": 1}\n{"name": "B", "age": 2}'
VALID = '{"name": "Alice", "age": 36}'


# --- _parse_json_response ----------------------------------------------------


def _resp(content: str) -> MagicMock:
    r = MagicMock()
    r.content = content
    return r


def test_parse_single_object_array_accepted() -> None:
    args, feedback = _parse_json_response(_resp(f"[{VALID}]"), "Person", "m", schema=Person)
    assert args == {"name": "Alice", "age": 36}
    assert feedback is None


def test_parse_two_objects_asks_for_single_object() -> None:
    args, feedback = _parse_json_response(_resp(TWO_OBJECTS), "Person", "m", schema=Person)
    assert args is None
    assert feedback is not None
    assert "2 separate JSON objects" in feedback
    assert "'Person'" in feedback


def test_parse_non_object_array_keeps_old_feedback() -> None:
    args, feedback = _parse_json_response(_resp("[1, 2]"), "Person", "m", schema=Person)
    assert args is None
    assert feedback is not None
    assert "valid JSON but not a JSON object" in feedback


def test_parse_empty_array_keeps_old_feedback() -> None:
    _, feedback = _parse_json_response(_resp("[]"), "Person", "m", schema=Person)
    assert feedback is not None
    assert "valid JSON but not a JSON object" in feedback


# --- end to end --------------------------------------------------------------


def _json_llm(contents: list[str]) -> MagicMock:
    llm = MagicMock()
    llm.ainvoke = AsyncMock(side_effect=[make_response(content=c) for c in contents])
    llm.bind_tools = MagicMock(side_effect=AssertionError("no tools in JSON mode"))
    return llm


def _all_text(call: Any) -> str:
    return "\n".join(str(m.content) for m in call.args[0])


@pytest.mark.asyncio
async def test_extract_data_two_objects_then_valid() -> None:
    llm = _json_llm([TWO_OBJECTS, VALID])
    result, _ = await extract_data(
        llm,
        Person,
        [HumanMessage(content="x")],
        mode=ExtractionMode.JSON,
        max_primary_retries=2,
        retry_config=_RETRY,
    )
    assert result is not None
    assert result.age == 36
    assert llm.ainvoke.await_count == 2
    assert "2 separate JSON objects" in _all_text(llm.ainvoke.await_args_list[1])


@pytest.mark.asyncio
async def test_extract_data_array_wrapped_object_first_attempt() -> None:
    llm = _json_llm([f"[{VALID}]"])
    result, _ = await extract_data(
        llm,
        Person,
        [HumanMessage(content="x")],
        mode=ExtractionMode.JSON,
        retry_config=_RETRY,
    )
    assert result is not None
    assert llm.ainvoke.await_count == 1


@pytest.mark.asyncio
async def test_agent_loop_two_objects_then_valid() -> None:
    llm = make_llm([make_response(content=TWO_OBJECTS), make_response(content=VALID)])
    result, _ = await extract_data_with_tools(
        llm,
        Person,
        "input",
        tools=[],
        final_answer_mode=ExtractionMode.JSON,
        max_validation_retries=1,
        retry_config=_RETRY,
    )
    assert result is not None
    assert result.name == "Alice"
    assert "2 separate JSON objects" in _all_text(llm.ainvoke.await_args_list[1])


@pytest.mark.asyncio
async def test_agent_loop_array_wrapped_object() -> None:
    llm = make_llm([make_response(content=json.dumps([{"name": "A", "age": 2}]))])
    result, _ = await extract_data_with_tools(
        llm,
        Person,
        "input",
        tools=[],
        final_answer_mode=ExtractionMode.JSON,
        retry_config=_RETRY,
    )
    assert result is not None
    assert result.age == 2


# --- schema-aware array handling ---------------------------------------------


class Item(BaseModel):
    description: str
    price: float


class Invoice(BaseModel):
    line_items: list[Item] = Field(default_factory=list)
    total: float | None = None


class Basket(BaseModel):
    goods: list[Item]


ONE_ITEM = '[{"description": "x", "price": 1.5}]'
TWO_ITEMS = '[{"description": "x", "price": 1.5}, {"description": "y", "price": 2}]'


def test_parse_array_for_multi_field_schema_without_shared_keys_is_rejected() -> None:
    args, feedback = _parse_json_response(_resp(ONE_ITEM), "Invoice", "m", schema=Invoice)
    assert args is None
    assert feedback is not None
    assert "valid JSON but not a JSON object" in feedback


def test_parse_array_for_single_list_field_schema_is_wrapped() -> None:
    args, feedback = _parse_json_response(_resp(TWO_ITEMS), "Basket", "m", schema=Basket)
    assert feedback is None
    assert args is not None
    assert len(args["goods"]) == 2


def test_parse_empty_array_for_single_list_field_schema_is_wrapped() -> None:
    args, feedback = _parse_json_response(_resp("[]"), "Basket", "m", schema=Basket)
    assert (args, feedback) == ({"goods": []}, None)


def test_parse_two_objects_for_multi_field_schema_gives_feedback() -> None:
    args, feedback = _parse_json_response(_resp(TWO_ITEMS), "Invoice", "m", schema=Invoice)
    assert args is None
    assert feedback is not None
    assert "2 separate JSON objects" in feedback


@pytest.mark.asyncio
@pytest.mark.parametrize("reply", [ONE_ITEM, TWO_ITEMS])
async def test_extract_data_list_accepts_bare_array(reply: str) -> None:
    llm = _json_llm([reply])
    result, _ = await extract_data_list(
        llm,
        Item,
        [HumanMessage(content="x")],
        mode=ExtractionMode.JSON,
        retry_config=_RETRY,
    )
    assert result is not None
    assert len(result) == len(json.loads(reply))
    assert llm.ainvoke.await_count == 1


@pytest.mark.asyncio
async def test_extract_data_single_list_field_schema_accepts_bare_array() -> None:
    llm = _json_llm([ONE_ITEM])
    result, _ = await extract_data(
        llm,
        Basket,
        [HumanMessage(content="x")],
        mode=ExtractionMode.JSON,
        retry_config=_RETRY,
    )
    assert result is not None
    assert result.goods[0].price == 1.5
