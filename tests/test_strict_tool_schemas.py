"""Strict tool binding: the schema tool really is strict, with safe fallbacks."""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any, Literal, Optional

import pydantic
import pytest
from langchain_core.messages import HumanMessage
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import BaseModel, Field

from saidex import ToolCallConfig, extract_data

from ._mock_llm import make_llm, make_response


class Color(Enum):
    RED = "red"
    BLUE = "blue"


class Nested(BaseModel):
    a: str
    b: Optional[int] = None  # noqa: UP045


class OptStr(BaseModel):
    x: Optional[str] = None  # noqa: UP045


class IntDefault(BaseModel):
    x: int = 0


class ListFactory(BaseModel):
    x: list[str] = Field(default_factory=list)


class WithNested(BaseModel):
    n: Nested


class OptNested(BaseModel):
    n: Optional[Nested] = None  # noqa: UP045


class ListNested(BaseModel):
    n: list[Nested]


class WithEnum(BaseModel):
    c: Color


class WithLiteral(BaseModel):
    c: Literal["a", "b"]


class WithDescription(BaseModel):
    x: str = Field(description="the x")


# (schema, args the model sends in strict style: optional fields as null)
SHAPES: list[tuple[type[BaseModel], dict[str, Any]]] = [
    (OptStr, {"x": None}),
    (IntDefault, {"x": 3}),
    (ListFactory, {"x": ["q"]}),
    (WithNested, {"n": {"a": "s", "b": None}}),
    (OptNested, {"n": None}),
    (ListNested, {"n": [{"a": "s", "b": None}]}),
    (WithEnum, {"c": "red"}),
    (WithLiteral, {"c": "a"}),
    (WithDescription, {"x": "v"}),
]
IDS = [s.__name__ for s, _ in SHAPES]


def _msgs() -> list[HumanMessage]:
    return [HumanMessage(content="text")]


def _bound_tool(llm: Any, call: int = 0) -> dict[str, Any]:
    tool: dict[str, Any] = llm.bind_tools.call_args_list[call].args[0][0]
    return tool


def _objects(node: Any) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    if isinstance(node, dict):
        if "properties" in node:
            found.append(node)
        for value in node.values():
            found.extend(_objects(value))
    elif isinstance(node, list):
        for item in node:
            found.extend(_objects(item))
    return found


def _keys(node: Any) -> set[str]:
    out: set[str] = set()
    if isinstance(node, dict):
        out |= set(node)
        for v in node.values():
            out |= _keys(v)
    elif isinstance(node, list):
        for v in node:
            out |= _keys(v)
    return out


@pytest.mark.parametrize(("schema", "args"), SHAPES, ids=IDS)
async def test_strict_tool_is_really_strict(schema: type[BaseModel], args: dict[str, Any]) -> None:
    llm = make_llm([make_response([{"name": schema.__name__, "args": args, "id": "1"}])])

    result, stats = await extract_data(llm, schema, _msgs())

    assert stats.success
    assert result == schema(**args)
    tool = _bound_tool(llm)
    assert tool["function"]["strict"] is True
    params = tool["function"]["parameters"]
    for obj in _objects(params):
        assert obj["additionalProperties"] is False
        assert obj["required"] == list(obj["properties"])
    assert "default" not in _keys(params)
    assert llm.bind_tools.call_args.kwargs["strict"] is True


async def test_optional_fields_are_nullable() -> None:
    llm = make_llm([make_response([{"name": "OptNested", "args": {"n": None}, "id": "1"}])])
    await extract_data(llm, OptNested, _msgs())

    params = _bound_tool(llm)["function"]["parameters"]
    branches = params["properties"]["n"]["anyOf"]
    assert {"type": "null"} in branches
    inner = next(b for b in _objects(params) if "a" in b["properties"])
    assert {"type": "null"} in inner["properties"]["b"]["anyOf"]


async def test_field_named_default_survives() -> None:
    class Odd(BaseModel):
        default: str

    llm = make_llm([make_response([{"name": "Odd", "args": {"default": "z"}, "id": "1"}])])
    result, _ = await extract_data(llm, Odd, _msgs())

    assert result == Odd(default="z")
    assert "default" in _bound_tool(llm)["function"]["parameters"]["properties"]


class DictTop(BaseModel):
    d: dict[str, int]


class DictInner(BaseModel):
    d: dict[str, int]


class DictNested(BaseModel):
    inner: DictInner


class DictInList(BaseModel):
    items: list[DictInner]


class DictOptional(BaseModel):
    inner: Optional[DictInner] = None  # noqa: UP045


@pytest.mark.parametrize(
    ("schema", "args"),
    [
        (DictTop, {"d": {"a": 1}}),
        (DictNested, {"inner": {"d": {"a": 1}}}),
        (DictInList, {"items": [{"d": {"a": 1}}]}),
        (DictOptional, {"inner": {"d": {"a": 1}}}),
    ],
)
async def test_dict_fields_fall_back_to_plain_tool_with_warning(
    schema: type[BaseModel], args: dict[str, Any], caplog: pytest.LogCaptureFixture
) -> None:
    llm = make_llm([make_response([{"name": schema.__name__, "args": args, "id": "1"}])])

    with caplog.at_level(logging.WARNING, logger="saidex.extractor"):
        result, stats = await extract_data(llm, schema, _msgs())

    assert stats.success
    assert result == schema(**args)
    assert _bound_tool(llm) == convert_to_openai_tool(schema)
    assert "strict" not in _bound_tool(llm)["function"]
    assert llm.bind_tools.call_args.kwargs["strict"] is False
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("dict" in w and "'d'" in w and "COMPATIBLE" in w for w in warnings)


@pytest.mark.skipif(not hasattr(pydantic, "MISSING"), reason="needs pydantic >= 2.14")
async def test_missing_default_falls_back_to_plain_tool_with_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class WithMissing(BaseModel):
        x: int = Field(default=pydantic.MISSING)

    llm = make_llm([make_response([{"name": "WithMissing", "args": {}, "id": "1"}])])

    with caplog.at_level(logging.WARNING, logger="saidex.extractor"):
        await extract_data(llm, WithMissing, _msgs())

    assert _bound_tool(llm) == convert_to_openai_tool(WithMissing)
    assert any("MISSING" in r.getMessage() for r in caplog.records)


async def test_compatible_binds_plain_dict() -> None:
    llm = make_llm([make_response([{"name": "OptStr", "args": {}, "id": "1"}])])

    await extract_data(llm, OptStr, _msgs(), tool_config=ToolCallConfig.COMPATIBLE)

    assert _bound_tool(llm) == convert_to_openai_tool(OptStr)
    assert llm.bind_tools.call_args.kwargs == {"tool_choice": "auto"}


class _InvalidSchemaError(Exception):
    status_code = 400


async def test_rejected_strict_schema_relaxes_to_plain_tool() -> None:
    bound_bad = make_llm([_InvalidSchemaError("Invalid schema for function 'OptStr': bad")])
    bound_ok = make_llm([make_response([{"name": "OptStr", "args": {"x": "v"}, "id": "1"}])])
    llm = make_llm([])
    llm.bind_tools.side_effect = [
        bound_bad.bind_tools.return_value,
        bound_ok.bind_tools.return_value,
    ]

    result, stats = await extract_data(llm, OptStr, _msgs())

    assert stats.success
    assert result == OptStr(x="v")
    assert stats.primary_retries == 0
    first, second = llm.bind_tools.call_args_list
    assert first.args[0][0]["function"]["strict"] is True
    assert second.args[0][0] == convert_to_openai_tool(OptStr)
    assert "strict" not in second.kwargs
