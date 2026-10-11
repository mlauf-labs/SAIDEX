"""Strict tool binding: the schema tool really is strict, with safe fallbacks."""

from __future__ import annotations

import dataclasses
import logging
from enum import Enum
from typing import Any, Dict, Literal, Optional  # noqa: UP035

import pydantic
import pytest
from langchain_core.messages import HumanMessage
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from saidex import ExtractionMode, ToolCallConfig, extract_data
from saidex import extractor as _extractor

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


class _ClientError(Exception):
    def __init__(self, message: str, status_code: int | None = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


def _llm_with_binds(*results: Any) -> Any:
    """LLM whose n-th ``bind_tools`` call returns a bound model yielding results[n]."""
    bounds = [make_llm(r if isinstance(r, list) else [r]).bind_tools.return_value for r in results]
    llm = make_llm([])
    llm.bind_tools.side_effect = bounds
    return llm


def _ok(name: str, args: dict[str, Any]) -> Any:
    return make_response([{"name": name, "args": args, "id": "1"}])


@pytest.fixture(autouse=True)
def _reset_warned() -> None:
    _extractor._WARNED_FALLBACK.clear()


@pytest.mark.parametrize(
    "message",
    [
        "Schema is too complex for compilation.",
        "'minimum' is not supported",
        "Invalid schema for function 'OptStr': bad",
    ],
)
async def test_rejected_strict_schema_relaxes_keeping_other_flags(
    message: str, caplog: pytest.LogCaptureFixture
) -> None:
    llm = _llm_with_binds(_ClientError(message), _ok("OptStr", {"x": "v"}))

    with caplog.at_level(logging.WARNING, logger="saidex.extractor"):
        result, stats = await extract_data(llm, OptStr, _msgs())

    assert stats.success
    assert result == OptStr(x="v")
    assert stats.primary_retries == 0
    first, second = llm.bind_tools.call_args_list
    assert first.args[0][0]["function"]["strict"] is True
    assert second.args[0][0] == convert_to_openai_tool(OptStr)
    assert second.kwargs["tool_choice"] == "OptStr"
    assert second.kwargs["parallel_tool_calls"] is False
    assert second.kwargs.get("strict") is not True
    assert any(message.lower()[:12] in r.getMessage().lower() for r in caplog.records)


async def test_client_error_detected_by_message_marker_without_status() -> None:
    llm = _llm_with_binds(
        _ClientError("'minimum' is not supported", None), _ok("OptStr", {"x": "v"})
    )

    _, stats = await extract_data(llm, OptStr, _msgs())

    assert stats.success
    assert llm.bind_tools.call_count == 2


async def test_client_error_on_non_strict_tool_does_not_rebind() -> None:
    llm = _llm_with_binds(_ClientError("something odd"), _ok("OptStr", {}))

    _, stats = await extract_data(llm, OptStr, _msgs(), tool_config=ToolCallConfig.COMPATIBLE)

    assert not stats.success
    assert llm.bind_tools.call_count == 1


async def test_client_error_on_dict_fallback_does_not_rebind() -> None:
    llm = _llm_with_binds(_ClientError("something odd"), _ok("DictTop", {"d": {}}))

    _, stats = await extract_data(llm, DictTop, _msgs())

    assert not stats.success
    assert llm.bind_tools.call_count == 1


async def test_flag_rejection_after_strict_relax_still_uses_flag_relaxation() -> None:
    llm = _llm_with_binds(
        _ClientError("Schema is too complex"),
        _ClientError("Unsupported parameter: 'tool_choice'"),
        _ok("OptStr", {"x": "v"}),
    )

    _, stats = await extract_data(llm, OptStr, _msgs())

    assert stats.success
    third = llm.bind_tools.call_args_list[2]
    assert third.kwargs == {"tool_choice": "auto"}


async def test_flag_named_rejection_goes_straight_to_compatible() -> None:
    llm = _llm_with_binds(
        _ClientError("Unsupported parameter: 'strict'"), _ok("OptStr", {"x": "v"})
    )

    _, stats = await extract_data(llm, OptStr, _msgs())

    assert stats.success
    assert llm.bind_tools.call_count == 2
    assert llm.bind_tools.call_args_list[1].kwargs == {"tool_choice": "auto"}


class BareDict(BaseModel):
    d: dict  # type: ignore[type-arg]


class TypingDict(BaseModel):
    d: Dict[str, int]  # noqa: UP006


class AnyField(BaseModel):
    v: Any


class ObjectField(BaseModel):
    v: object


class TD(TypedDict):
    a: int


class WithTypedDict(BaseModel):
    t: TD


class Tree(BaseModel):
    children: list[Tree]


class Ping(BaseModel):
    pong: Optional[Pong] = None  # noqa: UP045


class Pong(BaseModel):
    ping: Optional[Ping] = None  # noqa: UP045


@dataclasses.dataclass
class DC:
    d: dict[str, int]


class WithDataclass(BaseModel):
    dc: DC


Ping.model_rebuild()


@pytest.mark.parametrize(
    "schema",
    [BareDict, TypingDict, AnyField, ObjectField, WithTypedDict, Tree, Ping, WithDataclass],
)
async def test_more_incompatible_shapes_fall_back(schema: type[BaseModel]) -> None:
    assert _extractor._strict_incompatibility(schema) is not None
    llm = make_llm([make_response([{"name": schema.__name__, "args": {}, "id": "1"}])])

    await extract_data(llm, schema, _msgs())

    assert _bound_tool(llm) == convert_to_openai_tool(schema)


def test_shared_but_non_recursive_model_is_compatible() -> None:
    class Twice(BaseModel):
        a: Nested
        b: Nested

    assert _extractor._strict_incompatibility(Twice) is None


async def test_stripped_defaults_are_appended_to_description() -> None:
    class Defaults(BaseModel):
        n: int = 0
        s: str = Field(default="x", description="Label.")
        o: Optional[int] = None  # noqa: UP045

    llm = make_llm([make_response([{"name": "Defaults", "args": {}, "id": "1"}])])
    await extract_data(llm, Defaults, _msgs())

    props = _bound_tool(llm)["function"]["parameters"]["properties"]
    assert props["n"]["description"] == "Default: 0"
    assert props["s"]["description"] == 'Label. Default: "x"'
    assert "description" not in props["o"]


async def test_fallback_warning_is_logged_once_per_schema(
    caplog: pytest.LogCaptureFixture,
) -> None:
    llm = make_llm([_ok("DictTop", {"d": {}}), _ok("DictTop", {"d": {}})])

    with caplog.at_level(logging.DEBUG, logger="saidex.extractor"):
        await extract_data(llm, DictTop, _msgs())
        await extract_data(llm, DictTop, _msgs())

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1


async def test_json_mode_unaffected() -> None:
    llm = make_llm([make_response(content='{"x": "v"}')])
    result, _ = await extract_data(llm, OptStr, _msgs(), mode=ExtractionMode.JSON)
    assert result == OptStr(x="v")
    llm.bind_tools.assert_not_called()
