"""Tests for ValidatorContext / with_context across every extraction entry point."""

from __future__ import annotations

import dataclasses
import functools
import json
from typing import Any

import pytest
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

import saidex
from saidex import (
    ExtractionMode,
    Tool,
    ValidationIssue,
    Validator,
    ValidatorContext,
    WithContext,
    extract_data,
    extract_data_list,
    extract_data_sync,
    extract_data_with_tools,
    extract_data_with_tools_sync,
    with_context,
)
from saidex.retry import RetryConfig
from tests._mock_llm import make_llm, make_response


class Item(BaseModel):
    name: str
    value: int


def test_with_context_wraps_the_callable() -> None:
    def fn(instance: Item, ctx: ValidatorContext) -> None:
        return None

    wrapped = with_context(fn)
    assert isinstance(wrapped, WithContext)
    assert wrapped.fn is fn


def test_validator_context_is_frozen() -> None:
    ctx = ValidatorContext(
        attempt=1, max_attempts=3, is_final=False, model_label="primary", is_fallback=False
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        ctx.attempt = 2  # type: ignore[misc]


def test_new_names_are_exported() -> None:
    for name in ("ValidatorContext", "WithContext", "with_context", "ContextValidator"):
        assert name in saidex.__all__


def test_validator_alias_is_subscriptable() -> None:
    assert Validator[Item] is not None


_NO_RETRY = RetryConfig(max_retries=0, retry_delays=[], retryable_exceptions=())
Ctx = tuple[int, int, bool, str, bool]


def _key(ctx: ValidatorContext) -> Ctx:
    return (ctx.attempt, ctx.max_attempts, ctx.is_final, ctx.model_label, ctx.is_fallback)


def _call(args: dict[str, Any], name: str = "Item", call_id: str = "c") -> dict[str, Any]:
    return {"name": name, "args": args, "id": call_id}


def _item_responses(n: int, value: int = 1) -> list[Any]:
    return [
        make_response([_call({"name": "a", "value": value}, call_id=f"c{i}")]) for i in range(n)
    ]


def _recording(seen: list[Ctx], *, reject: bool = True) -> WithContext[Any]:
    def validate(instance: Any, ctx: ValidatorContext) -> str | None:
        seen.append(_key(ctx))
        return "rejected" if reject else None

    return with_context(validate)


def _msgs() -> list[HumanMessage]:
    return [HumanMessage(content="extract")]


async def test_plain_validator_still_gets_one_argument() -> None:
    seen: list[Any] = []

    def plain(instance: Item) -> None:
        seen.append(instance)

    result, _ = await extract_data(
        make_llm(_item_responses(1)), Item, _msgs(), retry_config=_NO_RETRY, validator=plain
    )
    assert result is not None
    assert len(seen) == 1 and isinstance(seen[0], Item)


async def test_partial_and_callable_objects_stay_one_arg() -> None:
    seen: list[str] = []

    def tagged(tag: str, instance: Item) -> None:
        seen.append(tag)

    class CallableWithFn:
        fn = "not a context validator"

        def __call__(self, instance: Item) -> None:
            seen.append("object")

    await extract_data(
        make_llm(_item_responses(1)),
        Item,
        _msgs(),
        retry_config=_NO_RETRY,
        validator=functools.partial(tagged, "partial"),
    )
    await extract_data(
        make_llm(_item_responses(1)),
        Item,
        _msgs(),
        retry_config=_NO_RETRY,
        validator=CallableWithFn(),
    )
    assert seen == ["partial", "object"]


async def test_context_sequence_without_fallback() -> None:
    seen: list[Ctx] = []
    await extract_data(
        make_llm(_item_responses(3)),
        Item,
        _msgs(),
        max_primary_retries=3,
        retry_config=_NO_RETRY,
        validator=_recording(seen),
    )
    assert seen == [
        (1, 3, False, "primary", False),
        (2, 3, False, "primary", False),
        (3, 3, True, "primary", False),
    ]


async def test_async_context_validator() -> None:
    seen: list[Ctx] = []

    async def validate(instance: Item, ctx: ValidatorContext) -> None:
        seen.append(_key(ctx))

    result, _ = await extract_data(
        make_llm(_item_responses(1)),
        Item,
        _msgs(),
        max_primary_retries=1,
        retry_config=_NO_RETRY,
        validator=with_context(validate),
    )
    assert result is not None
    assert seen == [(1, 1, True, "primary", False)]


async def test_context_sequence_with_fallback() -> None:
    seen: list[Ctx] = []
    await extract_data(
        make_llm(_item_responses(2)),
        Item,
        _msgs(),
        fallback_llm_model=make_llm(_item_responses(2)),
        max_primary_retries=2,
        max_fallback_retries=2,
        retry_config=_NO_RETRY,
        validator=_recording(seen),
    )
    assert seen == [
        (1, 2, False, "primary", False),
        (2, 2, False, "primary", False),
        (1, 2, False, "fallback", True),
        (2, 2, True, "fallback", True),
    ]


async def test_primary_is_final_when_fallback_has_no_attempts() -> None:
    seen: list[Ctx] = []
    await extract_data(
        make_llm(_item_responses(1)),
        Item,
        _msgs(),
        fallback_llm_model=make_llm([]),
        max_primary_retries=1,
        max_fallback_retries=0,
        retry_config=_NO_RETRY,
        validator=_recording(seen),
    )
    assert seen == [(1, 1, True, "primary", False)]


async def test_parse_error_consumes_an_attempt() -> None:
    seen: list[Ctx] = []
    responses = [
        make_response([_call({"name": "a", "value": 1}, call_id="c0")]),
        make_response(invalid=True),
        make_response([_call({"name": "a", "value": 1}, call_id="c2")]),
    ]
    await extract_data(
        make_llm(responses),
        Item,
        _msgs(),
        max_primary_retries=3,
        retry_config=_NO_RETRY,
        validator=_recording(seen),
    )
    assert [c[0] for c in seen] == [1, 3]
    assert seen[-1][2] is True


async def test_context_validator_raise_and_bad_type() -> None:
    def raising(instance: Item, ctx: ValidatorContext) -> None:
        raise ValueError(f"no (attempt {ctx.attempt})")

    _, stats = await extract_data(
        make_llm(_item_responses(1)),
        Item,
        _msgs(),
        max_primary_retries=1,
        retry_config=_NO_RETRY,
        validator=with_context(raising),
    )
    assert stats.field_issues[-1].message == "no (attempt 1)"

    def bad(instance: Item, ctx: ValidatorContext) -> Any:
        return {"not": "valid"}

    with pytest.raises(TypeError):
        await extract_data(
            make_llm(_item_responses(1)),
            Item,
            _msgs(),
            retry_config=_NO_RETRY,
            validator=with_context(bad),
        )


async def test_lenient_on_final_attempt() -> None:
    def strict_then_lenient(instance: Item, ctx: ValidatorContext) -> ValidationIssue | None:
        if instance.value >= 100:
            return None
        return ValidationIssue(
            message="value too small",
            field_path="value",
            on_mismatch="flag" if ctx.is_final else "retry",
        )

    result, stats = await extract_data(
        make_llm(_item_responses(2)),
        Item,
        _msgs(),
        max_primary_retries=2,
        retry_config=_NO_RETRY,
        validator=with_context(strict_then_lenient),
    )
    assert result is not None and result.value == 1
    assert stats.success
    assert [i.attempt for i in stats.field_issues] == [0, 1]


async def test_list_items_receive_the_same_context() -> None:
    seen: list[tuple[str, Ctx]] = []

    def per_item(item: Item, ctx: ValidatorContext) -> None:
        seen.append((item.name, _key(ctx)))

    container_args = {"items": [{"name": "a", "value": 1}, {"name": "b", "value": 2}]}
    llm = make_llm([make_response([_call(container_args, name="ItemList")])])
    items, _ = await extract_data_list(
        llm,
        Item,
        _msgs(),
        max_primary_retries=1,
        retry_config=_NO_RETRY,
        validator=with_context(per_item),
    )
    assert items is not None
    assert seen == [
        ("a", (1, 1, True, "primary", False)),
        ("b", (1, 1, True, "primary", False)),
    ]


def test_sync_wrapper_passes_context() -> None:
    seen: list[Ctx] = []
    extract_data_sync(
        make_llm(_item_responses(1)),
        Item,
        _msgs(),
        max_primary_retries=1,
        retry_config=_NO_RETRY,
        validator=_recording(seen, reject=False),
    )
    assert seen == [(1, 1, True, "primary", False)]


class Answer(BaseModel):
    result: str


class _Args(BaseModel):
    name: str


async def _noop(**kwargs: Any) -> dict[str, Any]:
    return {"ok": True}


def _final(i: int) -> Any:
    return make_response([_call({"result": "x"}, name="Answer", call_id=f"f{i}")])


async def test_agent_tool_calling_contexts() -> None:
    seen: list[Ctx] = []
    await extract_data_with_tools(
        make_llm([_final(i) for i in range(5)]),
        Answer,
        "task",
        tools=[],
        max_validation_retries=2,
        retry_config=_NO_RETRY,
        validator=_recording(seen),
    )
    assert seen == [
        (1, 3, False, "primary", False),
        (2, 3, False, "primary", False),
        (3, 3, True, "primary", False),
    ]


async def test_agent_json_contexts() -> None:
    seen: list[Ctx] = []
    responses = [make_response(content=json.dumps({"result": "x"})) for _ in range(5)]
    await extract_data_with_tools(
        make_llm(responses),
        Answer,
        "task",
        tools=[],
        final_answer_mode=ExtractionMode.JSON,
        max_validation_retries=1,
        retry_config=_NO_RETRY,
        validator=_recording(seen),
    )
    assert seen == [(1, 2, False, "primary", False), (2, 2, True, "primary", False)]


async def test_agent_attempt_ignores_helper_tool_iterations() -> None:
    seen: list[Ctx] = []
    tool = Tool(name="lookup", description=".", parameters=_Args, handler=_noop)
    helper = make_response([_call({"name": "q"}, name="lookup", call_id="h")])
    responses = [helper, helper, _final(0), helper, _final(1)]
    await extract_data_with_tools(
        make_llm(responses),
        Answer,
        "task",
        tools=[tool],
        max_validation_retries=1,
        retry_config=_NO_RETRY,
        validator=_recording(seen),
    )
    assert [c[0] for c in seen] == [1, 2]


async def test_agent_fallback_contexts() -> None:
    seen: list[Ctx] = []
    await extract_data_with_tools(
        make_llm([_final(i) for i in range(5)]),
        Answer,
        "task",
        tools=[],
        fallback_llm_model=make_llm([_final(i) for i in range(5)]),
        max_validation_retries=1,
        retry_config=_NO_RETRY,
        validator=_recording(seen),
    )
    assert seen == [
        (1, 2, False, "primary", False),
        (2, 2, False, "primary", False),
        (1, 2, False, "fallback", True),
        (2, 2, True, "fallback", True),
    ]


async def test_agent_max_iterations_can_end_before_final() -> None:
    seen: list[Ctx] = []
    await extract_data_with_tools(
        make_llm([_final(i) for i in range(5)]),
        Answer,
        "task",
        tools=[],
        max_iterations=2,
        max_validation_retries=3,
        retry_config=_NO_RETRY,
        validator=_recording(seen),
    )
    assert seen == [(1, 4, False, "primary", False), (2, 4, False, "primary", False)]


def test_agent_sync_wrapper_passes_context() -> None:
    seen: list[Ctx] = []
    extract_data_with_tools_sync(
        make_llm([_final(0)]),
        Answer,
        "task",
        tools=[],
        max_validation_retries=0,
        retry_config=_NO_RETRY,
        validator=_recording(seen, reject=False),
    )
    assert seen == [(1, 1, True, "primary", False)]
