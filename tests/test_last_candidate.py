"""Tests for stats.last_candidate / stats.last_rejection across all entry points."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from saidex import (
    ExtractionEvent,
    ExtractionMode,
    ExtractorRunStats,
    GroundedStr,
    ValidationIssue,
    collect_stats,
    extract_data,
    extract_data_list,
    extract_data_sync,
    extract_data_with_tools,
)
from saidex.retry import RetryConfig
from tests._mock_llm import make_llm, make_response

_NO_RETRY = RetryConfig(max_retries=0, retry_delays=[], retryable_exceptions=())
SOURCE = "Invoice from ACME, amount 5."


class Item(BaseModel):
    name: str
    value: int


class Doc(BaseModel):
    vendor: GroundedStr
    amount: int


def _call(args: dict[str, Any], name: str = "Item", call_id: str = "c") -> dict[str, Any]:
    return {"name": name, "args": args, "id": call_id}


def _item(value: int, call_id: str = "c") -> Any:
    return make_response([_call({"name": "a", "value": value}, call_id=call_id)])


def _msgs(text: str = "extract") -> list[HumanMessage]:
    return [HumanMessage(content=text)]


def _reject_small(item: Item) -> str | None:
    return "value too small" if item.value < 100 else None


async def test_validator_rejection_sets_candidate() -> None:
    llm = make_llm([_item(1, "c0"), _item(2, "c1")])
    result, stats = await extract_data(
        llm, Item, _msgs(), max_primary_retries=2, retry_config=_NO_RETRY, validator=_reject_small
    )
    assert result is None
    assert stats.failure_reason == "validation_exhausted"
    assert stats.last_candidate == Item(name="a", value=2)
    assert [(i.message, i.attempt) for i in stats.last_rejection] == [("value too small", 1)]


async def test_grounding_rejection_sets_candidate() -> None:
    llm = make_llm([make_response([_call({"vendor": "Globex", "amount": 5}, name="Doc")])])
    _, stats = await extract_data(
        llm, Doc, _msgs(SOURCE), max_primary_retries=1, retry_config=_NO_RETRY
    )
    assert stats.last_candidate == Doc(vendor="Globex", amount=5)
    assert [i.category for i in stats.last_rejection] == ["grounding"]


async def test_flag_issues_not_in_last_rejection() -> None:
    def mixed(item: Item) -> list[ValidationIssue]:
        return [
            ValidationIssue(message="soft", field_path="name", on_mismatch="flag"),
            ValidationIssue(message="hard", field_path="value"),
        ]

    _, stats = await extract_data(
        make_llm([_item(1)]),
        Item,
        _msgs(),
        max_primary_retries=1,
        retry_config=_NO_RETRY,
        validator=mixed,
    )
    assert [i.message for i in stats.field_issues] == ["soft", "hard"]
    assert [i.message for i in stats.last_rejection] == ["hard"]


async def test_candidate_survives_later_parse_error() -> None:
    llm = make_llm([_item(1, "c0"), make_response(invalid=True)])
    _, stats = await extract_data(
        llm, Item, _msgs(), max_primary_retries=2, retry_config=_NO_RETRY, validator=_reject_small
    )
    assert stats.failure_reason == "parse_error"
    assert stats.last_candidate == Item(name="a", value=1)


async def test_candidate_survives_later_llm_error() -> None:
    llm = make_llm([_item(1, "c0"), RuntimeError("provider down")])
    _, stats = await extract_data(
        llm, Item, _msgs(), max_primary_retries=2, retry_config=_NO_RETRY, validator=_reject_small
    )
    assert stats.failure_reason == "llm_error"
    assert stats.last_candidate == Item(name="a", value=1)


async def test_no_schema_valid_attempt_gives_no_candidate() -> None:
    _, parse_stats = await extract_data(
        make_llm([make_response(invalid=True)] * 2),
        Item,
        _msgs(),
        max_primary_retries=2,
        retry_config=_NO_RETRY,
    )
    _, llm_stats = await extract_data(
        make_llm([RuntimeError("down")]), Item, _msgs(), retry_config=_NO_RETRY
    )
    for stats in (parse_stats, llm_stats):
        assert stats.last_candidate is None
        assert stats.last_rejection == ()


async def test_pydantic_failures_only_give_no_candidate() -> None:
    bad = make_response([_call({"name": "a", "value": "not-an-int"})])
    _, stats = await extract_data(
        make_llm([bad, bad]), Item, _msgs(), max_primary_retries=2, retry_config=_NO_RETRY
    )
    assert stats.failure_reason == "validation_exhausted"
    assert stats.last_candidate is None


async def test_success_after_rejection_has_no_candidate() -> None:
    llm = make_llm([_item(1, "c0"), _item(150, "c1")])
    result, stats = await extract_data(
        llm, Item, _msgs(), retry_config=_NO_RETRY, validator=_reject_small
    )
    assert result is not None
    assert stats.last_candidate is None
    assert stats.last_rejection == ()


async def test_fallback_candidate_overrides_primary() -> None:
    _, stats = await extract_data(
        make_llm([_item(1, "p")]),
        Item,
        _msgs(),
        fallback_llm_model=make_llm([_item(2, "f")]),
        max_primary_retries=1,
        max_fallback_retries=1,
        retry_config=_NO_RETRY,
        validator=_reject_small,
    )
    assert stats.last_candidate == Item(name="a", value=2)


async def test_fallback_without_candidate_keeps_primary() -> None:
    _, stats = await extract_data(
        make_llm([_item(1, "p")]),
        Item,
        _msgs(),
        fallback_llm_model=make_llm([make_response(invalid=True)]),
        max_primary_retries=1,
        max_fallback_retries=1,
        retry_config=_NO_RETRY,
        validator=_reject_small,
    )
    assert stats.failure_reason == "parse_error"
    assert stats.last_candidate == Item(name="a", value=1)
    assert [i.message for i in stats.last_rejection] == ["value too small"]


async def test_list_variant_unwraps_items() -> None:
    args = {"items": [{"name": "a", "value": 500}, {"name": "b", "value": 1}]}
    llm = make_llm([make_response([_call(args, name="ItemList")])])
    items, stats = await extract_data_list(
        llm, Item, _msgs(), max_primary_retries=1, retry_config=_NO_RETRY, validator=_reject_small
    )
    assert items is None
    assert stats.last_candidate == [Item(name="a", value=500), Item(name="b", value=1)]
    assert [(i.field_path, i.schema_name) for i in stats.last_rejection] == [("items -> 1", "Item")]


def test_sync_wrapper_exposes_candidate() -> None:
    _, stats = extract_data_sync(
        make_llm([_item(1)]),
        Item,
        _msgs(),
        max_primary_retries=1,
        retry_config=_NO_RETRY,
        validator=_reject_small,
    )
    assert stats.last_candidate == Item(name="a", value=1)


class Answer(BaseModel):
    result: str


def _final(result: str, call_id: str) -> Any:
    return make_response([_call({"result": result}, name="Answer", call_id=call_id)])


def _reject_answers(answer: Answer) -> str | None:
    return None if answer.result == "ok" else "not ok"


async def test_agent_tool_calling_candidate() -> None:
    llm = make_llm([_final("first", "f0"), _final("second", "f1")])
    _, stats = await extract_data_with_tools(
        llm,
        Answer,
        "task",
        tools=[],
        max_validation_retries=1,
        retry_config=_NO_RETRY,
        validator=_reject_answers,
    )
    assert stats.last_candidate == Answer(result="second")
    assert [i.message for i in stats.last_rejection] == ["not ok"]


async def test_agent_json_candidate() -> None:
    llm = make_llm([make_response(content=json.dumps({"result": "nope"}))] * 2)
    _, stats = await extract_data_with_tools(
        llm,
        Answer,
        "task",
        tools=[],
        final_answer_mode=ExtractionMode.JSON,
        max_validation_retries=1,
        retry_config=_NO_RETRY,
        validator=_reject_answers,
    )
    assert stats.last_candidate == Answer(result="nope")


async def test_agent_success_has_no_candidate() -> None:
    llm = make_llm([_final("bad", "f0"), _final("ok", "f1")])
    result, stats = await extract_data_with_tools(
        llm, Answer, "task", tools=[], retry_config=_NO_RETRY, validator=_reject_answers
    )
    assert result is not None
    assert stats.last_candidate is None and stats.last_rejection == ()


async def test_agent_fallback_without_candidate_keeps_primary() -> None:
    _, stats = await extract_data_with_tools(
        make_llm([_final("primary", "p")]),
        Answer,
        "task",
        tools=[],
        fallback_llm_model=make_llm([RuntimeError("down")]),
        max_validation_retries=0,
        retry_config=_NO_RETRY,
        validator=_reject_answers,
    )
    assert stats.failure_reason == "llm_error"
    assert stats.last_candidate == Answer(result="primary")


def test_add_combines_candidates() -> None:
    a = ExtractorRunStats(last_candidate=Answer(result="a"), last_rejection=())
    b = ExtractorRunStats()
    assert (a + b).last_candidate == Answer(result="a")
    c = ExtractorRunStats(last_candidate=Answer(result="c"))
    assert (a + c).last_candidate == Answer(result="c")
    ok = ExtractorRunStats(success=True)
    assert (a + ok).last_candidate is None


async def test_agent_fallback_candidate_overrides_primary() -> None:
    _, stats = await extract_data_with_tools(
        make_llm([_final("p", "p0")]),
        Answer,
        "task",
        tools=[],
        fallback_llm_model=make_llm([_final("f", "f0")]),
        max_validation_retries=0,
        retry_config=_NO_RETRY,
        validator=_reject_answers,
    )
    assert stats.fallback_used
    assert stats.last_candidate == Answer(result="f")


class _Outputs(AsyncCallbackHandler):
    def __init__(self) -> None:
        self.outputs: list[Any] = []

    async def on_chain_end(self, outputs: Any, **kw: Any) -> None:
        self.outputs.append(outputs)


async def test_event_exposes_candidate() -> None:
    events: list[ExtractionEvent] = []

    async def hook(event: ExtractionEvent) -> None:
        events.append(event)

    await extract_data(
        make_llm([_item(1)]),
        Item,
        _msgs(),
        max_primary_retries=1,
        retry_config=_NO_RETRY,
        validator=_reject_small,
        on_complete=hook,
    )
    assert events[0].last_candidate == Item(name="a", value=1)
    assert events[0].last_rejection == events[0].stats.last_rejection


async def test_trace_gets_only_a_flag() -> None:
    handler = _Outputs()
    await extract_data(
        make_llm([_item(1)]),
        Item,
        _msgs(),
        max_primary_retries=1,
        retry_config=_NO_RETRY,
        validator=_reject_small,
        callbacks=[handler],
    )
    outputs = handler.outputs[-1]
    assert outputs["has_last_candidate"] is True
    assert outputs["result"] is None
    assert "last_candidate" not in outputs


async def test_stats_sink_drops_candidate() -> None:
    async with collect_stats() as sink:
        _, stats = await extract_data(
            make_llm([_item(1)]),
            Item,
            _msgs(),
            max_primary_retries=1,
            retry_config=_NO_RETRY,
            validator=_reject_small,
        )
    recorded = sink.all()[0]
    assert stats.last_candidate is not None
    assert recorded.last_candidate is None  # type: ignore[union-attr]
    assert recorded.last_rejection == stats.last_rejection  # type: ignore[union-attr]


async def test_stats_with_candidate_is_hashable_and_not_in_repr() -> None:
    _, stats = await extract_data(
        make_llm([_item(1)]),
        Item,
        _msgs(),
        max_primary_retries=1,
        retry_config=_NO_RETRY,
        validator=_reject_small,
    )
    assert stats.last_candidate is not None
    assert isinstance(hash(stats), int)
    assert "last_candidate" not in repr(stats)
    assert "Item(" not in repr(stats)
