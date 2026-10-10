"""Tests for the on_exhausted salvage hook across all extraction entry points."""

from __future__ import annotations

import json
from typing import Any

import pytest
from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, ConfigDict, Field, computed_field

from saidex import (
    ExtractionEvent,
    ExtractionMode,
    FieldIssue,
    ValidationIssue,
    extract_data,
    extract_data_from_text,
    extract_data_list,
    extract_data_list_sync,
    extract_data_sync,
    extract_data_with_tools,
    extract_data_with_tools_sync,
    issue_item_index,
)
from saidex.retry import RetryConfig
from tests._mock_llm import make_llm, make_response

_NO_RETRY = RetryConfig(max_retries=0, retry_delays=[], retryable_exceptions=())
SOURCE = "Alice said: we ship on Monday. Bob said: budget is fine."


class Finding(BaseModel):
    speaker: str
    quote: str


class Findings(BaseModel):
    findings: list[Finding]


def _call(args: dict[str, Any], name: str, call_id: str = "c") -> dict[str, Any]:
    return {"name": name, "args": args, "id": call_id}


def _findings_resp(quotes: list[str], call_id: str = "c") -> Any:
    args = {"findings": [{"speaker": "x", "quote": q} for q in quotes]}
    return make_response([_call(args, "Findings", call_id)])


def _verbatim(doc: Findings) -> list[ValidationIssue]:
    return [
        ValidationIssue(
            message="quote is not verbatim in the source",
            field_path=f"findings -> {i} -> quote",
            error_type="not_verbatim",
        )
        for i, f in enumerate(doc.findings)
        if f.quote not in SOURCE
    ]


def _drop_rejected(doc: Findings, issues: tuple[FieldIssue, ...]) -> Findings:
    bad = {int(i.field_path.split(" -> ")[1]) for i in issues}
    return Findings(findings=[f for i, f in enumerate(doc.findings) if i not in bad])


QUOTES = ["we ship on Monday", "budget is huge", "budget is fine"]


def _msgs() -> list[HumanMessage]:
    return [HumanMessage(content=SOURCE)]


async def _run(llm: Any, **kwargs: Any) -> Any:
    return await extract_data(
        llm,
        Findings,
        _msgs(),
        max_primary_retries=kwargs.pop("max_primary_retries", 2),
        retry_config=_NO_RETRY,
        validator=kwargs.pop("validator", _verbatim),
        **kwargs,
    )


async def test_without_hook_behaviour_is_unchanged() -> None:
    result, stats = await _run(make_llm([_findings_resp(QUOTES, f"c{i}") for i in range(2)]))
    assert result is None
    assert stats.success is False and stats.salvaged is False


async def test_hook_drops_invalid_items() -> None:
    result, stats = await _run(
        make_llm([_findings_resp(QUOTES, f"c{i}") for i in range(2)]),
        on_exhausted=_drop_rejected,
    )
    assert result is not None
    assert [f.quote for f in result.findings] == ["we ship on Monday", "budget is fine"]
    assert stats.success is True and stats.salvaged is True
    assert stats.failure_reason is None
    assert any(i.error_type == "not_verbatim" for i in stats.field_issues)
    assert stats.last_candidate is not None  # pre-salvage value kept for auditing


async def test_async_hook() -> None:
    async def drop(doc: Findings, issues: tuple[FieldIssue, ...]) -> Findings:
        return _drop_rejected(doc, issues)

    result, stats = await _run(
        make_llm([_findings_resp(QUOTES, f"c{i}") for i in range(2)]), on_exhausted=drop
    )
    assert result is not None and stats.salvaged


async def test_hook_returning_none_keeps_failure() -> None:
    result, stats = await _run(
        make_llm([_findings_resp(QUOTES, f"c{i}") for i in range(2)]),
        on_exhausted=lambda doc, issues: None,
    )
    assert result is None
    assert stats.failure_reason == "validation_exhausted" and not stats.salvaged


async def test_returning_candidate_unchanged_is_salvaged() -> None:
    result, stats = await _run(
        make_llm([_findings_resp(QUOTES, f"c{i}") for i in range(2)]),
        on_exhausted=lambda doc, issues: doc,
    )
    assert result is not None and len(result.findings) == 3
    assert stats.salvaged


async def test_hook_not_called_without_candidate() -> None:
    calls: list[Any] = []

    def hook(doc: Findings, issues: tuple[FieldIssue, ...]) -> None:
        calls.append(doc)

    await _run(make_llm([make_response(invalid=True)] * 2), on_exhausted=hook)
    await _run(make_llm([RuntimeError("down")]), on_exhausted=hook)
    assert calls == []


async def test_hook_runs_when_last_attempt_was_a_parse_error() -> None:
    calls: list[Findings] = []

    def hook(doc: Findings, issues: tuple[FieldIssue, ...]) -> Findings:
        calls.append(doc)
        return _drop_rejected(doc, issues)

    result, stats = await _run(
        make_llm([_findings_resp(QUOTES, "c0"), make_response(invalid=True)]), on_exhausted=hook
    )
    assert len(calls) == 1
    assert result is not None and stats.salvaged


async def test_hook_runs_once_after_fallback() -> None:
    calls: list[Findings] = []

    def hook(doc: Findings, issues: tuple[FieldIssue, ...]) -> Findings:
        calls.append(doc)
        return _drop_rejected(doc, issues)

    result, stats = await _run(
        make_llm([_findings_resp(QUOTES, "p")]),
        max_primary_retries=1,
        fallback_llm_model=make_llm([_findings_resp(["budget is huge"], "f")]),
        max_fallback_retries=1,
        on_exhausted=hook,
    )
    assert len(calls) == 1
    assert [f.quote for f in calls[0].findings] == ["budget is huge"]  # fallback's candidate
    assert result is not None and result.findings == []
    assert stats.fallback_used and stats.salvaged


async def test_invalid_salvage_keeps_run_failed() -> None:
    class Strict(BaseModel):
        value: int = Field(ge=0)

    def reject(_: Strict) -> str:
        return "never"

    def corrupt(doc: Strict, issues: tuple[FieldIssue, ...]) -> Strict:
        return doc.model_copy(update={"value": -5})  # bypasses validation

    llm = make_llm([make_response([_call({"value": 1}, "Strict")])])
    result, stats = await extract_data(
        llm,
        Strict,
        _msgs(),
        max_primary_retries=1,
        retry_config=_NO_RETRY,
        validator=reject,
        on_exhausted=corrupt,
    )
    assert result is None
    assert stats.success is False and stats.salvaged is False
    assert stats.field_issues[-1].error_type == "on_exhausted_invalid"
    assert stats.field_issues[-1].field_path == "<on_exhausted>"


async def test_salvage_with_aliased_fields() -> None:
    class Aliased(BaseModel):
        model_config = ConfigDict(populate_by_name=False)
        full_name: str = Field(alias="fullName")

    llm = make_llm([make_response([_call({"fullName": "Ada"}, "Aliased")])])
    result, stats = await extract_data(
        llm,
        Aliased,
        _msgs(),
        max_primary_retries=1,
        retry_config=_NO_RETRY,
        validator=lambda a: "no",
        on_exhausted=lambda a, issues: a,
    )
    assert result is not None and result.full_name == "Ada"
    assert stats.salvaged


async def test_hook_exception_propagates() -> None:
    def boom(doc: Findings, issues: tuple[FieldIssue, ...]) -> Findings:
        raise RuntimeError("hook bug")

    with pytest.raises(RuntimeError, match="hook bug"):
        await _run(make_llm([_findings_resp(QUOTES, f"c{i}") for i in range(2)]), on_exhausted=boom)


async def test_hook_wrong_return_type_raises() -> None:
    with pytest.raises(TypeError, match="on_exhausted"):
        await _run(
            make_llm([_findings_resp(QUOTES, f"c{i}") for i in range(2)]),
            on_exhausted=lambda doc, issues: {"findings": []},
        )


async def test_on_complete_and_trace_see_salvage() -> None:
    events: list[ExtractionEvent] = []

    async def hook(event: ExtractionEvent) -> None:
        events.append(event)

    class Outputs(AsyncCallbackHandler):
        def __init__(self) -> None:
            self.outputs: list[Any] = []

        async def on_chain_end(self, outputs: Any, **kw: Any) -> None:
            self.outputs.append(outputs)

    handler = Outputs()
    await _run(
        make_llm([_findings_resp(QUOTES, f"c{i}") for i in range(2)]),
        on_exhausted=_drop_rejected,
        on_complete=hook,
        callbacks=[handler],
    )
    assert events[0].result is not None and events[0].stats.salvaged
    assert handler.outputs[-1]["salvaged"] is True
    assert handler.outputs[-1]["success"] is True


async def test_salvage_with_computed_field_and_extra_forbid() -> None:
    class Priced(BaseModel):
        model_config = ConfigDict(extra="forbid")
        net: int

        @computed_field  # type: ignore[prop-decorator]
        @property
        def gross(self) -> int:
            return self.net * 2

    llm = make_llm([make_response([_call({"net": 5}, "Priced")])])
    result, stats = await extract_data(
        llm,
        Priced,
        _msgs(),
        max_primary_retries=1,
        retry_config=_NO_RETRY,
        validator=lambda p: "no",
        on_exhausted=lambda p, issues: p,
    )
    assert result is not None and result.gross == 10
    assert stats.salvaged


async def test_from_text_forwards_hook() -> None:
    result, stats = await extract_data_from_text(
        make_llm([_findings_resp(QUOTES, f"c{i}") for i in range(2)]),
        Findings,
        SOURCE,
        max_primary_retries=2,
        retry_config=_NO_RETRY,
        validator=_verbatim,
        on_exhausted=_drop_rejected,
    )
    assert result is not None and len(result.findings) == 2
    assert stats.salvaged


def _issue(path: str) -> FieldIssue:
    return FieldIssue(
        schema_name="X", field_path=path, category="value", error_type="e", message="m", attempt=0
    )


@pytest.mark.parametrize(
    ("path", "list_field", "expected"),
    [
        ("items -> 3", "items", 3),
        ("items -> 12 -> quote", "items", 12),
        ("findings -> 0 -> quote", "findings", 0),
        ("findings -> 0 -> quote", "items", None),
        ("itemsX -> 1", "items", None),
        ("<external validator>", "items", None),
    ],
)
def test_issue_item_index(path: str, list_field: str, expected: int | None) -> None:
    assert issue_item_index(_issue(path), list_field) == expected


def _drop_items(items: list[Finding], issues: tuple[FieldIssue, ...]) -> list[Finding]:
    bad = {issue_item_index(i) for i in issues}
    return [item for index, item in enumerate(items) if index not in bad]


def _not_verbatim_item(f: Finding) -> str | None:
    return None if f.quote in SOURCE else "quote is not verbatim"


def _list_resp(quotes: list[str]) -> Any:
    args = {"items": [{"speaker": "x", "quote": q} for q in quotes]}
    return make_response([_call(args, "FindingList")])


async def test_list_hook_drops_items() -> None:
    items, stats = await extract_data_list(
        make_llm([_list_resp(QUOTES)]),
        Finding,
        _msgs(),
        max_primary_retries=1,
        retry_config=_NO_RETRY,
        validator=_not_verbatim_item,
        on_exhausted=_drop_items,
    )
    assert items is not None
    assert [f.quote for f in items] == ["we ship on Monday", "budget is fine"]
    assert stats.item_count == 2 and stats.salvaged
    assert stats.schema_name == "Finding"


async def test_list_hook_receives_item_schema_issues() -> None:
    seen: list[tuple[FieldIssue, ...]] = []

    def hook(items: list[Finding], issues: tuple[FieldIssue, ...]) -> None:
        seen.append(issues)

    await extract_data_list(
        make_llm([_list_resp(QUOTES)]),
        Finding,
        _msgs(),
        max_primary_retries=1,
        retry_config=_NO_RETRY,
        validator=_not_verbatim_item,
        on_exhausted=hook,
    )
    assert [(i.field_path, i.schema_name) for i in seen[0]] == [("items -> 1", "Finding")]


async def test_list_hook_may_return_empty_list() -> None:
    items, stats = await extract_data_list(
        make_llm([_list_resp(["nope"])]),
        Finding,
        _msgs(),
        max_primary_retries=1,
        retry_config=_NO_RETRY,
        validator=_not_verbatim_item,
        on_exhausted=lambda items, issues: [],
    )
    assert items == []
    assert stats.item_count == 0 and stats.salvaged


async def test_list_salvage_with_computed_field_on_forbid_item() -> None:
    class It(BaseModel):
        model_config = ConfigDict(extra="forbid")
        net: int

        @computed_field  # type: ignore[prop-decorator]
        @property
        def gross(self) -> int:
            return self.net * 2

    items, stats = await extract_data_list(
        make_llm([make_response([_call({"items": [{"net": 1}]}, "ItList")])]),
        It,
        _msgs(),
        max_primary_retries=1,
        retry_config=_NO_RETRY,
        validator=lambda i: "no",
        on_exhausted=lambda items, issues: items,
    )
    assert items is not None and items[0].gross == 2
    assert stats.salvaged


async def test_salvage_with_nested_computed_field_on_forbid_model() -> None:
    class Inner(BaseModel):
        model_config = ConfigDict(extra="forbid")
        net: int

        @computed_field  # type: ignore[prop-decorator]
        @property
        def gross(self) -> int:
            return self.net * 2

    class Outer(BaseModel):
        inner: Inner

    result, stats = await extract_data(
        make_llm([make_response([_call({"inner": {"net": 3}}, "Outer")])]),
        Outer,
        _msgs(),
        max_primary_retries=1,
        retry_config=_NO_RETRY,
        validator=lambda o: "no",
        on_exhausted=lambda o, issues: o,
    )
    assert result is not None and result.inner.gross == 6
    assert stats.salvaged


async def test_list_hook_returning_non_instances_raises() -> None:
    with pytest.raises(TypeError, match="on_exhausted.*Finding"):
        await extract_data_list(
            make_llm([_list_resp(QUOTES)]),
            Finding,
            _msgs(),
            max_primary_retries=1,
            retry_config=_NO_RETRY,
            validator=_not_verbatim_item,
            on_exhausted=lambda items, issues: [{"speaker": "x", "quote": "we ship on Monday"}],  # type: ignore[arg-type,return-value]
        )


def _agent_final(quotes: list[str], call_id: str) -> Any:
    return _findings_resp(quotes, call_id)


async def test_agent_tool_calling_salvage() -> None:
    llm = make_llm([_agent_final(QUOTES, f"f{i}") for i in range(2)])
    result, stats = await extract_data_with_tools(
        llm,
        Findings,
        SOURCE,
        tools=[],
        max_validation_retries=1,
        retry_config=_NO_RETRY,
        validator=_verbatim,
        on_exhausted=_drop_rejected,
    )
    assert result is not None and len(result.findings) == 2
    assert stats.salvaged and stats.success and stats.failure_reason is None


async def test_agent_json_salvage() -> None:
    content = json.dumps({"findings": [{"speaker": "x", "quote": q} for q in QUOTES]})
    llm = make_llm([make_response(content=content)] * 2)
    result, stats = await extract_data_with_tools(
        llm,
        Findings,
        SOURCE,
        tools=[],
        final_answer_mode=ExtractionMode.JSON,
        max_validation_retries=1,
        retry_config=_NO_RETRY,
        validator=_verbatim,
        on_exhausted=_drop_rejected,
    )
    assert result is not None and stats.salvaged


async def test_agent_salvage_after_fallback() -> None:
    calls: list[Findings] = []

    def hook(doc: Findings, issues: tuple[FieldIssue, ...]) -> Findings:
        calls.append(doc)
        return _drop_rejected(doc, issues)

    result, stats = await extract_data_with_tools(
        make_llm([_agent_final(QUOTES, "p")]),
        Findings,
        SOURCE,
        tools=[],
        fallback_llm_model=make_llm([RuntimeError("down")]),
        max_validation_retries=0,
        retry_config=_NO_RETRY,
        validator=_verbatim,
        on_exhausted=hook,
    )
    assert len(calls) == 1
    assert result is not None and stats.salvaged and stats.fallback_used


def test_sync_wrappers_forward_on_exhausted() -> None:
    result, stats = extract_data_sync(
        make_llm([_findings_resp(QUOTES)]),
        Findings,
        _msgs(),
        max_primary_retries=1,
        retry_config=_NO_RETRY,
        validator=_verbatim,
        on_exhausted=_drop_rejected,
    )
    assert result is not None and stats.salvaged

    items, list_stats = extract_data_list_sync(
        make_llm([_list_resp(QUOTES)]),
        Finding,
        _msgs(),
        max_primary_retries=1,
        retry_config=_NO_RETRY,
        validator=_not_verbatim_item,
        on_exhausted=_drop_items,
    )
    assert items is not None and list_stats.salvaged

    agent_result, agent_stats = extract_data_with_tools_sync(
        make_llm([_agent_final(QUOTES, "f")]),
        Findings,
        SOURCE,
        tools=[],
        max_validation_retries=0,
        retry_config=_NO_RETRY,
        validator=_verbatim,
        on_exhausted=_drop_rejected,
    )
    assert agent_result is not None and agent_stats.salvaged
