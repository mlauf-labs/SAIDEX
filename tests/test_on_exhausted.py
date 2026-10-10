"""Tests for the on_exhausted salvage hook across all extraction entry points."""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, ConfigDict, Field

from saidex import ExtractionEvent, FieldIssue, ValidationIssue, extract_data
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
