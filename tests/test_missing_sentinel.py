"""Pydantic's ``MISSING`` sentinel (pydantic >= 2.14) for fields absent from the source.

Schemas live behind ``if MISSING is not None`` so the module imports cleanly on older
pydantic, where every test is skipped instead.
"""

import json
import logging
from typing import Annotated, Any

import pydantic
import pytest
from langchain_core.messages import HumanMessage

from saidex import (
    ExtractionMode,
    Grounded,
    GroundedStr,
    extract_data,
    extract_data_from_text,
    extract_data_list,
    extract_data_with_tools,
    field_check,
)
from saidex import extractor as _extractor
from saidex.retry import RetryConfig

from ._mock_llm import make_llm, make_response

MISSING = getattr(pydantic, "MISSING", None)

requires_missing = pytest.mark.skipif(MISSING is None, reason="pydantic < 2.14 has no MISSING")

_NO_RETRY = RetryConfig(max_retries=0, retry_delays=[])

if MISSING is not None:

    class Note(pydantic.BaseModel):
        name: str
        note: str | MISSING = MISSING  # type: ignore[valid-type]

    class Checked(pydantic.BaseModel):
        name: str
        note: str | MISSING = field_check(Grounded(), default=MISSING)  # type: ignore[valid-type]

    class CheckedAnnotated(pydantic.BaseModel):
        note: Annotated[str | MISSING, Grounded()] = MISSING  # type: ignore[valid-type]

    class HoistedAlias(pydantic.BaseModel):
        note: GroundedStr | MISSING = MISSING  # type: ignore[valid-type]


def _call(name: str, args: dict[str, Any], call_id: str = "c1") -> Any:
    return make_response([{"name": name, "args": args, "id": call_id}])


def _msgs() -> list[HumanMessage]:
    return [HumanMessage(content="text")]


@requires_missing
async def test_tool_calling_omitted_field_is_missing() -> None:
    llm = make_llm([_call("Note", {"name": "a"})])

    result, stats = await extract_data(llm, Note, _msgs(), retry_config=_NO_RETRY)

    assert stats.success and result is not None
    assert result.note is MISSING
    assert "note" not in result.model_dump()
    assert "note" not in json.loads(result.model_dump_json())


@requires_missing
async def test_present_value_is_kept() -> None:
    llm = make_llm([_call("Note", {"name": "a", "note": "hi"})])

    result, _ = await extract_data(llm, Note, _msgs(), retry_config=_NO_RETRY)

    assert result is not None and result.note == "hi"
    assert result.model_dump() == {"name": "a", "note": "hi"}


@requires_missing
async def test_null_is_rejected_and_retried() -> None:
    llm = make_llm(
        [_call("Note", {"name": "a", "note": None}, "c1"), _call("Note", {"name": "a"}, "c2")]
    )

    result, stats = await extract_data(llm, Note, _msgs(), retry_config=_NO_RETRY)

    assert stats.success and result is not None
    assert result.note is MISSING
    assert stats.primary_retries == 1
    assert any("note" in str(i.field_path) for i in stats.field_issues)


@requires_missing
async def test_json_mode_omitted_field_is_missing() -> None:
    response = make_response(content=json.dumps({"name": "a"}))
    llm = make_llm([response])

    result, stats = await extract_data(
        llm, Note, _msgs(), mode=ExtractionMode.JSON, retry_config=_NO_RETRY
    )

    assert stats.success and result is not None
    assert result.note is MISSING


@requires_missing
async def test_agent_loop_final_answer_omitted_field_is_missing() -> None:
    llm = make_llm([_call("Note", {"name": "a"})])

    result, stats = await extract_data_with_tools(
        llm, Note, "input", tools=[], retry_config=_NO_RETRY
    )

    assert stats.success and result is not None
    assert result.note is MISSING


@requires_missing
async def test_extract_data_list_mixes_missing_and_values() -> None:
    items = [{"name": "a"}, {"name": "b", "note": "x"}]
    llm = make_llm([_call("NoteList", {"items": items})])

    result, stats = await extract_data_list(llm, Note, _msgs(), retry_config=_NO_RETRY)

    assert stats.success and result is not None
    assert [i.note for i in result] == [MISSING, "x"]
    assert [i.model_dump() for i in result] == [{"name": "a"}, {"name": "b", "note": "x"}]


@requires_missing
async def test_field_check_grounding_skips_absent_and_flags_bad_value() -> None:
    absent = make_llm([_call("Checked", {"name": "a"})])
    result, stats = await extract_data_from_text(
        absent, Checked, "ACME ships widgets", retry_config=_NO_RETRY
    )
    assert stats.success and result is not None and result.note is MISSING
    assert not stats.field_issues

    bad = make_llm([_call("Checked", {"name": "a", "note": "Globex"})])
    result, stats = await extract_data_from_text(
        bad, Checked, "ACME ships widgets", max_primary_retries=1, retry_config=_NO_RETRY
    )
    assert result is None
    assert any(i.category == "grounding" for i in stats.field_issues)


@requires_missing
async def test_annotated_grounded_inside_union_is_checked() -> None:
    bad = make_llm([_call("CheckedAnnotated", {"note": "Globex"})])

    result, stats = await extract_data_from_text(
        bad, CheckedAnnotated, "ACME ships widgets", max_primary_retries=1, retry_config=_NO_RETRY
    )

    assert result is None
    assert any(i.category == "grounding" for i in stats.field_issues)


@requires_missing
async def test_grounded_str_in_union_is_not_grounded() -> None:
    # Documents a gotcha (docs/schema-design.md, "Fields that may be missing"): Annotated
    # metadata inside a union is not hoisted, so GroundedStr | MISSING silently skips the check.
    llm = make_llm([_call("HoistedAlias", {"note": "Globex"})])

    result, stats = await extract_data_from_text(
        llm, HoistedAlias, "ACME ships widgets", retry_config=_NO_RETRY
    )

    assert stats.success and result is not None and result.note == "Globex"
    assert not stats.field_issues


@requires_missing
async def test_on_exhausted_salvage_keeps_missing() -> None:
    def reject(doc: Note) -> str:
        return "always rejected"

    llm = make_llm([_call("Note", {"name": "a"}, "c1"), _call("Note", {"name": "a"}, "c2")])

    result, stats = await extract_data(
        llm,
        Note,
        _msgs(),
        max_primary_retries=2,
        retry_config=_NO_RETRY,
        validator=reject,
        on_exhausted=lambda doc, issues: doc,
    )

    assert stats.salvaged and result is not None
    assert result.note is MISSING


@requires_missing
async def test_strict_binding_falls_back_to_non_strict(
    caplog: pytest.LogCaptureFixture,
) -> None:
    llm = make_llm([_call("Note", {"name": "a"})])
    _extractor._WARNED_FALLBACK.clear()

    with caplog.at_level(logging.WARNING, logger="saidex.extractor"):
        result, stats = await extract_data(llm, Note, _msgs(), retry_config=_NO_RETRY)

    assert stats.success and result is not None
    tool = llm.bind_tools.call_args.args[0][0]
    assert "strict" not in tool["function"]
    assert llm.bind_tools.call_args.kwargs["strict"] is False
    assert any(r.levelno == logging.WARNING for r in caplog.records)
