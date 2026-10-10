"""Tests for the external ``validator`` callable across all extraction entry points."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from saidex import (
    ExtractionMode,
    ValidationIssue,
    extract_data,
    extract_data_from_text,
    extract_data_list,
    extract_data_sync,
    extract_data_with_tools,
)
from saidex.retry import RetryConfig

# ---------------------------------------------------------------------------
# Helpers & schemas
# ---------------------------------------------------------------------------

_NO_NETWORK_RETRY = RetryConfig(
    max_retries=0,
    retry_delays=[],
    retryable_exceptions=(),
    rate_limit_exceptions=(),
)


class SimpleSchema(BaseModel):
    name: str
    value: int


def _tool_response(args: dict[str, Any], name: str = "SimpleSchema") -> MagicMock:
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


def _json_response(content: str) -> MagicMock:
    response = MagicMock()
    response.content = content
    response.tool_calls = []
    response.invalid_tool_calls = []
    return response


def _json_llm(contents: list[str]) -> MagicMock:
    llm = MagicMock()
    llm.ainvoke = AsyncMock(side_effect=[_json_response(c) for c in contents])
    llm.bind_tools = MagicMock(
        side_effect=AssertionError("bind_tools must not be called in JSON mode")
    )
    return llm


def _messages() -> list[HumanMessage]:
    return [HumanMessage(content="Extract data")]


def _reject_low(instance: SimpleSchema) -> None:
    """Sync validator that rejects values below 100 by raising."""
    if instance.value < 100:
        raise ValueError(f"value {instance.value} must be at least 100")


# ---------------------------------------------------------------------------
# extract_data — single shot, TOOL_CALLING
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_validator_rejects_once_then_passes() -> None:
    llm = _bound_llm(
        [
            _tool_response({"name": "Alice", "value": 5}),  # rejected by validator
            _tool_response({"name": "Alice", "value": 150}),  # accepted
        ]
    )

    result, stats = await extract_data(
        llm,
        SimpleSchema,
        _messages(),
        retry_config=_NO_NETWORK_RETRY,
        validator=_reject_low,
    )

    assert result is not None
    assert result.value == 150
    assert stats.success
    assert stats.primary_retries == 1
    # The rejection is recorded as a structured field issue.
    assert any(issue.error_type == "external_validator" for issue in stats.field_issues)


@pytest.mark.asyncio
async def test_validator_exhausts_retries_returns_none() -> None:
    llm = _bound_llm([_tool_response({"name": "Alice", "value": 1})] * 3)

    result, stats = await extract_data(
        llm,
        SimpleSchema,
        _messages(),
        max_primary_retries=3,
        retry_config=_NO_NETWORK_RETRY,
        validator=_reject_low,
    )

    assert result is None
    assert not stats.success
    assert stats.failure_reason == "validation_exhausted"
    assert stats.primary_retries == 3
    assert sum(i.error_type == "external_validator" for i in stats.field_issues) == 3


@pytest.mark.asyncio
async def test_validator_can_return_error_string() -> None:
    def reject_with_message(instance: SimpleSchema) -> str | None:
        if instance.value < 100:
            return "too small"
        return None

    llm = _bound_llm(
        [
            _tool_response({"name": "Alice", "value": 5}),
            _tool_response({"name": "Alice", "value": 200}),
        ]
    )

    result, stats = await extract_data(
        llm,
        SimpleSchema,
        _messages(),
        retry_config=_NO_NETWORK_RETRY,
        validator=reject_with_message,
    )

    assert result is not None
    assert result.value == 200
    assert stats.primary_retries == 1
    assert stats.field_issues[0].message == "too small"


@pytest.mark.asyncio
async def test_async_validator_supported() -> None:
    async def reject_low_async(instance: SimpleSchema) -> None:
        if instance.value < 100:
            raise ValueError("async rejection")

    llm = _bound_llm(
        [
            _tool_response({"name": "Alice", "value": 5}),
            _tool_response({"name": "Alice", "value": 150}),
        ]
    )

    result, stats = await extract_data(
        llm,
        SimpleSchema,
        _messages(),
        retry_config=_NO_NETWORK_RETRY,
        validator=reject_low_async,
    )

    assert result is not None
    assert result.value == 150
    assert stats.primary_retries == 1


@pytest.mark.asyncio
async def test_validator_not_invoked_on_pydantic_failure() -> None:
    seen: list[int] = []

    def record(instance: SimpleSchema) -> None:
        seen.append(instance.value)

    # First response fails Pydantic (value is not an int) — validator must not run.
    llm = _bound_llm(
        [
            _tool_response({"name": "Alice", "value": "not_an_int"}),
            _tool_response({"name": "Alice", "value": 42}),
        ]
    )

    result, stats = await extract_data(
        llm,
        SimpleSchema,
        _messages(),
        retry_config=_NO_NETWORK_RETRY,
        validator=record,
    )

    assert result is not None
    # Validator only ran for the structurally valid instance.
    assert seen == [42]
    assert stats.primary_retries == 1


@pytest.mark.asyncio
async def test_validator_triggers_fallback() -> None:
    primary = _bound_llm([_tool_response({"name": "Alice", "value": 1})] * 3)
    fallback = _bound_llm([_tool_response({"name": "Alice", "value": 999})])

    result, stats = await extract_data(
        primary,
        SimpleSchema,
        _messages(),
        fallback_llm_model=fallback,
        max_primary_retries=3,
        retry_config=_NO_NETWORK_RETRY,
        validator=_reject_low,
    )

    assert result is not None
    assert result.value == 999
    assert stats.fallback_used


# ---------------------------------------------------------------------------
# JSON mode
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_validator_json_mode() -> None:
    llm = _json_llm(
        [
            '{"name": "Alice", "value": 5}',
            '{"name": "Alice", "value": 150}',
        ]
    )

    result, stats = await extract_data(
        llm,
        SimpleSchema,
        _messages(),
        mode=ExtractionMode.JSON,
        retry_config=_NO_NETWORK_RETRY,
        validator=_reject_low,
    )

    assert result is not None
    assert result.value == 150
    assert stats.primary_retries == 1


# ---------------------------------------------------------------------------
# Batch — per-item validation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_validator_runs_per_item_in_batch() -> None:
    bad = _tool_response(
        {"items": [{"name": "A", "value": 150}, {"name": "B", "value": 5}]},
        name="SimpleSchemaList",
    )
    good = _tool_response(
        {"items": [{"name": "A", "value": 150}, {"name": "B", "value": 120}]},
        name="SimpleSchemaList",
    )
    llm = _bound_llm([bad, good])

    items, stats = await extract_data_list(
        llm,
        SimpleSchema,
        _messages(),
        retry_config=_NO_NETWORK_RETRY,
        validator=_reject_low,
    )

    assert items is not None
    assert [i.value for i in items] == [150, 120]
    assert stats.primary_retries == 1
    # The per-item message points at the offending index.
    assert any("items -> 1" in i.message for i in stats.field_issues)


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------


class FinalAnswer(BaseModel):
    result: str
    score: int = 0


def _agent_final(args: dict[str, Any], call_id: str = "c1") -> MagicMock:
    response = MagicMock()
    response.tool_calls = [{"name": "FinalAnswer", "args": args, "id": call_id}]
    response.invalid_tool_calls = []
    response.content = ""
    return response


def _agent_llm(responses: list[MagicMock]) -> MagicMock:
    bound = MagicMock()
    bound.ainvoke = AsyncMock(side_effect=responses)
    llm = MagicMock()
    llm.bind_tools = MagicMock(return_value=bound)
    llm.ainvoke = bound.ainvoke
    return llm


@pytest.mark.asyncio
async def test_validator_in_agent_loop_reject_then_pass() -> None:
    def reject_low_score(instance: FinalAnswer) -> None:
        if instance.score < 10:
            raise ValueError("score too low")

    llm = _agent_llm(
        [
            _agent_final({"result": "first", "score": 1}, "c1"),
            _agent_final({"result": "second", "score": 50}, "c2"),
        ]
    )

    result, stats = await extract_data_with_tools(
        llm,
        FinalAnswer,
        "do it",
        tools=[],
        retry_config=_NO_NETWORK_RETRY,
        validator=reject_low_score,
    )

    assert result is not None
    assert result.result == "second"
    assert stats.validation_retries == 1
    assert any(i.error_type == "external_validator" for i in stats.field_issues)


@pytest.mark.asyncio
async def test_validator_in_agent_loop_exhausts() -> None:
    def always_reject(instance: FinalAnswer) -> None:
        raise ValueError("never good enough")

    llm = _agent_llm([_agent_final({"result": "x", "score": 1}, f"c{i}") for i in range(10)])

    result, stats = await extract_data_with_tools(
        llm,
        FinalAnswer,
        "do it",
        tools=[],
        max_validation_retries=2,
        retry_config=_NO_NETWORK_RETRY,
        validator=always_reject,
    )

    assert result is None
    assert not stats.success


# ---------------------------------------------------------------------------
# Sync wrapper
# ---------------------------------------------------------------------------


def test_validator_sync_wrapper() -> None:
    llm = _bound_llm(
        [
            _tool_response({"name": "Alice", "value": 5}),
            _tool_response({"name": "Alice", "value": 150}),
        ]
    )

    result, stats = extract_data_sync(
        llm,
        SimpleSchema,
        _messages(),
        retry_config=_NO_NETWORK_RETRY,
        validator=_reject_low,
    )

    assert result is not None
    assert result.value == 150
    assert stats.primary_retries == 1


# ---------------------------------------------------------------------------
# ValidationIssue
# ---------------------------------------------------------------------------


def test_validation_issue_defaults() -> None:
    issue = ValidationIssue(message="bad")
    assert issue.field_path == "<external validator>"
    assert issue.category == "value"
    assert issue.error_type == "external_validator"
    assert issue.on_mismatch == "retry"
    assert issue.received is None


def test_validation_issue_rejects_unknown_on_mismatch() -> None:
    with pytest.raises(ValueError, match="on_mismatch"):
        ValidationIssue(message="bad", on_mismatch="ignore")  # type: ignore[arg-type]


def test_validation_issue_is_frozen() -> None:
    issue = ValidationIssue(message="bad")
    with pytest.raises(AttributeError):
        issue.message = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Structured issues — single shot
# ---------------------------------------------------------------------------


def _feedback_texts(llm: MagicMock) -> list[str]:
    """Human-turn texts the bound LLM saw on its last call."""
    bound = llm.bind_tools.return_value
    msgs = bound.ainvoke.call_args_list[-1].args[0]
    return [str(m.content) for m in msgs if isinstance(m, HumanMessage)]


def _two_rule_validator(instance: SimpleSchema) -> list[ValidationIssue]:
    if instance.value >= 100:
        return []
    return [
        ValidationIssue(
            message="value must be at least 100",
            field_path="value",
            error_type="min_value",
            received=str(instance.value),
        ),
        ValidationIssue(
            message="name must not be a placeholder",
            field_path="name",
            category="format",
            error_type="placeholder_name",
        ),
    ]


def _reject_then_accept_llm() -> MagicMock:
    return _bound_llm(
        [
            _tool_response({"name": "Alice", "value": 5}),
            _tool_response({"name": "Alice", "value": 150}),
        ]
    )


@pytest.mark.asyncio
async def test_structured_retry_issues_become_field_issues_and_one_retry() -> None:
    llm = _reject_then_accept_llm()

    result, stats = await extract_data(
        llm,
        SimpleSchema,
        _messages(),
        retry_config=_NO_NETWORK_RETRY,
        validator=_two_rule_validator,
    )

    assert result is not None and result.value == 150
    assert stats.primary_retries == 1
    assert [(i.field_path, i.category, i.error_type, i.attempt) for i in stats.field_issues] == [
        ("value", "value", "min_value", 0),
        ("name", "format", "placeholder_name", 0),
    ]
    assert stats.field_issues[0].received == "5"
    feedback = next(t for t in _feedback_texts(llm) if "additional validation check" in t)
    assert "  - 'value': value must be at least 100\n    Received: '5'" in feedback
    assert "  - 'name': name must not be a placeholder" in feedback


@pytest.mark.asyncio
async def test_flag_only_issues_are_recorded_without_retry() -> None:
    def advisory(instance: SimpleSchema) -> list[ValidationIssue]:
        return [ValidationIssue(message="unusually long", field_path="name", on_mismatch="flag")]

    llm = _bound_llm([_tool_response({"name": "Alice", "value": 5})])

    result, stats = await extract_data(
        llm, SimpleSchema, _messages(), retry_config=_NO_NETWORK_RETRY, validator=advisory
    )

    assert result is not None
    assert stats.primary_retries == 0
    assert [(i.field_path, i.message) for i in stats.field_issues] == [("name", "unusually long")]


@pytest.mark.asyncio
async def test_mixed_retry_and_flag_feedback_lists_only_retry() -> None:
    def mixed(instance: SimpleSchema) -> list[ValidationIssue]:
        flag = ValidationIssue(message="soft finding", field_path="name", on_mismatch="flag")
        if instance.value >= 100:
            return [flag]
        return [flag, ValidationIssue(message="too small", field_path="value")]

    llm = _reject_then_accept_llm()

    result, stats = await extract_data(
        llm, SimpleSchema, _messages(), retry_config=_NO_NETWORK_RETRY, validator=mixed
    )

    assert result is not None
    assert stats.primary_retries == 1
    # attempt 0: flag + retry; attempt 1: flag only
    assert [(i.message, i.attempt) for i in stats.field_issues] == [
        ("soft finding", 0),
        ("too small", 0),
        ("soft finding", 1),
    ]
    feedback = next(t for t in _feedback_texts(llm) if "additional validation check" in t)
    assert "too small" in feedback
    assert "soft finding" not in feedback


@pytest.mark.asyncio
async def test_empty_issue_list_is_accepted() -> None:
    llm = _bound_llm([_tool_response({"name": "Alice", "value": 5})])

    result, stats = await extract_data(
        llm,
        SimpleSchema,
        _messages(),
        retry_config=_NO_NETWORK_RETRY,
        validator=lambda _: [],
    )

    assert result is not None
    assert stats.field_issues == ()


@pytest.mark.asyncio
async def test_falsy_non_str_result_is_accepted() -> None:
    # v0.7.0 idiom: `x.bad and "msg"` returns False when everything is fine.
    llm = _bound_llm([_tool_response({"name": "Alice", "value": 150})])

    result, stats = await extract_data(
        llm,
        SimpleSchema,
        _messages(),
        retry_config=_NO_NETWORK_RETRY,
        validator=lambda inst: inst.value < 100 and "too small",
    )

    assert result is not None
    assert stats.primary_retries == 0


@pytest.mark.asyncio
async def test_single_issue_without_list_is_accepted() -> None:
    def single(instance: SimpleSchema) -> ValidationIssue | None:
        return None if instance.value >= 100 else ValidationIssue("too small", field_path="value")

    llm = _reject_then_accept_llm()

    result, stats = await extract_data(
        llm, SimpleSchema, _messages(), retry_config=_NO_NETWORK_RETRY, validator=single
    )

    assert result is not None
    assert [(i.field_path, i.message) for i in stats.field_issues] == [("value", "too small")]


@pytest.mark.asyncio
async def test_tuple_of_issues_accepted() -> None:
    def as_tuple(instance: SimpleSchema) -> tuple[ValidationIssue, ...]:
        if instance.value >= 100:
            return ()
        return (ValidationIssue("a", field_path="value"), ValidationIssue("b", field_path="name"))

    llm = _reject_then_accept_llm()

    result, stats = await extract_data(
        llm, SimpleSchema, _messages(), retry_config=_NO_NETWORK_RETRY, validator=as_tuple
    )

    assert result is not None
    assert [i.message for i in stats.field_issues] == ["a", "b"]


@pytest.mark.asyncio
async def test_async_validator_returning_issues() -> None:
    async def async_rules(instance: SimpleSchema) -> list[ValidationIssue]:
        return _two_rule_validator(instance)

    llm = _reject_then_accept_llm()

    result, stats = await extract_data(
        llm, SimpleSchema, _messages(), retry_config=_NO_NETWORK_RETRY, validator=async_rules
    )

    assert result is not None
    assert [i.field_path for i in stats.field_issues] == ["value", "name"]


@pytest.mark.asyncio
async def test_structured_issues_exhaust_retries_stamp_attempts() -> None:
    llm = _bound_llm([_tool_response({"name": "Alice", "value": 1})] * 3)

    result, stats = await extract_data(
        llm,
        SimpleSchema,
        _messages(),
        max_primary_retries=3,
        retry_config=_NO_NETWORK_RETRY,
        validator=_two_rule_validator,
    )

    assert result is None
    assert stats.failure_reason == "validation_exhausted"
    assert [i.attempt for i in stats.field_issues] == [0, 0, 1, 1, 2, 2]


@pytest.mark.parametrize(
    "bad_result",
    [True, {"field": "x"}, ["plain string in a list"], [ValidationIssue("ok"), 42]],
)
@pytest.mark.asyncio
async def test_invalid_result_type_raises_type_error(bad_result: object) -> None:
    llm = _bound_llm([_tool_response({"name": "Alice", "value": 5})])

    with pytest.raises(TypeError, match="external validator"):
        await extract_data(
            llm,
            SimpleSchema,
            _messages(),
            retry_config=_NO_NETWORK_RETRY,
            validator=lambda _: bad_result,  # type: ignore[arg-type,return-value]
        )


@pytest.mark.asyncio
async def test_legacy_string_feedback_text_unchanged() -> None:
    llm = _reject_then_accept_llm()

    await extract_data(
        llm,
        SimpleSchema,
        _messages(),
        retry_config=_NO_NETWORK_RETRY,
        validator=lambda inst: "too small" if inst.value < 100 else None,
    )

    assert (
        "The extracted data is structurally valid but failed an additional "
        "validation check:\n\ntoo small\n\n"
        "Return a corrected response that resolves the problem above."
    ) in _feedback_texts(llm)


@pytest.mark.asyncio
async def test_structured_issues_from_text() -> None:
    llm = _reject_then_accept_llm()

    result, stats = await extract_data_from_text(
        llm,
        SimpleSchema,
        "Alice has 150 points.",
        retry_config=_NO_NETWORK_RETRY,
        validator=_two_rule_validator,
    )

    assert result is not None
    assert {i.field_path for i in stats.field_issues} == {"value", "name"}


# ---------------------------------------------------------------------------
# Structured issues — agent loop
# ---------------------------------------------------------------------------


def _score_rules(instance: FinalAnswer) -> list[ValidationIssue]:
    issues = [ValidationIssue("result is terse", field_path="result", on_mismatch="flag")]
    if instance.score < 10:
        issues.append(ValidationIssue("score too low", field_path="score", error_type="min"))
    return issues


@pytest.mark.asyncio
async def test_agent_tool_calling_structured_issues() -> None:
    llm = _agent_llm(
        [
            _agent_final({"result": "first", "score": 1}, "c1"),
            _agent_final({"result": "second", "score": 50}, "c2"),
        ]
    )

    result, stats = await extract_data_with_tools(
        llm,
        FinalAnswer,
        "do it",
        tools=[],
        retry_config=_NO_NETWORK_RETRY,
        validator=_score_rules,
    )

    assert result is not None and result.result == "second"
    assert stats.validation_retries == 1
    assert [(i.field_path, i.attempt) for i in stats.field_issues] == [
        ("result", 0),
        ("score", 0),
        ("result", 1),
    ]


@pytest.mark.asyncio
async def test_agent_tool_calling_flag_only_accepts_first_answer() -> None:
    llm = _agent_llm([_agent_final({"result": "first", "score": 50}, "c1")])

    result, stats = await extract_data_with_tools(
        llm,
        FinalAnswer,
        "do it",
        tools=[],
        retry_config=_NO_NETWORK_RETRY,
        validator=_score_rules,
    )

    assert result is not None and result.result == "first"
    assert stats.validation_retries == 0
    assert [i.field_path for i in stats.field_issues] == ["result"]


@pytest.mark.asyncio
async def test_agent_json_mode_structured_issues() -> None:
    llm = _agent_llm(
        [
            _json_response('{"result": "first", "score": 1}'),
            _json_response('{"result": "second", "score": 50}'),
        ]
    )

    result, stats = await extract_data_with_tools(
        llm,
        FinalAnswer,
        "do it",
        tools=[],
        final_answer_mode=ExtractionMode.JSON,
        retry_config=_NO_NETWORK_RETRY,
        validator=_score_rules,
    )

    assert result is not None and result.result == "second"
    assert stats.validation_retries == 1
    assert [(i.field_path, i.attempt) for i in stats.field_issues] == [
        ("result", 0),
        ("score", 0),
        ("result", 1),
    ]


@pytest.mark.asyncio
async def test_agent_json_mode_flag_only_accepts_first_answer() -> None:
    llm = _agent_llm([_json_response('{"result": "first", "score": 50}')])

    result, stats = await extract_data_with_tools(
        llm,
        FinalAnswer,
        "do it",
        tools=[],
        final_answer_mode=ExtractionMode.JSON,
        retry_config=_NO_NETWORK_RETRY,
        validator=_score_rules,
    )

    assert result is not None and result.result == "first"
    assert stats.validation_retries == 0
