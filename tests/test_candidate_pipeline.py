"""Characterization + parity tests for the shared candidate-check pipeline.

The same two-step scenario — attempt 1 fails source grounding, attempt 2 fails
the external validator, attempt 3 passes — runs through every final-answer
path. All four paths must record the same issues with the same attempt stamps,
and each must keep its own feedback wording.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import HumanMessage, ToolMessage
from pydantic import BaseModel

from saidex import (
    ExtractionMode,
    GroundedStr,
    ValidationIssue,
    extract_data,
    extract_data_with_tools,
)
from saidex.grounding import Grounded, collect_field_check_issues, field_check
from saidex.retry import RetryConfig
from tests._mock_llm import make_llm, make_response

_NO_RETRY = RetryConfig(max_retries=0, retry_delays=[], retryable_exceptions=())
SOURCE = "Invoice from ACME, amount 5."


class Doc(BaseModel):
    vendor: GroundedStr
    amount: int


class FlagDoc(BaseModel):
    vendor: str = field_check(Grounded(on_mismatch="flag"))
    amount: int


BAD_VENDOR = {"vendor": "Globex", "amount": 5}
SMALL_AMOUNT = {"vendor": "ACME", "amount": 5}
GOOD = {"vendor": "ACME", "amount": 50}


def _min_amount(doc: Any) -> list[ValidationIssue]:
    if doc.amount >= 10:
        return []
    return [
        ValidationIssue(
            message="amount must be at least 10", field_path="amount", error_type="too_small"
        )
    ]


GROUNDING_FEEDBACK = collect_field_check_issues(Doc(**BAD_VENDOR), SOURCE, "Doc")[1]
VALIDATOR_FEEDBACK = "  - 'amount': amount must be at least 10"
EXPECTED_ISSUES = [
    ("vendor", "grounding", "grounding", 0),
    ("amount", "value", "too_small", 1),
]


def _tool_resp(args: dict[str, Any], call_id: str) -> MagicMock:
    return make_response([{"name": "Doc", "args": args, "id": call_id}])


def _json_resp(args: dict[str, Any]) -> MagicMock:
    return make_response(content=json.dumps(args))


def _last_messages(llm: MagicMock) -> list[Any]:
    return list(llm.bind_tools.return_value.ainvoke.call_args_list[-1].args[0])


def _issue_keys(stats: Any) -> list[tuple[str, str, str, int]]:
    return [(i.field_path, i.category, i.error_type, i.attempt) for i in stats.field_issues]


async def test_extract_data_tool_calling() -> None:
    llm = make_llm(
        [_tool_resp(BAD_VENDOR, "c1"), _tool_resp(SMALL_AMOUNT, "c2"), _tool_resp(GOOD, "c3")]
    )
    result, stats = await extract_data(
        llm, Doc, [HumanMessage(content=SOURCE)], retry_config=_NO_RETRY, validator=_min_amount
    )
    assert result is not None and result.amount == 50
    assert _issue_keys(stats) == EXPECTED_ISSUES
    texts = [str(m.content) for m in _last_messages(llm) if isinstance(m, HumanMessage)]
    assert GROUNDING_FEEDBACK in texts
    assert (
        "The extracted data is structurally valid but failed an additional validation check:"
        f"\n\n{VALIDATOR_FEEDBACK}\n\n"
        "Return a corrected response that resolves the problem above."
    ) in texts


async def test_extract_data_json_mode() -> None:
    llm = make_llm([_json_resp(BAD_VENDOR), _json_resp(SMALL_AMOUNT), _json_resp(GOOD)])
    result, stats = await extract_data(
        llm,
        Doc,
        [HumanMessage(content=SOURCE)],
        mode=ExtractionMode.JSON,
        retry_config=_NO_RETRY,
        validator=_min_amount,
    )
    assert result is not None
    assert _issue_keys(stats) == EXPECTED_ISSUES


async def test_agent_json_final_answer() -> None:
    llm = make_llm([_json_resp(BAD_VENDOR), _json_resp(SMALL_AMOUNT), _json_resp(GOOD)])
    result, stats = await extract_data_with_tools(
        llm,
        Doc,
        SOURCE,
        tools=[],
        final_answer_mode=ExtractionMode.JSON,
        retry_config=_NO_RETRY,
        validator=_min_amount,
    )
    assert result is not None
    assert _issue_keys(stats) == EXPECTED_ISSUES
    texts = [str(m.content) for m in _last_messages(llm) if isinstance(m, HumanMessage)]
    assert GROUNDING_FEEDBACK in texts
    assert (
        "The JSON output is structurally valid but failed an additional validation check:"
        f"\n\n{VALIDATOR_FEEDBACK}\n\n"
        "Please correct it and return only the fixed JSON object."
    ) in texts


async def test_agent_tool_calling_final_answer() -> None:
    llm = make_llm(
        [_tool_resp(BAD_VENDOR, "c1"), _tool_resp(SMALL_AMOUNT, "c2"), _tool_resp(GOOD, "c3")]
    )
    result, stats = await extract_data_with_tools(
        llm, Doc, SOURCE, tools=[], retry_config=_NO_RETRY, validator=_min_amount
    )
    assert result is not None
    assert _issue_keys(stats) == EXPECTED_ISSUES
    texts = [str(m.content) for m in _last_messages(llm) if isinstance(m, ToolMessage)]
    assert f"{GROUNDING_FEEDBACK}\n\nPlease call 'Doc' again with corrected values." in texts
    assert (
        "The final answer is structurally valid but failed an additional validation check:"
        f"\n\n{VALIDATOR_FEEDBACK}\n\n"
        "Please call 'Doc' again with corrected values."
    ) in texts


async def test_agent_tool_at_limit_texts_unchanged() -> None:
    # max_validation_retries=1: the second rejection happens at the budget limit.
    llm = make_llm([_tool_resp(BAD_VENDOR, "c1"), _tool_resp(SMALL_AMOUNT, "c2")])
    await extract_data_with_tools(
        llm,
        Doc,
        SOURCE,
        tools=[],
        max_validation_retries=1,
        retry_config=_NO_RETRY,
        validator=_min_amount,
    )
    # The loop mutates one message list in place, so the last call's list holds
    # every ToolMessage, including the one appended after the final LLM call.
    texts = [str(m.content) for m in _last_messages(llm) if isinstance(m, ToolMessage)]
    assert f"{GROUNDING_FEEDBACK}\n\nPlease call 'Doc' again with corrected values." in texts
    assert f"External validation failed: {VALIDATOR_FEEDBACK}" in texts


async def test_validator_skipped_when_grounding_rejects() -> None:
    seen: list[str] = []

    def record(doc: Any) -> None:
        seen.append(doc.vendor)

    llm = make_llm([_tool_resp(BAD_VENDOR, "c1"), _tool_resp(GOOD, "c2")])
    await extract_data(
        llm, Doc, [HumanMessage(content=SOURCE)], retry_config=_NO_RETRY, validator=record
    )
    assert seen == ["ACME"]


@pytest.mark.parametrize("agent", [False, True])
async def test_flag_grounding_and_validator_rejection_both_recorded(agent: bool) -> None:
    responses = [
        make_response([{"name": "FlagDoc", "args": BAD_VENDOR, "id": "c1"}]),
        make_response([{"name": "FlagDoc", "args": {"vendor": "ACME", "amount": 50}, "id": "c2"}]),
    ]
    llm = make_llm(responses)
    if agent:
        _, stats = await extract_data_with_tools(
            llm, FlagDoc, SOURCE, tools=[], retry_config=_NO_RETRY, validator=_min_amount
        )
    else:
        _, stats = await extract_data(
            llm,
            FlagDoc,
            [HumanMessage(content=SOURCE)],
            retry_config=_NO_RETRY,
            validator=_min_amount,
        )
    assert _issue_keys(stats)[:2] == [
        ("vendor", "grounding", "grounding", 0),
        ("amount", "value", "too_small", 0),
    ]
