"""Tests for reasoning-effort escalation across all extraction entry points."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from saidex import (
    ValidatorContext,
    extract_data,
    extract_data_list,
    extract_data_sync,
    with_context,
)
from saidex._reasoning import (
    _level_for_attempt,
    _profile_levels,
    _resolve_reasoning_levels,
    _with_reasoning_effort,
)
from saidex.retry import RetryConfig
from tests._mock_llm import make_llm, make_response


def _model(levels: list[str] | None = None) -> Any:
    model = MagicMock()
    model.profile = None if levels is None else {"reasoning_effort_levels": levels}
    return model


def test_none_means_no_levels() -> None:
    assert _resolve_reasoning_levels(_model(["low", "high"]), None, param="p") == ()


def test_explicit_list_without_profile_is_passed_through() -> None:
    assert _resolve_reasoning_levels(_model(), ["low", "xhigh"], param="p") == ("low", "xhigh")


def test_true_uses_profile_levels() -> None:
    assert _resolve_reasoning_levels(_model(["low", "medium", "high"]), True, param="p") == (
        "low",
        "medium",
        "high",
    )


def test_true_without_profile_levels_disables_escalation(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level("INFO", logger="saidex._reasoning")
    assert _resolve_reasoning_levels(_model(), True, param="reasoning_escalation") == ()
    assert _resolve_reasoning_levels(_model([]), True, param="reasoning_escalation") == ()
    assert any("reasoning_escalation" in r.message for r in caplog.records)


def test_unknown_level_with_profile_raises() -> None:
    with pytest.raises(ValueError, match="'turbo'.*low.*high"):
        _resolve_reasoning_levels(_model(["low", "high"]), ["low", "turbo"], param="p")


def test_empty_list_raises() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        _resolve_reasoning_levels(_model(), [], param="p")


def test_plain_string_is_rejected() -> None:
    with pytest.raises(ValueError, match="list"):
        _resolve_reasoning_levels(_model(), "high", param="p")  # type: ignore[arg-type]


def test_non_string_level_is_rejected() -> None:
    with pytest.raises(ValueError):
        _resolve_reasoning_levels(_model(), ["low", 3], param="p")  # type: ignore[list-item]


def test_profile_levels_tolerates_missing_or_odd_profiles() -> None:
    assert _profile_levels(object()) == ()
    assert _profile_levels(_model()) == ()
    weird = MagicMock()
    weird.profile = {"reasoning_effort_levels": "high"}
    assert _profile_levels(weird) == ()


@pytest.mark.parametrize(
    ("index", "expected"), [(0, "low"), (1, "medium"), (2, "high"), (3, "high"), (9, "high")]
)
def test_level_for_attempt(index: int, expected: str) -> None:
    assert _level_for_attempt(("low", "medium", "high"), index) == expected


def test_level_for_attempt_without_levels() -> None:
    assert _level_for_attempt((), 0) is None


def test_with_reasoning_effort_binds_only_when_set() -> None:
    runnable = MagicMock()
    assert _with_reasoning_effort(runnable, None) is runnable
    runnable.bind.assert_not_called()
    _with_reasoning_effort(runnable, "high")
    runnable.bind.assert_called_once_with(reasoning_effort="high")


@pytest.mark.parametrize("bad", [1, False, 2.0])
def test_non_sequence_spec_is_rejected(bad: Any) -> None:
    with pytest.raises(ValueError):
        _resolve_reasoning_levels(_model(), bad, param="p")


_NO_RETRY = RetryConfig(
    max_retries=0, retry_delays=[], retryable_exceptions=(), rate_limit_exceptions=()
)


class Item(BaseModel):
    name: str
    value: int


def _msgs() -> list[HumanMessage]:
    return [HumanMessage(content="extract")]


def _item(value: Any = 1, call_id: str = "c") -> Any:
    return make_response([{"name": "Item", "args": {"name": "a", "value": value}, "id": call_id}])


def _efforts(llm: Any) -> list[str]:
    """reasoning_effort values bound on the tool-bound model, in call order."""
    return [c.kwargs["reasoning_effort"] for c in llm.bind_tools.return_value.bind.call_args_list]


def _reject(item: Item) -> str:
    return "no"


async def test_no_escalation_by_default() -> None:
    llm = make_llm([_item()])
    await extract_data(llm, Item, _msgs(), retry_config=_NO_RETRY)
    llm.bind_tools.return_value.bind.assert_not_called()


async def test_levels_escalate_per_attempt_and_last_repeats() -> None:
    llm = make_llm([_item(call_id=f"c{i}") for i in range(4)])
    _, stats = await extract_data(
        llm,
        Item,
        _msgs(),
        max_primary_retries=4,
        retry_config=_NO_RETRY,
        validator=_reject,
        reasoning_escalation=["low", "medium", "high"],
    )
    assert _efforts(llm) == ["low", "medium", "high", "high"]
    assert stats.reasoning_effort == "high"


async def test_parse_error_consumes_a_level() -> None:
    llm = make_llm([make_response(invalid=True), _item()])
    _, stats = await extract_data(
        llm, Item, _msgs(), retry_config=_NO_RETRY, reasoning_escalation=["low", "high"]
    )
    assert _efforts(llm) == ["low", "high"]
    assert stats.success and stats.reasoning_effort == "high"


async def test_true_uses_the_profile() -> None:
    llm = make_llm([_item(call_id="a"), _item(call_id="b")])
    llm.profile = {"reasoning_effort_levels": ["minimal", "low"]}
    await extract_data(
        llm,
        Item,
        _msgs(),
        max_primary_retries=2,
        retry_config=_NO_RETRY,
        validator=_reject,
        reasoning_escalation=True,
    )
    assert _efforts(llm) == ["minimal", "low"]


async def test_fallback_only_escalation() -> None:
    primary = make_llm([_item(call_id="p")])
    fallback = make_llm([_item(call_id="f1"), _item(call_id="f2")])
    _, stats = await extract_data(
        primary,
        Item,
        _msgs(),
        max_primary_retries=1,
        fallback_llm_model=fallback,
        max_fallback_retries=2,
        retry_config=_NO_RETRY,
        validator=_reject,
        fallback_reasoning_escalation=["medium", "high"],
    )
    primary.bind_tools.return_value.bind.assert_not_called()
    assert _efforts(fallback) == ["medium", "high"]
    assert stats.reasoning_effort == "high"


async def test_fallback_does_not_inherit_primary_levels() -> None:
    primary = make_llm([_item(call_id="p")])
    fallback = make_llm([_item(call_id="f")])
    _, stats = await extract_data(
        primary,
        Item,
        _msgs(),
        max_primary_retries=1,
        fallback_llm_model=fallback,
        max_fallback_retries=1,
        retry_config=_NO_RETRY,
        validator=_reject,
        reasoning_escalation=["high"],
    )
    assert _efforts(primary) == ["high"]
    fallback.bind_tools.return_value.bind.assert_not_called()
    assert stats.reasoning_effort is None


async def test_invalid_fallback_spec_fails_before_any_call() -> None:
    primary = make_llm([_item()])
    fallback = make_llm([])
    fallback.profile = {"reasoning_effort_levels": ["low", "high"]}
    with pytest.raises(ValueError, match="fallback_reasoning_escalation"):
        await extract_data(
            primary,
            Item,
            _msgs(),
            fallback_llm_model=fallback,
            retry_config=_NO_RETRY,
            fallback_reasoning_escalation=["turbo"],
        )
    primary.bind_tools.return_value.ainvoke.assert_not_called()


async def test_relaxed_rebind_keeps_the_level() -> None:
    rejection = TypeError("unexpected keyword argument 'strict'")
    llm = make_llm([rejection, _item()])
    await extract_data(llm, Item, _msgs(), retry_config=_NO_RETRY, reasoning_escalation=["high"])
    assert _efforts(llm) == ["high", "high"]


async def test_validator_context_carries_the_level() -> None:
    seen: list[str | None] = []

    def record(item: Item, ctx: ValidatorContext) -> str:
        seen.append(ctx.reasoning_effort)
        return "no"

    await extract_data(
        make_llm([_item(call_id="a"), _item(call_id="b")]),
        Item,
        _msgs(),
        max_primary_retries=2,
        retry_config=_NO_RETRY,
        validator=with_context(record),
        reasoning_escalation=["low", "high"],
    )
    assert seen == ["low", "high"]


async def test_list_variant_forwards() -> None:
    args = {"items": [{"name": "a", "value": 1}]}
    llm = make_llm([make_response([{"name": "ItemList", "args": args, "id": "c"}])])
    _, stats = await extract_data_list(
        llm, Item, _msgs(), retry_config=_NO_RETRY, reasoning_escalation=["low"]
    )
    assert _efforts(llm) == ["low"] and stats.reasoning_effort == "low"


async def test_json_mode_binds_the_raw_model() -> None:
    from saidex import ExtractionMode

    llm = make_llm([make_response(content='{"name": "a", "value": 1}')])
    await extract_data(
        llm,
        Item,
        _msgs(),
        mode=ExtractionMode.JSON,
        retry_config=_NO_RETRY,
        reasoning_escalation=["low"],
    )
    llm.bind.assert_called_once_with(reasoning_effort="low")


def test_sync_wrapper_forwards() -> None:
    llm = make_llm([_item()])
    _, stats = extract_data_sync(
        llm, Item, _msgs(), retry_config=_NO_RETRY, reasoning_escalation=["low"]
    )
    assert stats.reasoning_effort == "low"
