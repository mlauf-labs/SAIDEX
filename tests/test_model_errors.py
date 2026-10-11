"""Tests for ModelErrorKind / classify_model_error (provider-agnostic error kinds)."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock

import httpx
import openai
import pytest
from langchain_core import exceptions as lc
from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

import saidex
from saidex import (
    ExtractorRunStats,
    FailureReason,
    ModelErrorKind,
    classify_model_error,
    extract_data,
    extract_data_list,
    extract_data_with_tools,
)
from saidex import retry as retry_mod
from saidex.retry import RetryConfig, with_retry
from tests._mock_llm import make_llm, make_response

# The CI lowest-direct job re-runs the core suite on langchain-core 0.3.69, which
# has no standard model exceptions; the missing-module path is covered by
# test_missing_optional_modules_are_skipped via monkeypatching on newer versions.
if not hasattr(lc, "ModelError"):
    pytest.skip("langchain-core < 1.6 has no standard model exceptions", allow_module_level=True)


def _bare(cls: type[BaseException]) -> BaseException:
    """Instantiate an exception class without calling its (SDK-specific) __init__."""
    return cls.__new__(cls)


@pytest.mark.parametrize(
    ("cls", "kind"),
    [
        (lc.ModelRateLimitError, ModelErrorKind.RATE_LIMIT),
        (lc.ModelTimeoutError, ModelErrorKind.TIMEOUT),
        (lc.ModelConnectionError, ModelErrorKind.CONNECTION),
        (lc.ModelAPIError, ModelErrorKind.SERVER_ERROR),
        (lc.ModelAuthenticationError, ModelErrorKind.AUTHENTICATION),
        (lc.ModelPermissionDeniedError, ModelErrorKind.PERMISSION_DENIED),
        (lc.ModelInvalidRequestError, ModelErrorKind.INVALID_REQUEST),
        (lc.ModelNotFoundError, ModelErrorKind.NOT_FOUND),
        (lc.ContextOverflowError, ModelErrorKind.CONTEXT_OVERFLOW),
        (openai.RateLimitError, ModelErrorKind.RATE_LIMIT),
        (openai.APITimeoutError, ModelErrorKind.TIMEOUT),
        (openai.APIConnectionError, ModelErrorKind.CONNECTION),
        (openai.InternalServerError, ModelErrorKind.SERVER_ERROR),
        (openai.AuthenticationError, ModelErrorKind.AUTHENTICATION),
        (openai.PermissionDeniedError, ModelErrorKind.PERMISSION_DENIED),
        (openai.BadRequestError, ModelErrorKind.INVALID_REQUEST),
        (openai.NotFoundError, ModelErrorKind.NOT_FOUND),
        (httpx.ConnectError, ModelErrorKind.CONNECTION),
        (httpx.ReadTimeout, ModelErrorKind.TIMEOUT),
        (TimeoutError, ModelErrorKind.TIMEOUT),
        (asyncio.TimeoutError, ModelErrorKind.TIMEOUT),
        (ConnectionResetError, ModelErrorKind.CONNECTION),
        (RuntimeError, ModelErrorKind.OTHER),
        (ValueError, ModelErrorKind.OTHER),
    ],
)
def test_classify_model_error(cls: type[BaseException], kind: ModelErrorKind) -> None:
    assert classify_model_error(_bare(cls)) is kind


def test_openai_timeout_is_timeout_not_connection() -> None:
    assert issubclass(openai.APITimeoutError, openai.APIConnectionError)
    assert classify_model_error(_bare(openai.APITimeoutError)) is ModelErrorKind.TIMEOUT


def test_context_overflow_beats_bad_request() -> None:
    class OverflowLikeLangchainOpenAI(openai.BadRequestError, lc.ContextOverflowError):
        pass

    exc = _bare(OverflowLikeLangchainOpenAI)
    assert classify_model_error(exc) is ModelErrorKind.CONTEXT_OVERFLOW


def test_provider_subclass_of_both_sdk_and_core_class() -> None:
    class OpenAIRateLimitLike(openai.RateLimitError, lc.ModelRateLimitError):
        pass

    assert classify_model_error(_bare(OpenAIRateLimitLike)) is ModelErrorKind.RATE_LIMIT


def test_missing_optional_modules_are_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = retry_mod.importlib.import_module

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name in ("openai", "httpx", "httpx2") or name == "langchain_core.exceptions":
            raise ImportError(name)
        return real_import(name)

    monkeypatch.setattr(retry_mod.importlib, "import_module", fake_import)
    retry_mod._error_kind_table.cache_clear()
    try:
        assert classify_model_error(TimeoutError()) is ModelErrorKind.TIMEOUT
        assert classify_model_error(ConnectionError()) is ModelErrorKind.CONNECTION
        assert classify_model_error(RuntimeError()) is ModelErrorKind.OTHER
    finally:
        retry_mod._error_kind_table.cache_clear()


def test_model_error_kind_values_and_export() -> None:
    assert [k.value for k in ModelErrorKind] == [
        "rate_limit",
        "timeout",
        "connection",
        "server_error",
        "authentication",
        "permission_denied",
        "invalid_request",
        "not_found",
        "context_overflow",
        "other",
    ]
    assert "ModelErrorKind" in saidex.__all__
    assert "classify_model_error" in saidex.__all__


async def _fail_then_succeed(exc: BaseException, calls: list[int]) -> str:
    calls.append(1)
    if len(calls) == 1:
        raise exc
    return "ok"


@pytest.mark.parametrize(
    "cls",
    [lc.ModelAPIError, lc.ModelConnectionError, lc.ModelTimeoutError, openai.InternalServerError],
)
async def test_default_config_retries_retryable_errors(cls: type[BaseException]) -> None:
    config = RetryConfig(retry_delays=[0.0])
    calls: list[int] = []
    result = await with_retry(lambda: _fail_then_succeed(_bare(cls), calls), config)
    assert result.result == "ok" and result.retry_count == 1


@pytest.mark.parametrize(
    "cls",
    [
        lc.ModelAuthenticationError,
        lc.ModelInvalidRequestError,
        lc.ContextOverflowError,
        openai.BadRequestError,
    ],
)
async def test_default_config_does_not_retry_permanent_errors(cls: type[BaseException]) -> None:
    config = RetryConfig(retry_delays=[0.0])
    calls: list[int] = []
    with pytest.raises(cls):
        await with_retry(lambda: _fail_then_succeed(_bare(cls), calls), config)
    assert len(calls) == 1


def test_default_rate_limit_exceptions_include_core_class() -> None:
    config = RetryConfig()
    assert lc.ModelRateLimitError in config.rate_limit_exceptions
    assert openai.RateLimitError in config.rate_limit_exceptions


def test_default_retryable_includes_httpx2_when_installed() -> None:
    httpx2 = pytest.importorskip("httpx2")
    assert httpx2.ConnectError in RetryConfig().retryable_exceptions


_NO_RETRY = RetryConfig(
    max_retries=0, retry_delays=[], retryable_exceptions=(), rate_limit_exceptions=()
)


class Item(BaseModel):
    name: str
    value: int


def _msgs() -> list[HumanMessage]:
    return [HumanMessage(content="extract")]


def _item_call(value: Any = 1) -> Any:
    return make_response([{"name": "Item", "args": {"name": "a", "value": value}, "id": "c"}])


async def test_extract_data_records_kind_of_invocation_error() -> None:
    _, stats = await extract_data(
        make_llm([lc.ModelRateLimitError("slow down")]), Item, _msgs(), retry_config=_NO_RETRY
    )
    assert stats.failure_reason is FailureReason.LLM_ERROR
    assert stats.llm_error_kind is ModelErrorKind.RATE_LIMIT


async def test_bind_time_rejection_is_invalid_request() -> None:
    llm = MagicMock()
    llm.bind_tools = MagicMock(side_effect=TypeError("unexpected keyword 'strict'"))
    _, stats = await extract_data(llm, Item, _msgs(), retry_config=_NO_RETRY)
    assert stats.llm_error_kind is ModelErrorKind.INVALID_REQUEST


async def test_no_kind_on_success_or_non_llm_failure() -> None:
    _, ok = await extract_data(make_llm([_item_call()]), Item, _msgs(), retry_config=_NO_RETRY)
    _, invalid = await extract_data(
        make_llm([_item_call("x")]), Item, _msgs(), max_primary_retries=1, retry_config=_NO_RETRY
    )
    assert ok.llm_error_kind is None
    assert invalid.failure_reason is FailureReason.VALIDATION_EXHAUSTED
    assert invalid.llm_error_kind is None


async def test_kind_follows_the_combined_reason() -> None:
    _, stats = await extract_data(
        make_llm([lc.ModelRateLimitError("x")]),
        Item,
        _msgs(),
        fallback_llm_model=make_llm([_item_call("x")]),
        max_fallback_retries=1,
        retry_config=_NO_RETRY,
    )
    assert stats.failure_reason is FailureReason.VALIDATION_EXHAUSTED
    assert stats.llm_error_kind is None


async def test_fallback_llm_error_kind_wins() -> None:
    _, stats = await extract_data(
        make_llm([_item_call("x")]),
        Item,
        _msgs(),
        max_primary_retries=1,
        fallback_llm_model=make_llm([lc.ModelTimeoutError("t")]),
        retry_config=_NO_RETRY,
    )
    assert stats.failure_reason is FailureReason.LLM_ERROR
    assert stats.llm_error_kind is ModelErrorKind.TIMEOUT


async def test_salvaged_run_has_no_llm_error_kind() -> None:
    _, stats = await extract_data(
        make_llm([_item_call(), lc.ModelConnectionError("down")]),
        Item,
        _msgs(),
        max_primary_retries=2,
        retry_config=_NO_RETRY,
        validator=lambda item: "no",
        on_exhausted=lambda item, issues: item,
    )
    assert stats.salvaged and stats.failure_reason is None
    assert stats.llm_error_kind is None


async def test_list_variant_keeps_kind() -> None:
    _, stats = await extract_data_list(
        make_llm([lc.ModelNotFoundError("no model")]), Item, _msgs(), retry_config=_NO_RETRY
    )
    assert stats.llm_error_kind is ModelErrorKind.NOT_FOUND


async def test_agent_loop_records_kind() -> None:
    _, stats = await extract_data_with_tools(
        make_llm([lc.ModelAPIError("boom")]), Item, "task", tools=[], retry_config=_NO_RETRY
    )
    assert stats.failure_reason is FailureReason.LLM_ERROR
    assert stats.llm_error_kind is ModelErrorKind.SERVER_ERROR


async def test_agent_fallback_kind_follows_reason() -> None:
    _, stats = await extract_data_with_tools(
        make_llm([lc.ModelAuthenticationError("no")]),
        Item,
        "task",
        tools=[],
        fallback_llm_model=make_llm([lc.ModelTimeoutError("t")]),
        retry_config=_NO_RETRY,
    )
    assert stats.llm_error_kind is ModelErrorKind.TIMEOUT


def test_add_combines_kinds() -> None:
    a = ExtractorRunStats(
        failure_reason=FailureReason.LLM_ERROR, llm_error_kind=ModelErrorKind.RATE_LIMIT
    )
    b = ExtractorRunStats(failure_reason=FailureReason.VALIDATION_EXHAUSTED)
    assert (a + b).llm_error_kind is None
    assert (b + a).llm_error_kind is ModelErrorKind.RATE_LIMIT
    assert (a + ExtractorRunStats(success=True)).llm_error_kind is None


async def test_trace_output_kind_is_plain_str() -> None:
    class Capture(AsyncCallbackHandler):
        def __init__(self) -> None:
            self.outputs: list[Any] = []

        async def on_chain_end(self, outputs: Any, **kw: Any) -> None:
            self.outputs.append(outputs)

    handler = Capture()
    await extract_data(
        make_llm([lc.ModelRateLimitError("x")]),
        Item,
        _msgs(),
        retry_config=_NO_RETRY,
        callbacks=[handler],
    )
    kind = handler.outputs[-1]["llm_error_kind"]
    assert type(kind) is str and kind == "rate_limit"
