"""Tests for ModelErrorKind / classify_model_error (provider-agnostic error kinds)."""

from __future__ import annotations

import asyncio

import httpx
import openai
import pytest
from langchain_core import exceptions as lc

import saidex
from saidex import ModelErrorKind, classify_model_error
from saidex import retry as retry_mod

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
