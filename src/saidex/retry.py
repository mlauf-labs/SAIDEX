"""Async retry utilities for LLM network calls."""

from __future__ import annotations

import asyncio
import functools
import importlib
import logging
import time
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

from .models import ModelErrorKind

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass(frozen=True)
class RetryResult(Generic[T]):
    """Result of a retried operation.

    Attributes:
        result: The successful return value.
        retry_count: How many retries were needed (0 = succeeded on first attempt).
    """

    result: T
    retry_count: int = 0


def _optional_classes(module: str, *names: str) -> tuple[type[BaseException], ...]:
    """Return the exception classes *names* from *module*, skipping what is missing.

    Optional providers (``openai``, ``httpx``, ``httpx2``) and newer
    langchain-core classes may not be installed; a missing module or name is
    simply left out.
    """
    try:
        mod = importlib.import_module(module)
    except ImportError:
        return ()
    found: list[type[BaseException]] = []
    for name in names:
        cls = getattr(mod, name, None)
        if isinstance(cls, type) and issubclass(cls, BaseException):
            found.append(cls)
    return tuple(found)


@functools.cache
def _error_kind_table() -> tuple[tuple[ModelErrorKind, tuple[type[BaseException], ...]], ...]:
    """Ordered ``(kind, classes)`` rows; the first matching row wins.

    Order matters: context overflow before invalid request (langchain-openai's
    overflow error is also an ``openai.BadRequestError``) and timeout before
    connection (``openai.APITimeoutError`` subclasses ``APIConnectionError``).
    """
    lc = "langchain_core.exceptions"
    return (
        (ModelErrorKind.CONTEXT_OVERFLOW, _optional_classes(lc, "ContextOverflowError")),
        (
            ModelErrorKind.RATE_LIMIT,
            _optional_classes(lc, "ModelRateLimitError")
            + _optional_classes("openai", "RateLimitError"),
        ),
        (
            ModelErrorKind.TIMEOUT,
            _optional_classes(lc, "ModelTimeoutError")
            + _optional_classes("openai", "APITimeoutError")
            + _optional_classes("httpx", "TimeoutException")
            + _optional_classes("httpx2", "TimeoutException")
            + (TimeoutError,),
        ),
        (
            ModelErrorKind.CONNECTION,
            _optional_classes(lc, "ModelConnectionError")
            + _optional_classes("openai", "APIConnectionError")
            + _optional_classes("httpx", "ConnectError", "RemoteProtocolError", "NetworkError")
            + _optional_classes("httpx2", "ConnectError", "RemoteProtocolError", "NetworkError")
            + (ConnectionError,),
        ),
        (
            ModelErrorKind.SERVER_ERROR,
            _optional_classes(lc, "ModelAPIError")
            + _optional_classes("openai", "InternalServerError"),
        ),
        (
            ModelErrorKind.AUTHENTICATION,
            _optional_classes(lc, "ModelAuthenticationError")
            + _optional_classes("openai", "AuthenticationError"),
        ),
        (
            ModelErrorKind.PERMISSION_DENIED,
            _optional_classes(lc, "ModelPermissionDeniedError")
            + _optional_classes("openai", "PermissionDeniedError"),
        ),
        (
            ModelErrorKind.NOT_FOUND,
            _optional_classes(lc, "ModelNotFoundError")
            + _optional_classes("openai", "NotFoundError"),
        ),
        (
            ModelErrorKind.INVALID_REQUEST,
            _optional_classes(lc, "ModelInvalidRequestError")
            + _optional_classes("openai", "BadRequestError"),
        ),
    )


def classify_model_error(exc: BaseException) -> ModelErrorKind:
    """Map a model-call exception to a provider-independent :class:`ModelErrorKind`.

    Uses langchain-core's standard model exceptions (langchain-core ≥ 1.6), which
    provider integrations raise alongside their SDK's own types, and falls back
    to the ``openai`` / ``httpx`` / ``httpx2`` types and builtin
    ``TimeoutError`` / ``ConnectionError``.  Modules that are not installed are
    skipped.

    Args:
        exc: The exception raised by a chat-model call.

    Returns:
        The matching kind, or :attr:`ModelErrorKind.OTHER`.
    """
    for kind, classes in _error_kind_table():
        if classes and isinstance(exc, classes):
            return kind
    return ModelErrorKind.OTHER


def _default_retryable_exceptions() -> tuple[type[Exception], ...]:
    """Return the default transient (retry-with-backoff) exception types.

    Follows langchain-core's ``is_retryable`` model exceptions (langchain-core
    >= 1.6: connection, timeout and 5xx server errors) plus the matching
    ``openai`` / ``httpx`` / ``httpx2`` types for older versions and direct SDK
    use.  Missing packages are skipped, so the library stays dependency-free.
    """
    network = ("ConnectError", "TimeoutException", "RemoteProtocolError")
    classes = (
        _optional_classes("httpx", *network)
        + _optional_classes("httpx2", *network)
        + _optional_classes(
            "openai", "APIConnectionError", "APITimeoutError", "InternalServerError"
        )
        + _optional_classes(
            "langchain_core.exceptions",
            "ModelConnectionError",
            "ModelTimeoutError",
            "ModelAPIError",
        )
    )
    return tuple(c for c in classes if issubclass(c, Exception))


def _default_rate_limit_exceptions() -> tuple[type[Exception], ...]:
    """Return the default rate-limit exception types (openai and langchain-core >= 1.6)."""
    classes = _optional_classes("openai", "RateLimitError") + _optional_classes(
        "langchain_core.exceptions", "ModelRateLimitError"
    )
    return tuple(c for c in classes if issubclass(c, Exception))


@dataclass
class RetryConfig:
    """Configuration for network-level retries inside LLM calls.

    Two independent retry strategies are applied:

    * **Transient errors** (connection resets, timeouts): retry up to
      ``max_retries`` times with the delays listed in ``retry_delays``.
    * **Rate-limit errors**: retry indefinitely (within
      ``rate_limit_max_duration_seconds``) every ``rate_limit_retry_interval``
      seconds.

    Attributes:
        max_retries: Maximum attempts for transient errors.
        retry_delays: Per-attempt sleep durations in seconds.  The last value
            is reused when the retry count exceeds the list length.
        retryable_exceptions: Exception types that trigger a transient retry.
            Defaults to langchain-core's standard model errors
            (``ModelConnectionError``, ``ModelTimeoutError``, ``ModelAPIError``
            for 5xx) plus common ``httpx`` / ``httpx2`` / ``openai`` network and
            server errors, for whichever packages are installed.
        rate_limit_exceptions: Exception types that trigger rate-limit retry
            logic.  Defaults to langchain-core's ``ModelRateLimitError`` and
            ``openai.RateLimitError`` for whichever are installed.
        rate_limit_retry_interval: Seconds to wait between rate-limit retries.
        rate_limit_max_duration_seconds: Total seconds before giving up on
            rate-limit retries and re-raising the exception.
    """

    max_retries: int = 3
    retry_delays: list[float] = field(default_factory=lambda: [1.0, 2.0, 4.0])
    retryable_exceptions: tuple[type[Exception], ...] = field(
        default_factory=_default_retryable_exceptions
    )
    rate_limit_exceptions: tuple[type[Exception], ...] = field(
        default_factory=_default_rate_limit_exceptions
    )
    rate_limit_retry_interval: float = 10.0
    rate_limit_max_duration_seconds: float = 900.0


#: Ready-to-use default configuration: langchain-core model errors, 5xx, openai, httpx/httpx2.
DEFAULT_RETRY_CONFIG = RetryConfig()


async def with_retry(
    operation: Callable[[], Coroutine[Any, Any, T]],
    config: RetryConfig,
    operation_name: str = "operation",
) -> RetryResult[T]:
    """Execute *operation* with automatic retries according to *config*.

    Args:
        operation: An async callable that takes no arguments and returns ``T``.
        config: Retry behaviour settings.
        operation_name: Human-readable label used in log messages.

    Returns:
        A :class:`RetryResult` containing the successful result and the number
        of retries that were needed.

    Raises:
        The original exception when all retry attempts are exhausted or the
        exception type is not retryable.
    """
    retry_count = 0
    rate_limit_start: float | None = None

    while True:
        try:
            result = await operation()
            return RetryResult(result=result, retry_count=retry_count)

        except Exception as exc:  # noqa: BLE001
            # --- Rate-limit handling ---
            if config.rate_limit_exceptions and isinstance(exc, config.rate_limit_exceptions):
                if rate_limit_start is None:
                    rate_limit_start = time.monotonic()
                elapsed = time.monotonic() - rate_limit_start
                if elapsed >= config.rate_limit_max_duration_seconds:
                    logger.error(
                        "%s: rate-limit retry budget exhausted after %.0fs — giving up.",
                        operation_name,
                        elapsed,
                    )
                    raise
                logger.warning(
                    "%s: rate limit hit (%.0fs elapsed); retrying in %.0fs …",
                    operation_name,
                    elapsed,
                    config.rate_limit_retry_interval,
                )
                await asyncio.sleep(config.rate_limit_retry_interval)
                continue

            # --- Transient-error handling ---
            if config.retryable_exceptions and isinstance(exc, config.retryable_exceptions):
                if retry_count >= config.max_retries:
                    logger.error(
                        "%s: transient error after %d attempt(s) — giving up: %s",
                        operation_name,
                        retry_count + 1,
                        exc,
                    )
                    raise
                delay_index = min(retry_count, len(config.retry_delays) - 1)
                delay = config.retry_delays[delay_index] if config.retry_delays else 1.0
                logger.warning(
                    "%s: attempt %d failed (%s); retrying in %.1fs …",
                    operation_name,
                    retry_count + 1,
                    type(exc).__name__,
                    delay,
                )
                await asyncio.sleep(delay)
                retry_count += 1
                continue

            raise
