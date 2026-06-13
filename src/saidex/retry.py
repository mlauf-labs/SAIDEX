"""Async retry utilities for LLM network calls."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

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


def _default_retryable_exceptions() -> tuple[type[Exception], ...]:
    """Return sensible defaults for retryable network exceptions.

    Attempts to import ``openai`` and ``httpx``; silently falls back to an
    empty tuple when neither is installed so the library stays dependency-free.
    """
    exceptions: list[type[Exception]] = []
    try:
        import httpx

        exceptions.extend([httpx.ConnectError, httpx.TimeoutException, httpx.RemoteProtocolError])
    except ImportError:
        pass
    try:
        import openai

        exceptions.extend([openai.APIConnectionError, openai.APITimeoutError])
    except ImportError:
        pass
    return tuple(exceptions)


def _default_rate_limit_exceptions() -> tuple[type[Exception], ...]:
    """Return rate-limit exception types when ``openai`` is installed."""
    try:
        import openai

        return (openai.RateLimitError,)
    except ImportError:
        return ()


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
            Defaults to common ``httpx`` / ``openai`` network errors when those
            packages are installed.
        rate_limit_exceptions: Exception types that trigger rate-limit retry
            logic.  Defaults to ``openai.RateLimitError`` when ``openai`` is
            installed.
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


#: Ready-to-use default configuration with OpenAI / httpx error handling.
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
