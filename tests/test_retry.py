"""Tests for the retry utilities."""

from __future__ import annotations

import pytest

from saidex.retry import RetryConfig, RetryResult, with_retry

NO_RETRY = RetryConfig(
    max_retries=0,
    retry_delays=[],
    retryable_exceptions=(),
    rate_limit_exceptions=(),
)

ONE_RETRY = RetryConfig(
    max_retries=1,
    retry_delays=[0.0],  # instant for tests
    retryable_exceptions=(ValueError,),
    rate_limit_exceptions=(),
)


# ---------------------------------------------------------------------------
# Success cases
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_success_no_retry() -> None:
    async def op() -> int:
        return 42

    result = await with_retry(op, NO_RETRY)
    assert result.result == 42
    assert result.retry_count == 0


@pytest.mark.asyncio
async def test_success_after_one_retry() -> None:
    call_count = 0

    async def op() -> str:
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise ValueError("transient")
        return "ok"

    result = await with_retry(op, ONE_RETRY)
    assert result.result == "ok"
    assert result.retry_count == 1


# ---------------------------------------------------------------------------
# Exhausted retries
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_raises_after_max_retries() -> None:
    async def op() -> None:
        raise ValueError("always fails")

    with pytest.raises(ValueError, match="always fails"):
        await with_retry(op, ONE_RETRY)


# ---------------------------------------------------------------------------
# Non-retryable exception passes through immediately
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_non_retryable_exception_raises_immediately() -> None:
    call_count = 0

    async def op() -> None:
        nonlocal call_count
        call_count += 1
        raise TypeError("not retryable")

    with pytest.raises(TypeError):
        await with_retry(op, ONE_RETRY)

    assert call_count == 1  # no retries attempted


# ---------------------------------------------------------------------------
# RetryResult
# ---------------------------------------------------------------------------


def test_retry_result_attributes() -> None:
    r: RetryResult[str] = RetryResult(result="hello", retry_count=2)
    assert r.result == "hello"
    assert r.retry_count == 2
