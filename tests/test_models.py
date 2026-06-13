"""Tests for StructuredOutputStats."""

from saidex.models import StructuredOutputStats


def test_total_retries() -> None:
    stats = StructuredOutputStats(primary_retries=2, fallback_retries=1)
    assert stats.total_retries == 3


def test_int_conversion() -> None:
    stats = StructuredOutputStats(primary_retries=2, fallback_retries=1)
    assert int(stats) == 3


def test_str_conversion() -> None:
    stats = StructuredOutputStats(primary_retries=2, fallback_retries=1)
    assert str(stats) == "3"


def test_comparison_with_int() -> None:
    stats = StructuredOutputStats(primary_retries=2)
    assert stats > 0
    assert stats >= 2
    assert stats < 5
    assert stats <= 2


def test_zero_retries() -> None:
    stats = StructuredOutputStats()
    assert stats.total_retries == 0
    assert not stats.fallback_used
    assert int(stats) == 0


def test_fallback_flag() -> None:
    stats = StructuredOutputStats(primary_retries=3, fallback_retries=1, fallback_used=True)
    assert stats.fallback_used is True
    assert stats.total_retries == 4


def test_immutability() -> None:
    stats = StructuredOutputStats(primary_retries=1)
    try:
        stats.primary_retries = 99  # type: ignore[misc]
        raise AssertionError("Should have raised")
    except (AttributeError, TypeError):
        pass
