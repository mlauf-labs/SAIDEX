"""Tests for ExtractDataStats, ExtractorRunStats, and FieldIssue."""

import pytest

from saidex.models import ExtractDataStats, ExtractorRunStats, FieldIssue


def _issue(field_path: str, attempt: int = 0, **kw: object) -> FieldIssue:
    base: dict[str, object] = {
        "schema_name": "S",
        "field_path": field_path,
        "category": "type",
        "error_type": "int_parsing",
        "message": "wrong type",
        "attempt": attempt,
    }
    base.update(kw)
    return FieldIssue(**base)  # type: ignore[arg-type]


def test_total_retries() -> None:
    stats = ExtractDataStats(primary_retries=2, fallback_retries=1)
    assert stats.total_retries == 3


def test_zero_retries() -> None:
    stats = ExtractDataStats()
    assert stats.total_retries == 0
    assert not stats.fallback_used
    assert stats.success is False
    assert stats.failure_reason is None
    assert stats.field_issues == ()
    assert stats.problem_fields == ()


def test_fallback_flag() -> None:
    stats = ExtractDataStats(primary_retries=3, fallback_retries=1, fallback_used=True)
    assert stats.fallback_used is True
    assert stats.total_retries == 4


def test_immutability() -> None:
    stats = ExtractDataStats(primary_retries=1)
    with pytest.raises((AttributeError, TypeError)):
        stats.primary_retries = 99  # type: ignore[misc]


def test_int_shims_removed() -> None:
    """The legacy int/comparison shims are gone (breaking change)."""
    stats = ExtractDataStats(primary_retries=2)
    with pytest.raises(TypeError):
        int(stats)
    with pytest.raises(TypeError):
        _ = stats > 0  # type: ignore[operator]


def test_problem_fields_dedupes_in_order() -> None:
    stats = ExtractDataStats(
        field_issues=(
            _issue("price"),
            _issue("name"),
            _issue("price", attempt=1),
        )
    )
    assert stats.problem_fields == ("price", "name")


def test_success_can_carry_self_corrected_issues() -> None:
    stats = ExtractDataStats(success=True, field_issues=(_issue("price"),))
    assert stats.success is True
    assert stats.problem_fields == ("price",)


def test_extractor_run_stats_shared_fields() -> None:
    stats = ExtractorRunStats(iterations=2, success=True, schema_name="S")
    assert stats.success is True
    assert stats.schema_name == "S"
    assert stats.problem_fields == ()


def test_extractor_run_stats_add_merges_issues() -> None:
    a = ExtractorRunStats(
        iterations=1,
        format_errors=1,
        field_issues=(_issue("a"),),
        schema_name="S",
    )
    b = ExtractorRunStats(
        iterations=2,
        format_errors=2,
        field_issues=(_issue("b"),),
        success=True,
        fallback_used=True,
    )
    merged = a + b
    assert merged.iterations == 3
    assert merged.format_errors == 3
    assert merged.fallback_used is True
    assert merged.success is True
    assert merged.schema_name == "S"
    assert merged.problem_fields == ("a", "b")
