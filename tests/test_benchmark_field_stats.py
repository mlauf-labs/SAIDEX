"""Tests for the benchmark field-issue aggregation helpers.

These cover ``benchmarks/_field_stats.py`` only, which depends on ``saidex``
alone (no LLM backend), so they run without network or provider packages.
"""

from __future__ import annotations

from dataclasses import dataclass

from benchmarks._field_stats import (
    format_console_summary,
    results_to_stats,
    summarize_results,
    summary_to_dict,
)
from saidex import FieldIssue


@dataclass
class _FakeResult:
    """Minimal stand-in for BenchmarkResult (duck-typed _ResultLike)."""

    success: bool
    schema_name: str
    field_issues: tuple[FieldIssue, ...] = ()


def _issue(field_path: str, schema: str = "Invoice", attempt: int = 0) -> FieldIssue:
    return FieldIssue(
        schema_name=schema,
        field_path=field_path,
        category="type",
        error_type="float_parsing",
        message="bad",
        attempt=attempt,
        received="x",
    )


def test_results_to_stats_projection() -> None:
    results = [
        _FakeResult(success=True, schema_name="Invoice"),
        _FakeResult(success=False, schema_name="Invoice", field_issues=(_issue("total"),)),
    ]
    stats = results_to_stats(results)
    assert len(stats) == 2
    assert stats[0].success is True
    assert stats[1].field_issues[0].field_path == "total"


def test_summarize_results_groups_per_schema() -> None:
    results = [
        _FakeResult(True, "Invoice"),
        _FakeResult(False, "Invoice", (_issue("total"),)),
        _FakeResult(False, "Invoice", (_issue("total"),)),
        _FakeResult(True, "Sentiment"),
    ]
    summary = summarize_results(results)
    by_name = {s.schema_name: s for s in summary.schemas}
    assert by_name["Invoice"].total_runs == 3
    assert by_name["Invoice"].failed_runs == 2
    assert by_name["Invoice"].field_problems[0].field_path == "total"
    assert by_name["Invoice"].field_problems[0].failed_runs_with_problem == 2
    assert by_name["Sentiment"].field_problems == ()


def test_summary_to_dict_is_json_serialisable() -> None:
    import json

    results = [_FakeResult(False, "Invoice", (_issue("total"),))]
    payload = summary_to_dict(summarize_results(results))
    # Round-trips through JSON without error.
    text = json.dumps(payload)
    assert "Invoice" in text
    assert payload["schemas"][0]["field_problems"][0]["field_path"] == "total"


def test_console_summary_empty_when_no_issues() -> None:
    results = [_FakeResult(True, "Invoice"), _FakeResult(True, "Sentiment")]
    assert format_console_summary(summarize_results(results)) == []


def test_console_summary_lists_problem_fields() -> None:
    results = [_FakeResult(False, "Invoice", (_issue("total"),))]
    lines = format_console_summary(summarize_results(results))
    assert any("FIELD ISSUE ANALYSIS" in line for line in lines)
    assert any("total" in line for line in lines)
