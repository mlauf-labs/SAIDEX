"""Tests for summarize_field_issues and the summary dataclasses."""

from __future__ import annotations

from saidex import (
    ExtractDataStats,
    ExtractorRunStats,
    FieldIssue,
    render_field_issue_report,
    summarize_field_issues,
)


def _issue(
    field_path: str,
    *,
    schema: str = "Invoice",
    category: str = "type",
    error_type: str = "int_parsing",
    attempt: int = 0,
    received: str | None = "x",
) -> FieldIssue:
    return FieldIssue(
        schema_name=schema,
        field_path=field_path,
        category=category,
        error_type=error_type,
        message="bad",
        attempt=attempt,
        received=received,
    )


def _run(
    *issues: FieldIssue,
    schema: str = "Invoice",
    success: bool = False,
) -> ExtractDataStats:
    return ExtractDataStats(
        success=success,
        schema_name=schema,
        field_issues=tuple(issues),
    )


def test_empty_input() -> None:
    summary = summarize_field_issues([])
    assert summary.schemas == ()


def test_single_failed_run_single_field() -> None:
    summary = summarize_field_issues([_run(_issue("price"))])
    assert len(summary.schemas) == 1
    schema = summary.schemas[0]
    assert schema.schema_name == "Invoice"
    assert schema.total_runs == 1
    assert schema.failed_runs == 1
    assert schema.success_rate == 0.0

    assert len(schema.field_problems) == 1
    fp = schema.field_problems[0]
    assert fp.field_path == "price"
    assert fp.total_occurrences == 1
    assert fp.runs_with_problem == 1
    assert fp.failed_runs_with_problem == 1
    assert fp.recovery_rate == 0.0
    assert fp.by_category == {"type": 1}
    assert fp.by_error_type == {"int_parsing": 1}


def test_self_corrected_field_has_full_recovery() -> None:
    # Field erred on attempt 0 and 1 but the run ultimately succeeded.
    run = _run(
        _issue("price", attempt=0),
        _issue("price", attempt=1),
        success=True,
    )
    summary = summarize_field_issues([run])
    fp = summary.schemas[0].field_problems[0]
    assert fp.total_occurrences == 2
    assert fp.runs_with_problem == 1
    assert fp.failed_runs_with_problem == 0
    assert fp.occurrences_on_successful_runs == 2
    assert fp.recovery_rate == 1.0
    assert fp.first_error_attempt_avg == 0.0  # earliest attempt per run is 0
    assert summary.schemas[0].success_rate == 1.0


def test_severity_orders_failing_field_first() -> None:
    # "name" fails in a run that recovers; "price" fails in runs that fail hard.
    runs = [
        _run(_issue("name"), success=True),
        _run(_issue("price"), success=False),
        _run(_issue("price"), success=False),
    ]
    summary = summarize_field_issues(runs)
    paths = [fp.field_path for fp in summary.schemas[0].field_problems]
    assert paths[0] == "price"  # higher severity (2 failed runs) sorts first
    price = summary.schemas[0].field_problems[0]
    assert price.failed_runs_with_problem == 2
    name = summary.schemas[0].field_problems[1]
    assert name.failed_runs_with_problem == 0
    assert price.severity > name.severity


def test_multi_schema_grouping_and_ordering() -> None:
    runs = [
        _run(_issue("price", schema="Invoice"), schema="Invoice", success=False),
        _run(_issue("price", schema="Invoice"), schema="Invoice", success=False),
        _run(_issue("mood", schema="Sentiment"), schema="Sentiment", success=True),
    ]
    summary = summarize_field_issues(runs)
    assert [s.schema_name for s in summary.schemas] == ["Invoice", "Sentiment"]
    assert summary.schemas[0].failed_runs == 2
    assert summary.schemas[1].failed_runs == 0


def test_bare_issues_count_occurrences_not_runs() -> None:
    summary = summarize_field_issues(
        [
            _issue("price", received="abc"),
            _issue("price", received="def"),
        ]
    )
    schema = summary.schemas[0]
    assert schema.total_runs == 0
    assert schema.failed_runs == 0
    fp = schema.field_problems[0]
    assert fp.total_occurrences == 2
    assert fp.runs_with_problem == 0
    assert fp.recovery_rate == 0.0
    assert set(fp.sample_received) == {"abc", "def"}


def test_mixed_stats_and_bare_issues() -> None:
    summary = summarize_field_issues(
        [
            _run(_issue("price"), success=False),
            _issue("price"),  # bare — adds an occurrence only
        ]
    )
    fp = summary.schemas[0].field_problems[0]
    assert fp.total_occurrences == 2
    assert fp.runs_with_problem == 1
    assert fp.failed_runs_with_problem == 1


def test_accepts_extractor_run_stats() -> None:
    stats = ExtractorRunStats(
        success=False,
        schema_name="Agent",
        field_issues=(_issue("answer", schema="Agent"),),
    )
    summary = summarize_field_issues([stats])
    assert summary.schemas[0].schema_name == "Agent"
    assert summary.schemas[0].field_problems[0].field_path == "answer"


def test_sample_received_capped_and_deduped() -> None:
    issues = [_issue("price", received=str(v)) for v in [1, 1, 2, 3, 4, 5, 6, 7]]
    summary = summarize_field_issues([_run(*issues, success=False)])
    fp = summary.schemas[0].field_problems[0]
    assert len(fp.sample_received) == 5
    assert fp.sample_received == ("1", "2", "3", "4", "5")


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------


def test_report_empty_summary() -> None:
    report = summarize_field_issues([]).to_markdown()
    assert report == "# Field issue report\n\n_No field issues recorded._"


def test_report_is_deterministic() -> None:
    runs = [
        _run(_issue("price"), success=False),
        _run(_issue("price"), success=False),
        _run(_issue("name", category="missing", error_type="missing"), success=True),
    ]
    summary = summarize_field_issues(runs)
    first = summary.to_markdown()
    second = render_field_issue_report(summary)
    assert first == second


def test_report_snapshot() -> None:
    runs = [
        _run(_issue("total", error_type="float_parsing", received="1.299,00"), success=False),
        _run(_issue("total", error_type="float_parsing", received="forty-two"), success=False),
        _run(
            _issue("currency", category="enum", error_type="enum", received="euros"),
            success=True,
        ),
    ]
    report = summarize_field_issues(runs).to_markdown()
    expected = (
        "# Field issue report\n"
        "\n"
        "## Invoice — 33% success (2/3 runs failed)\n"
        "\n"
        "| Field | Category | Top error | Hits | In failed runs | Recovery | Samples |\n"
        "| --- | --- | --- | ---: | ---: | ---: | --- |\n"
        "| total | type | float_parsing | 2 | 2 | 0% | `1.299,00`, `forty-two` |\n"
        "| currency | enum | enum | 1 | 0 | 100% | `euros` |\n"
    )
    assert report == expected


def test_report_schema_without_issues() -> None:
    summary = summarize_field_issues([_run(schema="Clean", success=True)])
    report = summary.to_markdown()
    assert "## Clean — 100% success (0/1 runs failed)" in report
    assert "_No field issues recorded._" in report


def test_report_escapes_pipe_in_sample() -> None:
    summary = summarize_field_issues([_run(_issue("raw", received="a|b"), success=False)])
    report = summary.to_markdown()
    assert r"`a\|b`" in report
