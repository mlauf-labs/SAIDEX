# Quality Analytics

Every extraction call returns stats that carry a structured list of
[`FieldIssue`][saidex.FieldIssue] objects — one per field-level validation
problem, tagged with the attempt it occurred in. A single run tells you whether
*that* extraction worked. Aggregate **many** runs and a different question gets
answered: **which fields does the model keep getting wrong, and why?**

That is what [`summarize_field_issues`][saidex.summarize_field_issues] is for.
Feed it the stats from a benchmark sweep or a labelled evaluation set and it
groups every issue per schema and per field, so you can find the weak field
descriptions and prompts and fix them.

## Collecting runs

`field_issues` are recorded even when a run ultimately **succeeds** — a field
that failed on attempt 0 but was corrected on attempt 1 still shows up. That is
deliberate: a field the model routinely fumbles (even if it recovers) is a
prompt-tuning signal.

```python
from saidex import extract_data_from_text, summarize_field_issues

runs = []
for text in dataset:                      # many varied inputs
    _, stats = await extract_data_from_text(llm, Invoice, text)
    runs.append(stats)

summary = summarize_field_issues(runs)
```

`summarize_field_issues` accepts a heterogeneous iterable: stats objects
([`ExtractDataStats`][saidex.ExtractDataStats] or
[`ExtractorRunStats`][saidex.ExtractorRunStats]) **and/or** bare
[`FieldIssue`][saidex.FieldIssue] instances. Stats objects contribute run
context (success/failure and run counts); bare issues count only toward the
occurrence metrics, because they carry no run outcome.

## Reading the summary

```python
for schema in summary.schemas:            # most problematic schema first
    print(f"{schema.schema_name}: {schema.success_rate:.0%} success "
          f"({schema.failed_runs}/{schema.total_runs} failed)")
    for fp in schema.field_problems:      # sorted by descending severity
        print(f"  {fp.field_path}: {fp.total_occurrences} issues, "
              f"{fp.failed_runs_with_problem} in failed runs, "
              f"recovery {fp.recovery_rate:.0%}, "
              f"top error {max(fp.by_error_type, key=fp.by_error_type.get)}")
```

### Per-field metrics

| Field | Meaning |
| --- | --- |
| `total_occurrences` | Every issue for this field, across all attempts and runs. |
| `runs_with_problem` | Runs in which the field failed at least once. |
| `failed_runs_with_problem` | Of those, how many runs ultimately failed. |
| `occurrences_on_successful_runs` | Issues that were self-corrected. |
| `recovery_rate` | Share of `runs_with_problem` that still succeeded. |
| `by_category` / `by_error_type` | Issue counts per category and raw Pydantic error type. |
| `first_error_attempt_avg` | Mean earliest attempt the field failed — low means up-front. |
| `sample_received` | Up to five distinct offending values. |
| `severity` | Sort score: dominated by `failed_runs_with_problem`, tie-broken by `total_occurrences`. |

A field with a **low `recovery_rate`** and a high `failed_runs_with_problem` is
where to spend your prompt-tuning effort first — it causes hard failures. A
field with a high `recovery_rate` is noisier than it is dangerous.

!!! note "Bare issues and run counts"
    A bare `FieldIssue` passed straight in is not a run, so it never increments
    `total_runs`, `failed_runs`, or `runs_with_problem` — only the
    occurrence-based fields. A schema seen only through bare issues reports
    `total_runs == 0` and `success_rate == 0.0`.

> **Runnable example:** [`examples/11_quality_analytics.py`](https://github.com/mlauf-labs/saidex/blob/main/examples/11_quality_analytics.py)
