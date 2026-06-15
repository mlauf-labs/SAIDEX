"""Example 11 — Find which fields a model keeps getting wrong.

A single extraction tells you whether *that* run worked.  Run many varied inputs
and aggregate their stats with ``summarize_field_issues`` to see, per schema and
per field, which fields cause trouble, why, and whether the model recovers — the
signal you need to improve a field's description or prompt.

This is the pattern a benchmark sweep uses: collect the ``stats`` from every run
into a list, then summarise once at the end.

Run:
    python examples/11_quality_analytics.py
"""

from __future__ import annotations

import asyncio

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from saidex import (
    ExtractDataStats,
    extract_data_from_text,
    summarize_field_issues,
)

# ---------------------------------------------------------------------------
# A schema whose field descriptions we want to stress-test
# ---------------------------------------------------------------------------


class Invoice(BaseModel):
    """A minimal invoice record."""

    invoice_number: str = Field(description="The invoice identifier")
    total: float = Field(description="Grand total in euros", ge=0)
    currency: str = Field(description="ISO 4217 currency code, e.g. 'EUR'")


# A few varied documents — some deliberately ambiguous, to provoke field issues.
DOCUMENTS = [
    "Invoice 2026-0001. Total: 1.299,00 EUR.",
    "INV-7782 — amount due 89 euros.",
    "Receipt no. A-55. Grand total forty-two euros and fifty cents.",
    "Invoice 2026-0002, total EUR 349.99.",
    "Bill #9001 — 12,50€.",
]


async def main() -> None:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    runs: list[ExtractDataStats] = []
    for text in DOCUMENTS:
        _, stats = await extract_data_from_text(llm, Invoice, text)
        runs.append(stats)

    summary = summarize_field_issues(runs)

    # Inspect the structured summary programmatically …
    for schema in summary.schemas:
        print(
            f"\n{schema.schema_name}: {schema.success_rate:.0%} success "
            f"({schema.failed_runs}/{schema.total_runs} runs failed)"
        )
        if not schema.field_problems:
            print("  No field issues — clean sweep.")
            continue
        for fp in schema.field_problems:
            top_error = max(fp.by_error_type, key=lambda k: fp.by_error_type[k])
            print(
                f"  {fp.field_path:<16} "
                f"issues={fp.total_occurrences:<3} "
                f"failed_runs={fp.failed_runs_with_problem:<3} "
                f"recovery={fp.recovery_rate:.0%}  "
                f"top_error={top_error}  "
                f"examples={list(fp.sample_received)}"
            )

    # … or render a deterministic Markdown report (great as a benchmark artifact).
    print("\n" + "=" * 60)
    print(summary.to_markdown())


if __name__ == "__main__":
    asyncio.run(main())
