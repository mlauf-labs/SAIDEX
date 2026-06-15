"""Example 3 — Use a fallback model when the primary model fails.

A cheap / fast primary model is tried first.  If it cannot produce a valid
response within the retry budget, a more capable fallback model takes over.

Run:
    python examples/03_with_fallback.py
"""

from __future__ import annotations

import asyncio
from typing import Annotated

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from saidex import extract_data_from_text

# ---------------------------------------------------------------------------
# Complex schema (more likely to trip up smaller models)
# ---------------------------------------------------------------------------


class FinancialReport(BaseModel):
    """Key figures extracted from a financial report excerpt."""

    company: str
    fiscal_year: int
    revenue_usd_millions: float = Field(description="Total revenue in millions USD")
    net_profit_usd_millions: float = Field(description="Net profit in millions USD")
    employees: int | None = Field(default=None, description="Number of full-time employees")
    yoy_revenue_growth_pct: Annotated[
        float | None,
        Field(description="Year-over-year revenue growth as percentage, e.g. 12.5 for +12.5%"),
    ] = None
    key_risks: list[str] = Field(
        description="Up to three key business risks mentioned in the report"
    )


# ---------------------------------------------------------------------------
# Sample text
# ---------------------------------------------------------------------------

REPORT_TEXT = """
Acme Corp Annual Report FY 2025

Acme Corporation reported record revenues of $4.82 billion for the fiscal year
ending December 2025, up 11.3 % compared to FY 2024.  Net profit reached
$612 million, reflecting a continued focus on operational efficiency.

The company employs approximately 28,400 full-time staff worldwide.

Management identified three key risks for the coming year:
  1. Increasing raw-material costs driven by global supply-chain disruption.
  2. Heightened regulatory scrutiny in the European Union.
  3. Rapid technology shifts requiring significant R&D investment.
"""


# ---------------------------------------------------------------------------
# Run extraction with fallback
# ---------------------------------------------------------------------------


async def main() -> None:
    # Primary: fast, cheap model
    primary = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    # Fallback: more capable model used only when primary fails
    fallback = ChatOpenAI(model="gpt-4o", temperature=0)

    report, stats = await extract_data_from_text(
        primary,
        FinancialReport,
        REPORT_TEXT,
        fallback_llm_model=fallback,
        max_primary_retries=2,
        max_fallback_retries=3,
    )

    if report is None:
        print(f"Extraction failed after {stats.total_retries} retries.")
        return

    print("Extracted financial report:")
    print(f"  Company:      {report.company}")
    print(f"  Fiscal year:  {report.fiscal_year}")
    print(f"  Revenue:      ${report.revenue_usd_millions:.0f}M")
    print(f"  Net profit:   ${report.net_profit_usd_millions:.0f}M")
    print(f"  Employees:    {report.employees:,}" if report.employees else "  Employees:    N/A")
    print(f"  YoY growth:   {report.yoy_revenue_growth_pct}%")
    print(f"  Key risks:    {report.key_risks}")
    print()
    print(
        f"Stats: primary_retries={stats.primary_retries}, "
        f"fallback_retries={stats.fallback_retries}, "
        f"fallback_used={stats.fallback_used}"
    )


if __name__ == "__main__":
    asyncio.run(main())
