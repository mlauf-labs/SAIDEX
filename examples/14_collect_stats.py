"""Example 14 — Observe extraction stats without threading them through calls.

`collect_stats()` is a scoped context manager that captures the stats of every
extraction inside its block — even ones buried deep in your own helper
functions — so you never have to return or thread a stats object up through
intermediate signatures.  `on_extraction()` registers a process-wide listener
for logging / tracing / metrics.

Run:
    python examples/14_collect_stats.py
"""

from __future__ import annotations

import asyncio

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from saidex import (
    ExtractionEvent,
    collect_stats,
    extract_data_from_text,
    on_extraction,
)


class Invoice(BaseModel):
    """A minimal invoice record."""

    invoice_number: str = Field(description="The invoice identifier")
    total: float = Field(description="Grand total in euros", ge=0)


DOCUMENTS = [
    "Invoice 2026-0001. Total: 1.299,00 EUR.",
    "INV-7782 — amount due 89 euros.",
    "Invoice 2026-0002, total EUR 349.99.",
]


async def parse_document(llm: ChatOpenAI, text: str) -> Invoice | None:
    """A deeply-nested helper that neither returns nor forwards any stats."""
    result, _ = await extract_data_from_text(llm, Invoice, text)
    return result


async def main() -> None:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # A process-wide listener — fires once per extraction, on success or failure.
    def log_extraction(event: ExtractionEvent) -> None:
        print(
            f"[listener] {event.schema_name}: "
            f"success={event.stats.success} issues={len(event.stats.field_issues)}"
        )

    unsubscribe = on_extraction(log_extraction)

    # The scoped sink captures every extraction inside the block — including the
    # ones parse_document runs, without parse_document exposing any stats.
    async with collect_stats() as sink:
        for text in DOCUMENTS:
            await parse_document(llm, text)

    unsubscribe()

    runs = sink.all()
    failures = sum(1 for s in runs if not s.success)
    print(f"\n{len(runs)} extractions, {failures} failed")
    for s in runs:
        print(f"  {s.schema_name:<8} success={s.success} fallback={s.fallback_used}")


if __name__ == "__main__":
    asyncio.run(main())
