"""Example 10 — Extract a list of items from one document in a single call.

Many documents contain repeated records: line items on an invoice, multiple
people in a transcript, several products on a page.  ``extract_data_list`` and
``extract_data_list_from_text`` return a precisely typed ``list[ModelT]`` from
a single LLM call, with per-item validation and the usual retry/fallback loop.

Run:
    python examples/10_batch_extraction.py
"""

from __future__ import annotations

import asyncio

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from saidex import extract_data_list_from_text

# ---------------------------------------------------------------------------
# Define the schema for a SINGLE item — not the list
# ---------------------------------------------------------------------------


class InvoiceLine(BaseModel):
    """A single line item on an invoice."""

    description: str = Field(description="What was purchased")
    quantity: int = Field(description="Number of units", ge=1)
    unit_price: float = Field(description="Price per unit", ge=0)


# ---------------------------------------------------------------------------
# Run batch extraction
# ---------------------------------------------------------------------------


async def main() -> None:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    document = """
    Invoice #2026-0042
      3x Mechanical keyboard ......... 89.00 each
      1x 27" 4K monitor .............. 349.99
      2x USB-C cable (2m) ............ 12.50 each
    """

    # Pass the per-item schema; the return type is list[InvoiceLine].
    lines, stats = await extract_data_list_from_text(llm, InvoiceLine, document)

    if lines is None:
        print(f"Extraction failed after {stats.total_retries} retries.")
        return

    print(f"Extracted {stats.item_count} line items:")
    total = 0.0
    for line in lines:
        subtotal = line.quantity * line.unit_price
        total += subtotal
        print(f"  {line.quantity:>2}x {line.description:<28} {subtotal:>10.2f}")
    print(f"{'TOTAL':>36} {total:>10.2f}")
    print(f"\nRetries needed: {stats.total_retries}")


if __name__ == "__main__":
    asyncio.run(main())
