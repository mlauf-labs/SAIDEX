"""Example 12 — External validator: cross-field rules that re-enter the retry loop.

Pydantic validators cover structural and field-level rules.  Some checks, though,
are inherently *cross-field*, *stateful*, or need *external context* — a database
lookup, a business rule, a reference-data check.  Pass a ``validator`` callable to
any extraction entry point: it receives the already-validated model and rejects it
by raising (or returning an error string).  A rejection is fed back into the same
field-level retry loop, exactly like a Pydantic failure, so the model self-corrects.

The validator may be **sync or async**, and is available on every entry point —
``extract_data``, ``extract_data_from_text``, the batch variants, the agent loop,
and all the ``*_sync`` wrappers.

Run:
    python examples/12_external_validator.py
"""

from __future__ import annotations

import asyncio

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from saidex import extract_data_from_text

# ---------------------------------------------------------------------------
# Schema — note there is NO cross-field check inside the model itself.
# ---------------------------------------------------------------------------


class InvoiceLine(BaseModel):
    """A single invoice line item."""

    description: str = Field(description="What was billed")
    amount: float = Field(description="Line total in EUR", ge=0)


class Invoice(BaseModel):
    """An invoice whose stated total must match the sum of its line items."""

    vendor: str = Field(description="Who issued the invoice")
    lines: list[InvoiceLine] = Field(description="All line items", min_length=1)
    total: float = Field(description="Stated grand total in EUR", ge=0)


# ---------------------------------------------------------------------------
# External validator — a cross-field rule Pydantic field validators can't express
# without baking the check into the schema.
# ---------------------------------------------------------------------------


def validate_invoice_total(inv: Invoice) -> None:
    """Reject the invoice when the line items do not sum to the stated total.

    Raising here feeds the message back to the model as a correction prompt and
    consumes one retry — just like a Pydantic ``ValidationError`` would.
    """
    line_sum = round(sum(line.amount for line in inv.lines), 2)
    if line_sum != round(inv.total, 2):
        raise ValueError(
            f"The line items sum to {line_sum:.2f} EUR but the stated total is "
            f"{inv.total:.2f} EUR. Re-read the document and make the total match "
            f"the sum of the line items."
        )


# ---------------------------------------------------------------------------
# Run extraction
# ---------------------------------------------------------------------------


async def main() -> None:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    document = """
    Invoice from Muster GmbH
      - Consulting services .... 1,200.00 EUR
      - Travel expenses ........   180.50 EUR
    Grand total: 1,380.50 EUR
    """

    invoice, stats = await extract_data_from_text(
        llm,
        Invoice,
        document,
        validator=validate_invoice_total,
    )

    if invoice is None:
        print(f"Extraction failed ({stats.failure_reason}) after {stats.total_retries} retries.")
        return

    print("Extracted invoice:")
    print(f"  Vendor: {invoice.vendor}")
    for line in invoice.lines:
        print(f"    - {line.description}: {line.amount:.2f} EUR")
    print(f"  Total:  {invoice.total:.2f} EUR")
    print(f"\nRetries needed: {stats.total_retries}")
    if stats.field_issues:
        print("Issues seen along the way:")
        for issue in stats.field_issues:
            print(f"  [{issue.error_type}] {issue.message}")


if __name__ == "__main__":
    asyncio.run(main())
