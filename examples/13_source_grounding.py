"""Example 13 — Source grounding: reject values that aren't in the text.

LLMs sometimes invent plausible-looking values. Mark a field as *grounded* and
SAIDEX verifies, after Pydantic validation, that the extracted value actually
appears in the source text. A value it cannot find is rejected and the existing
retry loop asks the model to correct it — an anti-hallucination guard.

Two equivalent, schema-visible surfaces:

* the ``Grounded`` Annotated marker — ``Annotated[str, Grounded()]``;
* the ``GroundedField`` helper — ``field: T = GroundedField(...)``.

Grounding is locale-aware: a number stored as ``1234.5`` is matched against
``1.234,50`` (de) or ``1,234.50`` (en). Point ``locale_field`` at a sibling
field (e.g. a country code) to drive the formatting from the extracted data, or
pass a fixed ``locale``.

By default a mismatch re-enters the retry loop; pass ``on_mismatch="flag"`` for
advisory grounding that keeps the value and only records a ``FieldIssue``.

Run:
    python examples/13_source_grounding.py
"""

from __future__ import annotations

import asyncio
from typing import Annotated

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from saidex import Grounded, GroundedField, IsoDateStr, extract_data_from_text

# ---------------------------------------------------------------------------
# Schema — grounded fields are obvious right where they are declared.
# ---------------------------------------------------------------------------


class Invoice(BaseModel):
    """An invoice whose key facts must be traceable to the document text."""

    # Marker form — must appear verbatim (normalised) in the text.
    vendor: Annotated[str, Grounded()] = Field(description="Who issued the invoice")
    # Helper form — same effect, thin wrapper around pydantic.Field.
    invoice_no: str = GroundedField(description="Invoice number")
    # Locale-aware: the number is matched in the document's locale, taken from
    # the (also extracted) country code field.
    total: float = GroundedField(locale_field="country", description="Grand total")
    country: str = Field(description="ISO country code of the issuer, e.g. DE")
    # Grounded ISO date — matched against localised surface forms (05.04.2024 …).
    issued_on: Annotated[IsoDateStr, Grounded(locale_field="country")] = Field(
        description="Invoice date (yyyy-mm-dd)"
    )
    # Not grounded — a free-form summary the model may phrase itself.
    note: str = Field(description="One-line summary")


async def main() -> None:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    document = """
    Rechnung Nr. R-2024-0815
    Aussteller: ACME GmbH (DE)
    Rechnungsdatum: 05.04.2024
    Gesamtbetrag: 1.234,50 EUR
    """

    invoice, stats = await extract_data_from_text(llm, Invoice, document)

    if invoice is None:
        print(f"Extraction failed ({stats.failure_reason}) after {stats.total_retries} retries.")
        return

    print("Extracted invoice (all grounded fields verified against the text):")
    print(f"  Vendor:     {invoice.vendor}")
    print(f"  Invoice no: {invoice.invoice_no}")
    print(f"  Total:      {invoice.total} ({invoice.country})")
    print(f"  Issued on:  {invoice.issued_on}")
    print(f"  Note:       {invoice.note}")
    print(f"\nRetries needed: {stats.total_retries}")
    if stats.field_issues:
        print("Issues seen along the way:")
        for issue in stats.field_issues:
            print(f"  [{issue.category}] {issue.field_path}: {issue.message}")


if __name__ == "__main__":
    asyncio.run(main())
