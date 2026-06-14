"""Invoice Benchmark 03 — Online shop invoice (small business, informal prose layout).

The text is deliberately written in a more prose-like, less structured format to test
how well models handle invoice data that isn't neatly tabulated.

Run standalone:
    python benchmarks/invoice_03.py
"""

from __future__ import annotations

import asyncio

from pydantic import BaseModel, Field
from saidex import ISODateStr

from ._base import BenchmarkScenario, run_all_models

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


class InvoiceData(BaseModel):
    """Key fields extracted from an invoice document."""

    contract_number: str = Field(description="Order number or job number, e.g. ORD-88124")
    customer_number: str = Field(description="Customer number, e.g. KNR-20091")
    invoice_date: ISODateStr = Field(description="Invoice date as yyyy-mm-dd, e.g. 2024-09-23")
    address: str = Field(description="Full delivery or billing address as a single string")
    billed_company_name: str = Field(description="Company name of the invoice recipient (the customer being billed)")
    issuing_company_name: str = Field(description="Company or shop name of the sender (seller)")
    total_amount: float = Field(description="Total amount of the order as a plain number, without currency symbol, e.g. 27.43")


# ---------------------------------------------------------------------------
# Test text
# ---------------------------------------------------------------------------

INVOICE_TEXT = """
Thank you for your order at HobbyBastel24!

Your package is on its way. Please keep this document as proof of purchase.

──────────────────────────────────────────────
Merchant Information
HobbyBastel24 – Owner: Sandra Pfeiffer
Gartenstraße 5, 04103 Leipzig, Germany
E-Mail: shop@hobbybastel24.de
Tel: +49 341 960 44 12
Small business under § 19 German VAT Act – VAT not charged
──────────────────────────────────────────────

Your Customer No.:  KNR-20091
Order Number:       ORD-88124
Order Date:         22 September 2024
Invoice Date:       23 September 2024

Delivery Address:
Ms. Julia Hartmann-Scheck
c/o Office Community Altstadt
Nikolaistraße 17, 2nd Floor
04109 Leipzig, Germany

──────────────────────────────────────────────
Order Summary
──────────────────────────────────────────────

  • 3x Fuse Bead Set "Pastel" (1,000 pieces, each € 3.49)        €  10.47
  • 1x Wooden Embroidery Frame 30 cm                              €   6.99
  • 2x Merino Felt Wool, 50g, Colour: Sky Blue (each € 2.99)     €   5.98
  • Shipping (DHL Standard)                                       €   3.99

                                    Total amount:                 €  27.43

Payment method: PayPal (Transaction: 9F812-XQ44T-0028)
Total amount settled: 22.09.2024

──────────────────────────────────────────────

Returns accepted within 14 days. Please contact us by e-mail beforehand.
We hope you enjoy your products!

Kind regards,
Sandra Pfeiffer & the HobbyBastel24 Team
"""


SCENARIO = BenchmarkScenario(
    name="Invoice 03 — Online Shop (small business)",
    description="Informal online-shop invoice; fields are embedded in prose rather than tabulated.",
    schema=InvoiceData,
    text=INVOICE_TEXT,
    system_prompt="You are an invoice data extraction assistant. Extract the requested fields exactly as they appear in the document.",
    expected={
        "contract_number": "ORD-88124",
        "customer_number": "KNR-20091",
        "invoice_date": "2024-09-23",
        "issuing_company_name": "HobbyBastel24",
        "total_amount": 27.43,
    },
)


async def main() -> None:
    print("Benchmark: Invoice 03 — Online Shop (small business)")
    await run_all_models(SCENARIO)


if __name__ == "__main__":
    asyncio.run(main())
