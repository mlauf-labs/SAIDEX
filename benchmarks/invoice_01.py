"""Invoice Benchmark 01 — SaaS subscription invoice (English, modern tech company).

Run standalone:
    python benchmarks/invoice_01.py
"""

from __future__ import annotations

import asyncio

from pydantic import BaseModel, Field

from ._base import BenchmarkScenario, run_all_models

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


class InvoiceData(BaseModel):
    """Key fields extracted from an invoice document."""

    contract_number: str | None = Field(None, description="Contract or subscription number, e.g. CTR-2024-00891")
    customer_number: str | None = Field(None, description="Customer ID or account number, e.g. CUST-4471")
    invoice_date: str | None = Field(None, description="Invoice issue date in ISO format if possible, e.g. 2024-03-15")
    address: str | None = Field(None, description="Full billing address of the customer as a single string")
    company_name: str | None = Field(None, description="Name of the company being billed")


# ---------------------------------------------------------------------------
# Test text
# ---------------------------------------------------------------------------

INVOICE_TEXT = """
INVOICE

TechFlow Solutions GmbH
Leopoldstraße 112
80802 Munich, Germany
VAT ID: DE298471023
support@techflow.io

─────────────────────────────────────────────────────────
BILL TO                                 INVOICE DETAILS
─────────────────────────────────────────────────────────
Nexora Digital AG                       Invoice No.:  INV-2024-03-0182
Attn: Accounts Payable                  Contract No.: CTR-2024-00891
Rosenthaler Straße 44, 3rd Floor        Customer No.: CUST-4471
10178 Berlin, Germany                   Invoice Date: 15 March 2024
                                        Due Date:     14 April 2024
─────────────────────────────────────────────────────────

SUBSCRIPTION DETAILS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Plan:         TechFlow Enterprise Cloud — Annual License
Period:       01 March 2024 – 28 February 2025
Seats:        50 active user seats

ITEM                                      QTY    UNIT PRICE       AMOUNT
─────────────────────────────────────────────────────────────────────────
Enterprise License (annual, per seat)      50      € 28.00/mo    € 16,800.00
Priority Support Add-on                     1     € 199.00/mo    €  2,388.00
SSO & Advanced Security Module              1     € 149.00/mo    €  1,788.00
─────────────────────────────────────────────────────────────────────────
                                                    Subtotal:    € 20,976.00
                                                    VAT 19%:     €  3,985.44
                                                    TOTAL DUE:   € 24,961.44
─────────────────────────────────────────────────────────────────────────

PAYMENT DETAILS
Bank: Deutsche Bank AG
IBAN: DE89 3704 0044 0532 0130 00
BIC:  COBADEFFXXX
Reference: INV-2024-03-0182 / CUST-4471

Please transfer the total amount within 30 days of the invoice date.
For questions contact billing@techflow.io or call +49 89 4521 9900.

Thank you for choosing TechFlow Solutions!
"""


SCENARIO = BenchmarkScenario(
    name="Invoice 01 — SaaS Subscription (EN)",
    description="English SaaS invoice with contract number, customer ID, address, company, and date.",
    schema=InvoiceData,
    text=INVOICE_TEXT,
    system_prompt="You are an invoice data extraction assistant. Extract the requested fields exactly as they appear in the document.",
    expected={
        "contract_number": "CTR-2024-00891",
        "customer_number": "CUST-4471",
        "invoice_date": "2024-03-15",
        "company_name": "Nexora Digital AG",
    },
)


async def main() -> None:
    print("Benchmark: Invoice 01 — SaaS Subscription (EN)")
    await run_all_models(SCENARIO)


if __name__ == "__main__":
    asyncio.run(main())
