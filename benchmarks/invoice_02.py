"""Invoice Benchmark 02 — Consulting firm invoice (complex layout with multiple service items).

Run standalone:
    python benchmarks/invoice_02.py
"""

from __future__ import annotations

import asyncio

from pydantic import BaseModel, Field
from saidex import CountryCodeStr, IsoDateStr, VatIdStr

from ._base import BenchmarkScenario, run_all_models

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


class InvoiceData(BaseModel):
    """Key fields extracted from an invoice document."""

    invoice_number: str = Field(description="The invoice's own number/ID, e.g. RE-2024-0619")
    contract_number: str = Field(description="Contract or project number, e.g. VTR-2024-0055")
    customer_number: str = Field(description="Customer number or client ID, e.g. KD-10234")
    invoice_date: IsoDateStr = Field(description="Invoice date as yyyy-mm-dd, e.g. 2024-06-01")
    address: str = Field(description="Full billing address of the customer as a single string")
    billed_company_name: str = Field(description="Company name of the invoice recipient (the customer being billed)")
    issuing_company_name: str = Field(description="Company name of the seller who issued the invoice")
    total_amount: float = Field(description="Total amount due as a plain number, without currency symbol or thousands separators, e.g. 20432.30")
    country: CountryCodeStr = Field(description="ISO 3166-1 alpha-2 country code of the recipient's billing address, e.g. DE")
    vat_id: VatIdStr = Field(description="The seller's VAT identification number, e.g. DE301847192")


# ---------------------------------------------------------------------------
# Test text
# ---------------------------------------------------------------------------

INVOICE_TEXT = """
                    ██████╗ CONSULTING
                  Berger & Partner Management Consulting GmbH
               Friedrichstraße 68 · 10117 Berlin · Germany
           Tel.: +49 30 8849 2210 · contact@berger-partner-consulting.de
                      Tax No.: 27/442/19872
                      VAT ID: DE 301 847 192

═══════════════════════════════════════════════════════════════════════

INVOICE No. RE-2024-0619
Issued on: 01 June 2024
Due by:    01 July 2024

═══════════════════════════════════════════════════════════════════════

Bill To:                                Order Details:
                                        ─────────────────────────────
MediCare Verwaltungs GmbH               Customer No.: KD-10234
Attn: Dr. Klaus Winterberg              Contract No.: VTR-2024-0055
Hauptstraße 220                         Project Code: PROJ-MCR-IT
69117 Heidelberg                        Service Period: 01 Apr – 31 May 2024
Germany

═══════════════════════════════════════════════════════════════════════

SERVICES RENDERED

No.   Description                                    Hrs.   Day Rate      Amount
────────────────────────────────────────────────────────────────────────────────
1     IT Strategy Workshops (4 days × 8 hrs)          32   € 1,850.00   €  7,400.00
2     Process Analysis & Documentation                18   €   185.00   €  3,330.00
3     Requirements Management (Sprint Coaching)       24   €   185.00   €  4,440.00
4     Final Presentation & Results Report              6   €   185.00   €  1,110.00
5     Travel & Accommodation Expenses (receipts)       —    Flat fee    €    890.00
────────────────────────────────────────────────────────────────────────────────
                                                       Net amount:    € 17,170.00
                                                       VAT 19%:       €  3,262.30
                                                       Total due:     € 20,432.30

═══════════════════════════════════════════════════════════════════════

Payment Instructions:
Please transfer the total amount of € 20,432.30 by 01 July 2024, quoting
invoice number RE-2024-0619, to the following bank account:

  Bank:      Commerzbank AG Berlin
  IBAN:      DE42 1004 0000 0287 8000 04
  BIC:       COBADEFFXXX
  Reference: RE-2024-0619 / VTR-2024-0055 / KD-10234

For billing enquiries please contact: billing@berger-partner-consulting.de

Kind regards,
Thomas Berger
Managing Director, Berger & Partner Management Consulting GmbH
"""


SCENARIO = BenchmarkScenario(
    name="Invoice 02 — Consulting",
    description="Consulting invoice with customer number, contract number, and Heidelberg address.",
    schema=InvoiceData,
    text=INVOICE_TEXT,
    system_prompt="You are an invoice data extraction assistant. Extract the requested fields exactly as they appear in the document.",
    expected={
        "invoice_number": "RE-2024-0619",
        "contract_number": "VTR-2024-0055",
        "customer_number": "KD-10234",
        "invoice_date": "2024-06-01",
        "billed_company_name": "MediCare Verwaltungs GmbH",
        "issuing_company_name": "Berger & Partner Management Consulting GmbH",
        "total_amount": 20432.30,
        "country": "DE",
        "vat_id": "DE301847192",
    },
)


async def main() -> None:
    print("Benchmark: Invoice 02 — Consulting")
    await run_all_models(SCENARIO)


if __name__ == "__main__":
    asyncio.run(main())
