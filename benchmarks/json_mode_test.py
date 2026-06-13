"""JSON Mode Benchmark — Extraction without tool calling.

This script directly compares ExtractionMode.TOOL_CALLING vs ExtractionMode.JSON
for the same invoice text and all three models. It demonstrates the difference in
behaviour, reliability, and retry count when the model is asked to produce raw JSON
instead of using the LLM's tool/function-calling capability.

Many smaller local models (like llama3.2:3b) do not reliably support tool calling,
making JSON mode the practical choice for those. This script makes that trade-off
visible with real measurements.

Run standalone:
    python benchmarks/json_mode_test.py
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from saidex import ExtractionMode, extract_from_text

from ._config import MODELS, make_llm

# ---------------------------------------------------------------------------
# Schema — reusing invoice fields for a direct apples-to-apples comparison
# ---------------------------------------------------------------------------


class InvoiceData(BaseModel):
    """Key fields extracted from an invoice document."""

    contract_number: str | None = Field(None, description="Contract or subscription number")
    customer_number: str | None = Field(None, description="Customer ID or account number")
    invoice_date: str | None = Field(None, description="Invoice issue date in ISO format if possible")
    address: str | None = Field(None, description="Full billing address of the customer")
    company_name: str | None = Field(None, description="Name of the company being billed")


# ---------------------------------------------------------------------------
# Invoice text (same as invoice_01 so results are directly comparable)
# ---------------------------------------------------------------------------

INVOICE_TEXT = """
INVOICE

TechFlow Solutions GmbH
Leopoldstraße 112
80802 Munich, Germany

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
Plan:         TechFlow Enterprise Cloud — Annual License
Period:       01 March 2024 – 28 February 2025
Seats:        50 active user seats

ITEM                                      QTY    UNIT PRICE       AMOUNT
─────────────────────────────────────────────────────────────────────────
Enterprise License (annual, per seat)      50      € 28.00/mo    € 16,800.00
Priority Support Add-on                     1     € 199.00/mo    €  2,388.00
─────────────────────────────────────────────────────────────────────────
                                                    TOTAL DUE:   € 22,573.44
─────────────────────────────────────────────────────────────────────────
"""

SYSTEM_PROMPT = "You are an invoice data extraction assistant. Extract the requested fields exactly as they appear in the document."


# ---------------------------------------------------------------------------
# Per-mode runner
# ---------------------------------------------------------------------------


async def run_mode(
    llm: ChatOpenAI,
    mode: ExtractionMode,
    model_name: str,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    try:
        value, stats = await extract_from_text(
            llm,
            InvoiceData,
            INVOICE_TEXT,
            mode=mode,
            system_prompt=SYSTEM_PROMPT,
        )
        duration = time.perf_counter() - t0
        return {
            "model": model_name,
            "mode": mode.value,
            "success": value is not None,
            "duration_s": round(duration, 2),
            "retries": stats.total_retries,
            "value": value.model_dump() if value else None,
        }
    except Exception as exc:  # noqa: BLE001
        duration = time.perf_counter() - t0
        return {
            "model": model_name,
            "mode": mode.value,
            "success": False,
            "duration_s": round(duration, 2),
            "retries": 0,
            "value": None,
            "error": str(exc),
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _print_result(r: dict[str, Any]) -> None:
    ok = "OK " if r["success"] else "FAIL"
    val = r.get("value") or {}
    fields = "  ".join(f"{k}={v}" for k, v in val.items() if v) if val else r.get("error", "—")
    print(
        f"  [{ok}] {r['model']:<20} mode={r['mode']:<12} "
        f"{r['duration_s']:>5.1f}s  retries={r['retries']}  {fields}"
    )


async def main() -> None:
    print("=" * 70)
    print("  JSON Mode vs Tool-Calling Mode — Invoice Extraction")
    print("  Same invoice text, same schema, three models, both modes")
    print("=" * 70)

    results: list[dict[str, Any]] = []

    for model_name in MODELS:
        llm = make_llm(model_name)
        for mode in (ExtractionMode.TOOL_CALLING, ExtractionMode.JSON):
            label = f"{model_name} [{mode.value}]"
            print(f"  Running {label}…", end=" ", flush=True)
            r = await run_mode(llm, mode, model_name)
            status = f"{r['duration_s']:.1f}s" if r["success"] else f"FAILED"
            print(status)
            results.append(r)

    print()
    print(f"{'=' * 70}")
    print("  Results summary")
    print(f"{'=' * 70}")
    for r in results:
        _print_result(r)

    print()
    print("Legend: mode=tool_calling uses LLM function/tool calling (may fail on small models)")
    print("        mode=json        injects JSON schema into prompt — works with any model")


if __name__ == "__main__":
    asyncio.run(main())
