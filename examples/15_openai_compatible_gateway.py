"""Example 15 — Tool calling against an OpenAI-compatible gateway.

Tool-calling mode binds the schema with OpenAI's most reliable flag set: a
forced ``tool_choice``, ``strict=True`` (structured outputs) and
``parallel_tool_calls=False``.  Several OpenAI-*compatible* gateways — endpoints
hosting Kimi/Moonshot, some Qwen or DeepSeek deployments — answer one or more of
those flags with an HTTP 400, even though the model calls tools perfectly well.

``ToolCallConfig`` decides which flags are sent.  ``None`` means *omit the
keyword argument entirely*, which is not the same as sending ``False``.

Run:
    python examples/15_openai_compatible_gateway.py
"""

from __future__ import annotations

import asyncio
import logging

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from saidex import ToolCallConfig, extract_data_from_text

# The automatic recovery announces itself with a warning — worth seeing.
logging.basicConfig(level=logging.INFO)

TEXT = "Invoice INV-2031 from Contoso Ltd, dated 2026-03-04, total 1,499.00 EUR."


class Invoice(BaseModel):
    """An invoice extracted from free text."""

    number: str = Field(description="Invoice number, e.g. 'INV-2031'")
    vendor: str = Field(description="Name of the issuing company")
    total_eur: float = Field(ge=0, description="Invoice total in EUR")


# ---------------------------------------------------------------------------
# A) Nothing to configure — automatic recovery
# ---------------------------------------------------------------------------


async def relies_on_auto_relax() -> None:
    """The default path: SAIDEX notices the rejection and relaxes the flags itself.

    On an HTTP 400 naming ``strict`` / ``tool_choice`` / ``parallel_tool_calls``
    the schema tool is re-bound **once** with ``ToolCallConfig.COMPATIBLE`` and
    the attempt continues — without consuming a validation retry.
    """
    llm = ChatOpenAI(
        model="kimi-k2",
        base_url="https://api.moonshot.ai/v1",
        api_key="your-key",
        temperature=0,
    )

    invoice, stats = await extract_data_from_text(llm, Invoice, TEXT)

    print("A) Auto-relax:")
    if invoice is None:
        print(f"   Failed ({stats.failure_reason}) after {stats.total_retries} retries.")
    else:
        print(f"   {invoice.model_dump()}  (retries: {stats.total_retries})")


# ---------------------------------------------------------------------------
# B) Configure the flags up front
# ---------------------------------------------------------------------------


async def configures_the_flags() -> None:
    """Skip the failed first request by telling SAIDEX what the gateway supports."""
    llm = ChatOpenAI(
        model="kimi-k2",
        base_url="https://api.moonshot.ai/v1",
        api_key="your-key",
        temperature=0,
    )

    invoice, stats = await extract_data_from_text(
        llm,
        Invoice,
        TEXT,
        tool_config=ToolCallConfig.COMPATIBLE,  # tool_choice="auto", no strict/parallel flags
    )

    print("\nB) Explicit COMPATIBLE preset:")
    if invoice is None:
        print(f"   Failed ({stats.failure_reason}) after {stats.total_retries} retries.")
    else:
        print(f"   {invoice.model_dump()}  (retries: {stats.total_retries})")


# ---------------------------------------------------------------------------
# C) Fine-grained: keep what the gateway supports, drop only what it rejects
# ---------------------------------------------------------------------------


async def tunes_individual_flags() -> None:
    """A gateway that honours a forced ``tool_choice`` but rejects ``strict``.

    Keeping the forced choice is worth it — it guarantees the model answers with
    a tool call instead of prose.
    """
    llm = ChatOpenAI(
        model="qwen-max",
        base_url="https://your-gateway.example/v1",
        api_key="your-key",
        temperature=0,
    )

    invoice, stats = await extract_data_from_text(
        llm,
        Invoice,
        TEXT,
        tool_config=ToolCallConfig(
            tool_choice="forced",
            strict=None,  # omit the parameter entirely (not: send `false`)
            parallel_tool_calls=None,
            auto_relax=False,  # fail loudly instead of degrading silently
        ),
    )

    print("\nC) Hand-tuned flags:")
    if invoice is None:
        print(f"   Failed ({stats.failure_reason}) after {stats.total_retries} retries.")
    else:
        print(f"   {invoice.model_dump()}  (retries: {stats.total_retries})")


async def main() -> None:
    # Each function needs real credentials for its gateway — enable the one you have.
    await relies_on_auto_relax()
    # await configures_the_flags()
    # await tunes_individual_flags()


if __name__ == "__main__":
    asyncio.run(main())
