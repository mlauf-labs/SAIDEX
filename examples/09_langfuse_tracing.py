"""Example 9 — Langfuse tracing (single extraction, retry, agent loop).

Observability is a common production requirement. Because SAIDEX is
LangChain-native, it integrates with `Langfuse <https://langfuse.com>`_ through
the standard LangChain callback handler — pass it via the ``callbacks``
parameter accepted by every public function. No SAIDEX-specific glue is needed.

This script traces three scenarios in one run:

  A) a single happy-path extraction,
  B) an extraction that triggers a validation retry (strict IBAN field),
  C) a multi-step agent loop that calls a tool before answering.

It is written to run even *without* Langfuse: if the package is missing or no
credentials are configured, tracing is skipped and the extractions still run.

Setup (uses Langfuse v3 — note the ``langfuse.langchain`` import):
    uv pip install "langfuse>=3" langchain-openai
    export LANGFUSE_PUBLIC_KEY="pk-lf-..."
    export LANGFUSE_SECRET_KEY="sk-lf-..."
    export LANGFUSE_HOST="https://cloud.langfuse.com"

Run:
    python examples/09_langfuse_tracing.py
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from saidex import IbanStr, Tool, extract_from_text, extract_with_tools

# ---------------------------------------------------------------------------
# Optional Langfuse wiring — degrade gracefully when it is unavailable
# ---------------------------------------------------------------------------


def build_langfuse_handler() -> tuple[Any | None, Any | None]:
    """Return ``(handler, client)`` if Langfuse is installed and configured.

    Returns:
        A tuple of the LangChain ``CallbackHandler`` and the Langfuse client,
        or ``(None, None)`` when tracing is unavailable.  Callers pass the
        handler straight into ``callbacks=[...]`` and flush the client at exit.
    """
    if not (os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")):
        print("[langfuse] credentials not set — running without tracing.\n")
        return None, None

    try:
        from langfuse import get_client
        from langfuse.langchain import CallbackHandler
    except ImportError:
        print('[langfuse] not installed — run: uv pip install "langfuse>=3"\n')
        return None, None

    return CallbackHandler(), get_client()


def callbacks_for(handler: Any | None) -> list[Any] | None:
    """Wrap a handler in the list SAIDEX expects, or ``None`` to skip tracing."""
    return [handler] if handler is not None else None


# ---------------------------------------------------------------------------
# Schemas and tools
# ---------------------------------------------------------------------------


class PersonInfo(BaseModel):
    """Information about a person extracted from free text."""

    name: str = Field(description="Full name of the person")
    age: int = Field(description="Age in years", ge=0, le=150)
    occupation: str = Field(description="Current job or profession")


class BankDetails(BaseModel):
    """Strict schema — a malformed IBAN forces a self-correcting retry."""

    holder: str = Field(description="Account holder name")
    iban: IbanStr = Field(description="IBAN, validated and normalised")


ORDERS = {"ORD-1042": {"status": "shipped", "eta": "2026-06-15", "carrier": "DHL"}}


class OrderStatusArgs(BaseModel):
    """Arguments for the get_order_status tool."""

    order_id: str = Field(description="Order id, e.g. ORD-1042")


async def order_status_handler(order_id: str) -> dict:
    return ORDERS.get(order_id, {"error": f"Order '{order_id}' not found"})


order_status_tool = Tool(
    name="get_order_status",
    description="Look up the status, ETA, and carrier of an order by its id.",
    parameters=OrderStatusArgs,
    handler=order_status_handler,
)


class TicketResolution(BaseModel):
    """Validated final answer the agent loop must produce."""

    order_id: str = Field(description="The order this ticket is about")
    current_status: str = Field(description="Order status as reported by the tool")
    customer_reply: str = Field(description="Friendly reply that can be sent to the customer")


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------


async def single_extraction(llm: ChatOpenAI, handler: Any | None) -> None:
    text = "Alice Müller, 34, is a software engineer based in Munich."
    person, stats = await extract_from_text(llm, PersonInfo, text, callbacks=callbacks_for(handler))
    print("A) single extraction:")
    print(f"   {person}  (retries: {stats.total_retries})\n")


async def extraction_with_retry(llm: ChatOpenAI, handler: Any | None) -> None:
    text = "Account holder: ACME GmbH. IBAN: DE89 3704 0044 0532 0130 00."
    result, stats = await extract_from_text(
        llm,
        BankDetails,
        text,
        callbacks=callbacks_for(handler),
        max_primary_retries=3,
    )
    print("B) extraction with possible retry:")
    print(f"   {result}  (retries: {stats.total_retries})\n")


async def agent_loop(llm: ChatOpenAI, handler: Any | None) -> None:
    ticket = "Where is my order ORD-1042? It's been two weeks!"
    resolution, stats = await extract_with_tools(
        llm,
        TicketResolution,
        ticket,
        tools=[order_status_tool],
        callbacks=callbacks_for(handler),
    )
    print("C) agent loop:")
    print(f"   {resolution}")
    print(f"   iterations={stats.iterations} tool_calls={stats.tool_calls}\n")


async def main() -> None:
    handler, client = build_langfuse_handler()
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    await single_extraction(llm, handler)
    await extraction_with_retry(llm, handler)
    await agent_loop(llm, handler)

    # Short-lived script: flush pending events before the process exits.
    if client is not None:
        client.flush()
        print("[langfuse] traces flushed — check your Langfuse dashboard.")


if __name__ == "__main__":
    asyncio.run(main())
