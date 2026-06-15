"""Example 8 — Agentic tool loop (extract_data_with_tools / run_extractor_agent).

Sometimes an extraction needs more than the text in the prompt: a database
lookup, an API call, or a resource that must be created first.
``extract_data_with_tools`` runs an agent loop in which the LLM may call your
:class:`saidex.Tool`s any number of times and then delivers a final answer
validated against your Pydantic schema.

Run:
    python examples/08_agent_tools.py
"""

from __future__ import annotations

import asyncio

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from saidex import Tool, extract_data_with_tools, run_extractor_agent

# ---------------------------------------------------------------------------
# Fake backend the tools will query
# ---------------------------------------------------------------------------

ORDERS = {
    "ORD-1042": {"status": "shipped", "eta": "2026-06-15", "carrier": "DHL"},
    "ORD-2077": {"status": "processing", "eta": None, "carrier": None},
}

CUSTOMERS = {
    "alice@example.com": {"name": "Alice Müller", "tier": "premium"},
}


# ---------------------------------------------------------------------------
# Define the tools: a Pydantic argument schema + an async handler each
# ---------------------------------------------------------------------------


class OrderStatusArgs(BaseModel):
    """Arguments for the get_order_status tool."""

    order_id: str = Field(description="Order id, e.g. ORD-1042")


async def order_status_handler(order_id: str) -> dict:
    order = ORDERS.get(order_id)
    if order is None:
        return {"error": f"Order '{order_id}' not found"}
    return {"order_id": order_id, **order}


class CustomerArgs(BaseModel):
    """Arguments for the get_customer tool."""

    email: str = Field(description="Customer email address")


async def customer_handler(email: str) -> dict:
    customer = CUSTOMERS.get(email.lower())
    if customer is None:
        return {"error": f"No customer with email '{email}'"}
    return customer


order_status_tool = Tool(
    name="get_order_status",
    description="Look up the current status, ETA, and carrier of an order by its id.",
    parameters=OrderStatusArgs,
    handler=order_status_handler,
)

customer_tool = Tool(
    name="get_customer",
    description="Look up a customer's name and support tier by email address.",
    parameters=CustomerArgs,
    handler=customer_handler,
)


# ---------------------------------------------------------------------------
# Define the final-answer schema
# ---------------------------------------------------------------------------


class TicketResolution(BaseModel):
    """Validated final answer the agent loop must produce."""

    order_id: str = Field(description="The order this ticket is about")
    current_status: str = Field(description="Order status as reported by the tools")
    customer_name: str = Field(description="Customer name from the customer lookup")
    customer_reply: str = Field(
        description="Friendly, complete reply that can be sent to the customer"
    )


# ---------------------------------------------------------------------------
# A) extract_data_with_tools — task as plain text
# ---------------------------------------------------------------------------


async def resolve_ticket_from_text() -> None:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    ticket = (
        "From: alice@example.com\n"
        "Subject: Where is my order?\n\n"
        "Hi, I ordered two weeks ago (order ORD-1042) and still have nothing. "
        "Can you tell me what's going on?"
    )

    resolution, stats = await extract_data_with_tools(
        llm,
        TicketResolution,
        ticket,
        tools=[order_status_tool, customer_tool],
    )

    print("A) extract_data_with_tools:")
    if resolution is None:
        print(f"   Failed after {stats.iterations} iterations.")
    else:
        print(f"   status={resolution.current_status!r} for {resolution.customer_name}")
        print(f"   reply: {resolution.customer_reply}")
    print(
        f"   stats: iterations={stats.iterations} tool_calls={stats.tool_calls} "
        f"validation_retries={stats.validation_retries} fallback_used={stats.fallback_used}"
    )


# ---------------------------------------------------------------------------
# B) run_extractor_agent — full control over the message list
# ---------------------------------------------------------------------------


async def resolve_ticket_with_history() -> None:
    """Use run_extractor_agent when you need to shape the conversation yourself."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    messages = [
        SystemMessage(
            content=(
                "You are a support agent. Use the available tools to gather "
                "facts before answering. Reply in the customer's language."
            )
        ),
        HumanMessage(content="Previous note: customer already received one apology voucher."),
        HumanMessage(content="Customer alice@example.com asks about order ORD-1042."),
    ]

    resolution, stats = await run_extractor_agent(
        llm,
        TicketResolution,
        messages,
        tools=[order_status_tool, customer_tool],
    )

    print("\nB) run_extractor_agent with custom history:")
    if resolution is None:
        print(f"   Failed after {stats.iterations} iterations.")
    else:
        print(f"   reply: {resolution.customer_reply}")


async def main() -> None:
    await resolve_ticket_from_text()
    await resolve_ticket_with_history()


if __name__ == "__main__":
    asyncio.run(main())
