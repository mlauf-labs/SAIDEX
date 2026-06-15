"""Example 2 — Extract structured output from a multi-turn chat history.

This example shows how to pass a full LangChain conversation (SystemMessage +
multiple HumanMessage / AIMessage turns) to ``run_structured_output`` directly.

Run:
    python examples/02_chat_history.py
"""

from __future__ import annotations

import asyncio
from enum import Enum

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from saidex import extract_data

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


class Sentiment(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    MIXED = "mixed"


class SupportTicketSummary(BaseModel):
    """Summary of a customer-support conversation."""

    topic: str = Field(description="Short topic or subject of the support request")
    sentiment: Sentiment = Field(description="Overall sentiment of the customer")
    resolved: bool = Field(description="Whether the issue was resolved in this conversation")
    action_items: list[str] = Field(
        description="List of follow-up actions required (empty list if none)"
    )
    priority: int = Field(description="Urgency priority 1 (low) to 5 (critical)", ge=1, le=5)


# ---------------------------------------------------------------------------
# Simulated conversation
# ---------------------------------------------------------------------------

MESSAGES = [
    SystemMessage(
        content=(
            "You are a support analyst. Analyse the following customer-support "
            "conversation and extract the required structured summary."
        )
    ),
    HumanMessage(content="Here is the conversation to analyse:"),
    AIMessage(
        content=(
            "Support: Hello, how can I help you today?\n"
            "Customer: My order #12345 hasn't arrived yet. It's been two weeks!\n"
            "Support: I'm sorry to hear that. Let me check the status …\n"
            "Support: It looks like the package was lost in transit. "
            "We will ship a replacement immediately.\n"
            "Customer: Thank you, but I needed it for an event yesterday. "
            "This is very frustrating.\n"
            "Support: I completely understand. I'll also apply a 15 % discount "
            "to your next order as an apology.\n"
            "Customer: Okay, I appreciate that."
        )
    ),
    HumanMessage(content="Please provide the structured summary now."),
]


# ---------------------------------------------------------------------------
# Run extraction
# ---------------------------------------------------------------------------


async def main() -> None:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    summary, stats = await extract_data(llm, SupportTicketSummary, MESSAGES)

    if summary is None:
        print(f"Extraction failed after {stats.total_retries} retries.")
        return

    print("Support ticket summary:")
    print(f"  Topic:        {summary.topic}")
    print(f"  Sentiment:    {summary.sentiment.value}")
    print(f"  Resolved:     {summary.resolved}")
    print(f"  Priority:     {summary.priority}/5")
    print(f"  Action items: {summary.action_items or 'none'}")
    print(f"\nStats: primary_retries={stats.primary_retries}, fallback_used={stats.fallback_used}")


if __name__ == "__main__":
    asyncio.run(main())
