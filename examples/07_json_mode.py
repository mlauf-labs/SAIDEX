"""Example 7 — Extraction without tool calling (JSON mode).

By default the library uses tool/function calling, which requires a capable
model.  ``ExtractionMode.JSON`` instead injects the schema's JSON Schema into
the prompt and parses the model's raw text reply as JSON.  This works with
*any* chat model — including local models served by Ollama, llama.cpp, or an
older API that does not support tool calling.

You can still use JSON mode with a tool-calling-capable model (e.g. to compare
behaviour or to work around provider-specific tool-calling quirks).

Run:
    python examples/07_json_mode.py
"""

from __future__ import annotations

import asyncio

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from saidex import ExtractionMode, extract_from_text, get_structured_data

# ---------------------------------------------------------------------------
# Define your schema
# ---------------------------------------------------------------------------


class Product(BaseModel):
    """A product extracted from a free-text description."""

    name: str = Field(description="Product name")
    price_eur: float = Field(ge=0, description="Price in EUR, e.g. 19.99")
    in_stock: bool = Field(description="Whether the product is currently available")
    tags: list[str] = Field(default_factory=list, description="Category or keyword tags")


# ---------------------------------------------------------------------------
# A) JSON mode with a standard OpenAI model
# ---------------------------------------------------------------------------


async def extract_with_openai_json() -> None:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    text = "The UltraWidget Pro costs 49.90 EUR, is in stock, and is a gadget and tool."

    product, stats = await extract_from_text(
        llm,
        Product,
        text,
        mode=ExtractionMode.JSON,  # <-- no tool calling
    )

    print("A) OpenAI in JSON mode:")
    if product is None:
        print(f"   Failed after {stats.total_retries} retries.")
    else:
        print(f"   {product.model_dump()}  (retries: {stats.total_retries})")


# ---------------------------------------------------------------------------
# B) JSON mode against a local, non-tool-calling model (e.g. Ollama)
# ---------------------------------------------------------------------------


async def extract_with_local_model() -> None:
    """Point ChatOpenAI at any OpenAI-compatible server (Ollama, vLLM, …).

    Many smaller local models do not support tool calling — JSON mode is the
    way to use them for structured extraction.
    """
    llm = ChatOpenAI(
        model="llama3.1",
        base_url="http://localhost:11434/v1",  # Ollama's OpenAI-compatible endpoint
        api_key="not-needed",
        temperature=0,
    )

    messages = [
        SystemMessage(content="Extract the product details from the user's message."),
        HumanMessage(content="SilentMouse M2 — 24,99 €, sold out. Tags: mouse, wireless."),
    ]

    product, stats = await get_structured_data(
        llm,
        Product,
        messages,
        mode=ExtractionMode.JSON,
    )

    print("\nB) Local model (Ollama) in JSON mode:")
    if product is None:
        print(f"   Failed after {stats.total_retries} retries.")
    else:
        print(f"   {product.model_dump()}  (retries: {stats.total_retries})")


async def main() -> None:
    await extract_with_openai_json()
    # Uncomment when a local server is running:
    # await extract_with_local_model()


if __name__ == "__main__":
    asyncio.run(main())
