"""Example 5 — Multimodal: extract structured data from images.

Demonstrates three scenarios:
  A) Extract from an image URL         → OpenAI GPT-4o (cloud)
  B) Extract from a local image file   → OpenAI GPT-4o (cloud, base64-encoded)
  C) Extract from an image URL         → self-hosted vLLM vision model (local)
  D) Extract from multiple images      → combined analysis into one schema

Requires:
    pip install langchain-openai

For scenario C also requires a running vLLM server:
    pip install vllm
    vllm serve Qwen/Qwen2-VL-7B-Instruct --port 8000

Run:
    python examples/05_image_analysis.py
"""

from __future__ import annotations

import asyncio
import base64
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from saidex import extract_data

# ---------------------------------------------------------------------------
# Helper functions — build LangChain multimodal messages from images
# ---------------------------------------------------------------------------


def image_message_from_url(
    url: str,
    prompt: str = "Analyse this image and extract the required information.",
    *,
    detail: str = "high",
) -> HumanMessage:
    """Create a multimodal HumanMessage from a public image URL.

    Args:
        url: Publicly accessible image URL (HTTPS).
        prompt: Text instruction accompanying the image.
        detail: OpenAI vision detail level — ``"low"``, ``"high"``, or
            ``"auto"`` (ignored by non-OpenAI models).

    Returns:
        A :class:`~langchain_core.messages.HumanMessage` with mixed
        text + image content.
    """
    return HumanMessage(
        content=[
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": url, "detail": detail}},
        ]
    )


def image_message_from_file(
    path: str | Path,
    prompt: str = "Analyse this image and extract the required information.",
    *,
    media_type: str | None = None,
) -> HumanMessage:
    """Create a multimodal HumanMessage from a local image file (base64-encoded).

    Args:
        path: Path to a local image file (JPEG, PNG, GIF, or WebP).
        prompt: Text instruction accompanying the image.
        media_type: MIME type such as ``"image/jpeg"`` or ``"image/png"``.
            Inferred from the file extension when not provided.

    Returns:
        A :class:`~langchain_core.messages.HumanMessage` with mixed
        text + image content.
    """
    path = Path(path)
    suffix_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    mime = media_type or suffix_map.get(path.suffix.lower(), "image/jpeg")
    image_b64 = base64.b64encode(path.read_bytes()).decode()
    data_url = f"data:{mime};base64,{image_b64}"
    return HumanMessage(
        content=[
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": data_url}},
        ]
    )


def multi_image_message(
    images: list[str | Path],
    prompt: str = "Analyse all images and extract the required information.",
) -> HumanMessage:
    """Create a HumanMessage that contains multiple images.

    Each item in *images* is treated as a URL when it starts with ``http``
    or ``https``, and as a local file path otherwise.

    Args:
        images: List of image URLs or local file paths.
        prompt: Single text instruction that applies to all images.

    Returns:
        A :class:`~langchain_core.messages.HumanMessage` with one text block
        followed by one image block per image.
    """
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for img in images:
        img_str = str(img)
        if img_str.startswith(("http://", "https://")):
            content.append({"type": "image_url", "image_url": {"url": img_str}})
        else:
            path = Path(img_str)
            suffix_map = {
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".png": "image/png",
                ".gif": "image/gif",
                ".webp": "image/webp",
            }
            mime = suffix_map.get(path.suffix.lower(), "image/jpeg")
            b64 = base64.b64encode(path.read_bytes()).decode()
            content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}})
    return HumanMessage(content=content)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class ReceiptItem(BaseModel):
    description: str
    quantity: int = Field(default=1, ge=1)
    unit_price: float = Field(ge=0)
    total: float = Field(ge=0)


class Receipt(BaseModel):
    """Structured data extracted from a receipt or invoice image."""

    vendor: str = Field(description="Name of the shop, restaurant, or vendor")
    date: str | None = Field(default=None, description="Date in ISO 8601 format (YYYY-MM-DD)")
    items: list[ReceiptItem] = Field(description="Line items on the receipt")
    subtotal: float | None = Field(default=None, ge=0, description="Subtotal before tax")
    tax: float | None = Field(default=None, ge=0, description="Tax amount")
    total: float = Field(ge=0, description="Final total amount paid")
    currency: str = Field(default="EUR", description="ISO 4217 currency code, e.g. EUR or USD")
    payment_method: str | None = Field(default=None, description="e.g. cash, credit card, PayPal")


class BusinessCard(BaseModel):
    """Contact information extracted from a business card image."""

    full_name: str
    job_title: str | None = None
    company: str | None = None
    email: str | None = None
    phone: str | None = None
    website: str | None = None
    address: str | None = None


class ChartData(BaseModel):
    """Key data points extracted from a chart or graph image."""

    chart_type: str = Field(description="e.g. bar chart, line chart, pie chart")
    title: str | None = None
    x_axis_label: str | None = None
    y_axis_label: str | None = None
    data_points: list[str] = Field(
        description="Key values or labels visible in the chart (up to 10)"
    )
    main_insight: str = Field(description="One-sentence summary of the chart's main message")


# ---------------------------------------------------------------------------
# Scenario A — Image from URL, OpenAI GPT-4o
# ---------------------------------------------------------------------------


async def scenario_a_url_openai() -> None:
    """Extract receipt data from a public image URL using GPT-4o."""
    print("=== Scenario A: Image URL → GPT-4o ===\n")

    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    # Public sample receipt image (replace with your own URL)
    receipt_url = "https://upload.wikimedia.org/wikipedia/commons/0/0b/ReceiptSwiss.jpg"

    messages = [
        SystemMessage(
            content=(
                "You are a precise receipt-parsing assistant. "
                "Extract all structured data from the receipt image exactly as shown. "
                "Use null for fields that are not visible."
            )
        ),
        image_message_from_url(receipt_url, prompt="Extract the receipt data from this image."),
    ]

    receipt, stats = await extract_data(llm, Receipt, messages)

    if receipt is None:
        print(f"  Failed after {stats.total_retries} retries.\n")
        return

    print(f"  Vendor:   {receipt.vendor}")
    print(f"  Date:     {receipt.date}")
    print(f"  Total:    {receipt.total} {receipt.currency}")
    print(f"  Items:    {len(receipt.items)}")
    for item in receipt.items:
        print(f"    - {item.description}: {item.total}")
    print(f"  Retries:  {stats.total_retries}\n")


# ---------------------------------------------------------------------------
# Scenario B — Local image file (base64), OpenAI GPT-4o
# ---------------------------------------------------------------------------


async def scenario_b_file_openai(image_path: str) -> None:
    """Extract business card data from a local image file using GPT-4o."""
    print(f"=== Scenario B: Local file ({image_path}) → GPT-4o ===\n")

    if not Path(image_path).exists():
        print("  Image file not found — skipping.\n")
        return

    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    messages = [
        SystemMessage(
            content=(
                "You are a contact-information extractor. "
                "Read all visible text from the business card image and "
                "populate the schema exactly. Use null for any field not present."
            )
        ),
        image_message_from_file(
            image_path,
            prompt="Extract the contact details from this business card.",
        ),
    ]

    card, stats = await extract_data(llm, BusinessCard, messages)

    if card is None:
        print(f"  Failed after {stats.total_retries} retries.\n")
        return

    print(f"  Name:     {card.full_name}")
    print(f"  Title:    {card.job_title}")
    print(f"  Company:  {card.company}")
    print(f"  Email:    {card.email}")
    print(f"  Phone:    {card.phone}")
    print(f"  Retries:  {stats.total_retries}\n")


# ---------------------------------------------------------------------------
# Scenario C — self-hosted vLLM vision model
# ---------------------------------------------------------------------------


async def scenario_c_vllm() -> None:
    """Extract structured data using a locally hosted vLLM vision model.

    Prerequisites:
        pip install vllm
        vllm serve Qwen/Qwen2-VL-7B-Instruct --port 8000

    Other popular vision models supported by vLLM:
        - Qwen/Qwen2-VL-7B-Instruct          (recommended, strong OCR)
        - Qwen/Qwen2.5-VL-7B-Instruct
        - llava-hf/llava-1.5-7b-hf
        - microsoft/Phi-3.5-vision-instruct
        - OpenGVLab/InternVL2-8B
    """
    print("=== Scenario C: Image URL → vLLM (local) ===\n")

    # Connect to vLLM server — it exposes an OpenAI-compatible API
    vllm = ChatOpenAI(
        model="Qwen/Qwen2-VL-7B-Instruct",  # must match the model loaded in vLLM
        base_url="http://localhost:8000/v1",  # vLLM default endpoint
        api_key="EMPTY",  # vLLM does not require a real key
        temperature=0,
        max_tokens=1024,
    )

    chart_url = (
        "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4b/"
        "Global_Carbon_Emission_by_Type_v2.png/1024px-Global_Carbon_Emission_by_Type_v2.png"
    )

    messages = [
        SystemMessage(
            content=(
                "You are a data analyst. Extract key facts and data points from "
                "the chart image. Be precise with numbers and labels."
            )
        ),
        image_message_from_url(
            chart_url,
            prompt="Extract the chart data and main insight from this image.",
            detail="auto",  # some vLLM models ignore this — safe to pass
        ),
    ]

    chart, stats = await extract_data(vllm, ChartData, messages)

    if chart is None:
        print(
            f"  Failed after {stats.total_retries} retries. "
            f"Is the vLLM server running on localhost:8000?\n"
        )
        return

    print(f"  Chart type:    {chart.chart_type}")
    print(f"  Title:         {chart.title}")
    print(f"  X-axis:        {chart.x_axis_label}")
    print(f"  Y-axis:        {chart.y_axis_label}")
    print(f"  Data points:   {chart.data_points}")
    print(f"  Main insight:  {chart.main_insight}")
    print(f"  Retries:       {stats.total_retries}\n")


# ---------------------------------------------------------------------------
# Scenario D — Multiple images in one request + fallback model
# ---------------------------------------------------------------------------


async def scenario_d_multi_image() -> None:
    """Analyse multiple images in a single message with a fallback model."""
    print("=== Scenario D: Multiple images → GPT-4o-mini + GPT-4o fallback ===\n")

    primary = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    fallback = ChatOpenAI(model="gpt-4o", temperature=0)

    image_urls = [
        "https://upload.wikimedia.org/wikipedia/commons/0/0b/ReceiptSwiss.jpg",
    ]

    messages = [
        SystemMessage(
            content=(
                "You are a receipt-parsing assistant. "
                "Analyse the provided receipt image(s) and extract the combined totals."
            )
        ),
        multi_image_message(
            image_urls,
            prompt="Extract the receipt data from these images.",
        ),
    ]

    receipt, stats = await extract_data(
        primary,
        Receipt,
        messages,
        fallback_llm_model=fallback,
        max_primary_retries=2,
    )

    if receipt is None:
        print(f"  Failed after {stats.total_retries} retries.\n")
        return

    print(f"  Vendor:        {receipt.vendor}")
    print(f"  Total:         {receipt.total} {receipt.currency}")
    print(f"  Fallback used: {stats.fallback_used}")
    print(f"  Retries:       {stats.total_retries}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    await scenario_a_url_openai()
    await scenario_b_file_openai("business_card.jpg")  # put your own file here
    # await scenario_c_vllm()   # uncomment when vLLM server is running
    await scenario_d_multi_image()


if __name__ == "__main__":
    asyncio.run(main())
