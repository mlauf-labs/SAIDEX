"""Example 4 — Custom retry configuration and Langfuse tracing.

Demonstrates:
- Custom network-level retry configuration (RetryConfig)
- Optional Langfuse integration via LangChain callbacks
- Using ``create_instance_safe`` as a standalone helper

Run:
    python examples/04_custom_retry_config.py
"""

from __future__ import annotations

import asyncio

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from saidex import RetryConfig, create_instance_safe, extract_data_from_text

# ---------------------------------------------------------------------------
# Custom retry config: more patient with rate limits, fewer transient retries
# ---------------------------------------------------------------------------

CONSERVATIVE_RETRY = RetryConfig(
    max_retries=2,
    retry_delays=[2.0, 5.0],  # longer waits between transient retries
    rate_limit_retry_interval=30.0,  # wait 30 s between rate-limit retries
    rate_limit_max_duration_seconds=300.0,  # give up after 5 minutes
)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


class ProductReview(BaseModel):
    product_name: str
    rating: int = Field(ge=1, le=5, description="Star rating 1–5")
    pros: list[str] = Field(description="Positive aspects mentioned")
    cons: list[str] = Field(description="Negative aspects mentioned")
    summary: str = Field(description="One-sentence summary of the review")


# ---------------------------------------------------------------------------
# Standalone usage of create_instance_safe
# ---------------------------------------------------------------------------


def demonstrate_create_instance_safe() -> None:
    """Show how create_instance_safe works without an LLM."""
    print("=== create_instance_safe demo ===\n")

    # Valid data
    instance, error = create_instance_safe(
        ProductReview,
        product_name="Widget Pro",
        rating=4,
        pros=["great build quality", "easy to use"],
        cons=["a bit pricey"],
        summary="A solid product worth the investment for professionals.",
    )
    if instance:
        print(f"Success: {instance.product_name} ({instance.rating}/5)")
    else:
        print(f"Error:\n{error}")

    print()

    # Invalid data — rating out of range + wrong type for pros
    bad_instance, bad_error = create_instance_safe(
        ProductReview,
        product_name="Widget Pro",
        rating=7,  # invalid: must be 1-5
        pros="great build",  # invalid: must be a list
        cons=[],
        summary="Good.",
    )
    if bad_instance is None:
        print("Validation error (as expected):")
        print(bad_error)


# ---------------------------------------------------------------------------
# LLM extraction with custom retry and optional Langfuse tracing
# ---------------------------------------------------------------------------


async def main() -> None:
    demonstrate_create_instance_safe()

    print("\n=== LLM extraction with custom RetryConfig ===\n")

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    review_text = """
    I've been using the NoisePro 3000 headphones for a month now.
    The sound quality is incredible and the noise cancellation is best-in-class.
    Battery life lasts a full 30 hours — impressive!
    On the downside, they feel a bit plasticky for the price, and the ear cups
    can get warm after long sessions.  Overall I'd give it 4 stars.
    """

    # Optional: add Langfuse tracing by installing langfuse and passing callbacks
    # from langfuse.callback import CallbackHandler
    # callbacks = [CallbackHandler()]
    callbacks = None

    review, stats = await extract_data_from_text(
        llm,
        ProductReview,
        review_text,
        callbacks=callbacks,
        retry_config=CONSERVATIVE_RETRY,
    )

    if review is None:
        print(f"Extraction failed after {stats.total_retries} retries.")
        return

    print(f"Product:  {review.product_name}")
    print(f"Rating:   {'★' * review.rating}{'☆' * (5 - review.rating)} ({review.rating}/5)")
    print(f"Pros:     {', '.join(review.pros)}")
    print(f"Cons:     {', '.join(review.cons)}")
    print(f"Summary:  {review.summary}")
    print(f"\nRetries:  {stats.total_retries}")


if __name__ == "__main__":
    asyncio.run(main())
