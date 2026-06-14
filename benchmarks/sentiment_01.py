"""Sentiment Benchmark 01 — Enthusiastic positive customer review (German).

Long, detailed, genuinely friendly customer feedback for a hotel stay.

Run standalone:
    python benchmarks/sentiment_01.py
"""

from __future__ import annotations

import asyncio
from typing import Literal

from pydantic import BaseModel, Field

from ._base import AtLeast, BenchmarkScenario, run_all_models

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


class SentimentResult(BaseModel):
    """Sentiment analysis result for a customer feedback text."""

    sentiment: Literal["positive", "neutral", "negative"] = Field(
        description="Overall sentiment of the text"
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidence score between 0.0 (uncertain) and 1.0 (very certain)"
    )
    key_phrases: list[str] = Field(
        description="3–6 key phrases or sentences that most strongly reflect the sentiment"
    )
    summary: str = Field(
        description="One to two sentence summary of the overall tone and main points"
    )
    emotions: list[str] = Field(
        description="Dominant emotions expressed (e.g. joy, gratitude, excitement, frustration)"
    )


# ---------------------------------------------------------------------------
# Test text  — positive / friendly
# ---------------------------------------------------------------------------

REVIEW_TEXT = """
We spent three nights at Hotel Seegarten last weekend, and I honestly cannot find
the words to describe how wonderful the stay was. From the moment we arrived, we were
greeted by Mr. Mayer at the reception desk with a beaming smile — it was immediately
clear that the entire team here genuinely puts their heart into the job, rather than
simply going through the motions.

The room itself was a dream: enormous, spotlessly clean, with a breathtaking view of
the lake. The balcony quickly became a fixed part of our morning routine — coffee in
hand, watching the sun rise over the mountains, with nothing but the gentle lapping
of the water to break the silence. These are the moments you take holidays for.

The breakfast buffet was simply outstanding. Fresh rolls straight from the in-house
bakery, home-made jams in at least twelve varieties, scrambled eggs made to order,
local cheese and cold cuts, and a fruit salad that could not have been any fresher.
We spent at least ninety minutes there every morning because it was simply too good
to leave in a hurry. A special mention goes to Ms. Gruber, who looked after the
buffet and always noticed the instant something needed refilling.

On our first evening we dined at the hotel's own restaurant. The menu was compact,
but every dish was at a standard we honestly hadn't expected. My trout fillet with
herb butter and rosemary potatoes was cooked to perfection, and the wine recommended
by the sommelier was a real discovery. My partner is a vegetarian, and even she was
delighted — the vegetable risotto with truffle oil and Parmesan was her favourite
meal of the entire holiday.

The pool was always pleasantly warm and never overcrowded. The sun loungers on the
lakeside jetty were freshly laid with towels every morning — without having to get
up early to reserve a spot, which we know from other holidays can be a real headache.
One afternoon we treated ourselves to an hour in the wellness area: sauna, infrared
cabin, and a small steam room — everything immaculately maintained, peaceful, and
deeply relaxing.

What struck us most was how well the staff remembers the little details. On the third
day, Mr. Mayer at reception brought up our hiking trip from the day before and asked
whether we had made it to the viewpoint. That personal touch makes an enormous
difference. You feel not like a room number, but like a genuinely welcome guest.

The surroundings are fantastic as well: hiking trails start right from the hotel,
winding through dense forests and along babbling brooks. The local village festival
happened to take place on our last evening — the receptionist mentioned it and even
managed to get us tickets for the festival tent.

Conclusion: we will definitely be back. Hotel Seegarten is the finest hotel we have
stayed in during the last ten years. For anyone seeking peace, nature, and truly warm
hospitality — you will not be disappointed.
"""


SCENARIO = BenchmarkScenario(
    name="Sentiment 01 — Positive Hotel Review",
    description="Long, very positive and enthusiastic hotel review.",
    schema=SentimentResult,
    text=REVIEW_TEXT,
    system_prompt="Analyse the sentiment of the following customer feedback precisely and in detail.",
    # confidence: a clear-cut positive review — the model should be confident,
    # but the exact value is model-subjective, so check a sensible floor rather
    # than an arbitrary exact number (which would cause false negatives).
    expected={"sentiment": "positive", "confidence": AtLeast(0.8)},
)


async def main() -> None:
    print("Benchmark: Sentiment 01 — Positive Hotel Review")
    await run_all_models(SCENARIO)


if __name__ == "__main__":
    asyncio.run(main())
