"""Sentiment Benchmark 02 — Neutral product test report (German, tech magazine style).

Long, factual, balanced product review with no clear positive or negative lean.

Run standalone:
    python benchmarks/sentiment_02.py
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
# Test text  — neutral / factual
# ---------------------------------------------------------------------------

REVIEW_TEXT = """
Review: Soundcore Liberty 4 NC – True Wireless Earbuds with Active Noise Cancellation

Build Quality and Design

The Soundcore Liberty 4 NC come in a matte black plastic housing that feels solid
and has no sharp edges. The charging case is compact and fits easily into a trouser
pocket. The earbuds themselves are made of soft silicone and ship with four different
ear-tip sizes. Compared to direct competitors in the same price bracket, build quality
sits squarely in the middle of the field: unremarkable, with no stand-out highs or lows.

Sound Profile

On the default equaliser preset the earbuds exhibit a noticeably boosted bass response,
which is typical for this price range. Mids and highs are reproduced accurately without
excessive colouration. For classical and jazz music the sound feels somewhat compressed;
for pop, hip-hop, and electronic genres the profile works well. The bundled Soundcore
app provides a 22-band equaliser that allows the sound to be tailored to personal
preferences. An additional personalised HearID hearing profile can be generated — a
feature that produces measurable differences in practice.

Active Noise Cancellation (ANC)

The ANC performance is impressive for the approximate price of €60. Deep, constant
sounds such as aircraft engines or train travel are noticeably reduced. High-frequency
noises such as voices or keyboard clicks, however, are only partially dampened. In a
direct comparison with Apple's AirPods Pro (2nd generation) or Sony's WF-1000XM5 —
both significantly more expensive — the ANC performance clearly falls short. For
occasional use in a home office or on public transport the function is adequate; for
frequent flyers or professional use, an upgrade would be advisable.

Battery Life and Connectivity

The stated battery life of eight hours (earbuds, ANC off) was reproducible in our
testing. With ANC enabled, battery life drops to approximately five and a half hours,
which precisely matches the manufacturer's specification. The case provides three
further charging cycles, giving a theoretical total runtime of up to 32 hours.

The Bluetooth 5.3 connection was stable in our tests with no notable dropouts.
Multipoint connection to two devices simultaneously works, but switching between
devices can cause a brief delay of one to two seconds. Latency is not disruptive in
everyday use, but is not optimal for gaming or video editing where precise lip-sync
is required.

Call Quality

During phone calls, the voices of conversation partners were described as "clear but
occasionally robotic." Wind noise is partially transmitted by the microphones. The
result sits at an average level for this product category.

Conclusion

The Soundcore Liberty 4 NC deliver solid performance across all tested categories
without excelling in any particular area. For buyers seeking a reliable everyday
earbud under €70 with usable ANC, these earbuds offer fair value for money. Those
expecting audiophile quality or professional noise cancellation should plan a larger
budget.
"""


SCENARIO = BenchmarkScenario(
    name="Sentiment 02 — Neutral Tech Review",
    description="Factual, balanced tech magazine review — neither positive nor negative.",
    schema=SentimentResult,
    text=REVIEW_TEXT,
    system_prompt="Analyse the sentiment of the following text precisely. Pay particular attention to the overall tone: factually neutral, evaluative, or emotional?",
    # confidence: neutral is genuinely the hardest class, so models are less
    # certain here — use a lower floor than the clear-cut positive/negative cases.
    expected={"sentiment": "neutral", "confidence": AtLeast(0.6)},
)


async def main() -> None:
    print("Benchmark: Sentiment 02 — Neutral Tech Review")
    await run_all_models(SCENARIO)


if __name__ == "__main__":
    asyncio.run(main())
