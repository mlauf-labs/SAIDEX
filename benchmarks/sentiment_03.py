"""Sentiment Benchmark 03 — Angry, negative customer complaint (German).

Long, emotionally charged complaint letter to a telecom provider. Tests whether
the model can correctly classify an aggressively negative tone.

Run standalone:
    python benchmarks/sentiment_03.py
"""

from __future__ import annotations

import asyncio
from typing import Literal

from pydantic import BaseModel, Field

from ._base import BenchmarkScenario, run_all_models

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
        default_factory=list,
        description="Dominant emotions expressed (e.g. joy, gratitude, excitement, frustration)"
    )


# ---------------------------------------------------------------------------
# Test text  — negative / aggressive complaint
# ---------------------------------------------------------------------------

REVIEW_TEXT = """
Subject: Immediate Contract Termination and Notice of Legal Action —
Customer No. 88312-KD | Contract MB-7723

Dear Sir or Madam,

I am writing this letter because, after five months of sustained incompetence,
indifference, and demonstrable deception on the part of your company, I have no
other option remaining. What I have experienced over the past weeks exceeds anything
I have encountered in twenty years as a customer of various service providers.

The facts: on 15 January 2024 I contracted with your company for a fibre-optic
connection at 300 Mbit/s. According to the contract, activation was to take place
on 1 February. It is now 1 June — that is a delay of four months. Four months during
which I called your so-called "customer service" multiple times per week, receiving
a different, contradictory answer each time, and during which your system has
allegedly "escalated" my case at least seven times without anything whatsoever
happening.

On the first call I was told there was an error in the order system that would be
resolved within three business days. On the second call no-one knew anything about
an error. On the third call your representative — whose name I have noted: Ms. Köhler,
call of 22 February at approximately 14:45 — informed me that my order had already
been closed and completed. When I asked why I therefore had no internet access, I
was subjected to nineteen minutes of hold music, after which I was disconnected
without warning.

What I find particularly remarkable: despite having provided no service whatsoever,
your company somehow managed to issue me with an invoice for €49.99 in both March
and April, collecting both amounts via direct debit. That is a total of €99.98 I
have paid for nothing. I demand immediate reimbursement of both amounts, together
with compensation for the inconvenience and financial loss incurred — I was forced
to purchase a mobile data plan as a stopgap, which cost me an additional €68.

On 10 April I submitted a formal written complaint by registered post. This went
unanswered. On 25 April I sent an email to complaints@your-provider.com. No response.
My solicitor has informed me that this constitutes a breach of your statutory
obligation to respond.

Your helpline has come to symbolise for me everything that is wrong with modern
corporate structures: automated systems that conceal accountability, employees
without authority to make decisions, and processes that appear designed to string
customers along until they give up. I will not give up.

With immediate effect, I am terminating my contract without notice on the grounds
of your sustained failure to provide the contracted service. Should I not receive
written confirmation of the termination along with reimbursement of the amounts
stated above by 15 June 2024, I will file a claim in the competent civil court and
additionally lodge a complaint with the national telecommunications regulator.

I am available for further correspondence exclusively in writing.

Yours, with the last remnants of my patience,
Andreas Klopfer
"""


SCENARIO = BenchmarkScenario(
    name="Sentiment 03 — Angry Customer Complaint",
    description="Very long, aggressively negative complaint letter to a telecommunications provider.",
    schema=SentimentResult,
    text=REVIEW_TEXT,
    system_prompt="Analyse the sentiment of the following text precisely. Pay particular attention to the overall tone: factually neutral, evaluative, or emotional?",
    expected={"sentiment": "negative", "confidence": 0.98},
)


async def main() -> None:
    print("Benchmark: Sentiment 03 — Angry Customer Complaint")
    await run_all_models(SCENARIO)


if __name__ == "__main__":
    asyncio.run(main())
