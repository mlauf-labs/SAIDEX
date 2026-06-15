"""Example 1 — Extract a Pydantic model from a plain text string.

Run:
    python examples/01_extract_from_text.py
"""

from __future__ import annotations

import asyncio
from typing import Annotated

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from saidex import extract_data_from_text

# ---------------------------------------------------------------------------
# Define your schema
# ---------------------------------------------------------------------------


class PersonInfo(BaseModel):
    """Information about a person extracted from text."""

    name: str = Field(description="Full name of the person")
    age: int = Field(description="Age in years", ge=0, le=150)
    occupation: str = Field(description="Current job or profession")
    city: Annotated[str | None, Field(description="City of residence")] = None


# ---------------------------------------------------------------------------
# Run extraction
# ---------------------------------------------------------------------------


async def main() -> None:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    text = """
    Alice Müller, 34, is a software engineer based in Munich.
    She has been working at a tech startup for three years.
    """

    person, stats = await extract_data_from_text(llm, PersonInfo, text)

    if person is None:
        print(f"Extraction failed after {stats.total_retries} retries.")
        return

    print("Extracted person:")
    print(f"  Name:       {person.name}")
    print(f"  Age:        {person.age}")
    print(f"  Occupation: {person.occupation}")
    print(f"  City:       {person.city}")
    print(f"\nRetries needed: {stats.total_retries}")


if __name__ == "__main__":
    asyncio.run(main())
