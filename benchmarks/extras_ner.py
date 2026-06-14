"""Extras Benchmark — Named Entity Recognition (business press release).

Extracts structured lists of persons, organizations, locations, and dates
from a realistic press release text.

Run standalone:
    python benchmarks/extras_ner.py
"""

from __future__ import annotations

import asyncio

from pydantic import BaseModel, Field
from saidex import ISODateStr

from ._base import BenchmarkScenario, run_all_models

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


class NamedEntities(BaseModel):
    """Named entities extracted from a text."""

    persons: list[str] = Field(
        description="Full names of all persons mentioned (first name + last name where available)"
    )
    organizations: list[str] = Field(
        description="Names of companies, institutions, agencies, associations, or other organizations"
    )
    locations: list[str] = Field(
        description="Cities, countries, regions, addresses, or other geographical locations"
    )
    dates: list[str] = Field(
        description="All dates or time references mentioned (as they appear in the text)"
    )
    monetary_values: list[str] = Field(
        description="All monetary amounts mentioned with their currency, e.g. '€ 450 million'"
    )
    purchase_price_eur_millions: int = Field(
        description="The acquisition purchase price expressed as a whole number of millions of EUR, e.g. 450 for '€ 450 million'"
    )
    isin_codes: list[str] = Field(
        description="All ISIN securities identifiers mentioned, e.g. 'DE000A2YNH52'"
    )
    announcement_date: ISODateStr = Field(
        description="The date the press release was issued, as yyyy-mm-dd"
    )


# ---------------------------------------------------------------------------
# Test text — German business press release
# ---------------------------------------------------------------------------

NER_TEXT = """
PRESS RELEASE

InnovateTech AG Acquires Hamburg-Based AI Start-Up NeuroLayer GmbH —
Europe's Largest AI Acquisition of 2024

Frankfurt am Main, 8 March 2024 — Listed technology group InnovateTech AG
(ISIN: DE000A2YNH52, Frankfurt Stock Exchange) today announced the completion of its
acquisition of NeuroLayer GmbH, a Hamburg-based specialist in industrial computer vision
solutions. The purchase price amounts to € 450 million in cash, financed through a
combination of existing credit facilities and bridge financing arranged by
Commerzbank AG and Deutsche Bank AG.

NeuroLayer was founded in 2019 by Dr. Amara Osei and Prof. Dr. Kai Brinkmann at the
University of Hamburg and currently employs 280 staff at locations in Hamburg
(headquarters), Munich, and Warsaw. Over the past five years the company has developed
technologies for automated quality control in manufacturing, which are today deployed
by customers including Volkswagen AG, Siemens AG, and Bosch GmbH.

"With the acquisition of NeuroLayer we significantly strengthen our position as a leading
provider of AI infrastructure in Europe," said Dr. Susanne Reimer, Chief Executive Officer
of InnovateTech AG, at a press conference in Frankfurt. "NeuroLayer's technology is an
ideal complement to our portfolio and opens up new markets in the automotive and process
manufacturing industries."

Dr. Amara Osei, co-founder and outgoing CEO of NeuroLayer, will join the executive board
of InnovateTech AG as Chief Technology Officer (CTO) following the merger. Prof. Dr. Kai
Brinkmann is stepping back from day-to-day operations but will remain with the company
as a strategic adviser.

The European Commission in Brussels cleared the transaction on 22 February 2024 following
a review under EU merger control regulations. The Federal Cartel Office in Bonn had already
granted its approval on 15 January 2024.

For the 2024 financial year, InnovateTech AG expects the integration of NeuroLayer to
contribute additional revenue of approximately € 85 million and EBITDA synergies of
€ 12–18 million, expected to be fully realised from financial year 2025.

InnovateTech AG was advised legally by Freshfields Bruckhaus Deringer LLP (Frankfurt) and
financially by Goldman Sachs International (London). NeuroLayer was advised by
Hengeler Mueller (Hamburg) and Lazard GmbH (Munich).

About InnovateTech AG:
InnovateTech AG is a European technology group headquartered in Frankfurt am Main with
over 4,500 employees across 12 countries. The company develops AI-based software platforms
for industry, logistics, and financial services. Revenue for financial year 2023 totalled
€ 1.2 billion.

Press Contact:
Tobias Wendt, Head of Corporate Communications
InnovateTech AG, Taunusanlage 18, 60325 Frankfurt am Main
E-mail: press@innovatetech.de | Tel.: +49 69 7140 2200

Investor Relations:
Claudia Hartmann, Director Investor Relations
E-mail: ir@innovatetech.de | Tel.: +49 69 7140 2350
"""


SCENARIO = BenchmarkScenario(
    name="Extras — Named Entity Recognition",
    description="Business press release about a corporate acquisition — extraction of persons, companies, locations, dates, and monetary values.",
    schema=NamedEntities,
    text=NER_TEXT,
    system_prompt="Extract all named entities from the following text. Be thorough and include every mentioned person, organization, location, date, and monetary value.",
    expected={
        # list[str]: subset check — each expected entry must appear in the list.
        "persons": ["Dr. Amara Osei", "Dr. Susanne Reimer", "Prof. Dr. Kai Brinkmann"],
        "organizations": ["InnovateTech AG", "NeuroLayer GmbH", "Commerzbank AG"],
        "locations": ["Frankfurt am Main", "Hamburg", "Munich"],
        "monetary_values": ["450 million", "85 million", "1.2 billion"],
        "isin_codes": ["DE000A2YNH52"],
        # int scalar: numeric comparison (tolerant of int/float), the purchase
        # price normalised to millions of EUR — '€ 450 million' -> 450.
        "purchase_price_eur_millions": 450,
        # ISODateStr: validated as a real yyyy-mm-dd calendar date.
        "announcement_date": "2024-03-08",
    },
)


async def main() -> None:
    print("Benchmark: Extras — Named Entity Recognition")
    await run_all_models(SCENARIO)


if __name__ == "__main__":
    asyncio.run(main())
