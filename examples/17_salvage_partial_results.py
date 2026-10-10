"""Example 17 — Salvage a partial result with ``on_exhausted``.

A findings extractor: every finding carries a verbatim quote, and an external
validator rejects quotes that are not in the source text.  When the model keeps
paraphrasing one quote, the whole run would fail and the valid findings would be
lost.  ``on_exhausted`` runs once after every retry has failed and keeps the
findings that passed — the run then succeeds with ``stats.salvaged=True``.

Run:
    python examples/17_salvage_partial_results.py
"""

from __future__ import annotations

import asyncio

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from saidex import FieldIssue, ValidationIssue, extract_data_from_text, issue_item_index

TRANSCRIPT = """\
Interviewer: How do you handle conflicts in the team?
Candidate: I set up a short one-on-one first and listen before I propose anything.
Interviewer: And deadlines?
Candidate: We cut scope early instead of working weekends.
"""


class Finding(BaseModel):
    """One observation about the candidate, backed by a verbatim quote."""

    topic: str = Field(description="What the finding is about")
    quote: str = Field(description="Exact sentence from the transcript, copied verbatim")


class Findings(BaseModel):
    """All findings from the interview."""

    findings: list[Finding] = Field(description="Every finding, each with a verbatim quote")


def quotes_are_verbatim(doc: Findings) -> list[ValidationIssue]:
    """Reject every finding whose quote is not literally in the transcript."""
    return [
        ValidationIssue(
            message="This quote is not verbatim in the transcript; copy the exact sentence.",
            field_path=f"findings -> {index} -> quote",
            error_type="quote_not_verbatim",
            received=finding.quote,
        )
        for index, finding in enumerate(doc.findings)
        if finding.quote not in TRANSCRIPT
    ]


def keep_verbatim_findings(doc: Findings, issues: tuple[FieldIssue, ...]) -> Findings | None:
    """After all retries: drop the rejected findings, keep the rest."""
    rejected = {issue_item_index(issue, "findings") for issue in issues}
    kept = [f for index, f in enumerate(doc.findings) if index not in rejected]
    return Findings(findings=kept) if kept else None


async def main() -> None:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    result, stats = await extract_data_from_text(
        llm,
        Findings,
        TRANSCRIPT,
        max_primary_retries=2,
        validator=quotes_are_verbatim,
        on_exhausted=keep_verbatim_findings,
    )
    if result is None:
        print(f"No usable findings ({stats.failure_reason}).")
        return
    label = "salvaged" if stats.salvaged else "clean"
    print(f"{len(result.findings)} finding(s), {label} result:")
    for finding in result.findings:
        print(f"  - {finding.topic}: “{finding.quote}”")
    for issue in stats.last_rejection:
        print(f"  dropped {issue.field_path}: {issue.received}")


if __name__ == "__main__":
    asyncio.run(main())
