"""Extras Benchmark — Meeting notes extraction (German).

Tests structured extraction from an unstructured meeting transcript:
participants, decisions, action items (with owners and deadlines), and next meeting date.

Run standalone:
    python benchmarks/extras_meeting.py
"""

from __future__ import annotations

import asyncio

from pydantic import BaseModel, Field
from saidex import ISODateStr

from ._base import BenchmarkScenario, run_all_models

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


class ActionItem(BaseModel):
    """A single action item from a meeting."""

    owner: str = Field(description="Person responsible for completing the task")
    task: str = Field(description="Description of the task to be done")
    due_date: ISODateStr = Field(description="Deadline as yyyy-mm-dd if mentioned, e.g. 2024-04-05")


class MeetingNotes(BaseModel):
    """Structured extraction of key information from a meeting transcript or notes."""

    title: str = Field(description="Short title or subject of the meeting")
    participants: list[str] = Field(description="List of all participants mentioned")
    decisions: list[str] = Field(description="List of explicit decisions made during the meeting")
    action_items: list[ActionItem] = Field(description="List of action items assigned to specific people")
    open_questions: list[str] = Field(
        description="Topics or questions that were raised but not resolved"
    )
    next_meeting: ISODateStr = Field(description="Date of the next meeting as yyyy-mm-dd if mentioned")


# ---------------------------------------------------------------------------
# Test text — informal meeting notes from a product team
# ---------------------------------------------------------------------------

MEETING_TEXT = """
Besprechungsprotokoll – Sprint-Review & Planning
Datum: Donnerstag, 4. April 2024, 14:00–16:15 Uhr
Raum: Konferenzraum „Weser" / Remote via Teams

Anwesend:
  - Sandra Köhler (Product Ownerin)
  - Markus Engel (Tech Lead)
  - Jana Berger (Backend-Entwicklerin)
  - Felix Wörner (Frontend-Entwickler)
  - Dr. Priya Nair (Data Engineer, remote)
  - Tom Schuster (QA-Lead)
  - Entschuldigt: Lena Fischer (Designerin) – Krankheit

Moderation: Sandra Köhler
Protokollant: Felix Wörner

──────────────────────────────────────────────

TOP 1 – Sprint-Review (Sprint 18, 21.03.–03.04.2024)

Sandra eröffnet die Runde und bedankt sich beim Team für den erfolgreichen Sprint.
Von 34 geplanten Story Points wurden 31 abgeschlossen.

Abgenommene Features:
  • CSV-Import für Nutzerdaten (Jana, Felix) – vollständig deployed, Smoke-Tests bestanden
  • Dashboard-Filterkomponente (Felix) – deployed, allerdings Performance-Probleme bei
    Datensätzen > 10.000 Einträgen (siehe TOP 3)
  • Automatisierter Nightly-Report per E-Mail (Priya) – in Staging, noch nicht live

Nicht abgeschlossen:
  • OAuth-Integration mit Azure AD – auf Sprint 19 verschoben (Aufwand unterschätzt)

──────────────────────────────────────────────

TOP 2 – Kundenfeedback Q1

Sandra präsentiert aggregiertes Feedback aus 12 Kundengesprächen. Die wichtigsten Punkte:
  1. Kunden wünschen sich einen Dark Mode für das Dashboard.
  2. Exportfunktion (PDF, Excel) wird als „dringend notwendig" eingestuft.
  3. Ladezeiten auf mobilen Geräten sind laut 8 von 12 Kunden „zu langsam".

Entscheidung: Dark Mode wird in Sprint 20 als eigenständiges Epic eingeplant.
Entscheidung: Export-Funktion (PDF + Excel) hat höchste Priorität für Q2.

Sandra wird bis zum 10. April ein priorisiertes Backlog für Q2 erstellen.

──────────────────────────────────────────────

TOP 3 – Performance-Problem Dashboard-Filter

Markus erklärt, dass die Filterkomponente bei großen Datensätzen alle Datensätze
serverseitig lädt statt zu paginieren. Priya schlägt vor, auf cursor-based pagination
umzusteigen und einen dedizierten Index auf der Spalte `created_at` anzulegen.

Entscheidung: Markus und Jana übernehmen das Refactoring in Sprint 19.
Ziel: Filteranfragen sollen unter 200ms liegen (aktuell: ~1,8s bei 50.000 Einträgen).

Aufgaben:
  → Jana: DB-Index anlegen und Query optimieren, bis 12. April
  → Markus: Frontend-seitige Paginierungslogik, bis 12. April
  → Tom: Lasttests mit 100.000 Datensätzen erstellen und auführen, bis 15. April

──────────────────────────────────────────────

TOP 4 – Infrastruktur & Deployment

Priya berichtet, dass der Staging-Server seit zwei Wochen intermittierend ausfällt.
Ursache noch unklar – vermutlich Speicherleck im Logging-Service.

Offene Frage: Sollen wir auf Kubernetes migrieren? Markus sieht mittelfristig keine
Alternative, möchte aber erst eine Kosten-Nutzen-Analyse erstellen. Entscheidung wird
auf das nächste Management-Meeting vertagt.

Entscheidung: Priya analysiert den Logging-Service bis 8. April und schlägt
einen Fix oder Workaround vor.

──────────────────────────────────────────────

TOP 5 – Sprint 19 Planning (04.04.–17.04.2024)

Committed Stories: 30 Story Points
  - OAuth Azure AD Integration (Markus, Jana) – 13 SP
  - Dashboard Performance Refactoring (Markus, Jana) – 8 SP
  - Automatisierter Nightly-Report auf LIVE schalten (Priya, Tom) – 5 SP
  - Technische Schulden: Logging-Service Diagnose + Fix (Priya) – 4 SP

──────────────────────────────────────────────

Diverses / AOB

Sandra erinnert daran, dass am 18. April das Quartalsgespräch mit dem Vorstand stattfindet.
Markus soll eine technische Status-Präsentation vorbereiten (max. 10 Folien).

Lena Fischer wird gebeten, die Dark-Mode-Mockups bis zum Sprint-Start (20. April) zu liefern.
(Sandra informiert Lena separat per E-Mail.)

──────────────────────────────────────────────

Nächstes Meeting: Sprint-Review Sprint 19 am Donnerstag, 18. April 2024, 14:00 Uhr

Ende der Sitzung: 16:15 Uhr
"""


SCENARIO = BenchmarkScenario(
    name="Extras — Meeting Notes Extraction (DE)",
    description="Unstructured meeting minutes: participants, decisions, action items with owners and deadlines.",
    schema=MeetingNotes,
    text=MEETING_TEXT,
    system_prompt="Du bist ein Assistent, der Besprechungsprotokolle strukturiert auswertet. Extrahiere alle relevanten Informationen präzise.",
    expected={
        # Scalar: lenient string comparison.
        "next_meeting": "2024-04-18",
        # int: number of elements expected in the list field.
        "participants": 6,
        "decisions": 5,
        "open_questions": 1,
        # list of dicts: element count + owner/due_date per item.
        "action_items": [
            {"owner": "Sandra", "due_date": "2024-04-10"},
            {"owner": "Jana", "due_date": "2024-04-12"},
            {"owner": "Markus", "due_date": "2024-04-12"},
            {"owner": "Tom", "due_date": "2024-04-15"},
            {"owner": "Priya", "due_date": "2024-04-08"},
        ],
    },
)


async def main() -> None:
    print("Benchmark: Extras — Meeting Notes Extraction (DE)")
    await run_all_models(SCENARIO)


if __name__ == "__main__":
    asyncio.run(main())
