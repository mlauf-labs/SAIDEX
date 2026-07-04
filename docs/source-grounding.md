# Source Grounding

→ [Documentation index](index.md)

> **Runnable example:** [`examples/13_source_grounding.py`](../examples/13_source_grounding.py)
> **Tests:** [`tests/test_grounding.py`](../tests/test_grounding.py), [`tests/test_grounding_integration.py`](../tests/test_grounding_integration.py)

LLMs sometimes invent plausible-looking values — a vendor name, an ID, a total —
that are **not present** in the document. Mark a field as **grounded** and SAIDEX
verifies, after Pydantic validation, that the extracted value actually appears in
the source text. A value it cannot find is rejected and the **existing retry
loop** asks the model to correct it, exactly like a Pydantic failure.

Grounding needs no new function arguments: it is driven entirely by markers in
your schema. The source text is taken transiently from the human-turn message(s)
of the run (it is **not** retained — that is what `capture_source_text` controls).

---

## Two ways to mark a field

Both surfaces are equivalent — they attach the same `Grounded` check to the
field's metadata — so pick whichever reads best. Grounded fields stay obvious
right where they are declared.

```python
from typing import Annotated
from pydantic import BaseModel, Field
from saidex import Grounded, GroundedField, GroundedStr

class Invoice(BaseModel):
    vendor: Annotated[str, Grounded()]                 # Annotated marker
    invoice_no: str = GroundedField(description="No.")  # Field helper
    alias: GroundedStr                                  # ready-made alias
    note: str = Field(description="Free summary")       # not grounded
```

---

## How it fits the retry loop

```text
LLM produces output
        │
        ▼
  Pydantic validation
        │
        ├─ fails → field-level error fed back, retry ◄────────────┐
        │                                                          │
        └─ passes                                                  │
              │                                                    │
              ▼                                                    │
     grounding checks (per field)                                  │
              │                                                    │
              ├─ value not in text → fed back, retry ──────────────┤
              │                                                     │
              └─ all present                                        │
                    │                                               │
                    ▼                                               │
              validator(instance) (if any)                          │
                    │                                               │
                    ├─ rejects → fed back, retry ───────────────────┘
                    │
                    └─ accepts → returned ✅
```

A grounding failure:

- consumes one validation retry (`stats.primary_retries` / `validation_retries`),
- records a structured [`FieldIssue`](quality-analytics.md) with
  `category="grounding"` and the field path (e.g. `items -> 2 -> vendor`),
- triggers the **fallback model** when the primary exhausts its retries,
- sets `stats.failure_reason = "validation_exhausted"` if all attempts fail.

Grounding runs **before** an [external validator](external-validators.md) and only
on structurally valid instances.

---

## Matching semantics

The default `mode="normalized"` compares **case-, whitespace- and
diacritics-insensitively** (so `acme  gmbh` matches `ACME GmbH`, and `Müller`
matches `Muller`). Non-breaking and thin spaces are treated as ordinary spaces.

```python
Grounded()                # normalized (default)
Grounded(mode="exact")    # require the verbatim str(value) to be present
```

Grounding applies to **scalar leaf fields** (`str`, numbers, dates); container
and nested-model fields are walked through to reach the scalars inside them.

---

## Retry vs. flag (`on_mismatch`)

By default a grounding failure **re-enters the retry loop** so the model can
correct the value. Set `on_mismatch="flag"` for **advisory** grounding: the
extracted value is kept, no retry is consumed, and the mismatch is only recorded
as a `FieldIssue(category="grounding")` for later inspection.

```python
class Invoice(BaseModel):
    vendor: Annotated[str, Grounded()]                      # "retry" (default)
    summary: str = GroundedField(on_mismatch="flag")        # advisory only
```

Both modes record a `FieldIssue`; only `"retry"` produces feedback to the model
and consumes an attempt. A schema can mix the two — retry failures drive the
correction prompt, flag failures ride along as recorded issues.

---

## Locale-aware values

A canonical value rarely matches its **surface form** in the document, and the
difference is locale dependent:

| Value | Surface forms matched |
| --- | --- |
| `1234.5` | `1.234,50` (de) · `1,234.50` (en) · `1 234,50` (fr) · `1'234.50` (ch) |
| `-1234.5` | `-1.234,50` · `(1.234,50)` · `1.234,50-` |
| `"2024-04-05"` | `05.04.2024` · `5. April 2024` · `April 5, 2024` |

Tell grounding which locale to render for:

```python
class Invoice(BaseModel):
    country: str                                          # e.g. "DE"
    total: float = GroundedField(locale_field="country")  # dynamic: from a sibling
    tax: float = GroundedField(locale="de")               # static: fixed hint
```

- `locale_field` reads the hint from a sibling field at runtime (a `CountryCodeStr`
  or `LanguageCodeStr` works well); it can read a sibling because checks run over
  the **fully validated instance**.
- `locale` is a fixed hint for when the language is known up front.
- The hint accepts a **country or language code**. When it is missing or unknown,
  grounding falls back to a broad set of conventions, so a wrong hint never causes
  a false rejection.

> Currency symbols need no special handling: the number `1234.5` rendered as
> `1.234,50` is found as a substring of `€1.234,50` / `EUR 1.234,50`.

---

## Build your own check — the `FieldCheck` pattern

Grounding is the first built-in check on a reusable mechanism. Any
`FieldCheck` subclass can be attached the same two ways — as an Annotated marker,
or via the `field_check` helper — and SAIDEX runs it over the validated instance
with an `ExtractionContext` (source text, the containing instance, the field
path).

```python
from typing import Annotated
from pydantic import BaseModel
from saidex import FieldCheck, ExtractionContext, field_check

class InCatalogue(FieldCheck):
    def check(self, value, ctx: ExtractionContext) -> str | None:
        if str(value) not in KNOWN_SKUS:
            return f"'{value}' is not a known SKU."
        return None

class Order(BaseModel):
    sku: Annotated[str, InCatalogue()]            # marker form
    alt_sku: str = field_check(InCatalogue())     # helper form
```

A check returns `None` to accept or an LLM-friendly message to reject; rejections
re-enter the same retry loop as grounding. Return `None` when `ctx.source_text`
is missing so the check is skipped during plain model construction.

---

## Availability

Grounding works across every extraction entry point and both extraction modes
(`TOOL_CALLING` and `JSON`), including the batch variants (checked **per item**)
and the agent loop (checked on the **final answer**), plus all the `*_sync`
wrappers. There are no new parameters to pass — just mark the fields.

---

## Out of scope (tracked separately)

Fuzzy matching, value-form localisation (number words, boolean yes/no), and
phone-number surface forms are tracked as follow-up issues on the project
tracker.
