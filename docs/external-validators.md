# External Validators

→ [Documentation index](index.md)

> **Runnable example:** [`examples/12_external_validator.py`](../examples/12_external_validator.py)
> **Tests:** [`tests/test_external_validator.py`](../tests/test_external_validator.py)

[Pydantic validators](validators.md) cover structural and field-level rules that
can be expressed *inside the schema*.  Some checks, though, are inherently
**cross-field**, **stateful**, or need **external context** — a database lookup, a
business rule, a reference-data check, a running total that must reconcile.

Every extraction entry point accepts an optional **`validator`** callable.  It
receives the model instance *after* it has already passed Pydantic validation and
either **accepts** it (returns `None`/empty string) or **rejects** it (raises any
exception, or returns a non-empty error message).  A rejection re-enters the
**existing field-level retry loop** with the message surfaced to the model — exactly
like a Pydantic `ValidationError` — so the model gets a chance to self-correct.

---

## Quick example

```python
from pydantic import BaseModel
from saidex import extract_data_from_text


class Invoice(BaseModel):
    vendor: str
    lines: list[InvoiceLine]
    total: float


def validate_invoice(inv: Invoice) -> None:
    line_sum = round(sum(line.amount for line in inv.lines), 2)
    if line_sum != round(inv.total, 2):
        raise ValueError(
            f"Line items sum to {line_sum} but the stated total is {inv.total}. "
            f"Make the total match the sum of the line items."
        )


invoice, stats = await extract_data_from_text(
    llm,
    Invoice,
    document,
    validator=validate_invoice,   # sync or async callable
)
```

---

## How it fits the retry loop

```
LLM produces output
        │
        ▼
  Pydantic validation
        │
        ├─ fails → field-level error fed back, retry ◄─────────┐
        │                                                       │
        └─ passes                                               │
              │                                                 │
              ▼                                                 │
        validator(instance)                                     │
              │                                                 │
              ├─ raises / returns message → fed back, retry ────┘
              │
              └─ returns None → accepted ✅
```

A validator rejection:

- consumes one validation retry (`stats.primary_retries` / `validation_retries`),
- records a structured [`FieldIssue`](quality-analytics.md) with
  `error_type="external_validator"` and `field_path="<external validator>"`,
- triggers the **fallback model** when the primary model exhausts its retries,
- sets `stats.failure_reason = "validation_exhausted"` if all attempts fail.

The validator only runs on **structurally valid** instances — a Pydantic failure
short-circuits before the validator is ever called.

---

## Signal: raise or return

Both styles work; pick whichever reads best.

```python
# Raise — reuse any existing validation helper that throws.
def validate(inv: Invoice) -> None:
    if inv.total < 0:
        raise ValueError("total must not be negative")

# Return a message — handy when you build the message conditionally.
def validate(inv: Invoice) -> str | None:
    if inv.total < 0:
        return "total must not be negative"
    return None
```

Write **specific, actionable** messages — the model reads them to know what to fix.

---

## Sync or async

The callable may be synchronous or a coroutine function; SAIDEX awaits it when
needed.  Use an async validator when the check itself does I/O (a database or
reference-data lookup):

```python
async def validate_against_catalogue(order: Order) -> None:
    known = await catalogue.exists(order.sku)
    if not known:
        raise ValueError(f"Unknown SKU '{order.sku}'. Use a SKU from the catalogue.")

order, stats = await extract_data_from_text(llm, Order, text, validator=validate_against_catalogue)
```

---

## Batch extraction — validated per item

For `extract_data_list` / `extract_data_list_from_text` the validator is applied to
**each extracted item** individually.  Failures are reported per item so the model
can pinpoint the offending record:

```python
items, stats = await extract_data_list_from_text(
    llm, InvoiceLine, document, validator=validate_line
)
# A rejected item surfaces as e.g. "items -> 2: amount must be positive"
```

---

## Agent loop

`extract_data_with_tools` / `run_extractor_agent` apply the validator to the
**final answer**.  A rejection is fed back as a tool result and the agent keeps
looping (up to `max_validation_retries`) so it can revise its answer.

---

## Availability

`validator` is available on every entry point and its `*_sync` wrapper:

`extract_data` · `extract_data_from_text` · `extract_data_list` ·
`extract_data_list_from_text` · `extract_data_with_tools` · `run_extractor_agent`
(and `extract_data_sync`, `extract_data_from_text_sync`, …).
