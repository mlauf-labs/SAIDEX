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
exception, or returns a non-empty error message).  To report several findings at
once, it can also return a list of [`ValidationIssue`](#structured-issues)
objects, each with its own field path and retry-or-flag behaviour.  A rejection re-enters the
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
  `error_type="external_validator"` and `field_path="<external validator>"`
  (or one `FieldIssue` per [`ValidationIssue`](#structured-issues)
  with your own path and rule id),
- triggers the **fallback model** when the primary model exhausts its retries,
- sets `stats.failure_reason = "validation_exhausted"` if all attempts fail.

The validator only runs on **structurally valid** instances — a Pydantic failure
short-circuits before the validator is ever called.

## When every attempt fails — `last_candidate`

If the retries run out, the call still returns `(None, stats)`. The last
instance that passed schema validation but was rejected by your validator (or
by source grounding) is not lost, though:

```python
result, stats = await extract_data(llm, Findings, messages, validator=check_quotes)
if result is None and stats.last_candidate is not None:
    findings = cast(Findings, stats.last_candidate)
    for issue in stats.last_rejection:          # the "retry" issues that rejected it
        print(issue.field_path, issue.message)
```

- `last_candidate` is the **most recent** schema-valid candidate. A later
  attempt that fails differently (malformed JSON, provider error) does not
  clear it, so it can be set whatever `failure_reason` says.
- With a fallback model, the fallback's candidate wins if it produced one.
- `last_rejection` holds only the issues that rejected it; `"flag"` issues stay
  in `field_issues`.
- On success both are empty (`None` / `()`).
- Batch variants expose a `list` of items, with paths like `items -> 3 -> quote`.

!!! warning "PII"
    The candidate is extracted source data. It stays on the stats object and is
    visible to `on_complete` hooks and listeners. The enclosing chain-run
    span's output only gets `has_last_candidate` (nested LLM spans still contain
    the raw model output), and `collect_stats()` sinks drop it.

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

## Structured issues

Return a list of `ValidationIssue` objects when one attempt can break several
independent rules.  Each issue becomes its own entry in `stats.field_issues`
(with your `field_path`, `category` and `error_type`), and the model gets
per-field feedback — the same treatment as Pydantic and `Grounded()` failures.

```python
from saidex import ValidationIssue, extract_data_from_text


def check_reply(reply: Reply) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    if INTERNAL_NOTE_MARKER in reply.body:
        issues.append(ValidationIssue(
            "Do not quote the internal note.",
            field_path="body", error_type="leaks_internal_note",
        ))
    if "refund" in reply.body and refund_pending:
        issues.append(ValidationIssue(
            "Do not promise a refund while approval is pending.",
            field_path="body", error_type="premature_refund",
        ))
    if reply.language != customer_language:
        issues.append(ValidationIssue(
            f"Reply in the customer's language ({customer_language}).",
            field_path="language", error_type="wrong_language",
            received=reply.language,
        ))
    if len(reply.body) > 1500:
        issues.append(ValidationIssue(
            "Reply is unusually long.",
            field_path="body", error_type="long_reply", on_mismatch="flag",
        ))
    return issues


reply, stats = await extract_data_from_text(llm, Reply, ticket, validator=check_reply)
```

The model receives the `"retry"` issues grouped by field path:

```text
The extracted data is structurally valid but failed an additional validation check:

  - 'body': Do not quote the internal note.
  - 'body': Do not promise a refund while approval is pending.
  - 'language': Reply in the customer's language (en).
    Received: 'de'
```

| Validator returns | Instance | Retry consumed | Recorded in `stats.field_issues` |
| --- | --- | --- | --- |
| `None` / `""` / `[]` | accepted | no | — |
| non-empty `str`, or raises | rejected | yes | one issue, `field_path="<external validator>"` |
| list of non-empty `str` | rejected | yes | one issue per string, `field_path="<external validator>"` |
| issues, at least one `on_mismatch="retry"` | rejected | yes | every issue, `flag` ones included |
| issues, all `on_mismatch="flag"` | accepted | no | every issue |

`ValidationIssue` fields:

| Field | Default | Meaning |
| --- | --- | --- |
| `message` | — (required) | What is wrong; shown to the model |
| `field_path` | `"<external validator>"` | Offending field, e.g. `"body"` or `"lines -> 0 -> price"` — relative to the validated model; batch extraction adds the `items -> {i}` prefix |
| `category` | `"value"` | Coarse bucket, same vocabulary as `FieldIssue.category` |
| `error_type` | `"external_validator"` | Free-form rule id — count violations per rule across runs |
| `on_mismatch` | `"retry"` | `"retry"` rejects and consumes a retry; `"flag"` only records |
| `received` | `None` | Optional offending value, shown to the model |

A single `ValidationIssue` (without a list) works too, and a list may mix plain
strings with issues.  Returning anything else — a dict, `True`, bytes — raises
`TypeError`: it is treated as a bug in the validator, not as a rejection, and
never consumes a retry.

`flag` issues mirror [`FieldCheck(on_mismatch="flag")`](source-grounding.md): use them
for soft findings you want to count in evals without burning a retry.

---

## Attempt context — strict early, lenient at the end

A validator only sees the instance. To change its behaviour per round, wrap it
with `with_context`; it then receives a `ValidatorContext` as second argument:

| Field | Meaning |
| --- | --- |
| `attempt` | 1-based position in the current model's budget (parse errors use up a slot too) |
| `max_attempts` | Size of that budget (`max_primary_retries` / `max_fallback_retries`; `max_validation_retries + 1` in the agent loop) |
| `is_final` | `True` only on the last allowed attempt of the **whole run** — not on the primary's last attempt when a fallback model will still run |
| `model_label` | `"primary"` or `"fallback"` |
| `is_fallback` | Whether the fallback model produced the candidate |

```python
from saidex import ValidationIssue, ValidatorContext, extract_data, with_context

def strict_then_lenient(inv: Invoice, ctx: ValidatorContext) -> ValidationIssue | None:
    if abs(sum(line.amount for line in inv.lines) - inv.total) < 0.01:
        return None
    return ValidationIssue(
        message="Line items do not add up to the total.",
        field_path="total",
        # Retry while there are attempts left; on the last one, accept and flag.
        on_mismatch="flag" if ctx.is_final else "retry",
    )

result, stats = await extract_data(
    llm, Invoice, messages, validator=with_context(strict_then_lenient)
)
```

Plain one-argument validators keep working unchanged. The wrapper is explicit
on purpose: SAIDEX never inspects a validator's signature. Batch extraction
passes the same context to every item of an attempt.

`ValidatorContext.attempt` is 1-based, while `FieldIssue.attempt` in the stats is
0-based: for the same round, `ctx.attempt == issue.attempt + 1`.

A variable annotated `Validator[...]` may now hold a `WithContext`, which is not
directly callable. Pass it to the API rather than calling it yourself.

!!! note "Agent loop"
    In `extract_data_with_tools` / `run_extractor_agent`, helper-tool calls
    consume `max_iterations` too. If `max_iterations` runs out first, the run
    ends without any validator call having seen `is_final=True`. If the model
    sends several final answers in one response, each of them may be checked
    with `is_final=True`.

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
**each extracted item** individually.  Every failing item becomes its own field
issue, located by its index, so the model can pinpoint the offending record:

```python
items, stats = await extract_data_list_from_text(
    llm, InvoiceLine, document, validator=validate_line
)
# A rejected item surfaces with field_path "items -> 2" — or "items -> 2 -> amount"
# when the validator returns ValidationIssue(..., field_path="amount").
```

One `"retry"` issue on any item rejects the whole batch attempt; `"flag"` issues
are recorded per item without a retry.

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
