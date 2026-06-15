# Batch Extraction

→ [Documentation index](index.md)

Many documents contain **repeated records** — line items on an invoice, several
people in a meeting transcript, multiple products on a page. Rather than wrapping
your model in a container schema by hand or running one extraction per record,
`extract_data_list` pulls them all in a **single LLM call** and returns a
precisely typed `list[ModelT]`.

> **Runnable example:** [`examples/10_batch_extraction.py`](../examples/10_batch_extraction.py)

---

## `extract_data_list_from_text` — plain text → list of models

Define the schema for **one** item; the function returns a list of them.

```python
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from saidex import extract_data_list_from_text


class InvoiceLine(BaseModel):
    description: str = Field(description="What was purchased")
    quantity: int = Field(description="Number of units", ge=1)
    unit_price: float = Field(description="Price per unit", ge=0)


lines, stats = await extract_data_list_from_text(llm, InvoiceLine, document)

if lines is not None:
    print(f"Extracted {stats.item_count} line items")
    for line in lines:
        print(line.description, line.quantity, line.unit_price)
```

### Full signature

```python
async def extract_data_list_from_text(
    llm_model:           Any,
    schema:              type[ModelT],   # the schema for ONE item
    text:                str,
    *,
    mode:                ExtractionMode = ExtractionMode.TOOL_CALLING,
    system_prompt:       str | None = None,
    callbacks:           list[Any] | None = None,
    fallback_llm_model:  Any = None,
    max_primary_retries: int = 3,
    max_fallback_retries: int = 3,
    retry_config:        RetryConfig | None = None,
) -> tuple[list[ModelT] | None, ExtractDataStats]:
```

---

## `extract_data_list` — chat history → list of models

Use `extract_data_list` when you already have a list of LangChain messages and
want full control over the conversation:

```python
from langchain_core.messages import SystemMessage, HumanMessage
from saidex import extract_data_list

messages = [
    SystemMessage(content="Extract every attendee mentioned in the transcript."),
    HumanMessage(content=transcript),
]

people, stats = await extract_data_list(llm, Attendee, messages)
```

### Full signature

```python
async def extract_data_list(
    llm_model:           Any,
    schema:              type[ModelT],   # the schema for ONE item
    messages:            list[BaseMessage],
    *,
    mode:                ExtractionMode = ExtractionMode.TOOL_CALLING,
    callbacks:           list[Any] | None = None,
    fallback_llm_model:  Any = None,
    max_primary_retries: int = 3,
    max_fallback_retries: int = 3,
    retry_config:        RetryConfig | None = None,
) -> tuple[list[ModelT] | None, ExtractDataStats]:
```

---

## How it works

Internally the per-item *schema* is wrapped in a one-field container model
(`items: list[schema]`) and driven through the **exact same pipeline** as
[`extract_data`](extraction.md). That means everything you already rely on
applies unchanged:

- **Both modes** — tool calling (default) and raw [JSON mode](extraction-modes.md).
- **Per-item validation** — each element is validated independently. When item 3
  has a bad field, the error feedback points the model straight at it with a path
  like `items -> 2 -> quantity`, and only the [retry loop](retry-and-fallback.md)
  needed to fix it runs.
- **Fallback model** — an optional `fallback_llm_model` takes over if the primary
  exhausts its retry budget.

---

## The result

The return value mirrors the single-item API: a `(value, stats)` tuple where the
value is `None` on total failure. On success it is a `list` — **possibly empty**
when the document genuinely contains no records.

```python
lines, stats = await extract_data_list_from_text(llm, InvoiceLine, document)

if lines is None:
    raise RuntimeError("Extraction failed")  # all retries exhausted

print(f"{stats.item_count} items, {stats.total_retries} retries")
```

`ExtractDataStats.item_count` reports how many items were returned (it is `0` for
single-item extraction). All other fields — `total_retries`, `fallback_used` —
behave exactly as in [Retry & Fallback](retry-and-fallback.md).

---

## Synchronous usage

Both entry points have thin synchronous wrappers —
`extract_data_list_sync` and `extract_data_list_from_text_sync` — for callers
that are not running inside an event loop:

```python
from saidex import extract_data_list_from_text_sync

lines, stats = extract_data_list_from_text_sync(llm, InvoiceLine, document)
```

They accept the same arguments and must **not** be called from within a running
event loop. See [Observability → Async and sync usage](observability.md#async-and-sync-usage).

---

## Related

- [Extraction](extraction.md) — the single-item `extract_data` / `extract_data_from_text`
- [Extraction Modes](extraction-modes.md) — tool calling vs. raw JSON
- [Retry & Fallback](retry-and-fallback.md) — retry budgets and fallback models
- [Schema Design](schema-design.md) — writing schemas the LLM can fill reliably
