# Observability

→ [Documentation index](index.md)

This document covers tracing, callbacks, logging, the standalone
`create_instance_safe` utility, and async / sync usage.

---

## Callbacks

Both `extract_data_from_text` and `extract_data` accept a `callbacks`
parameter.  Pass any list of LangChain `BaseCallbackHandler` instances to
get automatic tracing of every LLM call — including retry attempts.

```python
result, stats = await extract_data(
    llm,
    MySchema,
    messages,
    callbacks=[my_handler],
)
```

---

## Langfuse

[Langfuse](https://langfuse.com) provides open-source LLM observability with
traces, spans, costs, and prompt management. Configure the keys via environment
variables (or the `Langfuse(...)` client) and attach the LangChain handler:

```bash
uv pip install "langfuse>=3"
```

```python
from langfuse.langchain import CallbackHandler

handler = CallbackHandler()  # reads LANGFUSE_* env vars

result, stats = await extract_data(
    llm,
    MySchema,
    messages,
    callbacks=[handler],
)
```

Each retry is recorded as a separate LLM span nested within the parent trace.
The validation error messages that trigger retries appear as the input to each
retry span — this makes it easy to see exactly what the LLM was correcting.

!!! tip "Full walkthrough"
    See [Langfuse Tracing](langfuse-tracing.md) for an end-to-end, runnable
    guide covering single extractions, retries, the agent loop, and session /
    user / tag tracking.

---

## LangSmith

[LangSmith](https://smith.langchain.com) is LangChain's native tracing platform.
Enable it via environment variables — no code changes needed:

```bash
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY=ls-...
export LANGCHAIN_PROJECT=my-project   # optional
```

Alternatively, use the `traceable` decorator from `langsmith.run_helpers`:

```python
from langsmith.run_helpers import traceable

@traceable(name="extract-invoice")
async def extract_invoice(text: str) -> Invoice | None:
    result, _ = await extract_data_from_text(llm, Invoice, text)
    return result
```

---

## Logging

The library uses the standard Python `logging` module under the
`saidex` logger hierarchy.

### Enable debug output

```python
import logging

# All loggers:
logging.basicConfig(level=logging.DEBUG)

# Only saidex:
logging.getLogger("saidex").setLevel(logging.DEBUG)
```

### Log levels used

| Level | Events |
| --- | --- |
| `DEBUG` | Successful extraction, start of each extraction attempt |
| `INFO` | Switching to fallback model, network retry successes |
| `WARNING` | Validation failures, invalid tool calls, rate-limit waits |
| `ERROR` | LLM invocation errors, all retries exhausted |

Debug output example for a single extraction with one retry:

```
DEBUG  saidex.extractor  Attempt 1/3 — schema: Invoice
WARNING saidex.extractor  Validation failed on attempt 1: ...
DEBUG  saidex.extractor  Attempt 2/3 — schema: Invoice
DEBUG  saidex.extractor  Extraction succeeded after 1 retry
```

### Structured logging

For JSON-based log aggregation (Datadog, Loki, CloudWatch), use a
`logging.Formatter` that outputs JSON, or integrate `structlog`:

```python
import logging
import structlog

structlog.configure(
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
)

log = structlog.get_logger("saidex")
```

---

## Safe model instantiation (`create_instance_safe`)

`create_instance_safe` is the building block used by the extraction loop.
It wraps Pydantic model instantiation and converts any `ValidationError` into
a structured, human-readable error string — the same format that gets sent
to the LLM in the retry message.

It is also useful standalone: when you already have LLM output as a
dictionary and want safe validation with rich error messages.

```python
from saidex import create_instance_safe
from pydantic import BaseModel

class Product(BaseModel):
    name:     str
    price:    float
    in_stock: bool

raw = {"name": "Widget", "price": "not-a-float", "in_stock": True}
instance, error = create_instance_safe(Product, **raw)

if instance is None:
    print(error)
    # TYPE ERRORS:
    #   - 'price': wrong type. Expected: float, Received: str = 'not-a-float'
    # REQUIRED CHANGES:
    #   1. Correct the data types of the specified fields.
    # SCHEMA 'Product':
    #   Required (3): name, price, in_stock
```

### Signature

```python
def create_instance_safe(
    schema: type[ModelT],
    **kwargs: Any,
) -> tuple[ModelT | None, str]:
```

Returns `(instance, "")` on success, `(None, error_string)` on failure.

### Use in tests

```python
from saidex import create_instance_safe

def test_clean_money_validator() -> None:
    instance, err = create_instance_safe(Invoice, net="€ 100,00", gross="119.00")
    assert instance is not None
    assert instance.net == 100.0

def test_missing_required_field() -> None:
    instance, err = create_instance_safe(Invoice, net=100.0)
    assert instance is None
    assert "gross" in err
```

This lets you test the exact error text the LLM will receive and verify that
your validator error messages are clear and actionable.

---

## Async and sync usage

The library is **async-first**.  Both `extract_data_from_text` and
`extract_data` are coroutines that must be `await`ed.

### Standard async usage

```python
import asyncio
from saidex import extract_data_from_text

async def main() -> None:
    result, stats = await extract_data_from_text(llm, MySchema, text)
    ...

asyncio.run(main())
```

### Calling from synchronous code

If you are in a purely synchronous context (a CLI script, a Django view, a
background job), use the **synchronous wrappers** instead of managing the event
loop yourself.  Every async entry point has a `*_sync` counterpart that mirrors
its signature exactly and runs the coroutine to completion internally:

| Async | Synchronous wrapper |
| --- | --- |
| `extract_data_from_text` | `extract_data_from_text_sync` |
| `extract_data` | `extract_data_sync` |
| `extract_data_with_tools` | `extract_data_with_tools_sync` |
| `run_extractor_agent` | `run_extractor_agent_sync` |

```python
from saidex import extract_data_from_text_sync

# No async/await, no asyncio.run — just call it.
result, stats = extract_data_from_text_sync(llm, MySchema, text)

# The agent loop has a sync wrapper too:
from saidex import extract_data_with_tools_sync

result, stats = extract_data_with_tools_sync(llm, MySchema, text, tools=[my_tool])
```

The wrappers delegate to `asyncio.run`, so call them only from code that is
**not** already inside an event loop.  If a running loop is detected they raise
a clear `RuntimeError` (rather than deadlocking) telling you to `await` the
async function directly.

### Jupyter / IPython

Jupyter notebooks already run an event loop, so the sync wrappers would raise.
Use `await` directly in a cell instead:

```python
result, stats = await extract_data_from_text(llm, MySchema, text)
```

### FastAPI integration

FastAPI routes are async by default — use `await` as normal:

```python
from fastapi import FastAPI
from saidex import extract_data_from_text

app = FastAPI()

@app.post("/extract")
async def extract_endpoint(text: str) -> dict:
    result, stats = await extract_data_from_text(llm, MySchema, text)
    if result is None:
        return {"error": "extraction failed", "retries": stats.total_retries}
    return result.model_dump()
```

---

## Measuring quality over time

Track extraction quality metrics in production:

```python
import time
from dataclasses import dataclass, field

@dataclass
class ExtractionMetrics:
    total:       int   = 0
    failures:    int   = 0
    fallbacks:   int   = 0
    total_time:  float = 0.0
    total_retries: int = 0

metrics = ExtractionMetrics()

async def extract_tracked(text: str) -> MySchema | None:
    t0 = time.monotonic()
    result, stats = await extract_data_from_text(llm, MySchema, text)
    elapsed = time.monotonic() - t0

    metrics.total += 1
    metrics.total_time  += elapsed
    metrics.total_retries += stats.total_retries
    if result is None:
        metrics.failures += 1
    if stats.fallback_used:
        metrics.fallbacks += 1

    return result
```

---

## Related

- [Extraction](extraction.md) — the two extraction functions
- [Retry & Fallback](retry-and-fallback.md) — understanding retry stats
- [Validators](validators.md) — testing validators with `create_instance_safe`
