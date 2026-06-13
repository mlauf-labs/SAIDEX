# Retry & Fallback

→ [Documentation index](index.md)

Reliable extraction requires two separate retry layers that work independently:

| Layer | What it handles | Configuration |
| --- | --- | --- |
| **Validation retries** | LLM returned wrong schema / invalid values | `max_primary_retries`, `max_fallback_retries` |
| **Network retries** | Connection errors, timeouts, rate limits | `RetryConfig` |

On top of both layers, a **fallback model** can take over when the primary
model exhausts its validation retry budget.

---

## Validation retries

> **Runnable example:** [`examples/03_with_fallback.py`](../examples/03_with_fallback.py)

When Pydantic validation fails, the library builds a structured, field-level
error report and appends it to the conversation as a `HumanMessage`:

```
"The output format is not correct. Please correct it based on the following errors:

MISSING REQUIRED FIELDS:
  - 'revenue_usd_millions': required field is missing

TYPE ERRORS:
  - 'fiscal_year': wrong type. Expected: integer, Received: str = '2025'

REQUIRED CHANGES:
  1. Add all missing required fields.
  2. Correct the data types of the specified fields."
```

The LLM receives this message and gets another attempt.  This loop continues
up to `max_primary_retries` times.

```python
result, stats = await get_structured_data(
    llm,
    MySchema,
    messages,
    max_primary_retries=5,   # allow 5 correction attempts (default: 3)
)
```

The same mechanism applies to invalid tool calls (malformed JSON): the library
detects them, includes the parse error in the retry message, and tries again.

---

## Fallback model

> **Runnable example:** [`examples/03_with_fallback.py`](../examples/03_with_fallback.py)

When the primary model exhausts all its retries, an optional fallback model
takes over.  It receives a **clean context** — the original messages plus one
sentence noting that a previous attempt failed — so the accumulated retry
noise from the primary model does not confuse it.

```python
from langchain_openai import ChatOpenAI
from saidex import extract_from_text

primary  = ChatOpenAI(model="gpt-4o-mini", temperature=0)  # fast + cheap
fallback = ChatOpenAI(model="gpt-4o",      temperature=0)  # more capable

result, stats = await extract_from_text(
    primary,
    FinancialReport,
    report_text,
    fallback_llm_model=fallback,
    max_primary_retries=2,   # fail fast on primary …
    max_fallback_retries=3,  # … be more patient on fallback
)

print(f"Fallback used:     {stats.fallback_used}")
print(f"Primary retries:   {stats.primary_retries}")
print(f"Fallback retries:  {stats.fallback_retries}")
print(f"Total retries:     {stats.total_retries}")
```

### Why use a fallback model?

- **Cost efficiency**: The primary model is cheaper.  The fallback only runs
  when actually needed.
- **Reliability**: Smaller models sometimes fail on complex or deeply-nested
  schemas.  A larger model handles them as a safety net.
- **Clean context**: The fallback starts fresh — without the failed attempts
  that could confuse it.

### What the fallback model receives

```python
# Original messages (unchanged):
[
    SystemMessage("Extract the financial report data."),
    HumanMessage(report_text),
]
# + one added hint:
HumanMessage(
    "A previous attempt to produce structured output for 'FinancialReport' "
    "failed after 2 retries. Please try again carefully, ensuring your "
    "response strictly follows the required JSON schema."
)
```

---

## Network retries (`RetryConfig`)

> **Runnable example:** [`examples/04_custom_retry_config.py`](../examples/04_custom_retry_config.py)

Network retries are handled separately from validation retries.  They catch
transient errors — connection resets, timeouts — and rate-limit responses
before the validation loop ever sees the result.

### Default configuration

When `retry_config` is not provided, the library uses `DEFAULT_RETRY_CONFIG`:

```python
RetryConfig(
    max_retries=3,
    retry_delays=[1.0, 2.0, 4.0],       # exponential back-off
    retryable_exceptions=(               # auto-detected from installed packages:
        httpx.ConnectError,              #   httpx (if installed)
        httpx.TimeoutException,
        openai.APIConnectionError,       #   openai (if installed)
        openai.APITimeoutError,
    ),
    rate_limit_exceptions=(openai.RateLimitError,),  # if openai installed
    rate_limit_retry_interval=10.0,      # seconds between rate-limit retries
    rate_limit_max_duration_seconds=900.0,   # give up after 15 minutes
)
```

### Custom configuration

```python
from saidex import RetryConfig, extract_from_text

config = RetryConfig(
    max_retries=2,
    retry_delays=[2.0, 5.0],             # longer waits
    rate_limit_retry_interval=30.0,      # wait 30 s between rate-limit retries
    rate_limit_max_duration_seconds=300.0,   # give up after 5 minutes
)

result, stats = await extract_from_text(
    llm, MySchema, text, retry_config=config
)
```

The example file [`04_custom_retry_config.py`](../examples/04_custom_retry_config.py)
shows this pattern alongside a `create_instance_safe` standalone demo and a
note on how to add Langfuse tracing via callbacks.

### Disabling network retries (e.g. in tests)

```python
NO_RETRY = RetryConfig(
    max_retries=0,
    retry_delays=[],
    retryable_exceptions=(),
    rate_limit_exceptions=(),
)

result, stats = await get_structured_data(
    llm, MySchema, messages, retry_config=NO_RETRY
)
```

### Custom exception types

```python
import httpx
import openai

config = RetryConfig(
    retryable_exceptions=(
        httpx.ConnectError,
        httpx.TimeoutException,
        openai.APIConnectionError,
        openai.APITimeoutError,
    ),
    rate_limit_exceptions=(openai.RateLimitError,),
)
```

### `RetryConfig` reference

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `max_retries` | `int` | `3` | Max transient-error retries per LLM call |
| `retry_delays` | `list[float]` | `[1.0, 2.0, 4.0]` | Seconds to wait before each retry |
| `retryable_exceptions` | `tuple[type[Exception], ...]` | httpx + openai network errors | Exceptions that trigger a transient retry |
| `rate_limit_exceptions` | `tuple[type[Exception], ...]` | `openai.RateLimitError` | Exceptions that trigger rate-limit wait |
| `rate_limit_retry_interval` | `float` | `10.0` | Seconds between rate-limit retries |
| `rate_limit_max_duration_seconds` | `float` | `900.0` | Total seconds before giving up on rate-limit retries |

---

## Full retry flow

```
get_structured_data(primary_llm, schema, messages)
│
│  Phase 1 — Primary model
│  ┌────────────────────────────────────────────────┐
│  │  for attempt in range(max_primary_retries):    │
│  │    try LLM call (with network retries)         │
│  │    validate response with Pydantic             │
│  │    if OK → return (instance, stats)            │
│  │    else   → append error message, retry        │
│  └────────────────────────────────────────────────┘
│         ↓ all attempts failed
│
│  Phase 2 — Fallback model (if configured)
│  ┌────────────────────────────────────────────────┐
│  │  reset to original messages + hint             │
│  │  for attempt in range(max_fallback_retries):   │
│  │    try LLM call (with network retries)         │
│  │    validate response with Pydantic             │
│  │    if OK → return (instance, stats)            │
│  │    else   → append error message, retry        │
│  └────────────────────────────────────────────────┘
│         ↓ all attempts failed
│
└─ return (None, stats)
```

---

## Error handling

Always check whether the result is `None` before using it:

```python
result, stats = await extract_from_text(llm, MySchema, text)

if result is None:
    logger.error(
        "Extraction failed after %d retries (fallback used: %s)",
        stats.total_retries,
        stats.fallback_used,
    )
    # Options:
    # - raise an exception
    # - return a default / partial result
    # - skip this record and continue
    raise RuntimeError("Could not extract structured data from the document.")

# result is guaranteed valid here
process(result)
```

### Tracking retry costs

`StructuredOutputStats` lets you track quality metrics over time:

```python
results = []
for text in documents:
    result, stats = await extract_from_text(llm, MySchema, text)
    results.append({
        "result": result,
        "retries": stats.total_retries,
        "fallback": stats.fallback_used,
        "failed": result is None,
    })

failures   = sum(1 for r in results if r["failed"])
avg_retries = sum(r["retries"] for r in results) / len(results)
print(f"Failure rate: {failures}/{len(results)}, avg retries: {avg_retries:.2f}")
```

---

## Related

- [Extraction](extraction.md) — the two extraction functions in detail
- [Schema Design](schema-design.md) — reducing retries by writing better schemas
- [Validators](validators.md) — reducing retries by cleaning LLM output in the schema
- [Observability](observability.md) — tracing retries with Langfuse / LangSmith
