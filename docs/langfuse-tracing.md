# Langfuse tracing

→ [Documentation index](index.md)

[Langfuse](https://langfuse.com) is an open-source LLM observability platform:
traces, spans, token costs, latencies, and prompt management. Because SAIDEX is
LangChain-native, it integrates through the standard LangChain **callback
handler** — no SAIDEX-specific glue and **no changes to the public API**.

This page is an end-to-end, copy-pasteable walkthrough. It shows what a trace
looks like for three scenarios:

1. a single happy-path extraction,
2. an extraction that needs a **validation retry**, and
3. a multi-step **agent loop** that calls tools before answering.

!!! note "Looking for the short version?"
    The [Observability](observability.md#langfuse) page has the one-paragraph
    summary. This page is the full guide.

---

## Install

Langfuse is only needed to *run* the example — it is **not** a SAIDEX runtime
dependency. Install it (plus a provider package) into your environment:

```bash
uv pip install "langfuse>=3" langchain-openai
```

!!! warning "Langfuse v2 vs v3"
    The import path and configuration changed in Langfuse **3.0**:

    | | v2 (old) | v3 (current) |
    | --- | --- | --- |
    | Import | `from langfuse.callback import CallbackHandler` | `from langfuse.langchain import CallbackHandler` |
    | Keys | passed to `CallbackHandler(...)` | configured on the `Langfuse(...)` client or via env vars |
    | Session / user / tags | `CallbackHandler(session_id=..., user_id=...)` | set on an enclosing span with `propagate_attributes(...)` |

    This guide uses **v3**.

---

## Configure credentials

Set your project keys once at startup. The recommended way is environment
variables, which the Langfuse client picks up automatically:

```bash
export LANGFUSE_PUBLIC_KEY="pk-lf-..."
export LANGFUSE_SECRET_KEY="sk-lf-..."
export LANGFUSE_HOST="https://cloud.langfuse.com"   # or your self-hosted URL
```

Or configure the client explicitly in code:

```python
from langfuse import Langfuse

Langfuse(
    public_key="pk-lf-...",
    secret_key="sk-lf-...",
    host="https://cloud.langfuse.com",
)
```

---

## 1. A single extraction

Attach the handler via the `callbacks` parameter — accepted by **all four**
public functions (`extract_data`, `extract_data_from_text`,
`extract_data_with_tools`, `run_extractor_agent`). Every LLM call SAIDEX makes is then
recorded automatically.

```python
import asyncio

from langchain_openai import ChatOpenAI
from langfuse import get_client
from langfuse.langchain import CallbackHandler
from pydantic import BaseModel, Field

from saidex import extract_data_from_text


class PersonInfo(BaseModel):
    """Information about a person extracted from free text."""

    name: str = Field(description="Full name of the person")
    age: int = Field(description="Age in years", ge=0, le=150)
    occupation: str = Field(description="Current job or profession")


async def main() -> None:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    handler = CallbackHandler()

    text = "Alice Müller, 34, is a software engineer based in Munich."

    person, stats = await extract_data_from_text(
        llm,
        PersonInfo,
        text,
        callbacks=[handler],
    )

    print(person, f"(retries: {stats.total_retries})")

    # Short-lived script: flush pending events before the process exits.
    get_client().flush()


if __name__ == "__main__":
    asyncio.run(main())
```

**What the trace shows:** one root trace containing a single LLM generation
with the prompt, the tool-call response, token usage, latency, and cost.

!!! tip "Always flush in short-lived scripts"
    Langfuse batches events in the background. In a CLI script or one-off job,
    call `get_client().flush()` before exit so nothing is lost. Long-running
    servers can rely on the periodic background flush.

---

## 2. A retry

When the model returns JSON that fails Pydantic validation, SAIDEX feeds the
field-level error back as a new message and asks the model to correct it (see
[Retry & Fallback](retry-and-fallback.md)). **Each attempt is a separate LLM
generation** nested inside the same trace.

The walkthrough is identical to the single-extraction example above — you do not
configure retries for the tracer, they appear automatically. To make a retry
likely, use a strict schema and a slightly ambiguous input:

```python
from saidex import IbanStr  # validated IBAN field type


class BankDetails(BaseModel):
    """Strict schema — a malformed IBAN forces a self-correcting retry."""

    holder: str = Field(description="Account holder name")
    iban: IbanStr = Field(description="IBAN, validated and normalised")


async def main() -> None:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    handler = CallbackHandler()

    text = "Account holder: ACME GmbH. IBAN: DE89 3704 0044 0532 0130 00."

    result, stats = await extract_data_from_text(
        llm,
        BankDetails,
        text,
        callbacks=[handler],
        max_primary_retries=3,
    )

    print(result, f"(retries: {stats.total_retries})")
    get_client().flush()
```

**What the trace shows:** the root trace now contains **N+1 generations** for
`N` retries. The validation error message that triggered each retry appears as
the *input* of the following generation — so you can read, span by span, exactly
what the model was asked to fix. `stats.total_retries` matches the number of
extra generations in the trace.

---

## 3. The agent loop

`extract_data_with_tools` / `run_extractor_agent` let the model call your tools any number
of times before producing a validated final answer (see
[Agent Loop](agent-loop.md)). Pass the same `callbacks` list and every iteration
— each tool-deciding LLM call — is captured as its own generation under one
trace.

```python
import asyncio

from langchain_openai import ChatOpenAI
from langfuse import get_client
from langfuse.langchain import CallbackHandler
from pydantic import BaseModel, Field

from saidex import Tool, extract_data_with_tools

ORDERS = {"ORD-1042": {"status": "shipped", "eta": "2026-06-15", "carrier": "DHL"}}


class OrderStatusArgs(BaseModel):
    order_id: str = Field(description="Order id, e.g. ORD-1042")


async def order_status_handler(order_id: str) -> dict:
    return ORDERS.get(order_id, {"error": f"Order '{order_id}' not found"})


order_status_tool = Tool(
    name="get_order_status",
    description="Look up the status, ETA, and carrier of an order by its id.",
    parameters=OrderStatusArgs,
    handler=order_status_handler,
)


class TicketResolution(BaseModel):
    order_id: str = Field(description="The order this ticket is about")
    current_status: str = Field(description="Order status as reported by the tool")
    customer_reply: str = Field(description="Friendly reply for the customer")


async def main() -> None:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    handler = CallbackHandler()

    ticket = "Where is my order ORD-1042? It's been two weeks!"

    resolution, stats = await extract_data_with_tools(
        llm,
        TicketResolution,
        ticket,
        tools=[order_status_tool],
        callbacks=[handler],
    )

    print(resolution)
    print(f"iterations={stats.iterations} tool_calls={stats.tool_calls}")
    get_client().flush()


if __name__ == "__main__":
    asyncio.run(main())
```

**What the trace shows:** one trace spanning the whole loop. You can follow the
sequence — model decides to call `get_order_status`, the tool result comes back,
the model calls the `TicketResolution` final-answer tool — and reconcile it with
`stats.iterations` and `stats.tool_calls`.

---

## Session, user, and tags

In Langfuse v3 these attributes are **not** passed to `CallbackHandler(...)`.
Because SAIDEX forwards only the `callbacks` list (not a full run config), set
them on an **enclosing span** with `propagate_attributes` — the handler attaches
to the currently active trace context automatically:

```python
from langfuse import get_client, propagate_attributes
from langfuse.langchain import CallbackHandler

langfuse = get_client()

with langfuse.start_as_current_observation(
    as_type="span", name="invoice-extraction"
) as span:
    with propagate_attributes(
        user_id="user-xyz",
        session_id="session-abc123",
        tags=["production", "v2"],
    ):
        handler = CallbackHandler()
        invoice, stats = await extract_data_from_text(
            llm, Invoice, text, callbacks=[handler]
        )
        span.update_trace(input={"text": text}, output=invoice)

langfuse.flush()
```

Every generation produced inside the block — including retries and agent-loop
iterations — is grouped under one trace tagged with the session and user.

---

## Companion example

A complete, runnable version of all three scenarios lives at
[`examples/09_langfuse_tracing.py`](https://github.com/mlauf-labs/saidex/blob/main/examples/09_langfuse_tracing.py).
It runs without Langfuse credentials (tracing is simply skipped if the keys are
unset) and falls back gracefully if `langfuse` is not installed.

---

## Related

- [Observability](observability.md) — callbacks, logging, LangSmith, metrics
- [Retry & Fallback](retry-and-fallback.md) — why retries appear as extra spans
- [Agent Loop](agent-loop.md) — the tool loop traced in scenario 3
