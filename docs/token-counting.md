# Counting tokens

→ [Documentation index](index.md)

Running extraction at scale costs money, and cost is measured in tokens. This
page shows how to measure the tokens an extraction consumes — including the extra
tokens spent on **retries** and the **agent loop**.

SAIDEX does not aggregate token usage in its stats objects. It doesn't need to:
because SAIDEX is LangChain-native and forwards its `callbacks` argument straight
to the model, token usage is captured with the **standard LangChain usage
callback** — the same integration point the [Langfuse guide](langfuse-tracing.md)
uses, with no SAIDEX-specific glue.

!!! note "Version requirement"
    The usage callback shown here (`get_usage_metadata_callback` /
    `UsageMetadataCallbackHandler`) was added in **`langchain-core` 0.3.49**.
    SAIDEX only pins `langchain-core >= 0.2`, so on an older version either
    upgrade, or fall back to a provider-specific callback (e.g. LangChain
    community's `get_openai_callback`).

---

## The short version

Wrap the extraction in `get_usage_metadata_callback()`. It registers itself for
every model call made inside the block, so you don't have to thread anything
through SAIDEX:

```python
from langchain_core.callbacks import get_usage_metadata_callback
from saidex import extract_data_from_text

with get_usage_metadata_callback() as cb:
    invoice, stats = await extract_data_from_text(llm, Invoice, document)

print(cb.usage_metadata)
# Illustrative — keyed by the model name the provider reports:
# {'gpt-4o-mini-2024-07-18': {
#     'input_tokens': 812, 'output_tokens': 143, 'total_tokens': 955,
#     'input_token_details': {'cache_read': 0}, 'output_token_details': {'reasoning': 0},
# }}
```

`usage_metadata` is a dict **keyed by model name**; each value carries
`input_tokens`, `output_tokens`, `total_tokens` and optional per-provider detail
sub-dicts. Everything the block runs — the first call, every retry, the fallback
model, each agent-loop step — accumulates into it automatically.

---

## Explicit handler

When you want to pass the handler yourself (for example to combine it with other
callbacks, or to scope accounting to a single call), construct a
`UsageMetadataCallbackHandler` and hand it to SAIDEX's `callbacks=` argument.
SAIDEX forwards it verbatim to the model:

```python
from langchain_core.callbacks import UsageMetadataCallbackHandler

usage = UsageMetadataCallbackHandler()

invoice, stats = await extract_data_from_text(
    llm, Invoice, document, callbacks=[usage]
)

print(usage.usage_metadata)
```

Every SAIDEX entry point accepts `callbacks` (`extract_data`,
`extract_data_from_text`, `extract_data_list`, the agent-loop runners and all the
`*_sync` wrappers), so the same handler works everywhere.

---

## Retries and the agent loop accumulate

A single extraction is rarely a single LLM call. The usage callback adds up
**every** call in scope, so the token count already includes the work that
SAIDEX's stats report separately:

| Stat | What it costs in tokens |
| --- | --- |
| `stats.primary_retries` / `stats.validation_retries` | each retry is another full LLM call |
| `stats.fallback_used` | the fallback model — it appears under its **own** model-name key |
| `stats.iterations` (agent loop) | every reasoning/tool step is a call |

Because the fallback model is a separate key, and a run may touch more than one
model, reduce across the dict to get a single figure:

```python
def total_tokens(usage: dict) -> int:
    """Sum total_tokens across every model that was called."""
    return sum(model["total_tokens"] for model in usage.values())

print(total_tokens(cb.usage_metadata))
```

Pair the totals with `stats` to reason about efficiency — e.g. tokens per
successful field, or how many tokens a retry-heavy run burned compared to a
clean one.

---

## Provider caveats

Token accounting is only as good as what the model reports, and providers differ:

- **Usage must be reported.** The callback records an entry only when the model
  attaches both `AIMessage.usage_metadata` **and** a `model_name` in
  `response_metadata`. Mainstream integrations (OpenAI, Anthropic, Google, …) do;
  some community/self-hosted wrappers don't — then `usage_metadata` stays empty.
- **Detail varies.** Some providers report only a total, or omit the
  `input_token_details` / `output_token_details` breakdowns (cached, reasoning,
  audio tokens). Treat those sub-dicts as best-effort.
- **Streaming.** A few models emit usage only when explicitly asked (e.g.
  `stream_usage=True`). SAIDEX consumes the final message rather than streaming
  tokens to you, so this rarely bites, but it's worth knowing if you configure
  the model to stream.

---

## Already using Langfuse?

If you trace with [Langfuse](langfuse-tracing.md), token usage **and** cost are
captured automatically as part of each span — no extra callback needed. The
manual approach on this page is for when you want the raw numbers without a
tracing backend.
