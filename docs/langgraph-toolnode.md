# LangGraph: SaidexToolNode

→ [Documentation index](index.md)

`SaidexToolNode` is a drop-in replacement for LangGraph's `ToolNode` that
repairs and validates tool calls **before** executing them. When a call is
wrong, either the agent fixes it from SAIDEX's structured feedback or SAIDEX
corrects it itself — the agent loop keeps running instead of breaking on a
malformed call.

```bash
pip install "saidex[langgraph]"
```

```python
from saidex.langgraph import SaidexToolNode

tool_node = SaidexToolNode(tools)          # instead of ToolNode(tools)
graph.add_node("tools", tool_node)
```

Core `saidex` never imports `langgraph` — this integration lives entirely in
`saidex.langgraph` and is only pulled in when you import it.

## Why

The stock `ToolNode` executes whatever it is given:

- **`invalid_tool_calls` are silently ignored.** A tool call whose argument
  JSON did not parse never gets a `ToolMessage` — and an unanswered tool-call
  id makes OpenAI-compatible providers reject the *next* model turn.
- **Validation feedback is generic.** A schema mismatch surfaces as
  `"Please fix the error and try again."` with a raw exception message.

`SaidexToolNode` guarantees **every tool-call id receives exactly one
`ToolMessage`**, and invalid calls get SAIDEX's structured, field-level
feedback (missing fields, type errors, allowed enum values, required changes).

## Who fixes an invalid call

| | Agent corrects (default) | SAIDEX corrects |
| --- | --- | --- |
| Enable | `SaidexToolNode(tools)` | `SaidexToolNode(tools, correction_model=llm)` |
| Who fixes the call | the agent, in its next turn | a correction LLM call inside the node |
| Visible in graph state | yes — the bad call, its error `ToolMessage`, the agent's new call | no — only the corrected call and its result |
| Cost | one extra agent turn with the full history; the error stays in context | one small correction call (a cheaper model is fine); nothing added to later turns |

```python
from saidex.langgraph import SaidexToolNode

tool_node = SaidexToolNode(
    tools,
    correction_model=small_fast_llm,   # SAIDEX corrects; omit to let the agent do it
    max_correction_retries=2,
)
```

Use `correction_mode=ExtractionMode.JSON` for a correction model without tool
calling. If correction fails, the call falls through to `on_unresolved` (by
default the agent gets feedback), so both modes combine naturally.

### What the correction model sees

- the tool name and description, the invalid arguments and the structured
  validation errors;
- the last `correction_context_messages` (default `10`) messages before the
  `AIMessage`, without system messages, each cut to 2,000 characters, plus the
  `AIMessage`'s own text. `0` turns the conversation context off.

The context is rendered as text inside one prompt rather than passed as chat
messages: the history references tools the correction model does not have
bound, which providers reject.

**Privacy.** With a `correction_model`, the correction model receives up to
`correction_context_messages` earlier messages, including tool results. Set
`correction_context_messages=0` if that content must not reach the correction
model (for example because it is served by a different provider).

### Missing required fields go to the agent

A call that lacks a required field is **not** sent to the correction model —
it has no reliable source for a value the agent never sent and would invent
one. Such calls are answered with feedback so the agent fills them in.
Two exceptions:

- the arguments contain a key the schema does not know (e.g. `town` instead of
  `city`) — the field is most likely just misnamed, so it is corrected;
- `correct_missing_fields=True` lets the correction model fill missing values
  from the conversation.

## What happens per call

1. **Repair** — `invalid_tool_calls` entries are recovered deterministically:
   `<think>` blocks are stripped (`strip_thinking=True`) and malformed JSON is
   fixed with json-repair (`json_repair=True`).
2. **Validate** — arguments are checked against the tool's Pydantic schema
   (`tool.tool_call_schema`, which excludes injected arguments such as
   `InjectedState` / `InjectedToolCallId`), then by your tool validator, if any.
   Tools with non-Pydantic schemas and unknown tool names are passed through
   to the executor unchanged.
3. **Correct** — with a `correction_model`, correctable calls are corrected
   concurrently (bounded by `config["max_concurrency"]`).
4. **Resolve** — calls that are still invalid follow `on_unresolved`.
5. **Execute** — every valid call runs on an internal stock `ToolNode`
   (parallel execution, `handle_tool_errors`, `Command` returns and
   `InjectedState` keep working). Calls are independent: valid calls run even
   when a sibling call in the same message is answered with feedback.

## Calls that stay invalid: `on_unresolved`

| Value | Behaviour |
| --- | --- |
| `"feedback"` (default) | The call is **not** executed; a `ToolMessage(status="error")` with structured guidance lets the agent fix it next turn. |
| `"raise"` | `ToolCallValidationError` is raised **before any call executes**. Correction (if enabled) has already been tried. `.calls` holds the unresolved calls that have an id (calls without an id count as failures but cannot be listed). |
| a hook | Called once per unresolved call with an `UnresolvedToolCall`; sync or async. |

The hook receives the tool name and id, the call's arguments (as they were
before any correction attempt), the raw payload, the field issues, the error
text, whether correction ran, the conversation and the graph state. It returns:

- a `dict` — salvaged arguments, re-validated, then executed (`outcome="salvaged"`);
  if they are still invalid the call gets the default feedback and a warning is logged;
- a `str` — your own feedback text for the agent;
- `None` — the default feedback.

An exception raised by the hook propagates and stops the run.

```python
from saidex import UnresolvedToolCall


def fill_user_id(call: UnresolvedToolCall) -> dict | None:
    if call.tool_name == "get_orders" and call.args is not None:
        return {**call.args, "user_id": call.state["user_id"]}
    return None


tool_node = SaidexToolNode(tools, on_unresolved=fill_user_id)
```

Prefer `"feedback"` in agent loops: when the node raises, LangGraph does not
apply its updates, and a checkpointed thread is left with unanswered tool-call
ids that make the next model call fail.

## Tool validators

Pydantic validators on a tool's argument schema run as usual — once before
execution and again when the tool runs. They receive no context (LangChain
calls them without one at execution), and should be free of side effects.

For checks that need context, pass per-tool validators:

```python
from saidex import ToolCallValidatorContext, ValidationIssue


def order_belongs_to_user(args, ctx: ToolCallValidatorContext):
    if not args.order_id.startswith(ctx.state["user_id"]):
        return ValidationIssue(message="order does not belong to the user", field_path="order_id")
    return None


tool_node = SaidexToolNode(tools, validators={"cancel_order": order_belongs_to_user})
```

A validator is called as `fn(args, ctx)` and returns what an `extract_data`
`validator=` returns: `None` to accept, a `str` or `ValidationIssue`s to
reject (`on_mismatch="flag"` only records the issue), or raises to reject. It
runs after Pydantic validation on the model's own arguments
(`ctx.phase == "initial"`), inside the correction cycle (`"correction"`, with
`attempt`, `max_attempts` and `is_final`) and on arguments returned by an
`on_unresolved` hook (`"salvage"`) — never at execution time.

When several calls are corrected at once, their validators can run
concurrently, so a validator should not share mutable state without protection.

A validator may call LangGraph's `interrupt()`, for example to let a human
approve a call: the graph pauses as with any node (a checkpointer is
required). On resume LangGraph runs the node again from the start, so earlier
validators and corrections run again too.

## Clean message history

- **Correction traffic never enters state.** The correction conversation is
  private to the node — you pay correction tokens once, not on every later turn.
- **`sanitize_messages=True`** (default) returns an updated copy of the
  `AIMessage` (same `id`) carrying the repaired, corrected or salvaged
  arguments — in `tool_calls`, in tool-call content blocks (`tool_use`,
  `tool_call`, `invalid_tool_call`, `function_call`) and in OpenAI's raw
  `additional_kwargs["tool_calls"]`. The `add_messages` reducer (used by
  `MessagesState`) **replaces** the malformed message, so leaked
  chain-of-thought or broken JSON disappears from history. Calls answered with
  feedback stay untouched — their feedback message references the call id.
  Set `sanitize_messages=False` if your messages channel uses a plain append
  reducer, where the update would **duplicate** the message.

  Provider integrations send the `tool_calls` arguments when a content block
  has the same id (`langchain-anthropic` prefers `tool_calls` over a matching
  `tool_use` block), so the corrected arguments reach the provider either way;
  rewriting the blocks keeps the stored state clean too.

  Corrected and salvaged arguments are stored in JSON form, as a model would
  send them (a `date` as `"2026-10-11"`, a `Decimal` as a string), so the next
  model call can serialize the history; the tool itself still receives the
  typed values. `SecretStr` fields appear masked (`"**********"`) in the
  stored arguments.

## Errors raised by a tool

Exceptions raised while a tool runs are handled exactly like the stock node
does, through `handle_tool_errors` (forwarded unchanged): by default only
argument-validation errors become an error `ToolMessage` and every other
exception stops the run; `handle_tool_errors=True` turns every exception into
an error `ToolMessage`. `interrupt()` always propagates. SAIDEX does not try to
correct runtime failures — that is the tool's responsibility.

## Sync graphs

`invoke()` runs repair, validation, correction and `on_unresolved` through a
blocking bridge — it also works inside an already running event loop, e.g.
`graph.invoke(...)` from a Jupyter cell. Valid calls then execute on the stock
node's sync path: a thread pool bounded by `config["max_concurrency"]`
(`max_concurrency=1` runs them one after another). As with the stock node,
async-only tools need `ainvoke`.

- **Event loop.** Repair, validation, correction and hooks run on their own
  event loop (on a worker thread when a loop is already running), so
  `on_unresolved` hooks and tool validators must not rely on objects bound to
  the caller's event loop, such as an async client created on it.
- **`max_concurrency` of 0 or below.** A `config["max_concurrency"]` of 0 or a
  negative value leaves corrections unbounded, but the stock sync executor
  rejects it when valid calls run — exactly as with the stock `ToolNode`.

## Edge cases, documented honestly

- **A tool call with no usable `id` cannot be answered.** A `ToolMessage`
  requires a `tool_call_id`, so such a call is dropped: not executed, not
  answered, and (with `sanitize_messages=True`) removed from the sanitized
  message. A warning is logged and `ToolCallStats.outcome` is `"dropped"`.
  With `on_unresolved="raise"` it still counts as a failure.
- **Correction LLM calls perform no network-level retries.** The
  `max_correction_retries` budget governs *validation* attempts only. A
  transient provider error (e.g. an HTTP 429) degrades that call straight to
  `on_unresolved` instead of crashing the node.

## Observability

Each invocation emits a `ToolNodeEvent` with per-call `ToolCallStats`:
`outcome` (`executed`, `corrected`, `salvaged`, `feedback`, `dropped`),
`repaired`, `prevalidated`, `correction_attempted`, `correction_retries` and
`field_issues` (including tool-validator issues):

```python
from saidex import collect_stats, on_tool_node

async with collect_stats() as sink:
    result = await graph.ainvoke(inputs)
for stats in sink:          # ToolNodeStats alongside extraction stats
    ...

sub = on_tool_node(lambda event: print(event.stats.feedback_count))
```

`on_extraction()` listeners do **not** receive tool-node events. No
`ToolNodeEvent` is emitted on the `"raise"` path.

Correction calls receive the node's callbacks: they nest under the node's span
in Langfuse/LangSmith as a `saidex.extract_data` chain run, and LangChain's
`get_usage_metadata_callback()` counts their tokens, keyed by model name — so a
separate correction model's cost is visible on its own (see
[Counting tokens](token-counting.md)).

## Migrating from `on_invalid`

`on_invalid` still works but emits a `DeprecationWarning`; it will be removed
in the release after next:

| Before | After |
| --- | --- |
| `on_invalid="feedback"` | omit it (no `correction_model`) |
| `on_invalid="correct", correction_model=llm` | `correction_model=llm` |
| `on_invalid="raise"` | `on_unresolved="raise"` |

Behaviour changes compared to 0.7: a `correction_model` alone now enables
correction; calls missing a required field go to the agent instead of the
correction model; the correction prompt carries conversation context
(`correction_context_messages=0` restores the old prompt); corrections run
concurrently; content blocks are sanitized too; `invoke()` executes tools on
the stock sync path, so async-only tools no longer run there — they need
`ainvoke()`, as with the stock node; `max_correction_retries=0` together with
a `correction_model` raises `ValueError` (it never called the model in 0.7
either).

## Scope

`SaidexToolNode` fixes tool calls that are *present but wrong* on the last
`AIMessage` of a graph whose tools node receives the state. Tool calls lost
during model **streaming** happen one layer earlier and cannot be recovered by
any tool node.

**Send-API fan-out payloads are passed straight through.** `create_react_agent`
(`version="v2"`) and langchain's `create_agent` dispatch each tool call
separately via `Send`, so a tool node never sees the whole `AIMessage`.
Validating right after the model call is tracked in
[#104](https://github.com/mlauf-labs/saidex/issues/104).

---

## Related

- [`examples/16_langgraph_toolnode.py`](../examples/16_langgraph_toolnode.py) — a complete runnable graph
- [Observability](observability.md) — `collect_stats`, `on_tool_node`, callbacks
- [Agent Loop](agent-loop.md) — SAIDEX's own tool loop, for extractions that need lookups before a final answer
