# Design: LangChain callback events for the agent loop and retry loops

**Date:** 2026-07-06
**Status:** Approved (pending spec review)
**Branch:** `feature/callback-tool-events`

---

## Problem

SAIDEX forwards the caller's `callbacks` only to `llm.ainvoke(...)`, so a
LangChain-based tracer (Langfuse, LangSmith, Arize, …) sees **only the LLM
generations**. Two gaps follow:

1. **Tool executions are invisible.** In the agent loop, a validated tool call
   invokes the caller's handler through a bare `await tool_obj.execute(tc_args)`
   ([`src/saidex/extractor.py`](../../../src/saidex/extractor.py) line ~1292).
   No `on_tool_start` / `on_tool_end` / `on_tool_error` events fire, so no tracer
   records the tools. `ExtractorRunStats.tool_calls` counts them internally, but
   nothing is emitted.
2. **Generations are flat.** Each LLM call is reported as an independent run with
   no enclosing parent, so retries and agent-loop iterations appear side by side
   rather than nested under one logical operation.

The docs already *promise* the missing piece:
[`docs/langfuse-tracing.md`](../../langfuse-tracing.md) scenario 3 tells the reader
they can "follow the sequence — model decides to call `get_order_status`, the tool
result comes back". Today those tool executions are **not** spans; the reader can
only infer them from the next generation's input. This design makes that promise
true.

## Goals

- Fire LangChain `on_tool_start` / `on_tool_end` / `on_tool_error` for every
  **helper** tool executed in the agent loop.
- Wrap the agent loop and the `extract_data` / `extract_data_list` retry loops in
  an enclosing **chain run** so every generation and every tool span nests under
  one parent, and run-level metrics can be attached at the end.
- Stay **tracer-agnostic** and reuse the existing `callbacks=` API — no new
  runtime dependency, no public API change.
- Preserve the hard invariant that **observability can never break a run**: a
  faulty callback handler is logged and swallowed, exactly like `on_complete` and
  the `on_extraction` observers today.

## Non-goals

- No native OpenTelemetry instrumentation in the library.
- No extension of the SAIDEX-specific `ExtractionEvent` / `on_extraction` event
  system with tool events (it does not reach LangChain tracers; it stays
  complementary and unchanged).
- No change to the public API surface (`src/saidex/__init__.py` exports are
  untouched). The new module is private.

## Chosen approach

Use the idiomatic LangChain callback-manager mechanics:

```python
cm = AsyncCallbackManager.configure(inheritable_callbacks=callbacks)
chain_run = await cm.on_chain_start({"name": "saidex.agent_loop"}, inputs, metadata=...)
# LLM calls:  config={"callbacks": chain_run.get_child()}   → generations nest
# Tools:      child = chain_run.get_child(); tool_run = await child.on_tool_start(...)
#             ... run handler ... await tool_run.on_tool_end(output) / on_tool_error(exc)
await chain_run.on_chain_end(outputs)   # or on_chain_error(exc)
```

`AsyncCallbackManager.configure`, `get_child`, `on_chain_start`, `on_tool_start`
are stable `langchain-core >= 0.2` APIs — already a core dependency.

Rejected alternatives: native OTel (new dependency, bypasses `callbacks=`);
extending `on_extraction` with tool events (never reaches LangChain tracers).

## Architecture

### 1. New private module `src/saidex/_callbacks.py`

Encapsulates all callback-manager plumbing so `extractor.py` stays readable.

- **`_ChainRun.start(callbacks, *, name, inputs, metadata) -> _ChainRun`** —
  builds `AsyncCallbackManager.configure(inheritable_callbacks=callbacks)` and
  opens a run via `on_chain_start`. When `callbacks` is falsy it returns a
  **no-op** instance (all methods below become no-ops) so there is zero overhead
  and byte-for-byte unchanged behaviour when no callbacks are passed.
- **`.child_callbacks() -> AsyncCallbackManager | None`** — returns
  `run.get_child()` to hand to `llm.ainvoke(config={"callbacks": ...})`. Returns
  `None` in the no-op case, so the existing "plain `ainvoke` when no callbacks"
  path is preserved.
- **`.tool_span(name, args) -> AsyncContextManager[_ToolSpan]`** — fires
  `on_tool_start` **before** the handler runs (accurate span duration). The
  yielded `_ToolSpan` exposes `record_output(str)` and `record_error(Exception)`;
  on exit it fires `on_tool_end(output)` or, when an error was recorded,
  `on_tool_error(exc)`.
- **`.end(outputs: dict)` / `.error(exc)`** — closes the run via `on_chain_end`
  / `on_chain_error`.
- **Isolation:** every callback invocation is wrapped defensively (`try/except`,
  log at error level, swallow). A broken handler must never propagate out of the
  tracing helper.

The module is not exported from `__init__.py`.

### 2. `src/saidex/tools.py` — surface handler exceptions

To let `on_tool_error` receive the real exception, `Tool.execute` is split into
an internal step that reports *both* the string content and any handler
exception:

```python
@dataclass
class _ToolOutcome:
    content: str                      # goes into the ToolMessage (always a str)
    handler_error: Exception | None   # set only when the handler raised

async def _invoke(self, raw_args) -> _ToolOutcome: ...
async def execute(self, raw_args) -> str:            # unchanged behaviour
    return (await self._invoke(raw_args)).content
```

- **Invalid arguments** (a bad LLM tool call, not a handler crash) → `_ToolOutcome`
  with the existing `"Invalid arguments for tool '…': …"` content and
  `handler_error=None` → reported as `on_tool_end` (a normal tool result telling
  the model to fix its call).
- **Handler raises** → `_ToolOutcome` with the existing
  `"Tool '…' raised an error: …"` content and `handler_error=exc` → reported as
  `on_tool_error`. The loop still appends the error-string `ToolMessage` and
  continues unchanged.
- `execute` keeps its signature and behaviour; it is public and directly tested
  in `tests/test_agent_loop.py`. `_ToolOutcome` is private to `tools.py`.

### 3. `src/saidex/extractor.py` — open/close chain runs

- **`run_extractor_agent`** (the outer function spanning primary **and** fallback)
  opens the `saidex.agent_loop` chain run, threads the `_ChainRun` into
  `_run_extractor_agent_with_model` (used for both the generation `config` and the
  tool spans), and closes it in the existing `_finish` helper.
- Inside `_run_extractor_agent_with_model`:
  - LLM calls use `config={"callbacks": chain_run.child_callbacks()}` (falling
    back to plain `ainvoke` when `child_callbacks()` is `None`).
  - The helper-tool execution is wrapped:
    ```python
    async with chain_run.tool_span(tc_name, tc_args) as span:
        outcome = await tool_obj._invoke(tc_args)
        if outcome.handler_error is not None:
            span.record_error(outcome.handler_error)
        else:
            span.record_output(outcome.content)
    tool_result_messages.append(ToolMessage(content=outcome.content, tool_call_id=tc_id))
    ```
  - **The final-answer schema tool is not a tool span.** It is the loop's
    structured output and is reported through the chain run's `end(outputs=...)`.
  - An **unknown tool** call (the model hallucinated a tool name) is reported as a
    tool span whose output is the existing `"Unknown tool '…'"` message
    (`on_tool_end`, not `on_tool_error` — it is not a handler crash).
- **`extract_data`** opens the `saidex.extract_data` chain run, threads the
  `_ChainRun` into `_try_with_model` (primary and fallback) for nested
  generations, and closes it in `_finish`.
- **`extract_data_list`** opens a `saidex.extract_data_list` chain run and passes
  the trace to the inner `extract_data` through a new **private** parameter
  `_trace: _ChainRun | None = None`, mirroring the existing `_notify_observers`
  flag. When `_trace` is provided the inner `extract_data` reuses it (nesting its
  generations under the list run) and does **not** open or close a second chain
  run.

### 4. What is recorded (PII-light, matching current stance)

- **`on_chain_start` inputs / metadata (static, known at start):**
  `{"schema": schema.__name__, "mode": mode.value, "tools": [t.name for t in tools]}`
  (tools only for the agent loop). The raw input text is included **only** when
  `capture_source_text=True`, consistent with today's opt-in privacy default.
- **`on_chain_end` outputs (dynamic — metrics are only known at the end):** the
  validated result **plus** the same metrics `on_complete` already exposes.
  - Agent loop: `success, failure_reason, iterations, tool_calls,
    validation_retries, fallback_used, problem_fields, format_errors`.
  - `extract_data`: `success, failure_reason, total_retries, primary_retries,
    fallback_retries, fallback_used, problem_fields, format_errors`.

  (The base callback protocol has no "update metadata at end" call, so dynamic
  metrics ride in the `on_chain_end` outputs dict alongside the result rather than
  in start-time metadata.)

Resulting Langfuse/LangSmith tree for the agent loop:

```
saidex.agent_loop
├── generation (iteration 1)
├── tool: get_order_status        ← on_tool_start / on_tool_end
├── generation (iteration 2)
└── generation (final answer)     ← output on the parent via on_chain_end
```

### 5. Error / lifecycle handling

- Normal extraction *failures* (retries exhausted, `llm_error`, …) do **not**
  raise — the functions return `None`. These close the chain run with
  `end(outputs={... "success": False ...})`.
- `on_chain_error` is reserved for an unexpected exception propagating out of the
  wrapped body. The outer functions wrap their body so such an exception fires
  `error(exc)` and re-raises.

## Testing (TDD)

New file `tests/test_callbacks_tracing.py`, using a fake async callback handler
that records `(event_name, payload)` tuples. All tests run without API keys or
network (mock-LLM pattern from `tests/test_agent_loop.py`).

1. **Agent-loop event order & nesting** — one helper tool + final answer:
   `on_chain_start` → generation events → `on_tool_start` → `on_tool_end` → …
   → `on_chain_end`. Assert the tool span carries the tool name and the args/result.
2. **Handler crash** → `on_tool_error` fires with the raised exception; the loop
   still produces a final answer and does not crash.
3. **Unknown tool** → `on_tool_start` / `on_tool_end` with the `"Unknown tool"`
   output.
4. **`extract_data` retry nesting** → a validation retry produces multiple
   generations nested under one `saidex.extract_data` chain run; `on_chain_end`
   carries the retry metrics.
5. **Isolation** → a handler that raises inside `on_tool_start` (or `on_chain_end`)
   is swallowed; the extraction still returns its result.
6. **`callbacks=None`** → no manager is built and behaviour is identical (guards
   the zero-overhead path).
7. **Sync wrapper path** → callbacks still fire through `extract_data_with_tools_sync`.

All existing tests must remain green (the change is additive; mocks that ignore
`config` are unaffected).

## Docs & changelog

- Update [`docs/langfuse-tracing.md`](../../langfuse-tracing.md) — scenario 3 now
  describes real tool spans nested under a `saidex.agent_loop` span; scenarios 1–2
  mention the enclosing chain-run span.
- Update [`docs/observability.md`](../../observability.md) and
  [`docs/agent-loop.md`](../../agent-loop.md) to note that tool executions are now
  traced.
- Review [`examples/09_langfuse_tracing.py`](../../../examples/09_langfuse_tracing.py)
  for consistency.
- The change is user-relevant → a `feat(observability): …` Conventional Commit so
  Commitizen generates the `CHANGELOG.md` entry. The **trace shape** changes
  (generations now nest under a SAIDEX chain-run span) but the **public API does
  not**, so this is not a `BREAKING CHANGE`; the behaviour change is called out in
  the commit body and docs.

## Git workflow

- Branch: `feature/callback-tool-events` off `develop` (already created).
- Conventional Commits throughout; PR targets `develop`.
- CI gate before PR: `ruff check`, `ruff format --check`, `mypy src/`,
  `pytest --cov`.
```
