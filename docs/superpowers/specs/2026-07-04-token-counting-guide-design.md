# Token counting guide (GH #15)

> Docs-only. Add a guide explaining how to count the tokens consumed by an
> extraction.

## Finding that reframes the issue

The issue states *"The library already surfaces usage via its stats objects
(`StructuredOutputStats`, `AgentRunStats`)."* This is **inaccurate**:

- Those class names do not exist; the real stats types are `ExtractDataStats`
  and `ExtractorRunStats`.
- Neither carries any token field. A repo-wide search finds **no** token
  tracking of any kind — SAIDEX never reads `usage_metadata`.

So the guide cannot read tokens off SAIDEX stats. It is framed instead around the
**LangChain usage callback**, which SAIDEX supports because it forwards its
`callbacks=` argument straight to the model (the same mechanism the Langfuse
guide uses). Decision confirmed in brainstorming: **docs-only, via callbacks** —
no new SAIDEX code.

## Scope

- New page `docs/token-counting.md`, linked from the nav under *Reliability*.
- Show the two idiomatic capture surfaces (verified against the installed
  `langchain-core`):
  - `get_usage_metadata_callback()` context manager (auto-registers; wraps a
    whole extraction incl. retries/fallback/agent-loop).
  - `UsageMetadataCallbackHandler()` passed via SAIDEX's `callbacks=`.
- Explain that usage **accumulates** across every in-scope LLM call, so retries,
  the fallback model and agent-loop steps are already included.
- Provider caveats + the langchain-core **0.3.49** version requirement (SAIDEX
  only pins `>= 0.2`).
- Cross-link Langfuse tracing (which captures usage automatically).

Out of scope: implementing token tracking in the stats objects (would be a
separate `feat`); provider-specific callbacks beyond a passing mention.

## Verified API (langchain-core 1.4.0 installed; added in 0.3.49)

`callback.usage_metadata` is `dict[model_name, {input_tokens, output_tokens,
total_tokens, input_token_details?, output_token_details?}]`. The handler only
records an entry when the message carries **both** `usage_metadata` and
`response_metadata["model_name"]`. Multiple calls **add** (verified: two calls
→ doubled totals). The fallback model appears under its own model-name key.

## Acceptance criteria mapping

- New docs page → `docs/token-counting.md`.
- Concrete snippet reading token usage → context-manager + explicit-handler
  snippets printing `usage_metadata`.
- Retries/agent-loop accumulation + provider caveats → dedicated sections.
- Linked from navigation → `mkdocs.yml` nav entry under *Reliability*.
