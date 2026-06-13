# SAIDEX — Structured AI Data Extraction

[![CI](https://img.shields.io/github/actions/workflow/status/mlauf-labs/saidex/tests.yml?label=CI&logo=github)](https://github.com/mlauf-labs/saidex/actions)
[![PyPI](https://img.shields.io/pypi/v/saidex.svg)](https://pypi.org/project/saidex/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

Extract validated [Pydantic](https://docs.pydantic.dev/) models from LLM responses — with automatic retry, field-level error feedback, an optional fallback model, and an agentic tool loop.

Works with any LangChain-compatible model: GPT-4o, Claude, Gemini, and local models via Ollama or vLLM. No tool-calling support required — use `ExtractionMode.JSON` for any chat model.

## Installation

```bash
pip install saidex
```

With automatic OpenAI rate-limit handling:

```bash
pip install "saidex[openai]"
```

## Quick start

```python
import asyncio
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from saidex import extract_from_text

class PersonInfo(BaseModel):
    name: str
    age: int
    occupation: str

async def main():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    person, stats = await extract_from_text(
        llm,
        PersonInfo,
        "Alice Müller, 34, works as a software engineer in Munich.",
    )

    print(person)               # name='Alice Müller' age=34 occupation='software engineer'
    print(stats.total_retries)  # 0 — first attempt was valid

asyncio.run(main())
```

## Key features

| Feature | Detail |
| --- | --- |
| **Validated output** | Pydantic validation with field-level error feedback fed back to the LLM for self-correction |
| **Auto-retry** | Configurable retries with exponential back-off for transient errors and rate limits |
| **Fallback model** | Pair a cheap primary with a powerful fallback — pay for the big model only when needed |
| **No tool-calling required** | `ExtractionMode.JSON` works with any chat model including local models via Ollama / vLLM |
| **Agentic tool loop** | Give the LLM your own tools (lookups, API calls) — it calls them freely, then delivers a validated final answer |
| **Multimodal** | Pass images via standard LangChain messages to any vision-capable model |
| **LangChain-native** | Plugs into any LangChain chat model; supports callbacks (Langfuse, LangSmith, …) |

## Links

- [Full documentation](https://github.com/mlauf-labs/saidex/blob/main/docs/index.md)
- [Examples](https://github.com/mlauf-labs/saidex/tree/main/examples)
- [Changelog](https://github.com/mlauf-labs/saidex/blob/main/CHANGELOG.md)
- [GitHub repository](https://github.com/mlauf-labs/saidex)

## License

Copyright 2026 Martin Lauff. Licensed under the [Apache License, Version 2.0](https://github.com/mlauf-labs/saidex/blob/main/LICENSE).
