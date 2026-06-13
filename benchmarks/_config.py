"""Benchmark configuration: Ollama server URL, model list, and LLM factory."""

from __future__ import annotations

from langchain_openai import ChatOpenAI

OLLAMA_BASE_URL = "http://localhost:11434/v1"

MODELS: list[str] = [
    # --- Llama ---
    "llama3.2:3b",
    # --- LFM (Liquid Foundation Models) ---
    "lfm2.5:8b",
    "lfm2.5-thinking:1.2b",
    # --- Gemma 4 ---
    "gemma4:e2b",
    "gemma4:e4b",
    # --- Qwen 3.5 / 3.6 ---
    "qwen3.5:0.8b",
    "qwen3.5:2b",
    "qwen3.5:9b",
    # "qwen3.6:27b",
    # --- Granite 4 ---
    "granite4:350m",
    "granite4:1b",
    "granite4:3b",
    # --- Granite 4.1 ---
    "granite4.1:3b",
    "granite4.1:8b",
    # --- Ministral 3 ---
    "ministral-3:3b",
    "ministral-3:8b",
    # --- disabled ---
    # "qwen2.5:7b",
    # "gemma4:12b-mlx",
]

# Vision-capable models for the "vision" benchmark group.
#
# This list was derived by querying the Ollama server's `/api/show` endpoint
# and keeping only models whose `capabilities` include "vision". Re-run
# `python -m benchmarks.list_vision_models` after pulling/updating models to
# regenerate it.
#
# Not vision-capable (text only): llama3.2:3b, lfm2.5:8b, lfm2.5-thinking:1.2b,
# granite4:350m, granite4:1b, granite4:3b, granite4.1:3b, granite4.1:8b.
VISION_MODELS: list[str] = [
    "gemma4:e2b",
    "gemma4:e4b",
    "qwen3.5:0.8b",
    "qwen3.5:2b",
    "qwen3.5:9b",
    # "qwen3.6:27b",
    "ministral-3:3b",
    "ministral-3:8b",
]


def make_llm(model: str, temperature: float = 0.0, timeout: int = 120) -> ChatOpenAI:
    """Return a ChatOpenAI instance pointed at the local Ollama server."""
    return ChatOpenAI(
        model=model,
        base_url=OLLAMA_BASE_URL,
        api_key="ollama",
        temperature=temperature,
        timeout=timeout,
        max_retries=0
    )
