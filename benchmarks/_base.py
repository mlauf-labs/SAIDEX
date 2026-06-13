"""Shared base types and helpers for all benchmark scenarios."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generic, Type, TypeVar

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from saidex import (
    ExtractionMode,
    StructuredOutputStats,
    extract_from_text,
    get_structured_data,
)
from pydantic import BaseModel

from ._config import MODELS, make_llm

ModelT = TypeVar("ModelT", bound=BaseModel)

# Directory where rendered vision images are cached.
IMAGE_CACHE_DIR = Path(__file__).parent / "results" / "images"


@dataclass
class BenchmarkScenario(Generic[ModelT]):
    """Describes a single extraction scenario.

    When *vision* is ``True``, the scenario's :attr:`text` is rendered into a
    PNG image and sent to a vision-capable model as image input instead of
    being passed as plain text.
    """

    name: str
    description: str
    schema: Type[ModelT]
    text: str
    mode: ExtractionMode = ExtractionMode.TOOL_CALLING
    system_prompt: str | None = None
    expected: dict[str, Any] = field(default_factory=dict)
    vision: bool = False
    image_prompt: str = "Extract the requested fields from this document image."


@dataclass
class BenchmarkResult:
    """Result of running one scenario against one model."""

    scenario_name: str
    model: str
    success: bool
    duration_s: float
    retries: int
    value: BaseModel | None
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario_name,
            "model": self.model,
            "success": self.success,
            "duration_s": round(self.duration_s, 2),
            "retries": self.retries,
            "value": self.value.model_dump() if self.value else None,
            "error": self.error,
        }


async def warmup_model(model_name: str, *, timeout: int = 300) -> tuple[bool, float, str | None]:
    """Send a tiny prompt so Ollama loads the model into memory.

    This is called once before a model's benchmark scenarios run, so the first
    timed scenario does not also pay the model load cost. Returns
    ``(success, duration_seconds, error_message)``; warmup failures are
    non-fatal and simply reported by the caller.
    """
    llm = make_llm(model_name, timeout=timeout)
    t0 = time.perf_counter()
    try:
        await llm.ainvoke([HumanMessage(content="hi")])
        return True, time.perf_counter() - t0, None
    except Exception as exc:  # noqa: BLE001
        return False, time.perf_counter() - t0, str(exc)


def _render_scenario_image(scenario: BenchmarkScenario[Any]) -> Path:
    """Render *scenario*'s text to a cached PNG image and return its path."""
    from ._image import render_text_to_image

    safe_name = "".join(c if c.isalnum() else "_" for c in scenario.name)
    image_path = IMAGE_CACHE_DIR / f"{safe_name}.png"
    if not image_path.exists():
        render_text_to_image(scenario.text, image_path)
    return image_path


def _build_vision_messages(scenario: BenchmarkScenario[Any]) -> list[BaseMessage]:
    """Render the scenario text to an image and wrap it in a multimodal message."""
    from ._image import image_to_data_url

    image_path = _render_scenario_image(scenario)
    data_url = image_to_data_url(image_path)

    system = scenario.system_prompt or (
        "You are a precise document-extraction assistant. Read all visible text "
        "in the image and populate the schema exactly. Use null for fields that "
        "are not present in the image."
    )
    return [
        SystemMessage(content=system),
        HumanMessage(
            content=[
                {"type": "text", "text": scenario.image_prompt},
                {"type": "image_url", "image_url": {"url": data_url}},
            ]
        ),
    ]


async def run_scenario(
    scenario: BenchmarkScenario[ModelT],
    model_name: str,
) -> BenchmarkResult:
    """Run *scenario* with *model_name* and return a BenchmarkResult."""
    llm = make_llm(model_name)

    t0 = time.perf_counter()
    try:
        if scenario.vision:
            messages = _build_vision_messages(scenario)
            value, stats = await get_structured_data(
                llm, scenario.schema, messages, mode=scenario.mode
            )
        else:
            kwargs: dict[str, Any] = {"mode": scenario.mode}
            if scenario.system_prompt:
                kwargs["system_prompt"] = scenario.system_prompt
            value, stats = await extract_from_text(
                llm, scenario.schema, scenario.text, **kwargs
            )
        duration = time.perf_counter() - t0
        return BenchmarkResult(
            scenario_name=scenario.name,
            model=model_name,
            success=value is not None,
            duration_s=duration,
            retries=stats.total_retries,
            value=value,
        )
    except Exception as exc:  # noqa: BLE001
        duration = time.perf_counter() - t0
        return BenchmarkResult(
            scenario_name=scenario.name,
            model=model_name,
            success=False,
            duration_s=duration,
            retries=0,
            value=None,
            error=str(exc),
        )


def print_scenario_results(results: list[BenchmarkResult]) -> None:
    """Print a comparison table for a single scenario across all models."""
    if not results:
        return

    scenario_name = results[0].scenario_name
    print(f"\n{'=' * 70}")
    print(f"  {scenario_name}")
    print(f"{'=' * 70}")
    print(f"  {'Model':<20} {'Time(s)':>7}  {'Retries':>7}  {'OK':>4}  Result")
    print(f"  {'-' * 66}")

    for r in results:
        status = "YES" if r.success else "NO "
        if r.success and r.value:
            summary = _summarise(r.value)
        elif r.error:
            summary = f"ERROR: {r.error[:60]}"
        else:
            summary = "—"
        print(f"  {r.model:<20} {r.duration_s:>7.1f}  {r.retries:>7}  {status:>4}  {summary}")


def _summarise(model: BaseModel) -> str:
    """Return a one-line summary of a Pydantic model for table display."""
    d = model.model_dump()
    parts = []
    for k, v in d.items():
        if v is None:
            continue
        if isinstance(v, list):
            parts.append(f"{k}=[{len(v)}]")
        elif isinstance(v, str) and len(v) > 30:
            parts.append(f"{k}={v[:27]}…")
        else:
            parts.append(f"{k}={v}")
        if len(parts) >= 4:
            break
    return "  ".join(parts)


async def run_all_models(scenario: BenchmarkScenario[Any]) -> list[BenchmarkResult]:
    """Run *scenario* against every model in MODELS and print results."""
    results: list[BenchmarkResult] = []
    for model in MODELS:
        print(f"  Running [{scenario.name}] on {model}…", end=" ", flush=True)
        result = await run_scenario(scenario, model)
        status = f"{result.duration_s:.1f}s" if result.success else f"FAILED ({result.error})"
        print(status)
        results.append(result)
    print_scenario_results(results)
    return results
