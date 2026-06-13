"""Vision Benchmark — extract invoice data from a *rendered image* of the text.

This reuses the three invoice texts (and their expected values) from the
``invoice_*`` benchmarks. Instead of feeding the model the raw text, the text is
rendered to a PNG document image (see ``_image.py``) and passed as vision input,
so the model must perform OCR-style reading before extraction.

Vision models on Ollama frequently do not support tool calling, so these
scenarios use ``ExtractionMode.JSON`` by default.

Run standalone:
    python benchmarks/vision_invoice.py
"""

from __future__ import annotations

import asyncio

from saidex import ExtractionMode

from ._base import BenchmarkScenario
from ._config import VISION_MODELS
from .invoice_01 import SCENARIO as TEXT_INVOICE_01
from .invoice_02 import SCENARIO as TEXT_INVOICE_02
from .invoice_03 import SCENARIO as TEXT_INVOICE_03

_VISION_SYSTEM_PROMPT = (
    "You are a precise invoice-extraction assistant. The user provides an image "
    "of an invoice document. Read all visible text in the image and extract the "
    "requested fields exactly as they appear. Use null for fields that are not "
    "present in the image."
)

_IMAGE_PROMPT = "Extract the invoice fields from this document image."


def _to_vision_scenario(text_scenario: BenchmarkScenario) -> BenchmarkScenario:
    """Build a vision variant of an existing text invoice scenario."""
    return BenchmarkScenario(
        name=f"{text_scenario.name} [Vision]",
        description=f"Image-based OCR extraction — {text_scenario.description}",
        schema=text_scenario.schema,
        text=text_scenario.text,
        mode=ExtractionMode.JSON,
        system_prompt=_VISION_SYSTEM_PROMPT,
        expected=dict(text_scenario.expected),
        vision=True,
        image_prompt=_IMAGE_PROMPT,
    )


SCENARIO_01 = _to_vision_scenario(TEXT_INVOICE_01)
SCENARIO_02 = _to_vision_scenario(TEXT_INVOICE_02)
SCENARIO_03 = _to_vision_scenario(TEXT_INVOICE_03)

SCENARIOS = [SCENARIO_01, SCENARIO_02, SCENARIO_03]


async def main() -> None:
    from ._base import print_scenario_results, run_scenario

    print("Vision Benchmark: Invoice extraction from rendered images\n")
    for scenario in SCENARIOS:
        results = []
        for model in VISION_MODELS:
            print(f"  Running [{scenario.name}] on {model}…", end=" ", flush=True)
            result = await run_scenario(scenario, model)
            status = f"{result.duration_s:.1f}s" if result.success else f"FAILED ({result.error})"
            print(status)
            results.append(result)
        print_scenario_results(results)


if __name__ == "__main__":
    # Allow running directly: `python benchmarks/vision_invoice.py`
    import sys
    from pathlib import Path

    if __package__ is None:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    asyncio.run(main())
