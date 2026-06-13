"""Query the Ollama server and report which configured models support vision.

Ollama exposes each model's capabilities via the native ``/api/show`` endpoint.
A model can be used in the "vision" benchmark group when its capabilities list
contains ``"vision"``.

Usage:
    python -m benchmarks.list_vision_models

Copy the printed VISION_MODELS list into ``benchmarks/_config.py``.
"""

from __future__ import annotations

import json
import urllib.request

from ._config import MODELS, OLLAMA_BASE_URL

_NATIVE_BASE = OLLAMA_BASE_URL.rsplit("/v1", 1)[0]


def model_capabilities(model: str) -> list[str]:
    """Return the Ollama capability tags for *model* (e.g. ['completion', 'vision'])."""
    payload = json.dumps({"model": model}).encode()
    request = urllib.request.Request(
        f"{_NATIVE_BASE}/api/show",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        info = json.load(response)
    return list(info.get("capabilities", []))


def main() -> None:
    print(f"Server: {_NATIVE_BASE}\n")
    vision: list[str] = []
    for model in MODELS:
        try:
            caps = model_capabilities(model)
            error = ""
        except Exception as exc:  # noqa: BLE001
            caps, error = [], str(exc)
        has_vision = "vision" in caps
        if has_vision:
            vision.append(model)
        mark = "YES" if has_vision else "no "
        detail = f"ERROR: {error}" if error else ", ".join(caps)
        print(f"  {model:28} vision={mark}  [{detail}]")

    print("\nVISION_MODELS: list[str] = [")
    for model in vision:
        print(f'    "{model}",')
    print("]")


if __name__ == "__main__":
    main()
