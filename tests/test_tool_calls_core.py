"""Tests for saidex._tool_calls, the framework-agnostic tool-call core (no langgraph needed)."""

from __future__ import annotations

import subprocess
import sys


def test_core_module_does_not_import_langgraph() -> None:
    code = "import sys, saidex._tool_calls; print('langgraph' in sys.modules)"
    completed = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert completed.stdout.strip() == "False"
