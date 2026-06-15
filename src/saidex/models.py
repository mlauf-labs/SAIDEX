"""Return types and configuration enums for structured output extraction."""

from dataclasses import dataclass
from enum import Enum
from typing import Any


class ExtractionMode(str, Enum):
    """How the LLM is asked to produce structured output.

    Attributes:
        TOOL_CALLING: Bind the schema as an OpenAI-style tool and force the
            model to call it (``llm.bind_tools(..., tool_choice=...)``).
            Requires a model that supports tool/function calling.  This is the
            default and most reliable mode.
        JSON: Inject the schema's JSON Schema into the prompt and ask the model
            to reply with a single raw JSON object.  The response content is
            parsed directly.  Works with any chat model — including those that
            do **not** support tool calling — at the cost of slightly less
            structural guarantees.
    """

    TOOL_CALLING = "tool_calling"
    JSON = "json"


@dataclass(frozen=True)
class ExtractDataStats:
    """Statistics from a ``extract_data`` call.

    Tracks how many retries were needed and whether the fallback model was used.
    Backward-compatible with ``int`` so existing code that treated the old return
    value as a retry count continues to work.

    Attributes:
        primary_retries: Number of retries against the primary model.
        fallback_retries: Number of retries against the fallback model.
        fallback_used: Whether the fallback model was invoked at all.
        item_count: Number of items returned by a batch (``extract_data_list``)
            call.  Always ``0`` for single-item extraction.
    """

    primary_retries: int = 0
    fallback_retries: int = 0
    fallback_used: bool = False
    item_count: int = 0

    @property
    def total_retries(self) -> int:
        """Total retry count across primary and fallback."""
        return self.primary_retries + self.fallback_retries

    # ------------------------------------------------------------------
    # Backward-compatibility shims so code that does ``int(stats)`` or
    # ``if stats > 0`` keeps working without modification.
    # ------------------------------------------------------------------

    def __int__(self) -> int:
        return self.total_retries

    def __str__(self) -> str:
        return str(self.total_retries)

    def __lt__(self, other: object) -> bool:
        if isinstance(other, int):
            return self.total_retries < other
        return NotImplemented

    def __le__(self, other: object) -> bool:
        if isinstance(other, int):
            return self.total_retries <= other
        return NotImplemented

    def __gt__(self, other: object) -> bool:
        if isinstance(other, int):
            return self.total_retries > other
        return NotImplemented

    def __ge__(self, other: object) -> bool:
        if isinstance(other, int):
            return self.total_retries >= other
        return NotImplemented


@dataclass(frozen=True)
class ExtractorRunStats:
    """Statistics from an :func:`extract_data_with_tools` / :func:`run_extractor_agent` call.

    Attributes:
        iterations: Total LLM invocations performed in the agent loop.
        tool_calls: Total helper-tool calls executed (not counting final-answer calls).
        validation_retries: Number of times the final-answer schema failed validation
            and the loop continued.
        fallback_used: Whether the fallback model was invoked.
    """

    iterations: int = 0
    tool_calls: int = 0
    validation_retries: int = 0
    fallback_used: bool = False

    def __add__(self, other: Any) -> "ExtractorRunStats":
        """Merge two stats instances (used when combining primary + fallback)."""
        if not isinstance(other, ExtractorRunStats):
            return NotImplemented
        return ExtractorRunStats(
            iterations=self.iterations + other.iterations,
            tool_calls=self.tool_calls + other.tool_calls,
            validation_retries=self.validation_retries + other.validation_retries,
            fallback_used=self.fallback_used or other.fallback_used,
        )
