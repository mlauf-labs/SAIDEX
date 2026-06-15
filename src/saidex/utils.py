"""Utility helpers for safe Pydantic model instantiation."""

from __future__ import annotations

from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from .models import FieldIssue

MODEL_T = TypeVar("MODEL_T", bound=BaseModel)

# Maximum length of a captured ``received`` value before it is truncated.
_MAX_RECEIVED_LEN = 120


def create_instance_safe(
    schema: type[MODEL_T],
    **data: Any,
) -> tuple[MODEL_T | None, str | None]:
    """Instantiate *schema* from *data* without raising on validation errors.

    When validation succeeds the second tuple element is ``None``.  When it
    fails the first element is ``None`` and the second contains a
    human-readable, structured error description that can be sent back to an
    LLM as correction guidance.

    Args:
        schema: The Pydantic model class to instantiate.
        **data: Keyword arguments forwarded to the model constructor.

    Returns:
        ``(instance, None)`` on success, ``(None, error_text)`` on failure.
    """
    instance, _, error_text = create_instance_with_issues(schema, **data)
    return instance, error_text


def create_instance_with_issues(
    schema: type[MODEL_T],
    **data: Any,
) -> tuple[MODEL_T | None, list[FieldIssue], str | None]:
    """Like :func:`create_instance_safe` but also return structured issues.

    The third element is the same LLM-facing error string as
    :func:`create_instance_safe`; the second is a structured
    :class:`~saidex.models.FieldIssue` list describing the same failures, ready
    for aggregation.  ``attempt`` is left at ``0`` here — the caller (which owns
    the retry loop) sets the real attempt index.

    Args:
        schema: The Pydantic model class to instantiate.
        **data: Keyword arguments forwarded to the model constructor.

    Returns:
        ``(instance, [], None)`` on success;
        ``(None, issues, error_text)`` on validation failure.
    """
    try:
        return schema(**data), [], None
    except ValidationError as exc:
        issues = _collect_field_issues(schema, exc)
        return None, issues, _format_validation_error(schema, exc)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _categorize_error(error_type: str) -> str:
    """Map a raw Pydantic error type to a coarse, human-facing category."""
    if error_type == "missing":
        return "missing"
    if error_type in ("enum", "literal_error") or "enum" in error_type:
        return "enum"
    if "type" in error_type or "parsing" in error_type:
        return "type"
    if error_type in {
        "string_too_short",
        "string_too_long",
        "value_error",
        "greater_than",
        "less_than",
        "greater_than_equal",
        "less_than_equal",
    }:
        return "value"
    return "other"


def _truncate_received(value: Any) -> str | None:
    """Render an error's input value as a short string, or ``None`` if absent."""
    if value is None or value == "N/A":
        return None
    text = str(value)
    if len(text) > _MAX_RECEIVED_LEN:
        return text[: _MAX_RECEIVED_LEN - 1] + "…"
    return text


def _collect_field_issues(schema: type[BaseModel], exc: ValidationError) -> list[FieldIssue]:
    """Turn a :class:`~pydantic.ValidationError` into structured field issues."""
    issues: list[FieldIssue] = []
    for error in exc.errors():
        field_path = " -> ".join(str(loc) for loc in error["loc"])
        error_type = error["type"]
        received = None if error_type == "missing" else _truncate_received(error.get("input"))
        issues.append(
            FieldIssue(
                schema_name=schema.__name__,
                field_path=field_path,
                category=_categorize_error(error_type),
                error_type=error_type,
                message=error["msg"],
                attempt=0,
                received=received,
            )
        )
    return issues


def _format_validation_error(schema: type[BaseModel], exc: ValidationError) -> str:
    """Render a :class:`~pydantic.ValidationError` as structured guidance text."""
    missing: list[str] = []
    type_errs: list[str] = []
    enum_errs: list[str] = []
    value_errs: list[str] = []
    other_errs: list[str] = []

    for error in exc.errors():
        field_path = " -> ".join(str(loc) for loc in error["loc"])
        error_type = error["type"]
        error_msg = error["msg"]
        error_input = error.get("input", "N/A")

        if error_type == "missing":
            missing.append(f"  - '{field_path}': required field is missing")

        elif error_type in ("enum", "literal_error") or (
            "enum" in error_type and error.get("ctx", {}).get("expected")
        ):
            ctx = error.get("ctx", {})
            allowed = ctx.get("expected", "")
            if isinstance(allowed, (list, tuple)):
                allowed_str = ", ".join(f"'{v}'" for v in allowed)
            else:
                allowed_str = str(allowed) if allowed else "see schema"
            enum_errs.append(
                f"  - '{field_path}': invalid enum value.\n"
                f"    Received: '{error_input}'\n"
                f"    Allowed:  [{allowed_str}]"
            )

        elif (
            "type" in error_type
            or "parsing" in error_type
            or error_type
            in {
                "int_parsing",
                "float_parsing",
                "bool_parsing",
                "int_type",
                "float_type",
                "bool_type",
                "str_type",
                "int_parsing_below",
                "int_parsing_above",
            }
        ):
            expected = _infer_expected_type(error_type, error_msg, error.get("ctx", {}))
            type_errs.append(
                f"  - '{field_path}': wrong type."
                f" Expected: {expected},"
                f" Received: {type(error_input).__name__} = '{error_input}'"
            )

        elif error_type in {
            "string_too_short",
            "string_too_long",
            "value_error",
            "greater_than",
            "less_than",
        }:
            ctx = error.get("ctx", {})
            ctx_str = ", ".join(f"{k}={v}" for k, v in ctx.items()) if ctx else "none"
            value_errs.append(f"  - '{field_path}': {error_msg} (constraints: {ctx_str})")

        else:
            other_errs.append(f"  - '{field_path}': [{error_type}] {error_msg}")

    lines: list[str] = [f"Validation failed for '{schema.__name__}':"]

    if missing:
        lines += ["\nMISSING REQUIRED FIELDS:"] + missing
    if type_errs:
        lines += ["\nTYPE ERRORS:"] + type_errs
    if value_errs:
        lines += ["\nVALIDATION ERRORS:"] + value_errs
    if enum_errs:
        lines += ["\nENUM VALUE ERRORS:"] + enum_errs
    if other_errs:
        lines += ["\nOTHER ERRORS:"] + other_errs

    instructions: list[str] = []
    if missing:
        instructions.append("Add all missing required fields.")
    if type_errs:
        instructions.append("Correct the data types of the specified fields.")
    if enum_errs:
        instructions.append("Use only the allowed enum values listed above.")
    if value_errs:
        instructions.append("Ensure values comply with the validation constraints.")
    if other_errs:
        instructions.append("Fix the other validation errors listed above.")

    if instructions:
        lines.append("\nREQUIRED CHANGES:")
        for i, instr in enumerate(instructions, 1):
            lines.append(f"  {i}. {instr}")

    required_fields = [name for name, f in schema.model_fields.items() if f.is_required()]
    optional_fields = [name for name, f in schema.model_fields.items() if not f.is_required()]

    lines.append(f"\nSCHEMA '{schema.__name__}':")
    lines.append(f"  Required ({len(required_fields)}): {', '.join(required_fields) or 'none'}")
    if optional_fields:
        preview = optional_fields[:5]
        suffix = "…" if len(optional_fields) > 5 else ""
        lines.append(f"  Optional ({len(optional_fields)}): {', '.join(preview)}{suffix}")

    return "\n".join(lines)


def _infer_expected_type(error_type: str, error_msg: str, ctx: dict[str, Any]) -> str:
    """Derive a human-readable expected type string from Pydantic error metadata."""
    if ctx.get("expected"):
        return str(ctx["expected"])
    mapping = {
        "int": "integer",
        "float": "float",
        "bool": "boolean",
        "str": "string",
    }
    for key, label in mapping.items():
        if key in error_type or key in error_msg.lower():
            return label
    if "list" in error_msg.lower() or "array" in error_msg.lower():
        return "list/array"
    if "dict" in error_msg.lower() or "object" in error_msg.lower():
        return "dict/object"
    return error_msg.lower() or "unknown"
