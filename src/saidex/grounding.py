"""Source-grounded, schema-declared field checks (anti-hallucination).

Mark a field as *grounded* and SAIDEX verifies, after Pydantic validation, that
the extracted value actually appears in the source text.  A value that cannot be
found is rejected and the existing retry loop asks the model to correct it.

Two equivalent surfaces make the grounded fields obvious when reading a schema::

    from typing import Annotated
    from pydantic import BaseModel
    from saidex import Grounded, GroundedField

    class Invoice(BaseModel):
        vendor: Annotated[str, Grounded()]                 # marker
        number: str = GroundedField(description="Inv. no.")  # field helper
        total: float = GroundedField(locale_field="country")  # locale-aware

Both surfaces attach the same :class:`Grounded` instance to the field's
``FieldInfo.metadata``, so the engine has a single discovery path.

The mechanism is generic: any :class:`FieldCheck` subclass can be attached the
same way (use :func:`field_check` for the helper form), so grounding is just the
first built-in check on a reusable pattern.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from typing import Annotated, Any

from pydantic import BaseModel, Field

from ._localization import normalize_for_match, render_candidates
from .models import FieldIssue

__all__ = [
    "FieldCheck",
    "ExtractionContext",
    "Grounded",
    "GroundedField",
    "GroundedStr",
    "field_check",
    "collect_field_check_issues",
]

# Maximum length of a captured ``received`` value before it is truncated.
_MAX_RECEIVED_LEN = 120

# Leaf value types a field check is applied to. Containers/models are recursed
# into rather than checked directly.
_SCALAR_TYPES = (str, int, float, Decimal, bool, date, datetime)


@dataclass(frozen=True)
class ExtractionContext:
    """Run context handed to every :class:`FieldCheck`.

    Attributes:
        source_text: The source text the run was performed on, or ``None`` when
            unavailable (e.g. manual model construction). Checks must treat a
            ``None`` source as "skip".
        model: The fully validated instance that *directly* contains the field,
            so a check can read sibling fields (e.g. a country code). ``None``
            when a check is run outside the engine walk.
        field_name: Name of the field being checked.
        field_path: Arrow path to the field, e.g. ``"items -> 0 -> vendor"``.
    """

    source_text: str | None
    model: BaseModel | None
    field_name: str
    field_path: str


@dataclass(frozen=True)
class FieldCheck:
    """Base class for schema-declared, source-aware field checks.

    Attach an instance to a field either via the Annotated marker form
    (``Annotated[T, MyCheck()]``) or the helper form
    (``f: T = field_check(MyCheck(), ...)``); both place it in
    ``FieldInfo.metadata`` where the engine discovers it.
    """

    def check(self, value: Any, ctx: ExtractionContext) -> str | None:
        """Validate *value* against the run *ctx*.

        Args:
            value: The validated field value.
            ctx: The run context (source text, containing instance, field path).

        Returns:
            ``None`` to accept, or an LLM-friendly error message to reject.
        """
        raise NotImplementedError


@dataclass(frozen=True)
class Grounded(FieldCheck):
    """Require a field's value to appear in the source text.

    Attributes:
        mode: ``"normalized"`` (default) compares case/whitespace/diacritics
            insensitively and tries locale-aware surface forms of the value;
            ``"exact"`` requires the verbatim ``str(value)`` to be present.
        locale: A fixed locale hint (country or language code, e.g. ``"de"``)
            used to render numbers/dates. Ignored in ``"exact"`` mode.
        locale_field: Name of a sibling field whose value supplies the locale
            hint at runtime (e.g. a ``CountryCodeStr`` field). Takes precedence
            over *locale* when it resolves to a non-empty value.
        on_mismatch: What a failed check does. ``"retry"`` (default) feeds the
            failure back to the model and consumes a retry; ``"flag"`` keeps the
            extracted value and only records a :class:`~saidex.models.FieldIssue`
            (advisory grounding, no retry).
    """

    mode: str = "normalized"
    locale: str | None = None
    locale_field: str | None = None
    on_mismatch: str = "retry"

    def check(self, value: Any, ctx: ExtractionContext) -> str | None:
        if value is None or not ctx.source_text:
            return None
        if self.mode == "exact":
            return None if str(value) in ctx.source_text else self._error(value, ctx)
        normalized_text = normalize_for_match(ctx.source_text)
        for candidate in render_candidates(value, self._resolve_hint(ctx)):
            if normalize_for_match(candidate) in normalized_text:
                return None
        return self._error(value, ctx)

    def _resolve_hint(self, ctx: ExtractionContext) -> str | None:
        if self.locale_field is not None and ctx.model is not None:
            sibling = getattr(ctx.model, self.locale_field, None)
            if sibling:
                return str(sibling)
        return self.locale

    def _error(self, value: Any, ctx: ExtractionContext) -> str:
        return (
            f"Field '{ctx.field_path}' has value {str(value)!r}, which does not appear "
            f"in the source text. Only use values that are present in the text."
        )


GroundedStr = Annotated[str, Grounded()]
"""Convenience alias for ``Annotated[str, Grounded()]``."""


def field_check(check: FieldCheck, **field_kwargs: Any) -> Any:
    """Build a Pydantic field that carries a :class:`FieldCheck`.

    The returned value is a :class:`~pydantic.fields.FieldInfo` (as produced by
    :func:`pydantic.Field`) with *check* appended to its ``metadata`` so the
    engine discovers it the same way it discovers an Annotated marker.

    Args:
        check: The field check to attach.
        **field_kwargs: Forwarded verbatim to :func:`pydantic.Field`
            (``description``, ``default``, constraints, …).

    Returns:
        A field descriptor usable as a model field default.
    """
    info = Field(**field_kwargs)
    info.metadata.append(check)
    return info


def GroundedField(  # noqa: N802 — mirrors pydantic.Field's CapWords spelling
    *,
    mode: str = "normalized",
    locale: str | None = None,
    locale_field: str | None = None,
    on_mismatch: str = "retry",
    **field_kwargs: Any,
) -> Any:
    """Field helper that marks a field as source-grounded.

    Thin wrapper around :func:`field_check` for :class:`Grounded`; see those for
    the parameters. Any other keyword (``description``, ``default``, …) is
    forwarded to :func:`pydantic.Field`.

    Returns:
        A field descriptor usable as a model field default::

            total: float = GroundedField(locale_field="country", description="…")
    """
    return field_check(
        Grounded(mode=mode, locale=locale, locale_field=locale_field, on_mismatch=on_mismatch),
        **field_kwargs,
    )


def collect_field_check_issues(
    instance: BaseModel,
    source_text: str | None,
    schema_name: str,
) -> tuple[list[FieldIssue], str | None]:
    """Run every attached :class:`FieldCheck` over a validated *instance*.

    Walks *instance* recursively (nested models, list/tuple/dict items),
    discovering checks in each field's ``FieldInfo.metadata`` and running them
    with an :class:`ExtractionContext`. Failures become :class:`FieldIssue`s
    (``category="grounding"``) plus a single LLM-facing feedback message.

    Args:
        instance: The instance that already passed Pydantic validation.
        source_text: The source text to ground against; when falsy, no checks
            run (returns ``([], None)``).
        schema_name: Schema name stamped onto each issue.

    Returns:
        ``([], None)`` when everything passes (or there is no source text);
        otherwise ``(issues, feedback)``. ``issues`` covers every failed check
        (both ``"retry"`` and ``"flag"``) and carries ``attempt=0`` for the
        caller to re-stamp. ``feedback`` is built only from ``"retry"`` failures
        and is ``None`` when all failures are advisory (``"flag"``), so the
        caller records the issues without consuming a retry.
    """
    if not source_text:
        return [], None

    failures: list[_Failure] = []
    _walk(instance, source_text, "", failures)
    if not failures:
        return [], None

    issues = [
        FieldIssue(
            schema_name=schema_name,
            field_path=failure.field_path,
            category="grounding",
            error_type="grounding",
            message=failure.message,
            attempt=0,
            received=failure.received,
        )
        for failure in failures
    ]
    retry_messages = [f.message for f in failures if f.on_mismatch != "flag"]
    feedback = (
        (
            "Some extracted values could not be found in the source text:\n\n"
            + "\n".join(f"  - {message}" for message in retry_messages)
            + "\n\nReturn a corrected response that uses only values present in the source text."
        )
        if retry_messages
        else None
    )
    return issues, feedback


@dataclass(frozen=True)
class _Failure:
    """One failed field check, tagged with how the engine should react."""

    field_path: str
    message: str
    received: str | None
    on_mismatch: str


def _walk(
    obj: Any,
    source_text: str,
    path: str,
    failures: list[_Failure],
) -> None:
    """Recursively run field checks, accumulating :class:`_Failure` records."""
    if isinstance(obj, BaseModel):
        for name, info in type(obj).model_fields.items():
            value = getattr(obj, name)
            child_path = f"{path} -> {name}" if path else name
            checks = [m for m in info.metadata if isinstance(m, FieldCheck)]
            if checks and isinstance(value, _SCALAR_TYPES):
                ctx = ExtractionContext(
                    source_text=source_text,
                    model=obj,
                    field_name=name,
                    field_path=child_path,
                )
                for check in checks:
                    message = check.check(value, ctx)
                    if message:
                        failures.append(
                            _Failure(
                                field_path=child_path,
                                message=message,
                                received=_truncate(value),
                                on_mismatch=getattr(check, "on_mismatch", "retry"),
                            )
                        )
            _walk(value, source_text, child_path, failures)
    elif isinstance(obj, (list, tuple)):
        for index, item in enumerate(obj):
            _walk(item, source_text, f"{path} -> {index}", failures)
    elif isinstance(obj, dict):
        for key, item in obj.items():
            _walk(item, source_text, f"{path} -> {key}", failures)


def _truncate(value: Any) -> str | None:
    """Render *value* as a short string for the issue's ``received`` field."""
    text = str(value)
    if len(text) > _MAX_RECEIVED_LEN:
        return text[: _MAX_RECEIVED_LEN - 1] + "…"
    return text
