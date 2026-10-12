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

import typing
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from types import MappingProxyType
from typing import Annotated, Any, get_args, get_origin, get_type_hints

import typing_extensions
from pydantic import BaseModel, Field
from typing_extensions import is_typeddict

from ._localization import fuzzy_contains, normalize_for_match, render_candidates
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

# Shared read-only "no named sources" value.
_EMPTY_SOURCES: Mapping[str, str] = MappingProxyType({})

# ``type X = ...`` (3.12+) and ``typing_extensions.TypeAliasType(...)`` create
# instances of two different classes on 3.12/3.13, so both are checked.
_TYPE_ALIAS_TYPES: tuple[type[Any], ...] = tuple(
    {
        typing_extensions.TypeAliasType,
        getattr(typing, "TypeAliasType", typing_extensions.TypeAliasType),
    }
)


@dataclass(frozen=True)
class ExtractionContext:
    """Run context handed to every :class:`FieldCheck`.

    Attributes:
        source_text: The default source text of the run, or ``None`` when
            unavailable (e.g. manual model construction). Checks must treat a
            ``None`` source as "skip"; an empty string is real (empty) evidence.
        model: The fully validated instance that *directly* contains the field,
            so a check can read sibling fields (e.g. a country code). ``None``
            when a check is run outside the engine walk.
        field_name: Name of the field being checked.
        field_path: Arrow path to the field, e.g. ``"items -> 0 -> vendor"``.
        sources: Named source texts passed via ``grounding_source={...}``, for
            checks that ground a field in one named source. Empty by default.
    """

    source_text: str | None
    model: BaseModel | None
    field_name: str
    field_path: str
    sources: Mapping[str, str] = field(default_factory=lambda: _EMPTY_SOURCES)


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
            ``"exact"`` requires the verbatim ``str(value)`` to be present;
            ``"fuzzy"`` accepts an approximate match (see *threshold*), tolerating
            OCR noise and hyphenation across line breaks.
        locale: A fixed locale hint (country or language code, e.g. ``"de"``)
            used to render numbers/dates. Ignored in ``"exact"`` mode.
        locale_field: Name of a sibling field whose value supplies the locale
            hint at runtime (e.g. a ``CountryCodeStr`` field). Takes precedence
            over *locale* when it resolves to a non-empty value.
        on_mismatch: What a failed check does. ``"retry"`` (default) feeds the
            failure back to the model and consumes a retry; ``"flag"`` keeps the
            extracted value and only records a :class:`~saidex.models.FieldIssue`
            (advisory grounding, no retry).
        threshold: Minimum similarity in ``[0, 1]`` required in ``"fuzzy"`` mode
            (``1.0`` == exact). Ignored in the other modes. The ratio is
            length-sensitive, so short values need a lower threshold.
        source: Name of the source to check against, as passed via
            ``grounding_source={"name": text}``. ``None`` (default) checks
            against the run's default source text.
    """

    mode: str = "normalized"
    locale: str | None = None
    locale_field: str | None = None
    on_mismatch: str = "retry"
    threshold: float = 0.85
    source: str | None = None

    def check(self, value: Any, ctx: ExtractionContext) -> str | None:
        text = ctx.source_text if self.source is None else ctx.sources.get(self.source)
        if value is None or text is None:
            return None
        if self.mode == "exact":
            return None if str(value) in text else self._error(value, ctx)
        normalized_text = normalize_for_match(text)
        candidates = (
            normalize_for_match(c) for c in render_candidates(value, self._resolve_hint(ctx))
        )
        if self.mode == "fuzzy":
            matched = any(fuzzy_contains(c, normalized_text, self.threshold) for c in candidates)
        else:
            matched = any(c in normalized_text for c in candidates)
        return None if matched else self._error(value, ctx)

    def _resolve_hint(self, ctx: ExtractionContext) -> str | None:
        if self.locale_field is not None and ctx.model is not None:
            sibling = getattr(ctx.model, self.locale_field, None)
            if sibling:
                return str(sibling)
        return self.locale

    def _error(self, value: Any, ctx: ExtractionContext) -> str:
        where = "the source text" if self.source is None else f"the source text {self.source!r}"
        return (
            f"Field '{ctx.field_path}' has value {str(value)!r}, which does not appear "
            f"in {where}. Only use values that are present in the text."
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
    threshold: float = 0.85,
    source: str | None = None,
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
        Grounded(
            mode=mode,
            locale=locale,
            locale_field=locale_field,
            on_mismatch=on_mismatch,
            threshold=threshold,
            source=source,
        ),
        **field_kwargs,
    )


@dataclass(frozen=True)
class _GroundingSources:
    """The evidence one extraction run grounds its fields in.

    Attributes:
        default: Text for checks without a source name, or ``None`` to skip them.
        named: Read-only named sources for checks such as ``Grounded(source="mail")``.
    """

    default: str | None
    named: Mapping[str, str] = field(default_factory=lambda: _EMPTY_SOURCES)

    @property
    def is_empty(self) -> bool:
        """Whether no field check has anything to ground against."""
        return self.default is None and not self.named


_NO_GROUNDING = _GroundingSources(default=None)


def _resolve_grounding_sources(
    grounding_source: str | Mapping[str, str] | None, human_text: str
) -> _GroundingSources:
    """Resolve the caller's ``grounding_source`` into the run's grounding evidence.

    Args:
        grounding_source: ``None`` (use *human_text*), a string (the default
            source), or a mapping of named sources whose ``"default"`` key, if
            present, replaces *human_text* as the default source.
        human_text: The human-turn text of the messages; empty means "none".

    Returns:
        The resolved sources. An explicit source is kept even when empty.

    Raises:
        TypeError: If *grounding_source* is not ``None``, a ``str`` or a mapping
            of ``str`` names to ``str`` texts.
    """
    fallback = human_text or None
    if grounding_source is None:
        return _GroundingSources(default=fallback)
    if isinstance(grounding_source, str):
        return _GroundingSources(default=grounding_source)
    if not isinstance(grounding_source, Mapping):
        raise TypeError(
            "grounding_source must be a str, a Mapping[str, str] or None; "
            f"got {type(grounding_source).__name__}"
        )
    named: dict[str, str] = {}
    for name, text in grounding_source.items():
        if not isinstance(name, str) or not isinstance(text, str):
            raise TypeError(
                "grounding_source must map str names to str texts; "
                f"got {name!r}: {type(text).__name__}"
            )
        named[name] = text
    return _GroundingSources(default=named.get("default", fallback), named=MappingProxyType(named))


def _required_source_names(schema: type[BaseModel]) -> list[tuple[str, str]]:
    """List ``(field_path, source_name)`` for every field check that names a source.

    Walks *schema* statically — nested models, the type arguments of
    ``list``/``dict``/``tuple``/unions, ``type`` aliases and ``TypedDict`` fields —
    so every model the runtime walk can reach is covered. A check names a source
    through a ``str`` attribute ``source``.
    """
    found: list[tuple[str, str]] = []
    seen: set[type[BaseModel]] = set()
    seen_other: set[int] = set()

    def visit_model(model: type[BaseModel], prefix: str) -> None:
        if model in seen:
            return
        seen.add(model)
        for name, info in model.model_fields.items():
            path = f"{prefix} -> {name}" if prefix else name
            for check in info.metadata:
                source = getattr(check, "source", None)
                if isinstance(check, FieldCheck) and isinstance(source, str):
                    found.append((path, source))
            visit_type(info.annotation, path)

    def visit_type(annotation: Any, path: str) -> None:
        if isinstance(annotation, _TYPE_ALIAS_TYPES) or is_typeddict(annotation):
            # Aliases may be recursive (``type Tree = list[Tree] | Leaf``).
            if id(annotation) in seen_other:
                return
            seen_other.add(id(annotation))
            if not is_typeddict(annotation):
                visit_type(annotation.__value__, path)
                return
            try:
                hints = get_type_hints(annotation, include_extras=True)
            except (NameError, TypeError):
                # Unresolvable forward references: keep the raw annotations.
                hints = dict(getattr(annotation, "__annotations__", {}))
            for key, hint in hints.items():
                visit_type(hint, f"{path} -> {key}")
            return
        origin = get_origin(annotation)
        if origin is None and isinstance(annotation, type) and issubclass(annotation, BaseModel):
            visit_model(annotation, path)
            return
        if isinstance(origin, _TYPE_ALIAS_TYPES):
            visit_type(origin, path)
        for arg in get_args(annotation):
            visit_type(arg, path)

    visit_model(schema, "")
    return found


def _validate_source_names(schema: type[BaseModel], sources: _GroundingSources) -> None:
    """Reject a schema whose field checks name a source *sources* does not provide.

    Args:
        schema: The schema about to be extracted.
        sources: The resolved grounding sources of the run.

    Raises:
        ValueError: Naming the first field whose source is missing, so a named
            check never silently falls back to the whole prompt.
    """
    for path, name in _required_source_names(schema):
        if name not in sources.named:
            available = ", ".join(repr(n) for n in sources.named) or "none"
            raise ValueError(
                f"Field '{path}' of {schema.__name__} is grounded in source {name!r}, but "
                f"grounding_source provides no source with that name (available: {available}). "
                f"Pass grounding_source={{{name!r}: ...}}."
            )


def collect_field_check_issues(
    instance: BaseModel,
    source_text: str | None,
    schema_name: str,
    *,
    sources: Mapping[str, str] | None = None,
) -> tuple[list[FieldIssue], str | None]:
    """Run every attached :class:`FieldCheck` over a validated *instance*.

    Walks *instance* recursively (nested models, list/tuple/dict items),
    discovering checks in each field's ``FieldInfo.metadata`` and running them
    with an :class:`ExtractionContext`. Failures become :class:`FieldIssue`s
    (``category="grounding"``) plus a single LLM-facing feedback message.

    Args:
        instance: The instance that already passed Pydantic validation.
        source_text: The default source text to ground against; when falsy,
            un-named checks are skipped.
        schema_name: Schema name stamped onto each issue.
        sources: Named source texts for checks such as
            ``Grounded(source="mail")``. With neither *source_text* nor
            *sources*, no checks run (returns ``([], None)``).

    Returns:
        ``([], None)`` when everything passes (or there is no source text);
        otherwise ``(issues, feedback)``. ``issues`` covers every failed check
        (both ``"retry"`` and ``"flag"``) and carries ``attempt=0`` for the
        caller to re-stamp. ``feedback`` is built only from ``"retry"`` failures
        and is ``None`` when all failures are advisory (``"flag"``), so the
        caller records the issues without consuming a retry.
    """
    issues, _, feedback = _collect_field_checks(
        instance, source_text or None, schema_name, sources=sources
    )
    return issues, feedback


def _collect_field_checks(
    instance: BaseModel,
    source_text: str | None,
    schema_name: str,
    *,
    sources: Mapping[str, str] | None = None,
) -> tuple[list[FieldIssue], list[FieldIssue], str | None]:
    """Like :func:`collect_field_check_issues`, plus the rejecting subset.

    Args:
        instance: The instance that already passed Pydantic validation.
        source_text: The default source text; ``None`` skips un-named checks,
            while ``""`` is empty evidence that nothing can be grounded in.
        schema_name: Schema name stamped onto each issue.
        sources: Named source texts for checks that name their source.

    Returns:
        ``(issues, rejecting, feedback)`` where ``rejecting`` holds the issues of
        ``"retry"`` checks, i.e. the ones that reject the instance.
    """
    named = sources if sources is not None else _EMPTY_SOURCES
    if source_text is None and not named:
        return [], [], None

    failures: list[_Failure] = []
    _walk(instance, source_text, named, "", failures)
    if not failures:
        return [], [], None

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
    rejecting = [
        issue
        for issue, failure in zip(issues, failures, strict=True)
        if failure.on_mismatch != "flag"
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
    return issues, rejecting, feedback


@dataclass(frozen=True)
class _Failure:
    """One failed field check, tagged with how the engine should react."""

    field_path: str
    message: str
    received: str | None
    on_mismatch: str


def _walk(
    obj: Any,
    source_text: str | None,
    sources: Mapping[str, str],
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
                    sources=sources,
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
            _walk(value, source_text, sources, child_path, failures)
    elif isinstance(obj, (list, tuple)):
        for index, item in enumerate(obj):
            _walk(item, source_text, sources, f"{path} -> {index}", failures)
    elif isinstance(obj, dict):
        for key, item in obj.items():
            _walk(item, source_text, sources, f"{path} -> {key}", failures)


def _truncate(value: Any) -> str | None:
    """Render *value* as a short string for the issue's ``received`` field."""
    text = str(value)
    if len(text) > _MAX_RECEIVED_LEN:
        return text[: _MAX_RECEIVED_LEN - 1] + "…"
    return text
