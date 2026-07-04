"""Tests for source-grounded field validation.

Covers the two configuration surfaces (the ``Grounded`` Annotated marker and the
``GroundedField`` helper), the ``Grounded`` check itself, and the engine walk
that runs all field checks over a validated instance. Integration with the
extraction entry points lives in ``test_grounding_integration.py``.
"""

from __future__ import annotations

from typing import Annotated

import pytest
from pydantic import BaseModel

from saidex import ExtractionContext, FieldCheck, Grounded, GroundedField, GroundedStr
from saidex.grounding import collect_field_check_issues

# ---------------------------------------------------------------------------
# Configuration surfaces — both land the marker in FieldInfo.metadata
# ---------------------------------------------------------------------------


def _checks(model: type[BaseModel], name: str) -> list[FieldCheck]:
    return [m for m in model.model_fields[name].metadata if isinstance(m, FieldCheck)]


class TestConfigurationSurfaces:
    def test_annotated_marker_and_field_helper_are_both_discoverable(self) -> None:
        class Doc(BaseModel):
            a: Annotated[str, Grounded()]
            b: str = GroundedField(description="b")
            c: str

        assert len(_checks(Doc, "a")) == 1
        assert len(_checks(Doc, "b")) == 1
        assert _checks(Doc, "c") == []

    def test_grounded_str_alias(self) -> None:
        class Doc(BaseModel):
            a: GroundedStr

        assert len(_checks(Doc, "a")) == 1

    def test_field_helper_keeps_field_metadata(self) -> None:
        class Doc(BaseModel):
            b: str = GroundedField(description="the vendor")

        assert Doc.model_fields["b"].description == "the vendor"

    def test_json_schema_still_builds(self) -> None:
        class Doc(BaseModel):
            a: Annotated[str, Grounded()]
            b: str = GroundedField(description="b")

        assert set(Doc.model_json_schema()["properties"]) == {"a", "b"}


# ---------------------------------------------------------------------------
# Grounded.check
# ---------------------------------------------------------------------------


def _ctx(text: str | None, model: BaseModel | None = None) -> ExtractionContext:
    return ExtractionContext(source_text=text, model=model, field_name="f", field_path="f")


class TestGroundedCheck:
    def test_passes_when_value_present(self) -> None:
        assert Grounded().check("ACME GmbH", _ctx("Invoice from ACME GmbH")) is None

    def test_fails_when_value_absent(self) -> None:
        msg = Grounded().check("Globex", _ctx("Invoice from ACME GmbH"))
        assert msg is not None
        assert "Globex" in msg

    def test_case_and_whitespace_insensitive(self) -> None:
        assert Grounded().check("acme   gmbh", _ctx("... ACME GmbH ...")) is None

    def test_skips_when_no_source_text(self) -> None:
        assert Grounded().check("anything", _ctx(None)) is None

    def test_skips_when_value_is_none(self) -> None:
        assert Grounded().check(None, _ctx("text")) is None

    def test_exact_mode_requires_verbatim(self) -> None:
        assert Grounded(mode="exact").check("acme", _ctx("ACME")) is not None
        assert Grounded(mode="exact").check("ACME", _ctx("ACME")) is None

    def test_static_locale_formats_number(self) -> None:
        assert Grounded(locale="de").check(1234.5, _ctx("Summe: 1.234,50 EUR")) is None

    def test_locale_field_drives_number_format(self) -> None:
        class Inv(BaseModel):
            country: str
            total: float

        inst = Inv(country="DE", total=1234.5)
        ctx = ExtractionContext(
            source_text="Summe: 1.234,50 EUR",
            model=inst,
            field_name="total",
            field_path="total",
        )
        assert Grounded(locale_field="country").check(1234.5, ctx) is None

    def test_number_word_matches_in_document_language(self) -> None:
        assert Grounded(locale="en").check(2, _ctx("There are two invoices")) is None
        assert Grounded(locale="de").check(2, _ctx("Es gibt zwei Rechnungen")) is None

    def test_boolean_matches_localised_yes_no(self) -> None:
        assert Grounded(locale="de").check(True, _ctx("Bezahlt: ja")) is None
        assert Grounded(locale="en").check(False, _ctx("Paid: no")) is None

    def test_boolean_matches_check_glyph(self) -> None:
        assert Grounded().check(True, _ctx("Paid: ✓")) is None


# ---------------------------------------------------------------------------
# Engine walk
# ---------------------------------------------------------------------------


class TestEngineWalk:
    def test_flags_hallucinated_value(self) -> None:
        class Doc(BaseModel):
            vendor: Annotated[str, Grounded()]

        issues, feedback = collect_field_check_issues(
            Doc(vendor="Globex"), "Invoice from ACME GmbH", "Doc"
        )
        assert feedback is not None
        assert len(issues) == 1
        assert issues[0].category == "grounding"
        assert issues[0].error_type == "grounding"
        assert issues[0].field_path == "vendor"
        assert issues[0].received == "Globex"

    def test_passes_present_value(self) -> None:
        class Doc(BaseModel):
            vendor: Annotated[str, Grounded()]

        issues, feedback = collect_field_check_issues(
            Doc(vendor="ACME GmbH"), "Invoice from ACME GmbH", "Doc"
        )
        assert issues == []
        assert feedback is None

    def test_no_source_text_skips(self) -> None:
        class Doc(BaseModel):
            vendor: Annotated[str, Grounded()]

        issues, feedback = collect_field_check_issues(Doc(vendor="Globex"), None, "Doc")
        assert issues == []
        assert feedback is None

    def test_ungrounded_fields_are_ignored(self) -> None:
        class Doc(BaseModel):
            vendor: Annotated[str, Grounded()]
            note: str

        issues, _ = collect_field_check_issues(
            Doc(vendor="ACME", note="anything hallucinated"), "from ACME", "Doc"
        )
        assert issues == []

    def test_recurses_into_list_items(self) -> None:
        class Line(BaseModel):
            name: Annotated[str, Grounded()]

        class Order(BaseModel):
            lines: list[Line]

        inst = Order(lines=[Line(name="ACME"), Line(name="Globex")])
        issues, feedback = collect_field_check_issues(inst, "We bought from ACME", "Order")
        assert len(issues) == 1
        assert issues[0].field_path == "lines -> 1 -> name"

    def test_recurses_into_dict_values(self) -> None:
        class Line(BaseModel):
            name: Annotated[str, Grounded()]

        class Order(BaseModel):
            by_code: dict[str, Line]

        inst = Order(by_code={"x": Line(name="Globex")})
        issues, _ = collect_field_check_issues(inst, "We bought from ACME", "Order")
        assert len(issues) == 1
        assert issues[0].field_path == "by_code -> x -> name"

    def test_long_received_value_is_truncated(self) -> None:
        class Doc(BaseModel):
            blob: Annotated[str, Grounded()]

        long_value = "Z" * 500
        issues, _ = collect_field_check_issues(Doc(blob=long_value), "nothing here", "Doc")
        assert issues[0].received is not None
        assert len(issues[0].received) < len(long_value)
        assert issues[0].received.endswith("…")


def test_base_field_check_is_abstract() -> None:
    with pytest.raises(NotImplementedError):
        FieldCheck().check("x", _ctx("text"))


# ---------------------------------------------------------------------------
# on_mismatch="flag" — record an issue but do not drive a retry
# ---------------------------------------------------------------------------


class TestFlagMode:
    def test_field_helper_forwards_on_mismatch(self) -> None:
        class Doc(BaseModel):
            note: str = GroundedField(on_mismatch="flag")

        check = _checks(Doc, "note")[0]
        assert isinstance(check, Grounded)
        assert check.on_mismatch == "flag"

    def test_flag_failure_is_recorded_without_feedback(self) -> None:
        class Doc(BaseModel):
            note: Annotated[str, Grounded(on_mismatch="flag")]

        issues, feedback = collect_field_check_issues(
            Doc(note="Globex"), "Invoice from ACME GmbH", "Doc"
        )
        # The issue is recorded for observability...
        assert len(issues) == 1
        assert issues[0].category == "grounding"
        assert issues[0].field_path == "note"
        # ...but there is no retry feedback, so the value is kept.
        assert feedback is None

    def test_mixed_retry_and_flag(self) -> None:
        class Doc(BaseModel):
            vendor: Annotated[str, Grounded()]  # retry (default)
            note: Annotated[str, Grounded(on_mismatch="flag")]

        issues, feedback = collect_field_check_issues(
            Doc(vendor="Globex", note="Initech"), "Invoice from ACME GmbH", "Doc"
        )
        # Both failures are recorded...
        assert {i.field_path for i in issues} == {"vendor", "note"}
        # ...but only the retry field drives the feedback.
        assert feedback is not None
        assert "vendor" in feedback
        assert "note" not in feedback

    def test_default_on_mismatch_is_retry(self) -> None:
        assert Grounded().on_mismatch == "retry"
