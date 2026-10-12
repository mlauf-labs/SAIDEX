"""Tests for source-grounded field validation.

Covers the two configuration surfaces (the ``Grounded`` Annotated marker and the
``GroundedField`` helper), the ``Grounded`` check itself, and the engine walk
that runs all field checks over a validated instance. Integration with the
extraction entry points lives in ``test_grounding_integration.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Annotated, Any

import pytest
from pydantic import BaseModel

from saidex import ExtractionContext, FieldCheck, Grounded, GroundedField, GroundedStr
from saidex.grounding import (
    _collect_field_checks,
    _resolve_grounding_sources,
    _validate_source_names,
    collect_field_check_issues,
    field_check,
)


class _TreeNode(BaseModel):
    name: Annotated[str, Grounded(source="tree")]
    children: list[_TreeNode] = []


_TreeNode.model_rebuild()

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

    def test_fuzzy_mode_tolerates_ocr_typo(self) -> None:
        check = Grounded(mode="fuzzy", threshold=0.8)
        assert check.check("ACME Corporation", _ctx("Issued by ACME Corporatlon Inc.")) is None

    def test_fuzzy_mode_tolerates_hyphenation(self) -> None:
        check = Grounded(mode="fuzzy", threshold=0.8)
        assert check.check("ACME Corporation", _ctx("by ACME Corpor-\nation Inc.")) is None

    def test_fuzzy_mode_rejects_dissimilar_value(self) -> None:
        check = Grounded(mode="fuzzy", threshold=0.9)
        assert check.check("Globex", _ctx("Invoice from ACME GmbH")) is not None

    def test_default_mode_and_threshold(self) -> None:
        g = Grounded()
        assert g.mode == "normalized"
        assert g.threshold == 0.85

    def test_field_helper_forwards_threshold(self) -> None:
        class Doc(BaseModel):
            v: str = GroundedField(mode="fuzzy", threshold=0.7)

        check = _checks(Doc, "v")[0]
        assert isinstance(check, Grounded)
        assert check.mode == "fuzzy"
        assert check.threshold == 0.7


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


def test_collect_field_checks_separates_rejecting_issues() -> None:
    class Mixed(BaseModel):
        strict_vendor: GroundedStr
        soft_vendor: str = field_check(Grounded(on_mismatch="flag"))

    issues, rejecting, feedback = _collect_field_checks(
        Mixed(strict_vendor="Globex", soft_vendor="Initech"), "We bought from ACME", "Mixed"
    )

    assert [i.field_path for i in issues] == ["strict_vendor", "soft_vendor"]
    assert [i.field_path for i in rejecting] == ["strict_vendor"]
    assert feedback is not None
    assert (issues, feedback) == collect_field_check_issues(
        Mixed(strict_vendor="Globex", soft_vendor="Initech"), "We bought from ACME", "Mixed"
    )


# ---------------------------------------------------------------------------
# Named sources and explicit (possibly empty) sources — issue #91
# ---------------------------------------------------------------------------


def _named_ctx(sources: dict[str, str], default: str | None = None) -> ExtractionContext:
    return ExtractionContext(
        source_text=default, model=None, field_name="f", field_path="f", sources=sources
    )


class TestNamedSources:
    def test_context_sources_default_to_empty(self) -> None:
        assert dict(_ctx("text").sources) == {}

    def test_named_check_uses_its_source(self) -> None:
        ctx = _named_ctx({"mail": "order 32008"}, default="orders 32008 and 28870")
        assert Grounded(source="mail").check("32008", ctx) is None
        assert Grounded(source="mail").check("28870", ctx) is not None

    def test_unnamed_check_uses_default_source(self) -> None:
        ctx = _named_ctx({"mail": "order 32008"}, default="orders 32008 and 28870")
        assert Grounded().check("28870", ctx) is None

    def test_named_source_missing_from_context_skips(self) -> None:
        assert Grounded(source="mail").check("28870", _named_ctx({}, default="x")) is None

    def test_error_names_the_source(self) -> None:
        msg = Grounded(source="mail").check("28870", _named_ctx({"mail": "order 32008"}))
        assert msg is not None
        assert "the source text 'mail'" in msg

    def test_unnamed_error_message_unchanged(self) -> None:
        msg = Grounded().check("Globex", _ctx("ACME"))
        assert msg == (
            "Field 'f' has value 'Globex', which does not appear in the source text. "
            "Only use values that are present in the text."
        )

    def test_positional_mode_still_works(self) -> None:
        check = Grounded("exact")
        assert check.mode == "exact"
        assert check.source is None

    def test_field_helper_forwards_source(self) -> None:
        class Doc(BaseModel):
            v: str = GroundedField(source="mail")

        check = _checks(Doc, "v")[0]
        assert isinstance(check, Grounded)
        assert check.source == "mail"


class TestEmptySource:
    @pytest.mark.parametrize("mode", ["normalized", "exact", "fuzzy"])
    def test_empty_default_source_rejects_value_in_every_mode(self, mode: str) -> None:
        assert Grounded(mode=mode).check("32008", _ctx("")) is not None

    def test_empty_named_source_rejects_value(self) -> None:
        assert Grounded(source="mail").check("32008", _named_ctx({"mail": ""})) is not None

    def test_empty_source_still_accepts_none(self) -> None:
        assert Grounded().check(None, _ctx("")) is None

    def test_public_helper_still_skips_empty_source_text(self) -> None:
        class Doc(BaseModel):
            vendor: Annotated[str, Grounded()]

        assert collect_field_check_issues(Doc(vendor="Globex"), "", "Doc") == ([], None)


class TestEngineWalkWithSources:
    def test_named_sources_reach_nested_checks(self) -> None:
        class Line(BaseModel):
            order: Annotated[str, Grounded(source="mail")]

        class Doc(BaseModel):
            lines: list[Line]

        issues, feedback = collect_field_check_issues(
            Doc(lines=[Line(order="32008"), Line(order="28870")]),
            None,
            "Doc",
            sources={"mail": "order 32008"},
        )
        assert [i.field_path for i in issues] == ["lines -> 1 -> order"]
        assert feedback is not None

    def test_named_sources_run_without_a_default_source(self) -> None:
        class Doc(BaseModel):
            order: Annotated[str, Grounded(source="mail")]
            vendor: Annotated[str, Grounded()]

        issues, _ = collect_field_check_issues(
            Doc(order="28870", vendor="anything"), None, "Doc", sources={"mail": "order 32008"}
        )
        # The named check runs; the un-named one has no default source and skips.
        assert [i.field_path for i in issues] == ["order"]

    def test_custom_check_sees_sources(self) -> None:
        seen: list[dict[str, str]] = []

        class Spy(FieldCheck):
            def check(self, value: object, ctx: ExtractionContext) -> str | None:
                seen.append(dict(ctx.sources))
                return None

        class Doc(BaseModel):
            v: Annotated[str, Spy()]

        collect_field_check_issues(Doc(v="x"), "text", "Doc", sources={"mail": "m"})
        assert seen == [{"mail": "m"}]


class TestResolveGroundingSources:
    def test_none_uses_human_text(self) -> None:
        sources = _resolve_grounding_sources(None, "human")
        assert sources.default == "human"
        assert dict(sources.named) == {}

    def test_empty_human_text_resolves_to_none(self) -> None:
        sources = _resolve_grounding_sources(None, "")
        assert sources.default is None
        assert sources.is_empty

    def test_string_replaces_human_text(self) -> None:
        assert _resolve_grounding_sources("mail", "human").default == "mail"

    def test_explicit_empty_string_is_kept(self) -> None:
        sources = _resolve_grounding_sources("", "human")
        assert sources.default == ""
        assert not sources.is_empty

    def test_mapping_default_key_wins(self) -> None:
        sources = _resolve_grounding_sources({"default": "d", "mail": "m"}, "human")
        assert sources.default == "d"
        assert dict(sources.named) == {"default": "d", "mail": "m"}

    def test_mapping_without_default_falls_back_to_human_text(self) -> None:
        sources = _resolve_grounding_sources({"mail": "m"}, "human")
        assert sources.default == "human"
        assert dict(sources.named) == {"mail": "m"}

    def test_accepts_any_mapping(self) -> None:
        sources = _resolve_grounding_sources(MappingProxyType({"mail": "m"}), "")
        assert dict(sources.named) == {"mail": "m"}

    def test_named_sources_are_a_read_only_copy(self) -> None:
        given = {"mail": "m"}
        sources = _resolve_grounding_sources(given, "")
        given["mail"] = "changed"
        assert sources.named["mail"] == "m"
        with pytest.raises(TypeError):
            sources.named["mail"] = "x"  # type: ignore[index]

    @pytest.mark.parametrize("given", [123, ["mail"], b"mail"])
    def test_rejects_wrong_type(self, given: Any) -> None:
        with pytest.raises(TypeError, match="grounding_source"):
            _resolve_grounding_sources(given, "")

    @pytest.mark.parametrize("given", [{"mail": 1}, {1: "m"}, {"mail": None}])
    def test_rejects_non_str_names_or_texts(self, given: Any) -> None:
        with pytest.raises(TypeError, match="grounding_source"):
            _resolve_grounding_sources(given, "")


class TestValidateSourceNames:
    def test_known_names_pass(self) -> None:
        class Doc(BaseModel):
            order: Annotated[str, Grounded(source="mail")]

        _validate_source_names(Doc, _resolve_grounding_sources({"mail": "m"}, ""))

    def test_unknown_name_raises_with_path_and_available(self) -> None:
        class Doc(BaseModel):
            order: Annotated[str, Grounded(source="mail")]

        with pytest.raises(ValueError, match=r"'order' of Doc .*'mail'.*available: 'catalog'"):
            _validate_source_names(Doc, _resolve_grounding_sources({"catalog": "c"}, ""))

    @pytest.mark.parametrize("given", [None, "plain string"])
    def test_named_check_needs_a_mapping(self, given: str | None) -> None:
        class Doc(BaseModel):
            order: Annotated[str, Grounded(source="mail")]

        with pytest.raises(ValueError, match="available: none"):
            _validate_source_names(Doc, _resolve_grounding_sources(given, "human"))

    def test_unnamed_checks_need_nothing(self) -> None:
        class Doc(BaseModel):
            vendor: Annotated[str, Grounded()]

        _validate_source_names(Doc, _resolve_grounding_sources(None, ""))

    def test_unused_names_are_allowed(self) -> None:
        class Doc(BaseModel):
            vendor: str

        _validate_source_names(Doc, _resolve_grounding_sources({"x": "y"}, ""))

    def test_finds_names_inside_list_items(self) -> None:
        class Line(BaseModel):
            sku: Annotated[str, Grounded(source="catalog")]

        class Doc(BaseModel):
            lines: list[Line]

        with pytest.raises(ValueError, match=r"'lines -> sku'"):
            _validate_source_names(Doc, _resolve_grounding_sources({"mail": "m"}, ""))

    def test_finds_names_inside_dict_values(self) -> None:
        class Line(BaseModel):
            sku: Annotated[str, Grounded(source="catalog")]

        class Doc(BaseModel):
            by_key: dict[str, Line]

        with pytest.raises(ValueError, match=r"'by_key -> sku'"):
            _validate_source_names(Doc, _resolve_grounding_sources({}, ""))

    def test_finds_names_inside_optional_models(self) -> None:
        class Line(BaseModel):
            sku: Annotated[str, Grounded(source="catalog")]

        class Doc(BaseModel):
            maybe: Line | None = None

        with pytest.raises(ValueError, match=r"'maybe -> sku'"):
            _validate_source_names(Doc, _resolve_grounding_sources({}, ""))

    def test_recursive_model_terminates(self) -> None:
        _validate_source_names(_TreeNode, _resolve_grounding_sources({"tree": "t"}, ""))
        with pytest.raises(ValueError, match="'tree'"):
            _validate_source_names(_TreeNode, _resolve_grounding_sources({}, ""))

    def test_custom_check_with_source_attribute_is_validated(self) -> None:
        @dataclass(frozen=True)
        class InSource(FieldCheck):
            source: str = "catalog"

            def check(self, value: Any, ctx: ExtractionContext) -> str | None:
                return None

        class Doc(BaseModel):
            sku: Annotated[str, InSource()]

        with pytest.raises(ValueError, match="'catalog'"):
            _validate_source_names(Doc, _resolve_grounding_sources({}, ""))
