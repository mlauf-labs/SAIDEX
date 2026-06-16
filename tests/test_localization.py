"""Tests for the internal localisation helpers behind source grounding.

These are pure functions: text normalisation for matching, and locale-aware
candidate rendering that turns a canonical extracted value into the surface
forms it may take in the source document (decimal/grouping separators, date
formats, month names).
"""

from __future__ import annotations

from datetime import date, datetime

from saidex._localization import normalize_for_match, render_candidates

# ---------------------------------------------------------------------------
# normalize_for_match
# ---------------------------------------------------------------------------


class TestNormalizeForMatch:
    def test_lowercases_and_collapses_whitespace(self) -> None:
        assert normalize_for_match("ACME   GmbH") == "acme gmbh"

    def test_strips_surrounding_whitespace(self) -> None:
        assert normalize_for_match("  hello  ") == "hello"

    def test_folds_diacritics(self) -> None:
        assert normalize_for_match("Müller") == normalize_for_match("Muller")

    def test_treats_nbsp_as_regular_space(self) -> None:
        # NBSP (U+00A0) and narrow NBSP (U+202F) used as thousands separators.
        assert normalize_for_match("1 234") == "1 234"
        assert normalize_for_match("1 234") == "1 234"

    def test_keeps_number_separators(self) -> None:
        # Must not strip '.'/',' — they carry meaning for number matching.
        assert normalize_for_match("1.234,50") == "1.234,50"


# ---------------------------------------------------------------------------
# render_candidates — numbers
# ---------------------------------------------------------------------------


class TestNumberCandidates:
    def test_german_float(self) -> None:
        cands = render_candidates(1234.5, "de")
        assert "1.234,5" in cands
        assert "1234,5" in cands
        assert "1.234,50" in cands  # currency-style 2 decimals

    def test_english_float(self) -> None:
        cands = render_candidates(1234.5, "en")
        assert "1,234.5" in cands
        assert "1234.5" in cands

    def test_country_code_hint_is_accepted(self) -> None:
        # A CountryCodeStr value such as "DE" must drive the same formatting.
        cands = render_candidates(1234.5, "DE")
        assert "1.234,50" in cands

    def test_german_integer_grouping(self) -> None:
        cands = render_candidates(1234567, "de")
        assert "1.234.567" in cands
        assert "1234567" in cands

    def test_negative_accounting_forms(self) -> None:
        cands = render_candidates(-1234.5, "de")
        assert "-1.234,50" in cands
        assert "(1.234,50)" in cands

    def test_no_hint_covers_common_conventions(self) -> None:
        cands = render_candidates(1234.5, None)
        assert "1,234.5" in cands  # en
        assert "1.234,5" in cands  # de


# ---------------------------------------------------------------------------
# render_candidates — dates
# ---------------------------------------------------------------------------


class TestDateCandidates:
    def test_german_date_forms(self) -> None:
        cands = render_candidates(date(2024, 4, 5), "de")
        assert "05.04.2024" in cands
        assert "5. April 2024" in cands
        assert "2024-04-05" in cands

    def test_english_date_forms(self) -> None:
        cands = render_candidates(date(2024, 4, 5), "en")
        assert "April 5, 2024" in cands
        assert "2024-04-05" in cands

    def test_iso_date_string_gets_localised_forms(self) -> None:
        # A field typed as IsoDateStr holds a plain ISO string, not a date object.
        cands = render_candidates("2024-04-05", "de")
        assert "05.04.2024" in cands
        assert "5. April 2024" in cands

    def test_datetime_value_renders_date_forms(self) -> None:
        cands = render_candidates(datetime(2024, 4, 5, 13, 30), "de")
        assert "05.04.2024" in cands


# ---------------------------------------------------------------------------
# render_candidates — strings & booleans
# ---------------------------------------------------------------------------


class TestStringCandidates:
    def test_plain_string_is_returned_verbatim(self) -> None:
        assert render_candidates("ACME GmbH", None) == ["ACME GmbH"]

    def test_bool_is_stringified(self) -> None:
        assert render_candidates(True, None) == ["True"]
