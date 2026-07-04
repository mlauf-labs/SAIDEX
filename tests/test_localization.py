"""Tests for the internal localisation helpers behind source grounding.

These are pure functions: text normalisation for matching, and locale-aware
candidate rendering that turns a canonical extracted value into the surface
forms it may take in the source document (decimal/grouping separators, date
formats, month names).
"""

from __future__ import annotations

from datetime import date, datetime

from saidex._localization import fuzzy_contains, normalize_for_match, render_candidates

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
# render_candidates — number-words (small integers)
# ---------------------------------------------------------------------------


class TestNumberWordCandidates:
    def test_english_number_word(self) -> None:
        assert "two" in render_candidates(2, "en")

    def test_german_number_word(self) -> None:
        assert "zwei" in render_candidates(2, "de")

    def test_french_number_word(self) -> None:
        assert "deux" in render_candidates(2, "fr")

    def test_range_boundaries(self) -> None:
        assert "zero" in render_candidates(0, "en")
        assert "zwanzig" in render_candidates(20, "de")

    def test_out_of_range_integer_has_no_word(self) -> None:
        cands = render_candidates(21, "en")
        assert "21" in cands
        assert not any(cand.isalpha() for cand in cands)

    def test_non_whole_number_has_no_word(self) -> None:
        cands = render_candidates(2.5, "en")
        assert not any(cand.isalpha() for cand in cands)

    def test_bare_integer_string_gets_number_word(self) -> None:
        # A field typed as a plain string holding "2" still localises.
        assert "zwei" in render_candidates("2", "de")


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


# ---------------------------------------------------------------------------
# fuzzy_contains — approximate substring matching
# ---------------------------------------------------------------------------


class TestFuzzyContains:
    def test_exact_substring_matches_any_threshold(self) -> None:
        assert fuzzy_contains("acme", "invoice from acme gmbh", 0.99)

    def test_single_typo_within_threshold(self) -> None:
        # "corporatlon" is "corporation" with one substitution (i -> l).
        assert fuzzy_contains("corporation", "acme corporatlon inc", 0.85)

    def test_hyphenation_break_within_threshold(self) -> None:
        # A line-break hyphen inserts two chars ("- ") into the surface form.
        assert fuzzy_contains("corporation", "acme corpor- ation inc", 0.8)

    def test_dissimilar_does_not_match(self) -> None:
        assert not fuzzy_contains("globex", "invoice from acme gmbh", 0.8)

    def test_typo_below_threshold_is_rejected(self) -> None:
        # One edit in a 4-char needle is only ratio 0.75.
        assert not fuzzy_contains("acme", "invoice from acne gmbh", 0.9)

    def test_empty_needle_never_matches(self) -> None:
        assert not fuzzy_contains("", "anything at all", 0.1)


# ---------------------------------------------------------------------------
# render_candidates — booleans
# ---------------------------------------------------------------------------


class TestBooleanCandidates:
    def test_bool_still_includes_stringified_form(self) -> None:
        assert "True" in render_candidates(True, None)

    def test_true_localised_words(self) -> None:
        assert "ja" in render_candidates(True, "de")
        assert "yes" in render_candidates(True, "en")
        assert "oui" in render_candidates(True, "fr")

    def test_false_localised_words(self) -> None:
        assert "nein" in render_candidates(False, "de")
        assert "no" in render_candidates(False, "en")

    def test_true_check_glyphs(self) -> None:
        assert "✓" in render_candidates(True, "en")

    def test_false_has_no_true_word_or_glyph(self) -> None:
        cands = render_candidates(False, "en")
        assert "yes" not in cands
        assert "✓" not in cands

    def test_no_hint_covers_all_languages(self) -> None:
        cands = render_candidates(True, None)
        assert "True" in cands
        assert "yes" in cands
        assert "ja" in cands
