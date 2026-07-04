"""Locale-aware rendering and normalisation for source grounding.

Source grounding checks whether an extracted value can be found in the input
text.  The trouble is that the *canonical* value an LLM returns rarely matches
its *surface form* in the document, and the difference is locale dependent:

* ``1234.5`` may appear as ``1.234,50`` (de), ``1,234.50`` (en) or ``1 234,50`` (fr);
* a date stored as ``2024-04-05`` may read ``05.04.2024`` or ``5. April 2024``;
* a boolean ``True`` may read ``ja`` / ``yes`` / a ``✓`` glyph, and a small
  integer ``2`` may be spelled ``two`` / ``zwei`` / ``deux``.

:func:`render_candidates` turns a value plus an optional locale hint (a country
or language code) into the list of surface forms worth searching for.
:func:`normalize_for_match` canonicalises both the candidate and the source text
before the substring test (case, whitespace, NBSP, diacritics).

This module is internal; the public surface is :mod:`saidex.grounding`.
"""

from __future__ import annotations

import contextlib
import re
import unicodedata
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from typing import Any

__all__ = ["fuzzy_contains", "normalize_for_match", "render_candidates"]

_WHITESPACE_RE = re.compile(r"\s+")
_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_NUMERIC_RE = re.compile(r"^[+-]?\d+(?:\.\d+)?$")

# Decimal/grouping separators per locale key (lower-cased country *or* language
# code). The fallbacks below cover anything not listed.
_SEPARATORS: dict[str, tuple[str, str]] = {
    # decimal '.', grouping ','
    "en": (".", ","),
    "us": (".", ","),
    "gb": (".", ","),
    "au": (".", ","),
    "ca": (".", ","),
    "nz": (".", ","),
    "ie": (".", ","),
    "in": (".", ","),
    "jp": (".", ","),
    # decimal ',', grouping '.'
    "de": (",", "."),
    "at": (",", "."),
    "nl": (",", "."),
    "it": (",", "."),
    "es": (",", "."),
    "pt": (",", "."),
    "da": (",", "."),
    "tr": (",", "."),
    # decimal ',', grouping space
    "fr": (",", " "),
    "be": (",", " "),
    # decimal '.', grouping apostrophe
    "ch": (".", "'"),
}

# Common separator conventions, always appended as fallbacks so a wrong or
# missing locale hint never causes a false rejection.
_FALLBACK_SEPARATORS: list[tuple[str, str]] = [
    (".", ","),
    (",", "."),
    (",", " "),
    (".", "'"),
]

_MONTHS: dict[str, dict[str, list[str]]] = {
    "en": {
        "full": [
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ],
        "abbr": [
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ],
    },
    "de": {
        "full": [
            "Januar",
            "Februar",
            "März",
            "April",
            "Mai",
            "Juni",
            "Juli",
            "August",
            "September",
            "Oktober",
            "November",
            "Dezember",
        ],
        "abbr": [
            "Jan",
            "Feb",
            "Mär",
            "Apr",
            "Mai",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Okt",
            "Nov",
            "Dez",
        ],
    },
    "fr": {
        "full": [
            "janvier",
            "février",
            "mars",
            "avril",
            "mai",
            "juin",
            "juillet",
            "août",
            "septembre",
            "octobre",
            "novembre",
            "décembre",
        ],
        "abbr": [
            "janv",
            "févr",
            "mars",
            "avr",
            "mai",
            "juin",
            "juil",
            "août",
            "sept",
            "oct",
            "nov",
            "déc",
        ],
    },
}

_FALLBACK_LANGUAGES = ["en", "de", "fr"]

# Spelled-out cardinals 0..20 per language (index == value). Kept small on
# purpose: compound forms (twenty-one / einundzwanzig) are out of scope.
_NUMBER_WORDS: dict[str, list[str]] = {
    "en": [
        "zero",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
        "eleven",
        "twelve",
        "thirteen",
        "fourteen",
        "fifteen",
        "sixteen",
        "seventeen",
        "eighteen",
        "nineteen",
        "twenty",
    ],
    "de": [
        "null",
        "eins",
        "zwei",
        "drei",
        "vier",
        "fünf",
        "sechs",
        "sieben",
        "acht",
        "neun",
        "zehn",
        "elf",
        "zwölf",
        "dreizehn",
        "vierzehn",
        "fünfzehn",
        "sechzehn",
        "siebzehn",
        "achtzehn",
        "neunzehn",
        "zwanzig",
    ],
    "fr": [
        "zéro",
        "un",
        "deux",
        "trois",
        "quatre",
        "cinq",
        "six",
        "sept",
        "huit",
        "neuf",
        "dix",
        "onze",
        "douze",
        "treize",
        "quatorze",
        "quinze",
        "seize",
        "dix-sept",
        "dix-huit",
        "dix-neuf",
        "vingt",
    ],
}

# Yes/no surface words per language, indexed by the boolean value.
_BOOLEAN_WORDS: dict[str, dict[bool, list[str]]] = {
    "en": {True: ["yes"], False: ["no"]},
    "de": {True: ["ja"], False: ["nein"]},
    "fr": {True: ["oui"], False: ["non"]},
}

# Language-independent "ticked" marks that stand in for a True value in forms.
# Only True has glyphs — a False box is the absence of a mark, which cannot be
# matched. These are short, so they broaden matching (see module note).
_BOOLEAN_TRUE_GLYPHS: list[str] = ["✓", "✔", "☑", "x", "X"]


def normalize_for_match(text: str) -> str:
    """Canonicalise *text* for a locale-tolerant substring comparison.

    Applies, in order: Unicode NFKD (so NBSP/thin spaces become ordinary
    spaces), removal of combining marks (so ``Müller`` matches ``Muller``),
    case folding, and whitespace collapsing.  Number separators (``.``/``,``)
    are deliberately preserved.

    Args:
        text: The string to normalise.

    Returns:
        The normalised string.
    """
    decomposed = unicodedata.normalize("NFKD", text)
    without_marks = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    collapsed = _WHITESPACE_RE.sub(" ", without_marks)
    return collapsed.strip().casefold()


def fuzzy_contains(needle: str, haystack: str, threshold: float) -> bool:
    """Report whether *needle* appears in *haystack* allowing minor differences.

    Tolerant of the character-level noise typical of scanned/OCR'd text and of
    hyphenation across line breaks: an inserted, deleted or substituted character
    costs one edit rather than breaking the match. Both arguments are expected to
    be normalised already (see :func:`normalize_for_match`).

    The similarity is ``1 - d / len(needle)`` where ``d`` is the edit distance
    between *needle* and its best-matching substring of *haystack* (Sellers'
    approximate substring matching). A match requires similarity ``>= threshold``.

    Args:
        needle: The (normalised) surface form to look for.
        haystack: The (normalised) source text to search.
        threshold: Minimum similarity in ``[0, 1]``; ``1.0`` means exact.

    Returns:
        ``True`` when the best-matching substring is similar enough, else
        ``False``. An empty *needle* never matches.
    """
    if not needle:
        return False
    if needle in haystack:
        return True
    distance = _min_substring_distance(needle, haystack)
    similarity = 1.0 - distance / len(needle)
    return similarity >= threshold


def render_candidates(value: Any, locale_hint: str | None) -> list[str]:
    """Render *value* into the surface forms it may take in the source text.

    Args:
        value: The extracted, canonical value (str, number, date, …).
        locale_hint: A country or language code (e.g. ``"DE"``, ``"de"``,
            ``"en-US"``) used to pick number/date formats.  When ``None`` a
            broad set covering the common conventions is produced.

    Returns:
        A de-duplicated list of candidate strings, always including the plain
        ``str(value)``.
    """
    if isinstance(value, bool):
        return _boolean_candidates(value, locale_hint)
    if isinstance(value, (int, float, Decimal)):
        return _number_candidates(value, locale_hint)
    if isinstance(value, datetime):
        return _dedup([value.isoformat(), *_date_candidates(value.date(), locale_hint)])
    if isinstance(value, date):
        return _date_candidates(value, locale_hint)
    if isinstance(value, str):
        return _string_candidates(value, locale_hint)
    return [str(value)]


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _min_substring_distance(needle: str, haystack: str) -> int:
    """Edit distance between *needle* and its best-matching substring of *haystack*.

    Sellers' variant of Levenshtein: the first row is all zeros so a match may
    begin at any position, and the answer is the minimum of the final row.
    ``O(len(needle) * len(haystack))`` time, ``O(len(haystack))`` space.
    """
    previous = [0] * (len(haystack) + 1)
    for i in range(1, len(needle) + 1):
        current = [i] + [0] * len(haystack)
        needle_char = needle[i - 1]
        for j in range(1, len(haystack) + 1):
            substitution = previous[j - 1] + (0 if needle_char == haystack[j - 1] else 1)
            current[j] = min(previous[j] + 1, current[j - 1] + 1, substitution)
        previous = current
    return min(previous)


def _norm_hint(hint: str | None) -> str | None:
    """Reduce a locale hint to a lower-cased two-letter key (``de-DE`` → ``de``)."""
    if not hint:
        return None
    cleaned = hint.strip().lower()
    return cleaned[:2] if len(cleaned) >= 2 else None


def _separators_for(hint: str | None) -> list[tuple[str, str]]:
    """Return the (decimal, grouping) pairs to try for *hint*, with fallbacks."""
    pairs: list[tuple[str, str]] = []
    key = _norm_hint(hint)
    if key and key in _SEPARATORS:
        pairs.append(_SEPARATORS[key])
    for pair in _FALLBACK_SEPARATORS:
        if pair not in pairs:
            pairs.append(pair)
    return pairs


def _languages_for(hint: str | None) -> list[str]:
    """Return the month-name languages to try for *hint*, with fallbacks."""
    langs: list[str] = []
    key = _norm_hint(hint)
    if key and key in _MONTHS:
        langs.append(key)
    for lang in _FALLBACK_LANGUAGES:
        if lang not in langs:
            langs.append(lang)
    return langs


def _group(integer_part: str, grouping: str) -> str:
    """Insert *grouping* every three digits from the right."""
    if not grouping or len(integer_part) <= 3:
        return integer_part
    chunks: list[str] = []
    while len(integer_part) > 3:
        chunks.insert(0, integer_part[-3:])
        integer_part = integer_part[:-3]
    chunks.insert(0, integer_part)
    return grouping.join(chunks)


def _number_candidates(value: Any, hint: str | None) -> list[str]:
    """Render a number into grouped/ungrouped, multi-decimal, signed variants."""
    try:
        dec = Decimal(str(value))
    except InvalidOperation:
        return [str(value)]

    negative = dec < 0
    magnitude = abs(dec)
    exponent = dec.as_tuple().exponent
    natural = -exponent if isinstance(exponent, int) and exponent < 0 else 0

    frac_lengths: list[int] = []
    for length in (natural, 2, 0):
        if length not in frac_lengths:
            frac_lengths.append(length)

    out: list[str] = [str(value)]
    for decimal_sep, grouping_sep in _separators_for(hint):
        for frac in frac_lengths:
            formatted = f"{magnitude:.{frac}f}"
            int_str, _, frac_str = formatted.partition(".")
            for grouped in (False, True):
                grouped_int = _group(int_str, grouping_sep) if grouped else int_str
                body = grouped_int + (decimal_sep + frac_str if frac_str else "")
                if negative:
                    out += [f"-{body}", f"({body})", f"{body}-"]
                else:
                    out.append(body)
    out += _number_word_candidates(value, hint)
    return _dedup(out)


def _number_word_candidates(value: Any, hint: str | None) -> list[str]:
    """Spell out whole numbers 0..20 in the resolved languages.

    Returns ``[]`` for non-integers and values outside ``0..20`` so callers can
    unconditionally extend their candidate list.
    """
    try:
        dec = Decimal(str(value))
    except InvalidOperation:
        return []
    if dec != dec.to_integral_value() or not (0 <= dec <= 20):
        return []
    index = int(dec)
    out: list[str] = []
    for lang in _languages_for(hint):
        words = _NUMBER_WORDS.get(lang)
        if words:
            out.append(words[index])
    return out


def _boolean_candidates(value: bool, hint: str | None) -> list[str]:
    """Render a boolean into its stringified, localised and glyph surface forms."""
    out: list[str] = [str(value)]
    for lang in _languages_for(hint):
        words = _BOOLEAN_WORDS.get(lang)
        if words:
            out += words[value]
    if value:
        out += _BOOLEAN_TRUE_GLYPHS
    return _dedup(out)


def _date_candidates(value: date, hint: str | None) -> list[str]:
    """Render a date into numeric and month-name surface forms."""
    day, month, year = value.day, value.month, value.year
    out: list[str] = [
        value.isoformat(),  # 2024-04-05
        f"{day:02d}.{month:02d}.{year}",
        f"{day}.{month}.{year}",
        f"{day:02d}/{month:02d}/{year}",
        f"{day}/{month}/{year}",
        f"{month:02d}/{day:02d}/{year}",
        f"{month}/{day}/{year}",
        f"{day:02d}-{month:02d}-{year}",
    ]
    for lang in _languages_for(hint):
        months = _MONTHS.get(lang)
        if not months:
            continue
        full = months["full"][month - 1]
        abbr = months["abbr"][month - 1]
        out += [
            f"{day}. {full} {year}",  # 5. April 2024 (de)
            f"{day} {full} {year}",  # 5 April 2024
            f"{full} {day}, {year}",  # April 5, 2024 (en)
            f"{day}. {abbr} {year}",
            f"{day} {abbr} {year}",
        ]
    return _dedup(out)


def _string_candidates(value: str, hint: str | None) -> list[str]:
    """A string is itself a candidate; also expand ISO dates and bare numbers."""
    out: list[str] = [value]
    stripped = value.strip()
    if _ISO_DATE_RE.match(stripped):
        with contextlib.suppress(ValueError):
            out += _date_candidates(date.fromisoformat(stripped), hint)
    elif _NUMERIC_RE.match(stripped):
        out += _number_candidates(stripped, hint)
    return _dedup(out)


def _dedup(items: list[str]) -> list[str]:
    """De-duplicate while preserving first-seen order."""
    seen: dict[str, None] = {}
    for item in items:
        seen.setdefault(item, None)
    return list(seen)
