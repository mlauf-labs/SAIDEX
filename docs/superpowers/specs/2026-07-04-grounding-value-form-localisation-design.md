# Value-form localisation for grounding (GH #46)

> Follow-up to #43. Extend locale-aware candidate rendering to non-numeric
> surface forms: number-words and boolean yes/no.

## Problem

Source grounding accepts a value when one of its rendered *surface forms* is
found in the source text. `render_candidates` in `src/saidex/_localization.py`
today covers numbers and dates, but:

- **Booleans** render only to `["True"]` / `["False"]` — a document that writes
  "ja", "yes", "oui" or a check glyph is never matched.
- **Integers** render only to numeric forms — a document that spells out "two",
  "zwei" or "deux" is never matched.

Both are language-dependent and must reuse the existing `locale` / `locale_field`
hint mechanism.

## Scope

- Number-words for **integers 0–20** in **en / de / fr**.
- Boolean words for **True** (yes/ja/oui) and **False** (no/nein/non) in the same
  languages, plus language-independent True check glyphs (`✓ ✔ ☑ x X`).
- Tests for at least **English and German** (acceptance criteria).
- No public API change: `Grounded` / `GroundedField` / `locale` / `locale_field`
  are untouched. Only `_localization.py` grows more candidates.

Out of scope: integers > 20, compound number-words (twenty-one / einundzwanzig),
ordinals, other languages.

## Design

All changes live in `src/saidex/_localization.py`.

### Data tables

```python
_NUMBER_WORDS: dict[str, list[str]]   # lang -> [word for 0..20]
_BOOLEAN_WORDS: dict[str, dict[bool, list[str]]]  # lang -> {True: [...], False: [...]}
_BOOLEAN_TRUE_GLYPHS: list[str]       # language-independent: ✓ ✔ ☑ x X
```

Languages follow the existing `_FALLBACK_LANGUAGES = ["en", "de", "fr"]` and the
`_languages_for(hint)` resolver, so a `de` hint tries German first then falls
back, exactly like month names do today.

### Rendering

- `render_candidates` bool branch: return `[str(value)]` **plus** the boolean
  words for every language from `_languages_for(hint)`, plus (for `True`) the
  check glyphs. De-duplicated via the existing `_dedup`.
- New `_number_word_candidates(value, hint)`: returns `[]` unless the value is a
  whole number in `0..20`; otherwise the number-words for the resolved
  languages. Wired into `_number_candidates` (covers `int` values and bare
  integer strings routed through `_string_candidates`).

### Matching semantics (known trade-off)

Grounding matches by normalised **substring** (`normalize_for_match(candidate)
in normalized_text`). Short lexical candidates ("x", "ja", "no") therefore also
match inside unrelated words ("Jacke", "nowhere"). This is consistent with the
rest of the module (e.g. "2" matches inside "2024") and the feature is opt-in
per field, so we keep substring matching and **document the false-positive
caveat** in `docs/source-grounding.md`. Word-boundary matching is explicitly not
introduced here to avoid diverging from the module-wide matching contract.

## Tests

New cases in the localisation / grounding test suite:

- `2` (int) matches "two" (en hint) and "zwei" (de hint); does not match an
  unrelated document.
- `True` matches "ja" (de), "yes" (en) and a "✓" glyph; `False` matches "nein" /
  "no".
- Locale hint selects the right language first while fallbacks still match.

## Docs

`docs/source-grounding.md`: extend the locale section with a short note that
number-words and boolean yes/no (incl. check glyphs) are now recognised, and the
substring caveat.
