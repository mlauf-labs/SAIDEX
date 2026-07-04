# Fuzzy / edit-distance matching for grounded fields (GH #45)

> Follow-up to #43. Add an optional fuzzy matching mode to grounding so the
> source-text check tolerates OCR noise, hyphenation across line breaks, and
> minor surface differences. The default stays the exact normalised-substring
> match from #43.

## Problem

Source grounding rejects a value unless one of its normalised surface forms is
found as a **substring** of the normalised source text. Real documents — scanned
PDFs, OCR output, text with hyphenation at line breaks — introduce small
character-level noise that breaks an otherwise-correct match ("Corporation"
scanned as "Corporatlon", "Corpor-\nation").

## Decision (from brainstorming)

The issue sketch proposed a new `match="fuzzy"` parameter, but `Grounded`
**already** has a `mode` switch (`"normalized"` / `"exact"`). Fuzzy is a third
matching strategy, so it extends `mode` rather than adding a parallel,
overlapping parameter.

```python
vendor: Annotated[str, Grounded(mode="fuzzy", threshold=0.85)]
```

## Scope

- `mode` gains a `"fuzzy"` value; `Grounded` / `GroundedField` gain a
  `threshold: float = 0.85` (ignored unless `mode="fuzzy"`).
- Algorithm is **in-house, stdlib only** — no new runtime dependency
  (`pydantic` / `langchain-core` / `json-repair` stay the only runtime deps).
- Default behaviour (`"normalized"`, `"exact"`) is unchanged.

Out of scope: token-overlap matching, per-candidate thresholds, configurable
distance algorithms.

## Design

### API (`src/saidex/grounding.py`)

`Grounded` dataclass gains `threshold: float = 0.85`. `mode` docstring documents
the new `"fuzzy"` value. `GroundedField` forwards `threshold`.

`Grounded.check` branches on `mode`:

- `"exact"` — verbatim `str(value)` substring (unchanged).
- `"normalized"` (default) — any normalised candidate is a substring (unchanged).
- `"fuzzy"` — any normalised candidate `fuzzy_contains` the normalised source
  text at `>= threshold`.

### Fuzzy primitive (`src/saidex/_localization.py`)

Lives beside `normalize_for_match` (matching is this module's job). Exported as
`fuzzy_contains(needle, haystack, threshold) -> bool`.

Uses **Sellers' approximate substring matching** (a Levenshtein DP whose first
row is all zeros, so a match may begin at any position). The minimum of the
final row is the edit distance between `needle` and its best-matching substring
of `haystack`. Similarity ratio = `1 - best_distance / len(needle)`; match when
`ratio >= threshold`.

- Fast path: exact substring ⇒ `True` (ratio 1.0).
- Empty needle ⇒ `False` (never match on nothing).
- Complexity `O(len(needle) * len(haystack))` per candidate, `O(len(haystack))`
  space (rolling rows). `check` stops at the first matching candidate.

Insertions/deletions are handled naturally, so hyphenation (`corpor- ation`) and
inserted OCR characters cost one edit each rather than breaking the match.

### Threshold note

Ratio is length-sensitive: one edit in a 4-char word drops the ratio to 0.75,
while one edit in a 16-char string only reaches 0.94. `0.85` is a balanced
default; users tune it per field. Documented in `docs/source-grounding.md`.

## Tests

- `tests/test_localization.py`: `fuzzy_contains` — exact substring at high
  threshold, single-typo within threshold, dissimilar rejected, empty needle
  rejected.
- `tests/test_grounding.py`: `Grounded(mode="fuzzy", ...)` tolerates an OCR typo
  and a hyphenation break, rejects a dissimilar value below threshold; default
  `threshold` and `mode` unchanged; `GroundedField` forwards `threshold`.

## Docs

`docs/source-grounding.md`: add a "Fuzzy matching" subsection (when to use it,
the `mode="fuzzy"` / `threshold` API, the length-sensitivity note) and remove
fuzzy from the "Out of scope" list.
