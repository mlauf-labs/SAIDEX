# JSON-repair golden corpus

Each case is a pair: `<name>.txt` (raw model content, byte-exact) and either
`<name>.json` (expected parsed value) or `<name>.error` (name of the expected
exception type). `tests/test_json_repair_golden.py` asserts
`_extract_json_from_content(raw)` matches. `.gitattributes` marks this folder
`-text` so git never rewrites line endings.

Goldens were generated with **json-repair 0.64.0**. The test is skipped (not failed)
when an older json-repair is installed (the CI "lowest" job).

## Regenerating

```bash
uv run python tests/golden/json_repair/regenerate.py
git diff tests/golden/json_repair
```

A changed golden means a json-repair upgrade altered how broken LLM output is
parsed. Review every diff deliberately before committing; do not regenerate blindly.

## Surprising current behaviour (kept as golden)

- `two_objects`: two concatenated objects become a list of both, not "the first wins".
- `truncated_string`: the cut-off string is kept as-is (`"Mathematician and wri"`).
- `number_formats`: the leading `+` is dropped and `1.5e3` becomes the float `1500.0`.
- `unicode_escapes_broken`: json-repair drops the trailing `\n` escape inside a string
  (the valid twin `unicode_escapes` keeps it).
- `missing_leading_brace`: the missing `{` is added (json-repair 0.64 feature).
- `nested_errors`: the unclosed final object is closed silently.
