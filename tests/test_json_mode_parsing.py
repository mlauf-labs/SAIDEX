"""JSON-mode comment stripping (issue #101)."""

from __future__ import annotations

import pytest

from saidex.extractor import (
    _extract_json_from_content,
    _strip_json_comments,
)

# --- _strip_json_comments ---------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('{"a": 1} // trailing\n{"b": 2}', '{"a": 1} \n{"b": 2}'),
        ('{"a": /* years */ 36}', '{"a":   36}'),
        ('{"a": 1} /* never closed', '{"a": 1}  '),
        ('{"a": 1} // no newline', '{"a": 1} '),
        ('{"a": "http://x"}', '{"a": "http://x"}'),
        ('{"a": "a /* b */ c"}', '{"a": "a /* b */ c"}'),
        ("{'a': 'x // y /* z */'}", "{'a': 'x // y /* z */'}"),
        (r'{"a": "q\" // still string"}', r'{"a": "q\" // still string"}'),
        (r"{'a': 'it\'s // ok'}", r"{'a': 'it\'s // ok'}"),
        ("[1/*x*/2]", "[1 2]"),
    ],
)
def test_strip_json_comments(text: str, expected: str) -> None:
    assert _strip_json_comments(text) == expected


# --- _extract_json_from_content ----------------------------------------------


def test_extract_block_comment_keeps_value() -> None:
    assert _extract_json_from_content('{"name": "Al", "age": /* years */ 36}') == {
        "name": "Al",
        "age": 36,
    }


def test_extract_valid_json_with_comment_markers_in_string_unchanged() -> None:
    assert _extract_json_from_content('{"url": "http://x", "k": "//"}') == {
        "url": "http://x",
        "k": "//",
    }


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (
            """Here's the record: {"name": "O'Brien", "note": "see /* x */", "age": 36}""",
            {"name": "O'Brien", "note": "see /* x */", "age": 36},
        ),
        (
            """Here's the record: {"name": "O'Brien", "url": "https://x.com/a", "age": 36,}""",
            {"name": "O'Brien", "url": "https://x.com/a", "age": 36},
        ),
        (
            """Source: https://example.com {"name": "Ada", "age": 36}""",
            {"name": "Ada", "age": 36},
        ),
        (
            """Don't forget: {"name": "x", 'u': 'http://y', // c
}""",
            {"name": "x", "u": "http://y"},
        ),
    ],
)
def test_prose_before_json_does_not_corrupt_values(raw: str, expected: dict[str, object]) -> None:
    assert _extract_json_from_content(raw) == expected
