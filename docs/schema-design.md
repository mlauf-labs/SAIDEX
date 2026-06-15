# Schema Design

→ [Documentation index](index.md)

A well-designed schema is the most important factor for reliable LLM
extraction. The Pydantic model is converted to a JSON Schema that the LLM
must fill in — ambiguous types or missing descriptions are the most common
cause of retries and extraction failures.

---

## Supported field types

### Primitive types

```python
from pydantic import BaseModel

class Example(BaseModel):
    title:     str    # plain text
    count:     int    # whole number
    price:     float  # decimal number
    is_active: bool   # true / false
```

### Optional / nullable fields

Use `T | None` with a `None` default to tell the LLM it may return `null`
when a value is not present in the source text:

```python
class Example(BaseModel):
    nickname:   str | None   = None
    birth_year: int | None   = None
    score:      float | None = None
```

Without the `None` default, the field is still required.  With it, a missing
value is silently populated as `None` and no retry is needed.

### Lists

```python
class Example(BaseModel):
    tags:       list[str]       # zero-or-more strings
    scores:     list[float]     # zero-or-more numbers
    line_items: list[LineItem]  # zero-or-more nested models
```

Always specify the element type.  A bare `list` gives the LLM no guidance
on what to put in the array.

### Enums — constrained choices

Use `str` enums (or `Literal`) whenever the set of allowed values is fixed and
known.  The LLM receives the full list of allowed values in the schema, which
drastically reduces hallucinated values.  When the LLM returns an invalid
value the library sends the full enum list in the correction message.

```python
from enum import Enum

class Priority(str, Enum):
    LOW      = "low"
    MEDIUM   = "medium"
    HIGH     = "high"
    CRITICAL = "critical"

class Ticket(BaseModel):
    priority: Priority
    # LLM can only return: "low" | "medium" | "high" | "critical"
```

For short, fixed value sets `Literal` is more concise:

```python
from typing import Literal

class Measurement(BaseModel):
    unit:  Literal["kg", "g", "lb", "oz"]
    value: float
```

### Nested models

Break complex objects into nested `BaseModel` classes:

```python
class Address(BaseModel):
    street:  str
    city:    str
    country: str = Field(description="ISO 3166-1 alpha-2 country code, e.g. DE")
    zip:     str | None = None

class Person(BaseModel):
    name:    str
    address: Address
```

Avoid nesting more than 3 levels deep — deep hierarchies increase the
chance of structural errors.

### Typed dicts

Simple key-value maps work well when both key and value types are specified:

```python
class Example(BaseModel):
    translations:       dict[str, str]          # language → text
    scores_by_category: dict[str, float]        # category → score
    aliases:            dict[str, list[str]]    # name → list of aliases
```

### Constrained fields (`Field`)

Use Pydantic's `Field` to enforce numeric and length boundaries.  Constraint
violations are reported field-by-field in the error feedback, so the LLM can
correct each one.

```python
from pydantic import Field

class Product(BaseModel):
    price:        float = Field(ge=0,        description="Price in EUR, must be ≥ 0")
    discount_pct: int   = Field(ge=0, le=100, description="Discount 0–100 %")
    name:         str   = Field(min_length=1, max_length=120)
    rating:       float = Field(ge=1.0, le=5.0, description="Star rating 1.0–5.0")
```

---

## ⚠️ Types to avoid

### `object`, `dict`, `Any` — never use these as field types

These are the most common cause of extraction failures.

```python
from typing import Any

# WRONG — the LLM has no idea what structure to produce.
class Bad(BaseModel):
    data:    object           # ❌ completely untyped
    extra:   dict             # ❌ no key/value types
    value:   Any              # ❌ bypasses all validation
    payload: dict[str, Any]   # ❌ the Any makes the value unverifiable
```

**Why this breaks:**

When the LLM tool-calling schema contains `object` or `{}`, the model has
no guidance on what JSON structure to produce.  It may return a string, a
number, a nested object, or nothing at all — and Pydantic will accept all
of them because the type has no constraints.  The library cannot detect or
correct this.

```python
# GOOD — replace every untyped field with a concrete type
class Good(BaseModel):
    data:    MySubModel           # ✅ define a real Pydantic model
    extra:   dict[str, str]       # ✅ typed key and value
    count:   int                  # ✅ concrete primitive
    payload: dict[str, list[str]] # ✅ fully specified
```

### `datetime`, `date`, `time` — use `str` instead

Python's `datetime` objects serialise to ISO 8601 strings in JSON Schema, but
many LLMs produce inconsistent date formats (`"Jan 5"`, `"05/01/2026"`,
`"2026-01-05T00:00:00"`).  Use `str` with an explicit format description and,
if needed, a `@field_validator` that normalises the input:

```python
# RISKY — serialisation surprises with some models
class Event(BaseModel):
    start_date: date    # ❌ model may return "January 5" or "05.01.2026"

# SAFE — the LLM sees the exact expected format in the description
class Event(BaseModel):
    start_date: str | None = Field(
        default=None,
        description="Event start date as YYYY-MM-DD, e.g. '2026-01-05', or null",
    )   # ✅
```

For a ready-made type that enforces this format in one annotation, use
[`IsoDateStr`](built-in-types.md#isodatestr) (`from saidex import IsoDateStr`).
To instead *auto-convert* other notations, see
[Validators — normalising date formats](validators.md#example--normalising-date-formats),
which turns `DD.MM.YYYY` → `YYYY-MM-DD`.

### Complex `Union` types

Avoid `str | int | list[float]` — the LLM often picks the wrong branch.
Use separate, clearly named fields instead:

```python
# RISKY
class Bad(BaseModel):
    value: str | int | float    # ❌ which type should the LLM choose?

# GOOD — split into explicit fields
class Good(BaseModel):
    value_text:   str | None   = Field(default=None, description="Text value, if textual")
    value_number: float | None = Field(default=None, description="Numeric value, if numeric")
```

---

## Field descriptions are essential

The `description` string in `Field(description=...)` is passed directly to
the LLM as part of the JSON Schema.  It is the **primary signal** the model
uses to understand what to put in each field.

**Without a description the LLM guesses from the field name alone.**
Field names are short and often ambiguous.

### Bad vs. good descriptions

```python
# BAD — no description; the LLM guesses
class Invoice(BaseModel):
    amount:   float   # EUR? USD? incl. tax? subtotal?
    date:     str     # issue date? due date? what format?
    ref:      str     # invoice number? customer ref? PO number?
    status:   str     # anything could go here

# GOOD — every field is unambiguous
class Invoice(BaseModel):
    amount: float = Field(
        description=(
            "Total invoice amount in EUR including VAT. "
            "Use the final 'Total' or 'Gesamtbetrag' line. Example: 119.00"
        )
    )
    date: str | None = Field(
        default=None,
        description="Invoice issue date as YYYY-MM-DD. Null if not stated.",
    )
    ref: str = Field(
        description="Invoice reference number as printed, e.g. 'RE-2026-0042'",
    )
    status: Literal["draft", "sent", "paid", "overdue"] = Field(
        description="Current payment status",
    )
```

### Description checklist

| What to include | Example |
| --- | --- |
| **Unit** | `"Price in EUR"`, `"Weight in kg"`, `"Duration in seconds"` |
| **Format** for strings | `"ISO 8601: YYYY-MM-DD"`, `"E.164 phone: +49..."` |
| **Concrete example** | `"e.g. 42.50"`, `"e.g. 'gpt-4o'"` |
| **Null semantics** | `"null if not mentioned"`, `"null if the field is absent"` |
| **Scope / source** | `"Use the 'Total' line, not the subtotal"` |
| **Language / encoding** | `"Lowercase, no spaces"`, `"Preserve original casing"` |

---

## Working with complex schemas

### Enums with error feedback

When the LLM returns an invalid enum value the error report lists all allowed
values explicitly:

```python
class Status(str, Enum):
    OPEN        = "open"
    IN_PROGRESS = "in_progress"
    CLOSED      = "closed"

class Ticket(BaseModel):
    title:  str
    status: Status
```

If the LLM produces `"in progress"`, the retry message will say:

> *`status`: invalid enum value 'in progress'. Allowed values: open,
> in_progress, closed*

### Nested models

```python
class Author(BaseModel):
    name:  str
    email: str | None = None

class Article(BaseModel):
    title:      str
    author:     Author
    tags:       list[str]
    word_count: int
```

### Constrained types

```python
class Rating(BaseModel):
    score: float = Field(ge=0.0, le=10.0)
    label: str   = Field(min_length=1, max_length=50)
```

Constraint violations (`ge`, `le`, `min_length`, `max_length`) are reported
separately so the LLM understands the exact bounds it must respect.

---

## `model_config` — model-wide settings

`ConfigDict` sets defaults that apply to every field in the model — no
per-field validator code needed.  The two most useful settings for LLM
extraction are:

```python
from pydantic import BaseModel, ConfigDict

class MySchema(BaseModel):
    model_config = ConfigDict(
        str_strip_whitespace=True,   # trim accidental leading/trailing spaces
        extra="ignore",              # silently drop extra fields the LLM adds
    )

    name:  str
    value: float
```

### `str_strip_whitespace=True`

LLMs occasionally add stray spaces around values.  This option strips all
of them automatically:

```python
Product(name="  Widget Pro  ", sku=" W-001 ")
# → name='Widget Pro', sku='W-001'
```

### `extra="ignore"`

Silently drops any key the LLM includes that is not declared in the schema.
Prevents unexpected validation errors from extra fields:

```python
# LLM adds "notes" — silently dropped
Address(street="Main St", city="Springfield", notes="second floor")
```

For a complete treatment of `model_config` options, see
[Validators — `model_config`](validators.md#model_config--model-wide-settings).

---

## Further reading

- [Built-in Field Types & Validators](built-in-types.md) — reusable types the library ships (`IsoDateStr`, `IbanStr`, …)
- [Validators](validators.md) — clean messy LLM output inside the schema
- [Extraction](extraction.md) — how to pass messages and schemas to the library
- [Retry & Fallback](retry-and-fallback.md) — what happens when validation fails
