# Built-in Field Types & Validators

→ [Documentation index](index.md)

> **Source:** [`src/saidex/validators.py`](../src/saidex/validators.py)  
> **Tests:** [`tests/test_validators.py`](../tests/test_validators.py)

This page lists the **reusable field types and validator functions that SAIDEX
ships with**. Drop them straight into your extraction schemas — no need to
write your own validator for these common cases.

> This is the counterpart to [Pydantic Validators](validators.md): that page
> teaches you how to *write your own* validators; this page documents the ones
> the library *provides ready-made*. Reach for a built-in type first, and fall
> back to a custom validator only when no built-in fits.

All built-ins are exported from the package root:

```python
from saidex import (
    ISODateStr, IbanStr, VatIdStr, CountryCodeStr,
    CurrencyCodeStr, IsinStr, PhoneStr, LanguageCodeStr,
)
```

Each type `X` also exports its underlying `validate_x` function (e.g.
`validate_iban`) so you can reuse the check outside a schema or compose it into
your own `Annotated` type.

---

## Why use a built-in type?

A built-in like [`ISODateStr`](#isodatestr) gives you several things at once:

- **One annotation, every schema** — attach validation by typing a field, with
  no `@field_validator` method to copy around.
- **Stays a `str`** — the value round-trips as a string (never coerced into a
  `datetime.date` or other object), so `model_dump()` and JSON output stay
  predictable. Some types additionally **normalise** the string to a canonical
  form (e.g. `IbanStr` strips spaces and upper-cases) — noted per type below.
- **Clear, LLM-facing error messages** — on invalid input the validator raises
  `ValueError`, which [`create_instance_safe`](validators.md#testing-validators-independently)
  turns into structured correction guidance and feeds back into the retry loop
  (see [Retry & Fallback](retry-and-fallback.md)).
- **Tested** — every built-in is covered in
  [`tests/test_validators.py`](../tests/test_validators.py).

---

## Reference

| Type | Validator | Normalises? | Purpose |
| --- | --- | --- | --- |
| [`ISODateStr`](#isodatestr) | `validate_iso_date` | no (strict) | A `str` in ISO `yyyy-mm-dd` calendar-date format |
| [`IbanStr`](#ibanstr) | `validate_iban` | yes | IBAN with structural **and** mod-97 checksum validation |
| [`VatIdStr`](#vatidstr) | `validate_vat_id` | yes | EU-style VAT identification number (format check) |
| [`CountryCodeStr`](#countrycodestr) | `validate_country_code` | yes | ISO 3166-1 alpha-2 country code, validated against the official set |
| [`CurrencyCodeStr`](#currencycodestr) | `validate_currency_code` | yes | ISO 4217 currency code, validated against the official set |
| [`IsinStr`](#isinstr) | `validate_isin` | yes | ISIN with structural **and** Luhn check-digit validation |
| [`PhoneStr`](#phonestr) | `validate_phone` | yes | Phone number normalised to E.164 (`+` and country code) |
| [`LanguageCodeStr`](#languagecodestr) | `validate_language_code` | yes | ISO 639-1 language code, validated against the official set |

Every `validate_*` function returns the (optionally normalised) value on success
and raises `ValueError` on failure — see [`validate_iso_date`](#validate_iso_date)
for the patterns in which you can reuse them directly.

---

## `ISODateStr`

```python
ISODateStr = Annotated[str, AfterValidator(validate_iso_date)]
```

A drop-in replacement for `str` on any field that must hold a date in
`yyyy-mm-dd` form (four-digit year, **zero-padded** month and day). It validates
both the *shape* and that the value is a *real calendar date*, while keeping the
field a `str`.

### Usage

```python
from pydantic import BaseModel, Field
from saidex import ISODateStr

class MeetingNotes(BaseModel):
    # Required date field
    meeting_date: ISODateStr = Field(description="Meeting date as yyyy-mm-dd")

    # Optional date field — combine with `| None`
    next_meeting: ISODateStr | None = Field(
        default=None,
        description="Date of the next meeting as yyyy-mm-dd, or null",
    )
```

`ISODateStr` also works inside nested models and lists — for example an
`action_items: list[ActionItem]` where each `ActionItem.due_date` is
`ISODateStr | None`. Validation errors report the full path
(`action_items -> 0 -> due_date`) so the LLM knows exactly which item to fix.

### What it accepts and rejects

| Input | Result |
| --- | --- |
| `"2024-04-05"` | ✅ valid, returned unchanged |
| `None` (on an `ISODateStr \| None` field) | ✅ allowed |
| `"2024-4-5"` | ❌ wrong format — month/day must be zero-padded |
| `"April 5th"`, `"05.04.2024"`, `"31/12/2026"` | ❌ wrong format |
| `"2024-04-05T00:00:00"` | ❌ wrong format — date only, no time |
| `"2024-13-40"`, `"2024-02-30"` | ❌ right shape, but not a real calendar date |

The two rejection reasons produce **distinct** error messages so the model gets
actionable feedback:

```text
'2024-4-5' does not match the required date format yyyy-mm-dd
(four-digit year, zero-padded month and day, e.g. 2024-04-05).

'2024-13-40' has the right shape but is not a real calendar date.
Use a valid yyyy-mm-dd date, e.g. 2024-04-05.
```

> **Tip:** Still describe the format in `Field(description=...)`. The validator
> is the safety net that triggers a retry; a good description is what stops the
> model from producing a bad value in the first place. See
> [Schema Design — dates](schema-design.md#datetime-date-time--use-str-instead).

### Strict vs. lenient — which should you use?

`ISODateStr` is **strict**: it rejects anything that is not already
`yyyy-mm-dd` and lets the retry loop ask the model to correct it. This keeps
your data clean and surfaces genuinely ambiguous input.

If you would rather **auto-normalise** other notations (e.g. rewrite
`05.04.2024` → `2024-04-05`) instead of rejecting them, use a `BeforeValidator`
that reformats the string — see
[Validators — normalising date formats](validators.md#example--normalising-date-formats).
You can even combine both: a `BeforeValidator` to normalise, then
`validate_iso_date` as the final guard.

---

## `validate_iso_date`

```python
def validate_iso_date(value: str) -> str: ...
```

The plain function that powers `ISODateStr`. It returns the value unchanged when
valid and raises `ValueError` otherwise. Use it directly when you want to:

**Compose your own `Annotated` type** (e.g. normalise first, then validate):

```python
from typing import Annotated
from pydantic import BeforeValidator, AfterValidator
from saidex import validate_iso_date

def to_iso(v: str) -> str:
    ...  # rewrite DD.MM.YYYY → YYYY-MM-DD, then let validate_iso_date check it
    return v

LenientISODate = Annotated[str, BeforeValidator(to_iso), AfterValidator(validate_iso_date)]
```

**Validate inside an existing `@field_validator` or `@model_validator`:**

```python
from pydantic import BaseModel, field_validator
from saidex import validate_iso_date

class Booking(BaseModel):
    check_in: str
    check_out: str

    @field_validator("check_in", "check_out", mode="after")
    @classmethod
    def _check_dates(cls, v: str) -> str:
        return validate_iso_date(v)
```

**Validate outside of Pydantic** (plain input checking):

```python
from saidex import validate_iso_date

try:
    validate_iso_date(user_input)
except ValueError as exc:
    print(f"Bad date: {exc}")
```

---

## `IbanStr`

```python
IbanStr = Annotated[str, AfterValidator(validate_iban)]
```

An IBAN field that validates **both** structure and the ISO 7064 **mod-97 check
digits**, so a transposed or misread digit is caught — not just a wrong shape.
The value is **normalised**: surrounding spaces are removed and letters
upper-cased, so `model_dump()` yields the compact canonical form.

```python
from pydantic import BaseModel, Field
from saidex import IbanStr

class PaymentDetails(BaseModel):
    iban: IbanStr = Field(description="Payee IBAN")

PaymentDetails(iban="de89 3704 0044 0532 0130 00").iban
# -> "DE89370400440532013000"
```

| Input | Result |
| --- | --- |
| `"DE89 3704 0044 0532 0130 00"` | ✅ → `"DE89370400440532013000"` |
| `"GB82 WEST 1234 5698 7654 32"` | ✅ normalised |
| `"DE42 1004 0000 0287 8000 04"` | ❌ valid shape, **checksum fails** |
| `"not-an-iban"`, `"1234"` | ❌ not structurally an IBAN |

---

## `VatIdStr`

```python
VatIdStr = Annotated[str, AfterValidator(validate_vat_id)]
```

An EU-style VAT identification number: a 2-letter country prefix followed by
2–12 alphanumeric characters. Spaces, dots, and hyphens are stripped and the
value upper-cased. This is a **format** check (not a per-country length or
checksum check), which is enough to reject obviously wrong values from an LLM.

```python
from saidex import VatIdStr

class Company(BaseModel):
    vat_id: VatIdStr | None = Field(None, description="EU VAT ID, e.g. DE298471023")

# "DE 301 847 192" -> "DE301847192";  "ATU12345678" stays;  "12345" -> ValueError
```

---

## `CountryCodeStr`

```python
CountryCodeStr = Annotated[str, AfterValidator(validate_country_code)]
```

An ISO 3166-1 **alpha-2** country code, validated against the official set of
assigned codes and upper-cased. Because it checks the real set, common mistakes
are rejected: the United Kingdom is `"GB"`, not `"UK"`; `"XX"` is invalid.

```python
from saidex import CountryCodeStr

class Address(BaseModel):
    country: CountryCodeStr = Field(description="ISO 3166-1 alpha-2 code, e.g. DE")

Address(country="de").country   # -> "DE"
Address(country="UK")           # -> ValueError (use "GB")
```

> **Tip:** Tell the model the expected form in the description (“two-letter ISO
> code”). If your source text uses country *names*, pair this with a
> `BeforeValidator` that maps names → codes (see
> [Build your own validated type](#build-your-own-validated-type)).

---

## `CurrencyCodeStr`

```python
CurrencyCodeStr = Annotated[str, AfterValidator(validate_currency_code)]
```

An ISO 4217 **three-letter** currency code, validated against the set of active
codes (plus the standard precious-metal/supranational codes) and upper-cased.

```python
from saidex import CurrencyCodeStr

class Money(BaseModel):
    amount: float = Field(description="Numeric amount, no symbol")
    currency: CurrencyCodeStr = Field(description="ISO 4217 code, e.g. EUR")

Money(amount=49.99, currency="eur").currency   # -> "EUR"
Money(amount=1.0, currency="€")                # -> ValueError (use "EUR")
```

---

## `IsinStr`

```python
IsinStr = Annotated[str, AfterValidator(validate_isin)]
```

An ISIN (International Securities Identification Number) that validates **both**
structure (2-letter prefix, 9 alphanumeric characters, 1 check digit) **and** the
ISO 6166 **Luhn check digit**, catching transposed or misread characters. The
value is normalised (spaces removed, upper-cased).

```python
from saidex import IsinStr

class Security(BaseModel):
    isin: IsinStr = Field(description="12-character ISIN, e.g. US0378331005")

Security(isin="us 0378331005").isin   # -> "US0378331005"
```

| Input | Result |
| --- | --- |
| `"US0378331005"` (Apple) | ✅ valid |
| `"DE0005140008"` (Deutsche Bank) | ✅ valid |
| `"US0378331006"` | ❌ valid shape, **check digit fails** |
| `"12345"`, `"USABC"` | ❌ not structurally an ISIN |

> The check digit validates the *digits*, not the country prefix — by design,
> since valid prefixes include non-country codes like `XS` (Eurobonds). Pair it
> with a description if you also want a specific issuer country.

---

## `PhoneStr`

```python
PhoneStr = Annotated[str, AfterValidator(validate_phone)]
```

A phone number normalised to **E.164** (`+`, country code, then digits — max 15
total). Common separators (spaces, parentheses, dots, slashes, hyphens) are
removed, and a leading international `00` is converted to `+`.

```python
from saidex import PhoneStr

class Contact(BaseModel):
    phone: PhoneStr | None = Field(None, description="Phone in E.164 form, e.g. +496971402200")

Contact(phone="+49 69 7140 2200").phone   # -> "+496971402200"
Contact(phone="069 7140 2200")            # -> ValueError (no country code)
```

> E.164 **requires** a country code, so a bare national number is rejected. If
> your source text omits it, prepend the expected country code in a
> `BeforeValidator` (see [Build your own validated type](#build-your-own-validated-type)).

---

## `LanguageCodeStr`

```python
LanguageCodeStr = Annotated[str, AfterValidator(validate_language_code)]
```

An ISO 639-1 **two-letter** language code, validated against the official set and
lower-cased. Rejects names and wrong codes: English is `"en"`, not `"english"`
or `"gb"`.

```python
from saidex import LanguageCodeStr

class Page(BaseModel):
    language: LanguageCodeStr = Field(description="ISO 639-1 language code, e.g. en")

Page(language="EN").language   # -> "en"
```

---

## How built-ins fit the retry loop

Because every built-in raises `ValueError` on bad input, it plugs into the same
self-correction flow as any hand-written validator:

```text
LLM output → create_instance_safe(schema, **output)
                   │
                   ├─ ISODateStr / validate_iso_date runs
                   │
                   ├─ valid   → validated instance ✅
                   └─ invalid → ValueError
                                   │
                                   ▼
                         structured field-level error
                         ("action_items -> 0 -> due_date: …")
                                   │
                                   ▼
                         sent back to the LLM as a retry prompt
```

See [Retry & Fallback](retry-and-fallback.md) for the full end-to-end loop and
[Validators — testing](validators.md#testing-validators-independently) for how to
assert on the exact message the LLM receives.

---

## Build your own validated type

The built-ins on this page are nothing special — they are just `Annotated`
aliases wrapping a validator function. You can create your own in **your** code
the exact same way, and they behave identically: reusable across schemas, and
their `ValueError` messages flow into the retry loop via `create_instance_safe`.

### The recipe

```python
from typing import Annotated
from pydantic import AfterValidator

def validate_x(value: str) -> str:
    """Return the (optionally normalised) value, or raise ValueError."""
    if not is_acceptable(value):
        raise ValueError("…specific, actionable message telling the LLM the rule…")
    return normalise(value)          # or just `return value` to keep it as-is

XStr = Annotated[str, AfterValidator(validate_x)]
```

Three decisions shape the validator:

1. **`AfterValidator` vs `BeforeValidator`.** `AfterValidator` runs *after*
   Pydantic has coerced the value to the field's declared type — use it for
   strings you want to check and lightly normalise (case, whitespace).
   `BeforeValidator` runs on the *raw* value first — use it to reshape messy
   input (e.g. map a country *name* to a code) before the strict check. You can
   stack both: `Annotated[str, BeforeValidator(reshape), AfterValidator(check)]`.
2. **Strict vs normalising.** Return the value unchanged to be strict (like
   `ISODateStr`), or return a canonical form to normalise (like `IbanStr`).
   Document which you chose — callers rely on the output shape.
3. **The error message.** It is sent verbatim to the LLM. State the rule and give
   an example: `"…expected NNNNN or NNNNN-NNNN, e.g. 10115"`. See
   [Validators — error messages](validators.md#how-validators-interact-with-the-retry-loop).

### Example 1 — a strict pattern type

A reference/order ID that must look like `ORD-12345`:

```python
import re
from typing import Annotated
from pydantic import AfterValidator

_ORDER_ID_RE = re.compile(r"^ORD-\d{5}$")

def validate_order_id(value: str) -> str:
    if not isinstance(value, str) or not _ORDER_ID_RE.match(value):
        raise ValueError(
            f"'{value}' is not a valid order ID. Expected 'ORD-' followed by "
            f"exactly five digits, e.g. 'ORD-04812'."
        )
    return value

OrderId = Annotated[str, AfterValidator(validate_order_id)]
```

### Example 2 — a normalising, set-validated type

Validate against an allowed set and normalise. The same shape as
`CountryCodeStr` / `CurrencyCodeStr`:

```python
from typing import Annotated
from pydantic import AfterValidator

_PRIORITIES = {"low", "medium", "high", "critical"}

def validate_priority(value: str) -> str:
    norm = str(value).strip().lower()
    if norm not in _PRIORITIES:
        allowed = ", ".join(sorted(_PRIORITIES))
        raise ValueError(f"'{value}' is not a valid priority. Use one of: {allowed}.")
    return norm

Priority = Annotated[str, AfterValidator(validate_priority)]
```

> For a fixed, *small* set you usually want a `typing.Literal` or an `Enum`
> instead — Pydantic then advertises the allowed values in the JSON schema the
> LLM sees. Reach for a set-checking validator when the list is large (country
> codes), comes from data, or needs normalisation before the check.

### Example 3 — reshape first, then reuse a built-in

Map a country *name* to a code with a `BeforeValidator`, then hand off to the
library's `validate_country_code` for the authoritative check:

```python
from typing import Annotated, Any
from pydantic import BeforeValidator, AfterValidator
from saidex import validate_country_code

_NAME_TO_CODE = {"germany": "DE", "deutschland": "DE", "united kingdom": "GB"}

def name_to_code(value: Any) -> Any:
    if isinstance(value, str):
        return _NAME_TO_CODE.get(value.strip().lower(), value)
    return value

# Accepts "Germany", "DE", or "de"; always ends up validated and normalised to "DE".
LenientCountryCode = Annotated[str, BeforeValidator(name_to_code), AfterValidator(validate_country_code)]
```

### Where to put it

Define each type once in a small module (e.g. `myapp/schema_types.py`) and
import it wherever you declare schemas. That keeps the validation rule in a
single, testable place — exactly how SAIDEX keeps its own in
[`src/saidex/validators.py`](../src/saidex/validators.py).

### Test it without an LLM

Validators are plain functions, so test them directly — and assert on the
message the LLM would receive via `create_instance_safe`:

```python
import pytest
from saidex import create_instance_safe

def test_order_id_rejects_bad_value():
    with pytest.raises(ValueError, match="valid order ID"):
        validate_order_id("12345")

def test_schema_reports_field_path():
    instance, error = create_instance_safe(MyOrder, order_id="12345")
    assert instance is None
    assert "order_id" in error      # this text is sent back to the model
```

See [Validators — testing](validators.md#testing-validators-independently) for
more on testing validators and inspecting the retry message.

---

## Related

- [Pydantic Validators](validators.md) — write your own pre/post/model validators and reusable `Annotated` types
- [Schema Design](schema-design.md) — which Pydantic types to use, and why `str` beats `date` for dates
- [Retry & Fallback](retry-and-fallback.md) — how a raised `ValueError` becomes a correction prompt
- [`src/saidex/validators.py`](../src/saidex/validators.py) — source of all built-in types
- [`tests/test_validators.py`](../tests/test_validators.py) — tests for every built-in
