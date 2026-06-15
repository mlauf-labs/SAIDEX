# Pydantic Validators

→ [Documentation index](index.md)

> **Runnable example:** [`examples/06_pydantic_validators.py`](../examples/06_pydantic_validators.py)  
> **Tests:** [`tests/test_validators.py`](../tests/test_validators.py)

LLMs produce slightly inconsistent output.  A numeric field might arrive as
`"€ 1.234,56"`, a date as `"05.01.2026"` or `"January 5th"`, a boolean as
`"ja"`, a country code as `"germany"`.  Pydantic validators let you **clean,
normalise, and verify** all of this *inside the schema definition* — before
your application code ever sees the data and without burning extra LLM retries.

When a validator raises `ValueError` the library catches the resulting
`ValidationError`, converts it into a structured, field-level error message,
and sends it back to the LLM as a correction prompt.  Validators therefore
participate in the retry loop automatically.

---

## How validators interact with the retry loop

```
LLM produces tool-call JSON
         │
         ▼
  create_instance_safe(schema, **llm_output)
         │
         ├─ Pydantic runs all validators (see order below)
         │
         ├─ Success → return validated instance ✅
         │
         └─ ValidationError
                │
                ▼
       _format_validation_error()
       builds a human-readable field-level report
                │
                ▼
       HumanMessage appended to conversation
       "The output format is not correct. Please correct it …"
                │
                ▼
       LLM gets another attempt (up to max_primary_retries)
```

Every `ValueError` you raise inside a validator becomes part of that error
report and is sent to the LLM.  Write error messages that are **specific and
actionable** — the LLM reads them to know what to fix.

```python
# BAD — vague, LLM doesn't know what to do
raise ValueError("invalid date")

# GOOD — tells the LLM exactly what format is expected
raise ValueError(
    f"date '{v}' is not in the required ISO 8601 format. "
    "Expected YYYY-MM-DD, e.g. '2026-01-31'."
)
```

---

## Validator types and execution order

```
raw LLM value (dict from tool-call JSON)
        │
        ▼  1. @model_validator(mode='before')
        │     Receives the raw dict.
        │     Use to rename keys, fill missing defaults, or handle
        │     unexpected structures before any field is touched.
        │
        ▼  2. @field_validator(mode='before')   [per field]
        │     Receives the raw field value (str, int, None, …).
        │     Use to strip symbols, convert formats, map synonyms.
        │
        ▼  3. Pydantic type coercion
        │     "42.5" → 42.5 (float), "true" → True (bool), etc.
        │
        ▼  4. Field constraints  (ge=, le=, min_length=, …)
        │
        ▼  5. @field_validator(mode='after')    [per field]
        │     Receives the already-typed value.
        │     Use to normalise (lowercase, uppercase), enforce
        │     domain rules, or validate format of the final type.
        │
        ▼  6. @model_validator(mode='after')
        │     Receives the fully constructed model instance.
        │     Use for cross-field checks (end > start, total = sum).
        │
        ▼  7. @computed_field properties
        │     Derived read-only values — never asked from the LLM.
        │
        ▼  validated model instance ✅
```

---

## `@field_validator(mode='before')` — clean raw LLM output

`mode='before'` receives the **raw value exactly as the LLM returned it**,
before Pydantic attempts any type coercion.  The return value replaces the
original and is then passed to Pydantic for normal processing.

This is the right place to handle all LLM-specific quirks: stray currency
symbols, locale-specific number formats, alternative date notations, language
synonyms for booleans, etc.

### Syntax

```python
from typing import Any
from pydantic import BaseModel, field_validator

class MyModel(BaseModel):
    price: float

    @field_validator("price", mode="before")
    @classmethod
    def clean_price(cls, v: Any) -> Any:
        # v is whatever the LLM put in "price" — could be str, int, float, None
        if isinstance(v, str):
            # … clean it …
            return cleaned_value
        return v  # always return something; don't return None unless field is Optional
```

**Rules:**
- Must be a `@classmethod`.
- First argument is `cls`, second is the raw value (`v`).
- Return the cleaned value — Pydantic then coerces it to the declared type.
- Raise `ValueError` to reject a value and trigger a retry.
- Apply one validator to multiple fields: `@field_validator("a", "b", mode="before")`.

### Example — cleaning numeric and currency fields

LLMs frequently include units, symbols, or locale-specific separators in
numeric fields (`"€ 1.234,56"`, `"$42.50"`, `"100 EUR"`).

```python
import re
from typing import Any
from pydantic import BaseModel, Field, field_validator

class Invoice(BaseModel):
    net_amount:   float = Field(ge=0, description="Net amount in EUR, e.g. 100.00")
    gross_amount: float = Field(ge=0, description="Gross amount in EUR, e.g. 119.00")

    @field_validator("net_amount", "gross_amount", mode="before")
    @classmethod
    def clean_money(cls, v: Any) -> Any:
        """Handle '€ 1.234,56', '$42.50', '1 190,00 EUR', or plain 42.5."""
        if not isinstance(v, str):
            return v
        cleaned = re.sub(r"[€$£¥\s]", "", v)
        cleaned = re.sub(r"[A-Za-z]+", "", cleaned).strip()
        # European notation: "1.234,56" → "1234.56"
        if re.search(r"\d{1,3}(\.\d{3})*(,\d+)$", cleaned):
            cleaned = cleaned.replace(".", "").replace(",", ".")
        elif "," in cleaned and "." not in cleaned:
            cleaned = cleaned.replace(",", ".")
        return cleaned
```

See the complete `Invoice` model in
[`examples/06_pydantic_validators.py`](../examples/06_pydantic_validators.py)
for the full pattern including a `paid` boolean and a `date` field.

### Example — normalising date formats

```python
import re
from typing import Any
from pydantic import BaseModel, Field, field_validator

class Event(BaseModel):
    start_date: str = Field(description="Start date as YYYY-MM-DD")
    end_date:   str = Field(description="End date as YYYY-MM-DD")

    @field_validator("start_date", "end_date", mode="before")
    @classmethod
    def normalise_date(cls, v: Any) -> Any:
        """Accept DD.MM.YYYY, DD/MM/YYYY and convert to YYYY-MM-DD."""
        if not isinstance(v, str):
            return v
        v = v.strip()
        if re.match(r"^\d{4}-\d{2}-\d{2}$", v):
            return v  # already correct
        m = re.match(r"^(\d{1,2})[./](\d{1,2})[./](\d{4})$", v)
        if m:
            return f"{m.group(3)}-{m.group(2).zfill(2)}-{m.group(1).zfill(2)}"
        raise ValueError(
            f"'{v}' is not a recognised date format. "
            "Expected YYYY-MM-DD, DD.MM.YYYY, or DD/MM/YYYY."
        )
```

### Example — mapping boolean synonyms

Different LLMs use different words for true/false, especially in multilingual
contexts.

```python
from typing import Any
from pydantic import BaseModel, field_validator

class Task(BaseModel):
    done: bool

    @field_validator("done", mode="before")
    @classmethod
    def normalise_bool(cls, v: Any) -> Any:
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            if v.lower() in {"yes", "ja", "oui", "true", "1", "done", "completed"}:
                return True
            if v.lower() in {"no", "nein", "non", "false", "0", "open", "pending"}:
                return False
        raise ValueError(
            f"Cannot interpret '{v}' as a boolean. "
            "Use true/false, yes/no, or 1/0."
        )
```

### Example — normalising enum values

When the LLM returns a value that almost matches an enum entry (`"HIGH"`, `"hoch"`, `"3"`):

```python
from enum import Enum
from typing import Any
from pydantic import BaseModel, field_validator

class Priority(str, Enum):
    LOW      = "low"
    MEDIUM   = "medium"
    HIGH     = "high"
    CRITICAL = "critical"

class Ticket(BaseModel):
    priority: Priority

    @field_validator("priority", mode="before")
    @classmethod
    def normalise_priority(cls, v: Any) -> Any:
        synonyms: dict[str, str] = {
            "hoch": "high", "niedrig": "low", "mittel": "medium",
            "kritisch": "critical", "urgent": "critical",
            "1": "low", "2": "medium", "3": "high", "4": "critical",
        }
        if isinstance(v, str):
            normalised = v.strip().lower()
            return synonyms.get(normalised, normalised)
        return v
```

---

## `@field_validator(mode='after')` — rules on the typed value

`mode='after'` receives the value **after Pydantic has coerced it** to the
declared type.  At this point you can safely use type-specific operations
(`str.lower()`, arithmetic, etc.) without checking `isinstance`.

Use `mode='after'` for:
- Normalisation that requires the correct type (`str.lower()`, `str.strip()`)
- Domain rules with a clear error message
- Format validation on strings (regex, length, structure)

### Example — normalise strings and validate format

```python
import re
from pydantic import BaseModel, Field, field_validator

class ContactInfo(BaseModel):
    email:        str | None = None
    phone:        str | None = None
    country_code: str        = Field(description="ISO 3166-1 alpha-2, e.g. DE")
    website:      str | None = None
    iban:         str | None = Field(default=None, description="IBAN or null")

    @field_validator("email", mode="after")
    @classmethod
    def lowercase_email(cls, v: str | None) -> str | None:
        return v.lower() if v else v

    @field_validator("country_code", mode="after")
    @classmethod
    def validate_country_code(cls, v: str) -> str:
        v = v.upper().strip()
        if not re.match(r"^[A-Z]{2}$", v):
            raise ValueError(
                f"'{v}' is not a valid ISO 3166-1 alpha-2 country code. "
                "Expected exactly 2 uppercase letters, e.g. 'DE', 'US', 'FR'."
            )
        return v

    @field_validator("phone", mode="after")
    @classmethod
    def normalise_phone(cls, v: str | None) -> str | None:
        """Strip formatting — keep only digits and a single leading '+'."""
        if not v:
            return v
        digits = re.sub(r"[^\d+]", "", v)
        return re.sub(r"\++", "+", digits) or None

    @field_validator("website", mode="after")
    @classmethod
    def ensure_https(cls, v: str | None) -> str | None:
        if v and not v.startswith(("http://", "https://")):
            return f"https://{v}"
        return v

    @field_validator("iban", mode="after")
    @classmethod
    def validate_iban_format(cls, v: str | None) -> str | None:
        if not v:
            return v
        iban = v.replace(" ", "").upper()
        if not re.match(r"^[A-Z]{2}\d{2}[A-Z0-9]{1,30}$", iban):
            raise ValueError(
                f"'{v}' does not look like a valid IBAN. "
                "Expected format: DE89 3704 0044 0532 0130 00"
            )
        return iban
```

The complete `ContactInfo` model is in
[`examples/06_pydantic_validators.py`](../examples/06_pydantic_validators.py).

### Using the `info` parameter

Pass `info: FieldValidationInfo` as a third argument to access the field name
and other context inside the validator:

```python
from pydantic import BaseModel, FieldValidationInfo, field_validator

class Measurement(BaseModel):
    value: float
    unit:  str

    @field_validator("value", "unit", mode="after")
    @classmethod
    def log_field(cls, v: object, info: FieldValidationInfo) -> object:
        print(f"Validated field '{info.field_name}': {v!r}")
        return v
```

---

## `@model_validator(mode='before')` — reshape the whole input

`mode='before'` runs before **any** field validator.  It receives the raw
input as a `dict` (or whatever was passed) and must return a `dict` for
normal processing to continue.

Use it when:
- The LLM returns a flat string instead of a structured object
- Field names differ from what the LLM produces (aliasing / renaming)
- You need to supply computed defaults that depend on multiple raw fields
- The LLM wraps the result in an extra level of nesting

### Example — parse a flat string into a dict

```python
from typing import Any
from pydantic import BaseModel, model_validator

class Address(BaseModel):
    street:   str
    city:     str
    zip_code: str
    country:  str = "DE"

    @model_validator(mode="before")
    @classmethod
    def handle_flat_string(cls, data: Any) -> Any:
        """Accept 'Hauptstraße 1, 80331 München, DE' as well as a dict."""
        if not isinstance(data, str):
            return data
        parts = [p.strip() for p in data.split(",")]
        result: dict[str, Any] = {}
        if parts:
            result["street"] = parts[0]
        if len(parts) >= 2:
            zip_city = parts[1].split(None, 1)
            if len(zip_city) == 2:
                result["zip_code"], result["city"] = zip_city
            else:
                result["city"] = parts[1]
        if len(parts) >= 3:
            result["country"] = parts[2].upper()
        return result
```

```python
# Both are valid after the validator:
Address(street="Hauptstraße 1", city="München", zip_code="80331")
Address.model_validate("Hauptstraße 1, 80331 München, DE")
```

### Example — handle LLM field-name variations

Some LLMs occasionally return `"order_number"` when you asked for `"order_id"`,
or `"total_price"` instead of `"total"`:

```python
from typing import Any
from pydantic import BaseModel, model_validator

class Order(BaseModel):
    order_id: str
    total:    float

    @model_validator(mode="before")
    @classmethod
    def normalise_field_names(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        aliases = {
            "order_number": "order_id",
            "id":           "order_id",
            "total_price":  "total",
            "amount":       "total",
            "sum":          "total",
        }
        return {aliases.get(k, k): v for k, v in data.items()}
```

### Example — unwrap an extra nesting level

```python
from typing import Any
from pydantic import BaseModel, model_validator

class Product(BaseModel):
    name:  str
    price: float

    @model_validator(mode="before")
    @classmethod
    def unwrap_nested(cls, data: Any) -> Any:
        """If LLM wraps result as {'product': {...}}, unwrap it."""
        if isinstance(data, dict) and len(data) == 1:
            (key, value) = next(iter(data.items()))
            if key.lower() in {"product", "result", "data", "output"}:
                return value
        return data
```

---

## `@model_validator(mode='after')` — cross-field consistency rules

`mode='after'` runs after all field validators have completed.  The validator
receives **`self`** — the fully constructed model instance — and must return
`self` (or raise `ValueError`).

Use it for rules that span multiple fields: date ordering, totals matching
line items, mutually exclusive fields, conditional requirements.

### Example — date ordering

```python
from pydantic import BaseModel, Field, model_validator

class Contract(BaseModel):
    start_date:  str      = Field(description="Contract start as YYYY-MM-DD")
    end_date:    str      = Field(description="Contract end as YYYY-MM-DD")
    signed_date: str | None = Field(default=None, description="Signing date or null")

    @model_validator(mode="after")
    def check_date_order(self) -> "Contract":
        if self.end_date <= self.start_date:
            raise ValueError(
                f"end_date ({self.end_date}) must be after "
                f"start_date ({self.start_date})."
            )
        if self.signed_date and self.signed_date > self.start_date:
            raise ValueError(
                f"signed_date ({self.signed_date}) must not be after "
                f"start_date ({self.start_date}). Contracts are signed before they start."
            )
        return self
```

### Example — total matches line items (with tolerance)

```python
from pydantic import BaseModel, Field, model_validator

class LineItem(BaseModel):
    quantity:   int   = Field(ge=1)
    unit_price: float = Field(ge=0)

class Receipt(BaseModel):
    items:          list[LineItem] = Field(min_length=1)
    declared_total: float          = Field(ge=0)
    discount:       float          = Field(default=0.0, ge=0)

    @model_validator(mode="after")
    def validate_total(self) -> "Receipt":
        calculated = round(
            sum(i.quantity * i.unit_price for i in self.items) - self.discount, 2
        )
        if calculated == 0:
            return self
        deviation_pct = abs(self.declared_total - calculated) / calculated * 100
        if deviation_pct > 2:
            raise ValueError(
                f"declared_total ({self.declared_total}) differs from the "
                f"sum of line items ({calculated}) by {deviation_pct:.1f} %. "
                "Please recheck all quantities, unit prices, and the total."
            )
        return self
```

### Example — mutually exclusive fields

```python
from pydantic import BaseModel, model_validator

class Discount(BaseModel):
    percent_off:  float | None = None
    fixed_amount: float | None = None

    @model_validator(mode="after")
    def exactly_one_discount_type(self) -> "Discount":
        has_percent = self.percent_off is not None
        has_fixed   = self.fixed_amount is not None
        if has_percent and has_fixed:
            raise ValueError(
                "Provide either percent_off or fixed_amount — not both. "
                "Set the one that does not apply to null."
            )
        if not has_percent and not has_fixed:
            raise ValueError(
                "One of percent_off or fixed_amount must be provided."
            )
        return self
```

### Example — conditional required field

```python
from pydantic import BaseModel, model_validator

class Shipment(BaseModel):
    is_international: bool
    customs_code:     str | None = None

    @model_validator(mode="after")
    def customs_code_required_for_international(self) -> "Shipment":
        if self.is_international and not self.customs_code:
            raise ValueError(
                "customs_code is required when is_international is true. "
                "Provide the HS tariff code, e.g. '8471.30'."
            )
        return self
```

The `Order` / `LineItem` models in
[`examples/06_pydantic_validators.py`](../examples/06_pydantic_validators.py)
show a production-ready `model_validator(mode='after')` that verifies the
declared total against the sum of its line items.

---

## `@computed_field` — automatically derived values

`@computed_field` decorates a `@property` and marks it as a **derived,
read-only field**.  It is included in `model_dump()` and JSON output, but the
LLM is never asked to provide it — it is always calculated from the model's
other fields.

### Syntax

```python
from pydantic import BaseModel, computed_field

class MyModel(BaseModel):
    quantity:   int
    unit_price: float

    @computed_field            # declares it as a field
    @property
    def subtotal(self) -> float:    # return type annotation is required
        return round(self.quantity * self.unit_price, 2)
```

> **Important:** both `@computed_field` and `@property` must be present,
> and the return type annotation is required.

### Example — multiple computed fields

```python
from pydantic import BaseModel, Field, computed_field

class Product(BaseModel):
    name:      str
    price_net: float = Field(ge=0, description="Net price in EUR")
    tax_rate:  float = Field(ge=0, le=1, description="Tax rate, e.g. 0.19 for 19 %")
    quantity:  int   = Field(ge=1, description="Number of units")

    @computed_field
    @property
    def price_gross(self) -> float:
        """Net price including tax."""
        return round(self.price_net * (1 + self.tax_rate), 2)

    @computed_field
    @property
    def line_total(self) -> float:
        """Total for all units including tax."""
        return round(self.price_gross * self.quantity, 2)

    @computed_field
    @property
    def tax_amount(self) -> float:
        """Tax portion of line_total."""
        return round(self.line_total - self.price_net * self.quantity, 2)
```

```python
p = Product(name="Widget", price_net=100.0, tax_rate=0.19, quantity=3)
p.price_gross   # 119.0
p.line_total    # 357.0
p.tax_amount    # 57.0
p.model_dump()  # includes all computed fields
```

**Key property:** Pydantic correctly excludes `@computed_field` properties
from the tool-calling JSON Schema sent to the LLM.  The LLM only sees the
source fields it needs to fill in.

---

## `model_config` — model-wide settings

`ConfigDict` sets defaults that apply to every field in the model without
any per-field code.

```python
from pydantic import BaseModel, ConfigDict

class MySchema(BaseModel):
    model_config = ConfigDict(
        str_strip_whitespace=True,   # trim leading/trailing spaces from all str fields
        extra="ignore",              # silently drop extra keys the LLM adds
        str_to_lower=False,          # set True to auto-lowercase all strings
        populate_by_name=True,       # allow aliased fields by their Python name too
    )
```

| Option | Effect | Typical use |
| --- | --- | --- |
| `str_strip_whitespace=True` | Strips leading/trailing spaces from every `str` | Always useful |
| `extra="ignore"` | Drops extra keys silently | Prevents spurious validation errors |
| `str_to_lower=True` | Auto-lowercases all strings | When casing is irrelevant |
| `populate_by_name=True` | Allow both alias and Python name | When using `Field(alias=...)` |

---

## The `Annotated` approach — reusable validators

Instead of defining a validator inside every model that uses a type, you can
attach validators directly to a type annotation using `Annotated`.  This makes
them reusable across multiple models without duplication.

> SAIDEX already ships a few of these ready-made — e.g.
> [`IsoDateStr`](built-in-types.md#isodatestr), [`IbanStr`](built-in-types.md#ibanstr),
> and [`CountryCodeStr`](built-in-types.md#countrycodestr). See
> [Built-in Field Types & Validators](built-in-types.md) before writing your own.

```python
from typing import Annotated, Any
from pydantic import BaseModel, BeforeValidator, AfterValidator

def clean_money(v: Any) -> Any:
    import re
    if isinstance(v, str):
        return re.sub(r"[€$£¥,\s]", "", v).replace(",", ".")
    return v

def positive_float(v: float) -> float:
    if v < 0:
        raise ValueError(f"Value must be ≥ 0, got {v}")
    return v

# Reusable type: strip symbols, then enforce positivity
EuroAmount = Annotated[float, BeforeValidator(clean_money), AfterValidator(positive_float)]

# Use in any model — no validator methods needed
class Invoice(BaseModel):
    net:   EuroAmount
    gross: EuroAmount

class LineItem(BaseModel):
    unit_price: EuroAmount
    subtotal:   EuroAmount
```

```python
Invoice(net="€ 100,00", gross="119.00")   # ✅ both cleaned automatically
```

---

## Common patterns — quick reference

| Problem | Solution |
| --- | --- |
| `"€ 1.234,56"` → `1234.56` | `@field_validator("price", mode="before")` → strip symbols, normalise separators |
| `"05.01.2026"` → `"2026-01-05"` | `@field_validator("date", mode="before")` → regex-reformat |
| `"ja"` / `"yes"` → `True` | `@field_validator("active", mode="before")` → map known synonyms |
| `"de"` → `"DE"` | `@field_validator("country", mode="after")` → `.upper()` |
| `"Alice@Example.COM"` → `"alice@example.com"` | `@field_validator("email", mode="after")` → `.lower()` |
| `"example.com"` → `"https://example.com"` | `@field_validator("url", mode="after")` → prepend scheme |
| `"+49 (0) 30 123"` → `"+4930123"` | `@field_validator("phone", mode="after")` → `re.sub` |
| LLM returns string, object expected | `@model_validator(mode="before")` → parse string into dict |
| LLM uses wrong field names | `@model_validator(mode="before")` → rename keys |
| `end_date > start_date` | `@model_validator(mode="after")` → compare fields |
| `total ≈ sum(items)` | `@model_validator(mode="after")` → check with tolerance |
| Exactly one of two fields must be set | `@model_validator(mode="after")` → XOR check |
| Field required only when another is `True` | `@model_validator(mode="after")` → conditional check |
| `subtotal = qty × price` | `@computed_field` → never asked from LLM |
| Strip spaces from all strings | `model_config = ConfigDict(str_strip_whitespace=True)` |
| Ignore extra fields the LLM adds | `model_config = ConfigDict(extra="ignore")` |

---

## Testing validators independently

Validators can be tested without an LLM using `create_instance_safe` or
direct model instantiation.  This is fast, free, and fully deterministic.

> **Full test suite:** [`tests/test_validators.py`](../tests/test_validators.py)

```python
from saidex import create_instance_safe

# Happy path
invoice, err = create_instance_safe(
    Invoice,
    net="€ 100,00",
    gross="119.00",
    date="05.01.2026",
    paid="ja",
)
assert invoice is not None
assert invoice.net == 100.0
assert invoice.date == "2026-01-05"
assert invoice.paid is True

# Error path — check the message the LLM would receive
invoice, err = create_instance_safe(
    Invoice,
    net=-5.0,       # violates ge=0
    gross=0.0,
    date="2026-01-01",
    paid=True,
)
assert invoice is None
assert "net" in err        # error mentions the field
assert "≥ 0" in err or "ge" in err.lower()
```

The `error` string returned by `create_instance_safe` is exactly the text
sent to the LLM in the retry message — writing a test for it ensures the
correction guidance is clear.

`create_instance_safe` is also available as a public API:

```python
from saidex import create_instance_safe

instance, error_msg = create_instance_safe(MySchema, **raw_dict)
if instance is None:
    print(error_msg)   # structured, field-level error description
```

---

## Common pitfalls

### Returning `None` from a non-optional field

```python
# WRONG — will immediately fail the model if v is any falsy value
@field_validator("name", mode="before")
@classmethod
def clean_name(cls, v: Any) -> Any:
    if isinstance(v, str):
        return v.strip() or None   # ❌ returns None if strip() gives ""
    return v

# RIGHT — raise ValueError so the retry mechanism kicks in
@field_validator("name", mode="before")
@classmethod
def clean_name(cls, v: Any) -> Any:
    if isinstance(v, str):
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("name must not be empty")   # ✅
        return cleaned
    return v
```

### Not returning `self` from a model validator

```python
# WRONG
@model_validator(mode="after")
def check_dates(self) -> None:        # ❌ must return self
    if self.end < self.start:
        raise ValueError("end must be after start")

# RIGHT
@model_validator(mode="after")
def check_dates(self) -> "MyModel":   # ✅
    if self.end < self.start:
        raise ValueError("end must be after start")
    return self
```

### Using `mode='after'` to clean symbols (should be `mode='before'`)

```python
# WRONG — by 'after' time, "€ 42" has already failed float coercion
@field_validator("price", mode="after")   # ❌
@classmethod
def clean_price(cls, v: float) -> float:
    ...

# RIGHT — clean the raw string before Pydantic tries to make it a float
@field_validator("price", mode="before")  # ✅
@classmethod
def clean_price(cls, v: Any) -> Any:
    ...
```

### Vague error messages

```python
# WRONG — the LLM doesn't know what to do
raise ValueError("invalid value")   # ❌

# RIGHT — tell the LLM exactly what is wrong and what is expected
raise ValueError(
    f"'{v}' is not a valid ISO 3166-1 alpha-2 country code. "
    "Expected exactly 2 uppercase letters, e.g. 'DE', 'US', 'FR'."
)  # ✅
```

---

## Related

- [Built-in Field Types & Validators](built-in-types.md) — reusable types the library ships (`IsoDateStr`, `IbanStr`, …)
- [Schema Design](schema-design.md) — designing schemas the LLM can fill reliably
- [Retry & Fallback](retry-and-fallback.md) — how the validation loop works end-to-end
- [`examples/06_pydantic_validators.py`](../examples/06_pydantic_validators.py) — all patterns in one runnable file
- [`tests/test_validators.py`](../tests/test_validators.py) — full test suite for all validator types
