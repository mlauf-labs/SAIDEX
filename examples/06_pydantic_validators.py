"""Example 6 — Pydantic validators: pre-processing, post-processing, and cross-field rules.

LLMs produce slightly inconsistent output — wrong date formats, currency symbols
in numeric fields, mixed-case values, etc.  Pydantic validators let you clean,
normalise, and verify that output *before* the extracted model reaches your code.

Four validator types are demonstrated here:

  1. @field_validator(mode='before') — clean / normalise raw LLM output
  2. @field_validator(mode='after')  — apply business rules after type coercion
  3. @model_validator(mode='before') — reshape the whole input dict upfront
  4. @model_validator(mode='after')  — cross-field consistency checks

Run:
    python examples/06_pydantic_validators.py
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator, model_validator

# ===========================================================================
# 1. @field_validator(mode='before') — clean raw LLM output
# ===========================================================================
# 'before' runs on the raw value that came from the LLM, before Pydantic
# tries to coerce the type.  Use it to handle quirky LLM output formats.


class Invoice(BaseModel):
    """Invoice data with pre-validators that tolerate messy LLM output."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="ignore")

    vendor: str
    amount: float = Field(ge=0, description="Total amount in EUR including VAT")
    currency: str = Field(description="ISO 4217 code, e.g. EUR")
    date: str = Field(description="Issue date as YYYY-MM-DD")
    paid: bool = Field(description="Whether the invoice has been paid")

    @field_validator("amount", mode="before")
    @classmethod
    def clean_amount(cls, v: Any) -> Any:
        """Accept '€ 1.234,56', '$42.50', '1234.56 EUR', or a plain number."""
        if isinstance(v, (int, float)):
            return v
        if isinstance(v, str):
            # Remove currency symbols and alphabetic suffixes
            cleaned = re.sub(r"[€$£¥\s]", "", v)
            cleaned = re.sub(r"[A-Za-z]+", "", cleaned).strip()
            # Handle European decimal notation: "1.234,56" → "1234.56"
            if re.search(r"\d{1,3}(\.\d{3})*(,\d+)$", cleaned):
                cleaned = cleaned.replace(".", "").replace(",", ".")
            elif "," in cleaned and "." not in cleaned:
                cleaned = cleaned.replace(",", ".")
            return cleaned
        return v

    @field_validator("currency", mode="before")
    @classmethod
    def normalise_currency(cls, v: Any) -> Any:
        """Accept 'eur', ' EUR ', 'Euro' — always returns uppercase ISO code."""
        if isinstance(v, str):
            mapping = {
                "euro": "EUR",
                "dollar": "USD",
                "pound": "GBP",
                "yen": "JPY",
                "franc": "CHF",
            }
            cleaned = v.strip().upper()
            return mapping.get(cleaned.lower(), cleaned)
        return v

    @field_validator("date", mode="before")
    @classmethod
    def normalise_date(cls, v: Any) -> Any:
        """Accept common date formats and convert to YYYY-MM-DD."""
        if not isinstance(v, str):
            return v
        v = v.strip()
        # already correct
        if re.match(r"^\d{4}-\d{2}-\d{2}$", v):
            return v
        # DD.MM.YYYY or DD/MM/YYYY
        m = re.match(r"^(\d{1,2})[./](\d{1,2})[./](\d{4})$", v)
        if m:
            return f"{m.group(3)}-{m.group(2).zfill(2)}-{m.group(1).zfill(2)}"
        # MM/DD/YYYY (US style) — heuristic: first number > 12 means it's a day
        m = re.match(r"^(\d{1,2})/(\d{1,2})/(\d{4})$", v)
        if m and int(m.group(1)) <= 12:
            return f"{m.group(3)}-{m.group(1).zfill(2)}-{m.group(2).zfill(2)}"
        return v

    @field_validator("paid", mode="before")
    @classmethod
    def normalise_bool(cls, v: Any) -> Any:
        """Accept 'yes'/'no', 'ja'/'nein', '1'/'0', True/False."""
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            if v.lower() in {"yes", "ja", "true", "1", "bezahlt", "paid"}:
                return True
            if v.lower() in {"no", "nein", "false", "0", "offen", "unpaid"}:
                return False
        return v


# ===========================================================================
# 2. @field_validator(mode='after') — business rules after type coercion
# ===========================================================================
# 'after' runs once Pydantic has already coerced the raw value to the
# declared type.  Use it for domain-specific rules and normalisation.


class ContactInfo(BaseModel):
    """Contact record with post-validators for normalisation and formatting."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="ignore")

    full_name: str = Field(description="Full name, first and last")
    email: str | None = Field(default=None, description="Email address or null")
    phone: str | None = Field(default=None, description="Phone number or null")
    country_code: str = Field(description="ISO 3166-1 alpha-2 country code, e.g. DE")
    website: str | None = Field(default=None, description="URL or null")

    @field_validator("email", mode="after")
    @classmethod
    def lowercase_email(cls, v: str | None) -> str | None:
        return v.lower() if isinstance(v, str) else v

    @field_validator("country_code", mode="after")
    @classmethod
    def uppercase_country(cls, v: str) -> str:
        return v.upper()

    @field_validator("phone", mode="after")
    @classmethod
    def normalise_phone(cls, v: str | None) -> str | None:
        """Strip formatting characters; keep only digits and leading '+'."""
        if not isinstance(v, str):
            return v
        # Keep digits and a single leading +
        digits = re.sub(r"[^\d+]", "", v)
        # Collapse multiple + signs
        return re.sub(r"\++", "+", digits) or None

    @field_validator("website", mode="after")
    @classmethod
    def ensure_https(cls, v: str | None) -> str | None:
        """Prepend https:// if the LLM omitted the scheme."""
        if isinstance(v, str) and not v.startswith(("http://", "https://")):
            return f"https://{v}"
        return v

    @field_validator("full_name", mode="after")
    @classmethod
    def title_case_name(cls, v: str) -> str:
        return v.strip().title()


# ===========================================================================
# 3. @model_validator(mode='before') — reshape the entire input dict
# ===========================================================================
# 'before' on the model level runs before any field validator.
# Use it when the LLM might return a slightly different structure than expected.


class Address(BaseModel):
    """Address with a model-level pre-validator that handles structural quirks."""

    street: str
    city: str
    zip_code: str = Field(description="Postal / ZIP code")
    country: str = Field(default="DE", description="ISO 3166-1 alpha-2, e.g. DE")

    @model_validator(mode="before")
    @classmethod
    def handle_flat_string(cls, data: Any) -> Any:
        """If the LLM returns a plain string, parse it into fields."""
        if isinstance(data, str):
            # "Hauptstraße 1, 80331 München, DE"
            parts = [p.strip() for p in data.split(",")]
            result: dict[str, Any] = {}
            if parts:
                result["street"] = parts[0]
            if len(parts) >= 2:
                zip_city = parts[1].split(None, 1)
                if len(zip_city) == 2:
                    result["zip_code"] = zip_city[0]
                    result["city"] = zip_city[1]
                else:
                    result["city"] = parts[1]
            if len(parts) >= 3:
                result["country"] = parts[2].upper()
            return result
        return data


# ===========================================================================
# 4. @model_validator(mode='after') — cross-field consistency rules
# ===========================================================================
# 'after' runs once all fields are populated and validated.
# Use it for rules that span multiple fields.


class LineItem(BaseModel):
    description: str
    quantity: int = Field(ge=1)
    unit_price: float = Field(ge=0)

    @computed_field  # type: ignore[misc]
    @property
    def subtotal(self) -> float:
        """Automatically computed from quantity × unit_price."""
        return round(self.quantity * self.unit_price, 2)


class Order(BaseModel):
    """Order with cross-field checks: date ordering and total consistency."""

    order_id: str
    order_date: str = Field(description="Order date as YYYY-MM-DD")
    delivery_date: str | None = Field(
        default=None,
        description="Expected delivery date as YYYY-MM-DD, or null",
    )
    items: list[LineItem] = Field(min_length=1)
    declared_total: float = Field(
        ge=0,
        description="Total as printed on the document, in EUR",
    )
    discount: float = Field(default=0.0, ge=0, description="Discount amount in EUR")

    @computed_field  # type: ignore[misc]
    @property
    def calculated_total(self) -> float:
        """Sum of all line-item subtotals minus discount."""
        return round(sum(i.subtotal for i in self.items) - self.discount, 2)

    @model_validator(mode="after")
    def check_delivery_after_order(self) -> Order:
        if self.delivery_date and self.delivery_date < self.order_date:
            raise ValueError(
                f"delivery_date ({self.delivery_date}) must not be before "
                f"order_date ({self.order_date})"
            )
        return self

    @model_validator(mode="after")
    def check_total_matches_items(self) -> Order:
        """Warn when declared_total differs from the sum of line items by > 1 %."""
        if self.calculated_total == 0:
            return self
        deviation = abs(self.declared_total - self.calculated_total) / self.calculated_total
        if deviation > 0.01:
            raise ValueError(
                f"declared_total ({self.declared_total}) deviates more than 1 % "
                f"from calculated_total ({self.calculated_total}). "
                f"Check the line items and totals."
            )
        return self


# ===========================================================================
# Demo
# ===========================================================================


def demo_invoice() -> None:
    print("=== Invoice — pre-validators ===\n")

    raw_inputs = [
        # Clean European notation
        {
            "vendor": "Muster GmbH",
            "amount": "€ 1.190,00",
            "currency": "Euro",
            "date": "05.01.2026",
            "paid": "ja",
        },
        # US dollar, US date
        {
            "vendor": "Acme Corp",
            "amount": "$42.50",
            "currency": "dollar",
            "date": "01/05/2026",
            "paid": "yes",
        },
        # Already clean
        {
            "vendor": "Clean Ltd.",
            "amount": 99.0,
            "currency": "EUR",
            "date": "2026-03-15",
            "paid": False,
        },
    ]

    for raw in raw_inputs:
        inv = Invoice(**raw)
        print(f"  Vendor:   {inv.vendor}")
        print(f"  Amount:   {inv.amount} {inv.currency}")
        print(f"  Date:     {inv.date}")
        print(f"  Paid:     {inv.paid}")
        print()


def demo_contact() -> None:
    print("=== ContactInfo — post-validators ===\n")

    contact = ContactInfo(
        full_name="  alice müller  ",
        email="  Alice.Mueller@EXAMPLE.COM  ",
        phone="+49 (0) 30 / 123 456-78",
        country_code="de",
        website="example.com/alice",
    )
    print(f"  Name:    {contact.full_name}")
    print(f"  Email:   {contact.email}")
    print(f"  Phone:   {contact.phone}")
    print(f"  Country: {contact.country_code}")
    print(f"  Website: {contact.website}")
    print()


def demo_address_from_string() -> None:
    print("=== Address — model-level pre-validator (string input) ===\n")

    addr = Address.model_validate("Hauptstraße 42, 80331 München, DE")
    print(f"  Street:   {addr.street}")
    print(f"  City:     {addr.city}")
    print(f"  ZIP:      {addr.zip_code}")
    print(f"  Country:  {addr.country}")
    print()


def demo_order() -> None:
    print("=== Order — cross-field model validators ===\n")

    order = Order(
        order_id="ORD-2026-001",
        order_date="2026-01-10",
        delivery_date="2026-01-15",
        items=[
            LineItem(description="Widget A", quantity=2, unit_price=10.0),
            LineItem(description="Widget B", quantity=1, unit_price=25.0),
        ],
        declared_total=44.50,  # within 1 % tolerance of 45.00
        discount=0.50,
    )

    print(f"  Order ID:          {order.order_id}")
    print(f"  Items:             {len(order.items)}")
    for item in order.items:
        print(f"    {item.description}: {item.quantity} × {item.unit_price} = {item.subtotal}")
    print(f"  Discount:          {order.discount}")
    print(f"  Declared total:    {order.declared_total}")
    print(f"  Calculated total:  {order.calculated_total}")
    print()

    print("  Testing cross-field rule: delivery_date < order_date …")
    try:
        Order(
            order_id="BAD",
            order_date="2026-01-10",
            delivery_date="2026-01-05",  # before order date — should fail
            items=[LineItem(description="X", quantity=1, unit_price=1.0)],
            declared_total=1.0,
        )
        print("  ERROR: should have raised!")
    except Exception as exc:
        print(f"  Correctly rejected: {exc}\n")


if __name__ == "__main__":
    demo_invoice()
    demo_contact()
    demo_address_from_string()
    demo_order()
