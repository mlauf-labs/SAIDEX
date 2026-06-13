"""Tests for Pydantic validator patterns used with saidex.

Validates that the schemas in examples/06_pydantic_validators.py behave
correctly — both the happy path and expected rejection cases.
All tests run without an LLM; they use create_instance_safe or direct
model instantiation.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
from pydantic import ValidationError


def _load_example(name: str) -> object:
    """Load an examples/ file whose name starts with a digit via importlib."""
    path = Path(__file__).parent.parent / "examples" / name
    spec = importlib.util.spec_from_file_location("_example_validators", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


_ex = _load_example("06_pydantic_validators.py")

Address = _ex.Address  # type: ignore[attr-defined]
ContactInfo = _ex.ContactInfo  # type: ignore[attr-defined]
Invoice = _ex.Invoice  # type: ignore[attr-defined]
LineItem = _ex.LineItem  # type: ignore[attr-defined]
Order = _ex.Order  # type: ignore[attr-defined]

# Pydantic forward-references don't resolve automatically across importlib
# boundaries — rebuild models that reference other models from the same file.
Order.model_rebuild()  # type: ignore[attr-defined]

from saidex import create_instance_safe  # noqa: E402  (import after example-module setup)

# ===========================================================================
# Invoice — pre-validators
# ===========================================================================


class TestInvoicePreValidators:
    def test_clean_european_amount(self) -> None:
        inv = Invoice(
            vendor="X", amount="€ 1.190,00", currency="EUR", date="2026-01-01", paid=False
        )
        assert inv.amount == pytest.approx(1190.0)

    def test_clean_dollar_amount(self) -> None:
        inv = Invoice(vendor="X", amount="$42.50", currency="USD", date="2026-01-01", paid=False)
        assert inv.amount == pytest.approx(42.50)

    def test_clean_amount_with_spaces(self) -> None:
        inv = Invoice(
            vendor="X", amount="  1 234.56  ", currency="EUR", date="2026-01-01", paid=False
        )
        assert inv.amount == pytest.approx(1234.56)

    def test_plain_float_amount_unchanged(self) -> None:
        inv = Invoice(vendor="X", amount=99.0, currency="EUR", date="2026-01-01", paid=False)
        assert inv.amount == pytest.approx(99.0)

    def test_currency_normalised_to_uppercase(self) -> None:
        inv = Invoice(vendor="X", amount=1.0, currency="eur", date="2026-01-01", paid=False)
        assert inv.currency == "EUR"

    def test_currency_word_to_code(self) -> None:
        inv = Invoice(vendor="X", amount=1.0, currency="Euro", date="2026-01-01", paid=False)
        assert inv.currency == "EUR"

    def test_date_german_format(self) -> None:
        inv = Invoice(vendor="X", amount=1.0, currency="EUR", date="05.01.2026", paid=False)
        assert inv.date == "2026-01-05"

    def test_date_slash_format(self) -> None:
        inv = Invoice(vendor="X", amount=1.0, currency="EUR", date="31/12/2026", paid=False)
        assert inv.date == "2026-12-31"

    def test_date_already_iso_unchanged(self) -> None:
        inv = Invoice(vendor="X", amount=1.0, currency="EUR", date="2026-06-03", paid=False)
        assert inv.date == "2026-06-03"

    @pytest.mark.parametrize("raw", ["yes", "ja", "true", "1", "paid", "bezahlt"])
    def test_bool_truthy_strings(self, raw: str) -> None:
        inv = Invoice(vendor="X", amount=1.0, currency="EUR", date="2026-01-01", paid=raw)
        assert inv.paid is True

    @pytest.mark.parametrize("raw", ["no", "nein", "false", "0", "offen", "unpaid"])
    def test_bool_falsy_strings(self, raw: str) -> None:
        inv = Invoice(vendor="X", amount=1.0, currency="EUR", date="2026-01-01", paid=raw)
        assert inv.paid is False

    def test_negative_amount_rejected(self) -> None:
        instance, error = create_instance_safe(
            Invoice,
            vendor="X",
            amount=-5.0,
            currency="EUR",
            date="2026-01-01",
            paid=False,
        )
        assert instance is None
        assert error is not None
        assert "amount" in error


# ===========================================================================
# ContactInfo — post-validators
# ===========================================================================


class TestContactInfoPostValidators:
    def test_email_lowercased(self) -> None:
        c = ContactInfo(full_name="Alice", email="Alice@EXAMPLE.COM", country_code="DE")
        assert c.email == "alice@example.com"

    def test_email_none_allowed(self) -> None:
        c = ContactInfo(full_name="Alice", email=None, country_code="DE")
        assert c.email is None

    def test_country_uppercased(self) -> None:
        c = ContactInfo(full_name="Alice", country_code="de")
        assert c.country_code == "DE"

    def test_phone_normalised(self) -> None:
        c = ContactInfo(full_name="Alice", phone="+49 (0) 30 / 123 456-78", country_code="DE")
        assert c.phone == "+49030123456-78".replace("-", "")

    def test_phone_none_allowed(self) -> None:
        c = ContactInfo(full_name="Alice", phone=None, country_code="DE")
        assert c.phone is None

    def test_website_gets_https(self) -> None:
        c = ContactInfo(full_name="Alice", website="example.com", country_code="DE")
        assert c.website == "https://example.com"

    def test_website_with_existing_https_unchanged(self) -> None:
        c = ContactInfo(full_name="Alice", website="https://example.com", country_code="DE")
        assert c.website == "https://example.com"

    def test_name_title_cased(self) -> None:
        c = ContactInfo(full_name="  alice müller  ", country_code="DE")
        assert c.full_name == "Alice Müller"

    def test_str_strip_applied_via_config(self) -> None:
        c = ContactInfo(full_name="Bob", email="  bob@example.com  ", country_code="DE")
        # ConfigDict(str_strip_whitespace=True) strips before validator runs
        assert c.email == "bob@example.com"


# ===========================================================================
# Address — model-level pre-validator
# ===========================================================================


class TestAddressModelPreValidator:
    def test_dict_input_works_normally(self) -> None:
        addr = Address(street="Hauptstraße 1", city="München", zip_code="80331", country="DE")
        assert addr.city == "München"

    def test_string_input_parsed(self) -> None:
        addr = Address.model_validate("Hauptstraße 42, 80331 München, DE")
        assert addr.street == "Hauptstraße 42"
        assert addr.zip_code == "80331"
        assert addr.city == "München"
        assert addr.country == "DE"

    def test_string_input_country_uppercased(self) -> None:
        addr = Address.model_validate("Main St 1, 10001 New York, us")
        assert addr.country == "US"

    def test_create_instance_safe_with_dict(self) -> None:
        instance, error = create_instance_safe(
            Address,
            street="Elm St",
            city="Springfield",
            zip_code="12345",
            country="US",
        )
        assert instance is not None
        assert error is None


# ===========================================================================
# LineItem — computed_field
# ===========================================================================


class TestLineItemComputedField:
    def test_subtotal_computed(self) -> None:
        item = LineItem(description="Widget", quantity=3, unit_price=10.0)
        assert item.subtotal == pytest.approx(30.0)

    def test_subtotal_rounds_to_two_decimals(self) -> None:
        item = LineItem(description="Widget", quantity=3, unit_price=0.1)
        assert item.subtotal == pytest.approx(0.3, abs=1e-9)

    def test_subtotal_in_model_dump(self) -> None:
        item = LineItem(description="Widget", quantity=2, unit_price=5.5)
        dumped = item.model_dump()
        assert "subtotal" in dumped
        assert dumped["subtotal"] == pytest.approx(11.0)


# ===========================================================================
# Order — cross-field model validators
# ===========================================================================


def _valid_order(**overrides: object) -> dict:
    base: dict = {
        "order_id": "ORD-001",
        "order_date": "2026-01-10",
        "delivery_date": "2026-01-15",
        "items": [LineItem(description="X", quantity=2, unit_price=10.0)],
        "declared_total": 20.0,
        "discount": 0.0,
    }
    base.update(overrides)
    return base


class TestOrderModelValidators:
    def test_valid_order_accepted(self) -> None:
        order = Order(**_valid_order())
        assert order.calculated_total == pytest.approx(20.0)

    def test_delivery_before_order_rejected(self) -> None:
        with pytest.raises(ValidationError, match="delivery_date"):
            Order(
                **_valid_order(
                    order_date="2026-01-10",
                    delivery_date="2026-01-05",
                )
            )

    def test_delivery_same_as_order_accepted(self) -> None:
        order = Order(
            **_valid_order(
                order_date="2026-01-10",
                delivery_date="2026-01-10",
            )
        )
        assert order.delivery_date == "2026-01-10"

    def test_delivery_none_accepted(self) -> None:
        order = Order(**_valid_order(delivery_date=None))
        assert order.delivery_date is None

    def test_total_matches_items(self) -> None:
        order = Order(
            **_valid_order(
                items=[LineItem(description="A", quantity=3, unit_price=10.0)],
                declared_total=30.0,
            )
        )
        assert order.calculated_total == pytest.approx(30.0)

    def test_total_within_tolerance_accepted(self) -> None:
        # 1 % tolerance: declared 30.20 vs calculated 30.00 → 0.67 % — ok
        order = Order(
            **_valid_order(
                items=[LineItem(description="A", quantity=3, unit_price=10.0)],
                declared_total=30.20,
            )
        )
        assert order is not None

    def test_total_outside_tolerance_rejected(self) -> None:
        instance, error = create_instance_safe(
            Order,
            **_valid_order(
                items=[LineItem(description="A", quantity=3, unit_price=10.0)],
                declared_total=25.0,  # 16 % off — should fail
            ),
        )
        assert instance is None
        assert error is not None
        assert "declared_total" in error or "calculated_total" in error

    def test_discount_applied_to_calculated_total(self) -> None:
        order = Order(
            **_valid_order(
                items=[LineItem(description="A", quantity=2, unit_price=10.0)],
                discount=2.0,
                declared_total=18.0,
            )
        )
        assert order.calculated_total == pytest.approx(18.0)

    def test_empty_items_rejected(self) -> None:
        instance, error = create_instance_safe(Order, **_valid_order(items=[]))
        assert instance is None
        assert error is not None

    def test_create_instance_safe_captures_cross_field_error(self) -> None:
        instance, error = create_instance_safe(
            Order,
            **_valid_order(
                order_date="2026-01-10",
                delivery_date="2026-01-01",
            ),
        )
        assert instance is None
        assert error is not None
