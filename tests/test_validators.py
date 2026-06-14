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


# ===========================================================================
# ISODateStr — reusable yyyy-mm-dd date type exported by the library
# ===========================================================================

from pydantic import BaseModel  # noqa: E402

from saidex import ISODateStr, validate_iso_date  # noqa: E402


class _Event(BaseModel):
    starts_on: ISODateStr
    ends_on: ISODateStr | None = None


class TestISODateStr:
    @pytest.mark.parametrize("value", ["2024-04-05", "1992-12-31", "2000-01-01"])
    def test_valid_dates_pass_through_unchanged(self, value: str) -> None:
        # Returned as a plain str, not coerced to datetime.date.
        assert validate_iso_date(value) == value
        event = _Event(starts_on=value)
        assert event.starts_on == value
        assert isinstance(event.starts_on, str)

    def test_optional_none_allowed(self) -> None:
        event = _Event(starts_on="2024-04-05", ends_on=None)
        assert event.ends_on is None

    @pytest.mark.parametrize(
        "value",
        ["2024-4-5", "April 5th", "05.04.2024", "31/12/2026", "2024-04-05T00:00:00", ""],
    )
    def test_wrong_format_rejected(self, value: str) -> None:
        with pytest.raises(ValueError, match="format yyyy-mm-dd"):
            validate_iso_date(value)

    @pytest.mark.parametrize("value", ["2024-13-40", "2024-00-10", "2024-02-30"])
    def test_impossible_calendar_date_rejected(self, value: str) -> None:
        with pytest.raises(ValueError, match="not a real calendar date"):
            validate_iso_date(value)

    def test_create_instance_safe_emits_guidance(self) -> None:
        instance, error = create_instance_safe(_Event, starts_on="2024-13-40")
        assert instance is None
        assert error is not None
        assert "starts_on" in error
        assert "calendar date" in error


# ===========================================================================
# IbanStr / VatIdStr / CountryCodeStr / CurrencyCodeStr — library field types
# ===========================================================================

from saidex import (  # noqa: E402
    CountryCodeStr,
    CurrencyCodeStr,
    IbanStr,
    VatIdStr,
    validate_country_code,
    validate_currency_code,
    validate_iban,
    validate_isin,
    validate_language_code,
    validate_phone,
    validate_vat_id,
)


class _BankAccount(BaseModel):
    iban: IbanStr
    vat_id: VatIdStr | None = None
    country: CountryCodeStr | None = None
    currency: CurrencyCodeStr | None = None


class TestIbanStr:
    @pytest.mark.parametrize(
        "raw,normalised",
        [
            ("DE89 3704 0044 0532 0130 00", "DE89370400440532013000"),
            ("de89370400440532013000", "DE89370400440532013000"),
            ("GB82 WEST 1234 5698 7654 32", "GB82WEST12345698765432"),
        ],
    )
    def test_valid_iban_normalised(self, raw: str, normalised: str) -> None:
        assert validate_iban(raw) == normalised
        assert _BankAccount(iban=raw).iban == normalised

    def test_bad_checksum_rejected(self) -> None:
        # Structurally fine, but the mod-97 check digits are wrong (fabricated).
        with pytest.raises(ValueError, match="checksum"):
            validate_iban("DE42 1004 0000 0287 8000 04")

    @pytest.mark.parametrize("value", ["not-an-iban", "1234", "D8 89 ABC", ""])
    def test_malformed_iban_rejected(self, value: str) -> None:
        with pytest.raises(ValueError, match="structurally valid IBAN"):
            validate_iban(value)


class TestVatIdStr:
    @pytest.mark.parametrize(
        "raw,normalised",
        [
            ("DE298471023", "DE298471023"),
            ("DE 301 847 192", "DE301847192"),
            ("atu12345678", "ATU12345678"),
        ],
    )
    def test_valid_vat_normalised(self, raw: str, normalised: str) -> None:
        assert validate_vat_id(raw) == normalised

    @pytest.mark.parametrize("value", ["XX", "12345", "D1", ""])
    def test_invalid_vat_rejected(self, value: str) -> None:
        with pytest.raises(ValueError, match="VAT identification number"):
            validate_vat_id(value)


class TestCountryCodeStr:
    @pytest.mark.parametrize("raw,normalised", [("de", "DE"), ("US", "US"), (" gb ", "GB")])
    def test_valid_country_normalised(self, raw: str, normalised: str) -> None:
        assert validate_country_code(raw) == normalised

    @pytest.mark.parametrize("value", ["UK", "XX", "Germany", "D"])
    def test_invalid_country_rejected(self, value: str) -> None:
        with pytest.raises(ValueError, match="ISO 3166-1 alpha-2"):
            validate_country_code(value)


class TestCurrencyCodeStr:
    @pytest.mark.parametrize("raw,normalised", [("eur", "EUR"), ("USD", "USD"), (" gbp ", "GBP")])
    def test_valid_currency_normalised(self, raw: str, normalised: str) -> None:
        assert validate_currency_code(raw) == normalised

    @pytest.mark.parametrize("value", ["XYZ", "EURO", "€", ""])
    def test_invalid_currency_rejected(self, value: str) -> None:
        with pytest.raises(ValueError, match="ISO 4217"):
            validate_currency_code(value)


class TestIsinStr:
    @pytest.mark.parametrize(
        "raw,normalised",
        [
            ("US0378331005", "US0378331005"),  # Apple
            ("us 5949181045", "US5949181045"),  # Microsoft, lower + spaces
            ("DE0005140008", "DE0005140008"),  # Deutsche Bank
        ],
    )
    def test_valid_isin_normalised(self, raw: str, normalised: str) -> None:
        assert validate_isin(raw) == normalised

    def test_bad_check_digit_rejected(self) -> None:
        with pytest.raises(ValueError, match="Luhn checksum"):
            validate_isin("US0378331006")  # last digit wrong

    @pytest.mark.parametrize("value", ["12345", "USABC", "0378331005", ""])
    def test_malformed_isin_rejected(self, value: str) -> None:
        with pytest.raises(ValueError, match="structurally valid ISIN"):
            validate_isin(value)


class TestPhoneStr:
    @pytest.mark.parametrize(
        "raw,normalised",
        [
            ("+49 69 7140 2200", "+496971402200"),
            ("+1 (415) 555-0132", "+14155550132"),
            ("0049 30 1234567", "+49301234567"),  # 00 -> +
        ],
    )
    def test_valid_phone_normalised(self, raw: str, normalised: str) -> None:
        assert validate_phone(raw) == normalised

    @pytest.mark.parametrize("value", ["069 7140 2200", "abc", "+0123", ""])
    def test_invalid_phone_rejected(self, value: str) -> None:
        # National numbers without a country code are rejected (E.164 needs one).
        with pytest.raises(ValueError, match="E.164"):
            validate_phone(value)


class TestLanguageCodeStr:
    @pytest.mark.parametrize("raw,normalised", [("en", "en"), ("DE", "de"), (" fr ", "fr")])
    def test_valid_language_normalised(self, raw: str, normalised: str) -> None:
        assert validate_language_code(raw) == normalised

    @pytest.mark.parametrize("value", ["english", "gb", "xx", "e"])
    def test_invalid_language_rejected(self, value: str) -> None:
        with pytest.raises(ValueError, match="ISO 639-1"):
            validate_language_code(value)


class TestFieldTypesInModel:
    def test_full_valid_model_normalises_all_fields(self) -> None:
        acc = _BankAccount(
            iban="de89 3704 0044 0532 0130 00",
            vat_id="DE 298 471 023",
            country="de",
            currency="eur",
        )
        assert acc.iban == "DE89370400440532013000"
        assert acc.vat_id == "DE298471023"
        assert acc.country == "DE"
        assert acc.currency == "EUR"

    def test_create_instance_safe_reports_field_path(self) -> None:
        instance, error = create_instance_safe(
            _BankAccount, iban="DE89370400440532013000", country="UK"
        )
        assert instance is None
        assert error is not None
        assert "country" in error
