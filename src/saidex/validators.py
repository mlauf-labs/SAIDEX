"""Reusable Pydantic field validators and annotated types for extraction schemas.

These are building blocks for the *schemas* you ask an LLM to populate. When a
validator raises :class:`ValueError`, :func:`saidex.create_instance_safe` renders
it into structured correction guidance, so the retry loop can ask the model to
fix the offending field.

Example
-------
.. code-block:: python

    from pydantic import BaseModel, Field
    from saidex import ISODateStr

    class Event(BaseModel):
        starts_on: ISODateStr = Field(description="Start date")
        ends_on: ISODateStr | None = Field(None, description="End date, if any")

To build your own validated type, follow the same recipe: write a function that
returns the (optionally normalised) value or raises ``ValueError``, then wrap it
with :class:`pydantic.AfterValidator` in an ``Annotated`` alias. See the
"Build your own validated type" section of ``docs/built-in-types.md``.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Annotated

from pydantic import AfterValidator

from ._reference_data import ISO_639_1, ISO_3166_1_ALPHA_2, ISO_4217

__all__ = [
    "ISODateStr",
    "validate_iso_date",
    "IbanStr",
    "validate_iban",
    "VatIdStr",
    "validate_vat_id",
    "CountryCodeStr",
    "validate_country_code",
    "CurrencyCodeStr",
    "validate_currency_code",
    "IsinStr",
    "validate_isin",
    "PhoneStr",
    "validate_phone",
    "LanguageCodeStr",
    "validate_language_code",
]

_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_IBAN_RE = re.compile(r"^[A-Z]{2}\d{2}[A-Z0-9]{1,30}$")
_VAT_ID_RE = re.compile(r"^[A-Z]{2}[A-Z0-9]{2,12}$")
_ISIN_RE = re.compile(r"^[A-Z]{2}[A-Z0-9]{9}\d$")
_E164_RE = re.compile(r"^\+[1-9]\d{1,14}$")
_PHONE_STRIP_RE = re.compile(r"[\s()./-]")


def validate_iso_date(value: str) -> str:
    """Validate that *value* is a calendar date in ``yyyy-mm-dd`` format.

    The value is returned unchanged when valid (kept as a ``str`` rather than
    coerced to :class:`datetime.date`, so extracted values round-trip exactly as
    the model produced them). On failure a :class:`ValueError` with a clear,
    LLM-friendly message is raised. Two distinct failures are reported so the
    model gets actionable feedback:

    * wrong shape (e.g. ``"April 5th"`` or ``"2024-4-5"`` without zero-padding);
    * correct shape but an impossible calendar date (e.g. ``"2024-13-40"``).
    """
    if not isinstance(value, str) or not _ISO_DATE_RE.match(value):
        raise ValueError(
            f"'{value}' does not match the required date format yyyy-mm-dd "
            f"(four-digit year, zero-padded month and day, e.g. 2024-04-05)."
        )
    try:
        datetime.strptime(value, "%Y-%m-%d")
    except ValueError:
        raise ValueError(
            f"'{value}' has the right shape but is not a real calendar date. "
            f"Use a valid yyyy-mm-dd date, e.g. 2024-04-05."
        ) from None
    return value


ISODateStr = Annotated[str, AfterValidator(validate_iso_date)]
"""A ``str`` field constrained to the ISO ``yyyy-mm-dd`` calendar-date format."""


def validate_iban(value: str) -> str:
    """Validate and normalise an IBAN (International Bank Account Number).

    Whitespace is removed and letters upper-cased, then both the structural
    format *and* the ISO 7064 mod-97 check digits are verified. The normalised,
    space-free IBAN is returned. Raises :class:`ValueError` with LLM-friendly
    guidance on failure.
    """
    iban = re.sub(r"\s+", "", str(value)).upper()
    if not _IBAN_RE.match(iban):
        raise ValueError(
            f"'{value}' is not a structurally valid IBAN. Expected a 2-letter "
            f"country code, 2 check digits, then up to 30 alphanumeric characters, "
            f"e.g. 'DE89 3704 0044 0532 0130 00'."
        )
    # ISO 7064 mod-97: move the first four characters to the end, replace each
    # letter with two digits (A=10 … Z=35), then the integer must be ≡ 1 (mod 97).
    rearranged = iban[4:] + iban[:4]
    digits = "".join(str(int(ch, 36)) for ch in rearranged)
    if int(digits) % 97 != 1:
        raise ValueError(
            f"'{value}' has the right shape but its IBAN check digits are invalid "
            f"(failed the mod-97 checksum). Verify the digits were read correctly."
        )
    return iban


IbanStr = Annotated[str, AfterValidator(validate_iban)]
"""A ``str`` field constrained to a checksum-valid IBAN (normalised, no spaces)."""


def validate_vat_id(value: str) -> str:
    """Validate and normalise an EU-style VAT identification number.

    Spaces, dots, and hyphens are stripped and letters upper-cased, then the
    value is checked against the general EU pattern (2-letter country prefix
    followed by 2–12 alphanumeric characters). Returns the normalised VAT ID.
    Note: this is a *format* check, not a per-country length/checksum check.
    """
    vat = re.sub(r"[\s.\-]", "", str(value)).upper()
    if not _VAT_ID_RE.match(vat):
        raise ValueError(
            f"'{value}' is not a valid VAT identification number. Expected a "
            f"2-letter country prefix followed by 2–12 alphanumeric characters, "
            f"e.g. 'DE298471023' or 'ATU12345678'."
        )
    return vat


VatIdStr = Annotated[str, AfterValidator(validate_vat_id)]
"""A ``str`` field constrained to the EU VAT-ID format (normalised, upper-cased)."""


def validate_country_code(value: str) -> str:
    """Validate and normalise an ISO 3166-1 alpha-2 country code.

    The value is stripped and upper-cased, then checked against the official set
    of assigned alpha-2 codes (so ``"UK"`` and ``"XX"`` are rejected; the United
    Kingdom is ``"GB"``). Returns the normalised two-letter code.
    """
    code = str(value).strip().upper()
    if code not in ISO_3166_1_ALPHA_2:
        raise ValueError(
            f"'{value}' is not a valid ISO 3166-1 alpha-2 country code. Use the "
            f"official two-letter code, e.g. 'DE' (Germany), 'US', 'GB' (not 'UK')."
        )
    return code


CountryCodeStr = Annotated[str, AfterValidator(validate_country_code)]
"""A ``str`` field constrained to a valid ISO 3166-1 alpha-2 country code."""


def validate_currency_code(value: str) -> str:
    """Validate and normalise an ISO 4217 currency code.

    The value is stripped and upper-cased, then checked against the set of active
    ISO 4217 codes (plus the standard precious-metal/supranational codes).
    Returns the normalised three-letter code.
    """
    code = str(value).strip().upper()
    if code not in ISO_4217:
        raise ValueError(
            f"'{value}' is not a valid ISO 4217 currency code. Use the official "
            f"three-letter code, e.g. 'EUR', 'USD', 'GBP', 'JPY'."
        )
    return code


CurrencyCodeStr = Annotated[str, AfterValidator(validate_currency_code)]
"""A ``str`` field constrained to a valid ISO 4217 currency code."""


def validate_isin(value: str) -> str:
    """Validate and normalise an ISIN (International Securities Identification Number).

    Whitespace is removed and letters upper-cased, then both the structural
    format (2-letter country prefix, 9 alphanumeric characters, 1 check digit)
    *and* the ISO 6166 Luhn check digit are verified. Returns the normalised
    ISIN. Raises :class:`ValueError` with LLM-friendly guidance on failure.
    """
    isin = re.sub(r"\s+", "", str(value)).upper()
    if not _ISIN_RE.match(isin):
        raise ValueError(
            f"'{value}' is not a structurally valid ISIN. Expected a 2-letter "
            f"country code, 9 alphanumeric characters, and 1 check digit "
            f"(12 characters total), e.g. 'US0378331005'."
        )
    # ISO 6166: expand letters to digits (A=10 … Z=35), then a Luhn (mod-10)
    # checksum over the resulting digit string must be 0.
    digits = "".join(str(int(ch, 36)) for ch in isin)
    total = 0
    double = False
    for ch in reversed(digits):
        d = int(ch)
        if double:
            d *= 2
            if d > 9:
                d -= 9
        total += d
        double = not double
    if total % 10 != 0:
        raise ValueError(
            f"'{value}' has the right shape but its ISIN check digit is invalid "
            f"(failed the Luhn checksum). Verify the characters were read correctly."
        )
    return isin


IsinStr = Annotated[str, AfterValidator(validate_isin)]
"""A ``str`` field constrained to a checksum-valid ISIN (normalised, upper-cased)."""


def validate_phone(value: str) -> str:
    """Validate and normalise a phone number to E.164 format.

    Common separators (spaces, parentheses, dots, slashes, hyphens) are removed
    and a leading international ``00`` prefix is converted to ``+``. The result
    must be E.164: a ``+``, a non-zero leading digit, then up to 14 more digits.
    Returns the normalised number. A national number without a country code is
    rejected, since E.164 requires one.
    """
    s = _PHONE_STRIP_RE.sub("", str(value))
    if s.startswith("00"):
        s = "+" + s[2:]
    if not _E164_RE.match(s):
        raise ValueError(
            f"'{value}' is not a valid phone number in E.164 format. Expected a "
            f"'+', the country code, then the number with no spaces (max 15 digits "
            f"total), e.g. '+496971402200'. Include the country code."
        )
    return s


PhoneStr = Annotated[str, AfterValidator(validate_phone)]
"""A ``str`` field constrained to an E.164 phone number (normalised, ``+`` prefix)."""


def validate_language_code(value: str) -> str:
    """Validate and normalise an ISO 639-1 two-letter language code.

    The value is stripped and lower-cased, then checked against the official set
    of ISO 639-1 codes (so ``"english"`` and ``"gb"`` are rejected; English is
    ``"en"``). Returns the normalised two-letter code.
    """
    code = str(value).strip().lower()
    if code not in ISO_639_1:
        raise ValueError(
            f"'{value}' is not a valid ISO 639-1 language code. Use the official "
            f"two-letter code, e.g. 'en' (English), 'de' (German), 'fr' (French)."
        )
    return code


LanguageCodeStr = Annotated[str, AfterValidator(validate_language_code)]
"""A ``str`` field constrained to a valid ISO 639-1 language code."""
