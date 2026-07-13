"""Tests for create_instance_safe and _format_validation_error."""

from enum import Enum

from pydantic import BaseModel, Field

from saidex.utils import create_instance_safe, create_instance_with_issues

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


class Simple(BaseModel):
    name: str
    age: int


class WithOptional(BaseModel):
    name: str
    nickname: str | None = None


class WithEnum(BaseModel):
    class Status(str, Enum):
        ACTIVE = "active"
        INACTIVE = "inactive"

    status: Status


class WithConstraints(BaseModel):
    score: int = Field(ge=0, le=100)


class Nested(BaseModel):
    label: str
    count: int


class WithNested(BaseModel):
    title: str
    items: list[Nested]


class WithSchemaField(BaseModel):
    """A model whose field is literally named ``schema`` (regression, issue: schema= collision)."""

    schema: str
    table: str


# ---------------------------------------------------------------------------
# Success cases
# ---------------------------------------------------------------------------


def test_valid_simple() -> None:
    instance, error = create_instance_safe(Simple, name="Alice", age=30)
    assert instance is not None
    assert error is None
    assert instance.name == "Alice"
    assert instance.age == 30


def test_valid_with_optional_omitted() -> None:
    instance, error = create_instance_safe(WithOptional, name="Bob")
    assert instance is not None
    assert instance.nickname is None


def test_valid_with_optional_provided() -> None:
    instance, error = create_instance_safe(WithOptional, name="Bob", nickname="Bobby")
    assert instance is not None
    assert instance.nickname == "Bobby"


def test_valid_enum() -> None:
    instance, error = create_instance_safe(WithEnum, status="active")
    assert instance is not None
    assert instance.status == WithEnum.Status.ACTIVE


def test_valid_with_constraints() -> None:
    instance, error = create_instance_safe(WithConstraints, score=50)
    assert instance is not None
    assert error is None


def test_field_named_schema_via_create_instance_safe() -> None:
    """A model field literally named 'schema' must not collide with the schema parameter."""
    instance, error = create_instance_safe(WithSchemaField, schema="public", table="t")
    assert error is None
    assert instance is not None
    assert instance.schema == "public"
    assert instance.table == "t"


def test_field_named_schema_via_create_instance_with_issues() -> None:
    instance, issues, error = create_instance_with_issues(
        WithSchemaField, schema="public", table="t"
    )
    assert error is None
    assert issues == []
    assert instance is not None
    assert instance.schema == "public"


# ---------------------------------------------------------------------------
# Failure cases — error text content
# ---------------------------------------------------------------------------


def test_missing_required_field() -> None:
    instance, error = create_instance_safe(Simple, age=30)
    assert instance is None
    assert error is not None
    assert "MISSING" in error.upper()
    assert "name" in error


def test_wrong_type() -> None:
    instance, error = create_instance_safe(Simple, name="Alice", age="not_a_number")
    assert instance is None
    assert error is not None
    assert "age" in error


def test_invalid_enum_value() -> None:
    instance, error = create_instance_safe(WithEnum, status="pending")
    assert instance is None
    assert error is not None
    assert "pending" in error or "status" in error


def test_constraint_violation() -> None:
    instance, error = create_instance_safe(WithConstraints, score=150)
    assert instance is None
    assert error is not None
    assert "score" in error


def test_error_contains_schema_info() -> None:
    instance, error = create_instance_safe(Simple, age=30)
    assert instance is None
    assert error is not None
    assert "Simple" in error
    assert "name" in error


def test_multiple_errors_reported() -> None:
    instance, error = create_instance_safe(Simple)
    assert instance is None
    assert error is not None
    assert "name" in error
    assert "age" in error
