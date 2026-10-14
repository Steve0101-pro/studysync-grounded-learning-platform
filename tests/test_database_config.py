from sqlalchemy.dialects import postgresql
from sqlalchemy.schema import CreateIndex

from app.database import normalize_database_url
from app.models import User


def test_supabase_pooler_database_url_uses_asyncpg_and_required_tls():
    url = normalize_database_url(
        "postgresql://postgres.projectref:password@aws-0-us-east-1.pooler.supabase.com:6543/postgres"
    )

    assert url.drivername == "postgresql+asyncpg"
    assert url.query["ssl"] == "require"
    assert url.port == 6543
    assert url.password == "password"


def test_supabase_sslmode_is_translated_for_asyncpg():
    url = normalize_database_url(
        "postgresql://postgres:password@db.projectref.supabase.co:5432/postgres?sslmode=require"
    )

    assert url.query["ssl"] == "require"
    assert "sslmode" not in url.query


def test_database_url_rejects_unsupported_dialects():
    try:
        normalize_database_url("sqlite:///local.db")
    except ValueError as exc:
        assert "PostgreSQL" in str(exc)
    else:
        raise AssertionError("Expected unsupported database dialect to be rejected")


def test_users_have_case_insensitive_unique_email_index():
    index = next(index for index in User.__table__.indexes if index.name == "uq_users_email_lower")
    statement = str(CreateIndex(index).compile(dialect=postgresql.dialect()))

    assert "UNIQUE INDEX" in statement
    assert "lower(email)" in statement
