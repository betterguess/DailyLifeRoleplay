import os
from pathlib import Path
from urllib.parse import quote_plus
from contextlib import contextmanager
from collections.abc import Iterator

try:
    import streamlit as st
except Exception:  # pragma: no cover
    st = None

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker


DB_PATH = Path("data/app.db")


class Base(DeclarativeBase):
    pass


def _first_env(*names: str) -> str:
    for name in names:
        value = os.environ.get(name)
        if value is not None and value.strip():
            return value.strip()
    return ""


def _first_setting(*names: str) -> str:
    env_value = _first_env(*names)
    if env_value:
        return env_value

    if st is None:
        return ""

    for name in names:
        try:
            value = st.secrets.get(name)
        except Exception:
            value = None
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _database_url_from_env() -> str:
    direct_url = _first_setting("DATABASE_URL", "SQLALCHEMY_DATABASE_URL")
    if direct_url:
        return direct_url

    pg_host = _first_setting("POSTGRES_HOST", "PGHOST", "PGSQL_HOST", "PSQL_HOST")
    pg_user = _first_setting("POSTGRES_USER", "PGUSER", "PGSQL_USER", "PSQL_USER", "PSQL_User")
    pg_pass = _first_setting("POSTGRES_PASSWORD", "PGPASSWORD", "PGSQL_PASS", "PSQL_PASS", "PSQL_Pass")

    if pg_host and pg_user and pg_pass:
        pg_port = _first_setting("POSTGRES_PORT", "PGPORT", "PGSQL_PORT", "PSQL_PORT") or "5432"
        pg_db = _first_setting("POSTGRES_DB", "PGDATABASE", "PGSQL_DB", "PSQL_DB") or "dailyliferoleplay"
        pg_sslmode = _first_setting("POSTGRES_SSLMODE", "PGSSLMODE", "PGSQL_SSLMODE", "PSQL_SSLMODE")
        query = f"?sslmode={quote_plus(pg_sslmode)}" if pg_sslmode else ""
        return (
            "postgresql+psycopg://"
            f"{quote_plus(pg_user)}:{quote_plus(pg_pass)}@{pg_host}:{pg_port}/{quote_plus(pg_db)}{query}"
        )

    return f"sqlite:///{DB_PATH}"


DATABASE_URL = _database_url_from_env()


def _engine():
    if DATABASE_URL.startswith("sqlite"):
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return create_engine(DATABASE_URL, future=True)


ENGINE = _engine()
SessionLocal = sessionmaker(bind=ENGINE, future=True)


@contextmanager
def get_session() -> Iterator[Session]:
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
