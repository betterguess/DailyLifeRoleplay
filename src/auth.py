import base64
import hashlib
import hmac
import json
import os
import secrets
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus

try:
    import streamlit as st
except Exception:  # pragma: no cover
    st = None

from sqlalchemy import Integer, String, Text, create_engine, func, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column


DB_PATH = Path("data/app.db")
PBKDF2_ITERATIONS = 210_000

ROLE_PATIENT = "patient"
ROLE_THERAPIST = "therapist"
ROLE_MANAGER = "manager"
ROLE_DEVELOPER = "developer"

LOCAL_ROLES = {ROLE_PATIENT, ROLE_DEVELOPER}
STAFF_ROLES = {ROLE_THERAPIST, ROLE_MANAGER, ROLE_DEVELOPER}
ALL_ROLES = {ROLE_PATIENT, ROLE_THERAPIST, ROLE_MANAGER, ROLE_DEVELOPER}

PERMISSIONS = {
    ROLE_PATIENT: {"use_program"},
    ROLE_THERAPIST: {"use_program", "view_assigned_patients", "create_roleplay", "view_progress"},
    ROLE_MANAGER: {"view_all_therapists", "view_user_data", "view_collected_data", "view_progress"},
    ROLE_DEVELOPER: {"*"},
}


@dataclass(frozen=True)
class User:
    username: str
    display_name: str
    role: str
    auth_source: str
    therapist_username: str | None


class Base(DeclarativeBase):
    pass


class UserRow(Base):
    __tablename__ = "users"

    username: Mapped[str] = mapped_column(String, primary_key=True)
    display_name: Mapped[str] = mapped_column(String, nullable=False)
    role: Mapped[str] = mapped_column(String, nullable=False)
    auth_source: Mapped[str] = mapped_column(String, nullable=False)
    password_hash: Mapped[str | None] = mapped_column(Text, nullable=True)
    therapist_username: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[str] = mapped_column(String, nullable=False)


class ActivityLogRow(Base):
    __tablename__ = "activity_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String, nullable=False)
    event_type: Mapped[str] = mapped_column(String, nullable=False)
    payload: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[str] = mapped_column(String, nullable=False)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def init_auth_store() -> None:
    Base.metadata.create_all(ENGINE)
    _bootstrap_if_empty()


def _bootstrap_if_empty() -> None:
    with Session(ENGINE) as session:
        count = session.scalar(select(func.count()).select_from(UserRow)) or 0
        if int(count) > 0:
            return

    create_local_user(
        username="devadmin",
        password="changeme123",
        role=ROLE_DEVELOPER,
        display_name="Developer Admin",
    )


def _hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS)
    salt_b64 = base64.b64encode(salt).decode("ascii")
    digest_b64 = base64.b64encode(digest).decode("ascii")
    return f"pbkdf2_sha256${PBKDF2_ITERATIONS}${salt_b64}${digest_b64}"


def _verify_password(password: str, stored_hash: str | None) -> bool:
    if not stored_hash:
        return False
    try:
        algo, rounds_s, salt_b64, digest_b64 = stored_hash.split("$", 3)
        if algo != "pbkdf2_sha256":
            return False
        rounds = int(rounds_s)
        salt = base64.b64decode(salt_b64.encode("ascii"))
        expected = base64.b64decode(digest_b64.encode("ascii"))
        actual = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, rounds)
        return hmac.compare_digest(actual, expected)
    except Exception:
        return False


def create_local_user(
    username: str,
    password: str,
    role: str,
    display_name: str,
    therapist_username: str | None = None,
) -> None:
    if role not in LOCAL_ROLES:
        raise ValueError("Kun patient eller developer kan oprettes lokalt.")
    if role == ROLE_PATIENT and not therapist_username:
        raise ValueError("Patienter skal tildeles en terapeut.")
    if len(password) < 8:
        raise ValueError("Kodeord skal være mindst 8 tegn.")

    username = username.strip().lower()
    if not username:
        raise ValueError("Brugernavn må ikke være tomt.")

    row = UserRow(
        username=username,
        display_name=display_name.strip() or username,
        role=role,
        auth_source="local",
        password_hash=_hash_password(password),
        therapist_username=therapist_username.strip().lower() if therapist_username else None,
        created_at=_utc_now(),
    )
    with Session(ENGINE) as session:
        existing = session.get(UserRow, username)
        if existing:
            raise ValueError("Brugernavn findes allerede.")
        session.add(row)
        session.commit()


def _to_user(row: UserRow) -> User:
    return User(
        username=row.username,
        display_name=row.display_name,
        role=row.role,
        auth_source=row.auth_source,
        therapist_username=row.therapist_username,
    )


def authenticate_local_user(username: str, password: str) -> User | None:
    username = username.strip().lower()
    with Session(ENGINE) as session:
        row = session.get(UserRow, username)
    if row is None or row.auth_source != "local":
        return None
    if not _verify_password(password, row.password_hash):
        return None
    return _to_user(row)


def sso_domain_allowed(email: str) -> bool:
    required = os.environ.get("STAFF_EMAIL_DOMAIN", "").strip().lower()
    if not required:
        return True
    return email.strip().lower().endswith(f"@{required}")


def _role_overrides() -> dict[str, str]:
    raw = os.environ.get("STAFF_ROLE_OVERRIDES_JSON", "").strip()
    if not raw:
        return {}
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            cleaned: dict[str, str] = {}
            for key, value in data.items():
                k = str(key).strip().lower()
                v = str(value).strip().lower()
                if k and v in STAFF_ROLES:
                    cleaned[k] = v
            return cleaned
    except Exception:
        return {}
    return {}


def provision_sso_user(email: str, requested_role: str) -> User:
    email = email.strip().lower()
    if requested_role not in STAFF_ROLES:
        raise ValueError("Ugyldig ansat-rolle.")
    if not sso_domain_allowed(email):
        raise ValueError("Email-domænet er ikke tilladt for ansat-login.")

    role = _role_overrides().get(email, requested_role)
    display = email.split("@", 1)[0].replace(".", " ").title()

    with Session(ENGINE) as session:
        row = session.get(UserRow, email)
        if row is None:
            row = UserRow(
                username=email,
                display_name=display,
                role=role,
                auth_source="sso",
                password_hash=None,
                therapist_username=None,
                created_at=_utc_now(),
            )
            session.add(row)
        else:
            row.display_name = display
            row.role = role
            row.auth_source = "sso"
            row.password_hash = None
            row.therapist_username = None
        session.commit()
        session.refresh(row)

    return _to_user(row)


def has_permission(role: str, permission: str) -> bool:
    allowed = PERMISSIONS.get(role, set())
    return "*" in allowed or permission in allowed


def log_event(username: str, event_type: str, payload: dict[str, Any] | None = None) -> None:
    with Session(ENGINE) as session:
        session.add(
            ActivityLogRow(
                username=username,
                event_type=event_type,
                payload=json.dumps(payload or {}, ensure_ascii=False),
                created_at=_utc_now(),
            )
        )
        session.commit()


def list_users() -> list[User]:
    with Session(ENGINE) as session:
        rows = session.scalars(select(UserRow).order_by(UserRow.role, UserRow.username)).all()
    return [_to_user(r) for r in rows]


def get_staff_users() -> list[User]:
    return [u for u in list_users() if u.role in STAFF_ROLES]


def get_therapists() -> list[User]:
    return [u for u in list_users() if u.role == ROLE_THERAPIST]


def get_patients_for_therapist(therapist_username: str) -> list[User]:
    therapist_username = therapist_username.strip().lower()
    with Session(ENGINE) as session:
        rows = session.scalars(
            select(UserRow)
            .where(
                UserRow.role == ROLE_PATIENT,
                UserRow.therapist_username == therapist_username,
            )
            .order_by(UserRow.username)
        ).all()
    return [_to_user(r) for r in rows]


def get_activity_counts(usernames: list[str] | None = None) -> dict[str, int]:
    stmt = (
        select(ActivityLogRow.username, func.count(ActivityLogRow.id))
        .group_by(ActivityLogRow.username)
        .order_by(func.count(ActivityLogRow.id).desc())
    )
    if usernames:
        stmt = stmt.where(ActivityLogRow.username.in_(usernames))

    with Session(ENGINE) as session:
        rows = session.execute(stmt).all()
    return {username: int(count) for username, count in rows}
