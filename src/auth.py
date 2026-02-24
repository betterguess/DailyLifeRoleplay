import base64
import hashlib
import hmac
import json
import os
import secrets
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import Integer, String, Text, func, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Mapped, mapped_column

from src.db import Base, get_session

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


def init_auth_store() -> None:
    try:
        _bootstrap_if_empty()
    except SQLAlchemyError as exc:
        raise RuntimeError(
            "Databaseschema mangler eller er ikke opdateret. "
            "Kør 'alembic upgrade head' før app-start."
        ) from exc


def _bootstrap_if_empty() -> None:
    with get_session() as session:
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
    with get_session() as session:
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
    with get_session() as session:
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

    with get_session() as session:
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
    with get_session() as session:
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
    with get_session() as session:
        rows = session.scalars(select(UserRow).order_by(UserRow.role, UserRow.username)).all()
    return [_to_user(r) for r in rows]


def get_staff_users() -> list[User]:
    return [u for u in list_users() if u.role in STAFF_ROLES]


def get_therapists() -> list[User]:
    return [u for u in list_users() if u.role == ROLE_THERAPIST]


def get_patients_for_therapist(therapist_username: str) -> list[User]:
    therapist_username = therapist_username.strip().lower()
    with get_session() as session:
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

    with get_session() as session:
        rows = session.execute(stmt).all()
    return {username: int(count) for username, count in rows}
