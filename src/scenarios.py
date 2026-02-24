import glob
import json
from datetime import datetime
from typing import Any

import streamlit as st
from sqlalchemy import JSON, CheckConstraint, DateTime, Integer, String, func, select
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Mapped, mapped_column

from src.db import Base, get_session


SCENARIO_DIR = "scenarios"
SCENARIO_CONTENT_TYPE = JSON().with_variant(JSONB(), "postgresql")
SCENARIO_SCHEMA_VERSION = 1
SCENARIO_STATUSES = {"draft", "published", "archived"}


def ensure_schema_version(content: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(content)
    normalized.setdefault("schema_version", SCENARIO_SCHEMA_VERSION)
    return normalized


def validate_scenario_content(content: dict[str, Any]) -> None:
    if not isinstance(content, dict):
        raise ValueError("Scenario content skal vaere et JSON-objekt.")

    required_fields = ("id", "title")
    missing = [field for field in required_fields if not str(content.get(field, "")).strip()]
    if missing:
        raise ValueError(f"Mangler obligatoriske felter i content: {', '.join(missing)}")

    schema_version = content.get("schema_version", SCENARIO_SCHEMA_VERSION)
    if not isinstance(schema_version, int) or schema_version < 1:
        raise ValueError("content.schema_version skal vaere et positivt heltal.")


def prepare_scenario_content(content: dict[str, Any]) -> dict[str, Any]:
    normalized = ensure_schema_version(content)
    validate_scenario_content(normalized)
    return normalized


def load_scenarios() -> list[dict[str, Any]]:
    scenarios: list[dict[str, Any]] = []
    for file_path in glob.glob(f"{SCENARIO_DIR}/*.json"):
        with open(file_path, "r", encoding="utf-8") as infile:
            try:
                data = json.load(infile)
                scenarios.append(prepare_scenario_content(data))
            except Exception as exc:
                st.warning(f"Kunne ikke indlaese {file_path}: {exc}")
    return sorted(scenarios, key=lambda scenario: str(scenario.get("title", "")).lower())


class ScenarioRow(Base):
    __tablename__ = "scenarios"
    __table_args__ = (
        CheckConstraint("status in ('draft', 'published', 'archived')", name="ck_scenarios_status"),
        CheckConstraint("version >= 1", name="ck_scenarios_version_positive"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    slug: Mapped[str] = mapped_column(String, nullable=False, unique=True, index=True)
    title: Mapped[str] = mapped_column(String, nullable=False)
    creator: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False, server_default="draft")
    version: Mapped[int] = mapped_column(Integer, nullable=False, server_default="1")
    content: Mapped[dict[str, Any]] = mapped_column(SCENARIO_CONTENT_TYPE, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


def upsert_scenario(
    *,
    slug: str,
    title: str,
    creator: str,
    content: dict[str, Any],
    status: str = "draft",
    version: int = 1,
) -> ScenarioRow:
    if status not in SCENARIO_STATUSES:
        raise ValueError(f"Ugyldig status '{status}'. Tilladte vaerdier: {', '.join(sorted(SCENARIO_STATUSES))}")
    if version < 1:
        raise ValueError("version skal vaere >= 1.")

    normalized_content = prepare_scenario_content(content)
    if normalized_content.get("title") != title:
        normalized_content["title"] = title
    if normalized_content.get("id") != slug:
        normalized_content["id"] = slug

    with get_session() as session:
        existing = session.scalar(select(ScenarioRow).where(ScenarioRow.slug == slug))
        if existing is None:
            existing = ScenarioRow(
                slug=slug,
                title=title,
                creator=creator,
                status=status,
                version=version,
                content=normalized_content,
            )
            session.add(existing)
        else:
            existing.title = title
            existing.creator = creator
            existing.status = status
            existing.version = version
            existing.content = normalized_content
            existing.deleted_at = None

        try:
            session.commit()
        except SQLAlchemyError:
            session.rollback()
            raise
        session.refresh(existing)
        return existing
