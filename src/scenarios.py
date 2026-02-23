import glob
import json
from datetime import datetime
from typing import Any

import streamlit as st

from sqlalchemy import JSON, DateTime, Integer, String, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from src.db import Base


SCENARIO_DIR = "scenarios"
SCENARIO_CONTENT_TYPE = JSON().with_variant(JSONB(), "postgresql")


def load_scenarios():
    scenarios = []
    for file_path in glob.glob(f"{SCENARIO_DIR}/*.json"):
        with open(file_path, "r", encoding="utf-8") as infile:
            try:
                data = json.load(infile)
                scenarios.append(data)
            except Exception as exc:
                st.warning(f"Kunne ikke indlaese {file_path}: {exc}")
    return sorted(scenarios, key=lambda scenario: scenario["title"])


class ScenariosRow(Base):
    __tablename__ = "scenarios"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    slug: Mapped[str] = mapped_column(String, nullable=False, unique=True, index=True)
    creator: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False, server_default="draft")
    version: Mapped[int] = mapped_column(Integer, nullable=False, server_default="1")
    content: Mapped[dict[str, Any]] = mapped_column(SCENARIO_CONTENT_TYPE, nullable=False)
    scenario_metadata: Mapped[dict[str, Any] | None] = mapped_column("metadata", SCENARIO_CONTENT_TYPE, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    deleted_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        server_default=null,
        onupdate=func.now(),
    
    )
