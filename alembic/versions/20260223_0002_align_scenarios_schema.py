"""align scenarios schema with hybrid model

Revision ID: 20260223_0002
Revises: 9a8bb4362f61
Create Date: 2026-02-23 14:05:00
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision = "20260223_0002"
down_revision = "9a8bb4362f61"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    columns = {column["name"] for column in inspector.get_columns("scenarios")}

    with op.batch_alter_table("scenarios") as batch_op:
        if "name" in columns and "title" not in columns:
            batch_op.alter_column("name", new_column_name="title", existing_type=sa.String(), existing_nullable=False)

        if "metadata" in columns:
            batch_op.drop_column("metadata")

        if "deleted_at" not in columns:
            batch_op.add_column(sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True))

        batch_op.create_check_constraint(
            "ck_scenarios_status",
            "status in ('draft', 'published', 'archived')",
        )
        batch_op.create_check_constraint("ck_scenarios_version_positive", "version >= 1")


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    columns = {column["name"] for column in inspector.get_columns("scenarios")}

    with op.batch_alter_table("scenarios") as batch_op:
        batch_op.drop_constraint("ck_scenarios_version_positive", type_="check")
        batch_op.drop_constraint("ck_scenarios_status", type_="check")

        if "deleted_at" in columns:
            batch_op.drop_column("deleted_at")

        if "metadata" not in columns:
            batch_op.add_column(
                sa.Column(
                    "metadata",
                    sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql"),
                    nullable=True,
                )
            )

        if "title" in columns and "name" not in columns:
            batch_op.alter_column("title", new_column_name="name", existing_type=sa.String(), existing_nullable=False)
