"""Persist semantic parent and child study chunks."""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "0008_hierarchical_study_chunks"
down_revision = "0007_case_insensitive_email"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_constraint(
        "uq_study_content_location_chunk",
        "study_content_units",
        type_="unique",
    )
    op.add_column(
        "study_content_units",
        sa.Column(
            "chunk_level",
            sa.String(length=16),
            nullable=False,
            server_default="child",
        ),
    )
    op.add_column(
        "study_content_units",
        sa.Column(
            "parent_unit_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("study_content_units.id", ondelete="CASCADE"),
            nullable=True,
        ),
    )
    op.add_column(
        "study_content_units",
        sa.Column("char_start", sa.Integer(), nullable=True),
    )
    op.add_column(
        "study_content_units",
        sa.Column("char_end", sa.Integer(), nullable=True),
    )
    op.create_index(
        "ix_study_content_units_parent_unit_id",
        "study_content_units",
        ["parent_unit_id"],
    )
    op.create_unique_constraint(
        "uq_study_content_location_chunk",
        "study_content_units",
        [
            "source_id",
            "location_type",
            "location_number",
            "chunk_level",
            "chunk_index",
        ],
    )
    op.alter_column("study_content_units", "chunk_level", server_default=None)


def downgrade() -> None:
    op.drop_constraint(
        "uq_study_content_location_chunk",
        "study_content_units",
        type_="unique",
    )
    op.drop_index(
        "ix_study_content_units_parent_unit_id",
        table_name="study_content_units",
    )
    op.drop_column("study_content_units", "char_end")
    op.drop_column("study_content_units", "char_start")
    op.drop_column("study_content_units", "parent_unit_id")
    op.drop_column("study_content_units", "chunk_level")
    op.create_unique_constraint(
        "uq_study_content_location_chunk",
        "study_content_units",
        ["source_id", "location_type", "location_number", "chunk_index"],
    )
