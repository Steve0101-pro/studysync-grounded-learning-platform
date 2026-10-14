"""Add persistent study sources and page/slide content units."""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "0005_study_sources"
down_revision = "0004_password_reset_tokens"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "study_sources",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("filename", sa.String(255), nullable=False),
        sa.Column("content_type", sa.String(100), nullable=False),
        sa.Column("storage_path", sa.String(500), nullable=False),
        sa.Column("content_sha256", sa.String(64), nullable=False),
        sa.Column("page_count", sa.Integer(), nullable=False),
        sa.Column("unit_count", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("user_id", "content_sha256", name="uq_study_sources_user_checksum"),
    )
    op.create_index("ix_study_sources_user_id", "study_sources", ["user_id"])

    op.create_table(
        "study_content_units",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column(
            "source_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("study_sources.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("location_type", sa.String(16), nullable=False),
        sa.Column("location_number", sa.Integer(), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.UniqueConstraint(
            "source_id",
            "location_type",
            "location_number",
            "chunk_index",
            name="uq_study_content_location_chunk",
        ),
    )
    op.create_index("ix_study_content_units_source_id", "study_content_units", ["source_id"])


def downgrade() -> None:
    op.drop_index("ix_study_content_units_source_id", table_name="study_content_units")
    op.drop_table("study_content_units")
    op.drop_index("ix_study_sources_user_id", table_name="study_sources")
    op.drop_table("study_sources")
