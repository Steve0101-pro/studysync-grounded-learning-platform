"""Persist source-grounded study practice questions and answers."""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "0006_study_practice"
down_revision = "0005_study_sources"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "study_quiz_questions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column(
            "user_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("topic", sa.String(255), nullable=False),
        sa.Column("prompt", sa.Text(), nullable=False),
        sa.Column("options", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("answer_index", sa.Integer(), nullable=False),
        sa.Column(
            "source_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("study_sources.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("location_type", sa.String(16), nullable=False),
        sa.Column("location_number", sa.Integer(), nullable=False),
        sa.Column("answered_index", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("answered_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_study_quiz_questions_user_id", "study_quiz_questions", ["user_id"])
    op.create_index("ix_study_quiz_questions_topic", "study_quiz_questions", ["topic"])


def downgrade() -> None:
    op.drop_index("ix_study_quiz_questions_topic", table_name="study_quiz_questions")
    op.drop_index("ix_study_quiz_questions_user_id", table_name="study_quiz_questions")
    op.drop_table("study_quiz_questions")
