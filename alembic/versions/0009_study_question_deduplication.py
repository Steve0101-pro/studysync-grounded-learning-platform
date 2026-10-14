"""Prevent concurrent reuse of a source passage in practice questions."""

from alembic import op
import sqlalchemy as sa


revision = "0009_study_question_dedup"
down_revision = "0008_hierarchical_study_chunks"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "study_quiz_questions",
        sa.Column("deduplication_key", sa.String(length=64), nullable=True),
    )
    op.create_index(
        "uq_study_quiz_user_deduplication_key",
        "study_quiz_questions",
        ["user_id", "deduplication_key"],
        unique=True,
    )


def downgrade() -> None:
    op.drop_index(
        "uq_study_quiz_user_deduplication_key",
        table_name="study_quiz_questions",
    )
    op.drop_column("study_quiz_questions", "deduplication_key")
