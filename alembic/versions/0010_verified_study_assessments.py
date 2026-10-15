"""Store typed, verified study assessment items and graded responses."""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "0010_verified_study_assessments"
down_revision = "0009_study_question_dedup"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "study_quiz_questions",
        sa.Column(
            "assessment_type",
            sa.String(length=32),
            nullable=False,
            server_default="passage_selection",
        ),
    )
    op.add_column(
        "study_quiz_questions",
        sa.Column("difficulty", sa.String(length=16), nullable=False, server_default="medium"),
    )
    op.add_column(
        "study_quiz_questions",
        sa.Column(
            "assessment_data",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
    )
    op.add_column(
        "study_quiz_questions",
        sa.Column("answered_text", sa.Text(), nullable=True),
    )
    op.add_column(
        "study_quiz_questions",
        sa.Column("is_correct", sa.Boolean(), nullable=True),
    )
    op.alter_column("study_quiz_questions", "assessment_type", server_default=None)
    op.alter_column("study_quiz_questions", "difficulty", server_default=None)
    op.alter_column("study_quiz_questions", "assessment_data", server_default=None)


def downgrade() -> None:
    op.drop_column("study_quiz_questions", "is_correct")
    op.drop_column("study_quiz_questions", "answered_text")
    op.drop_column("study_quiz_questions", "assessment_data")
    op.drop_column("study_quiz_questions", "difficulty")
    op.drop_column("study_quiz_questions", "assessment_type")
