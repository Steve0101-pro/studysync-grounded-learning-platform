"""Add timed, reportable study mock exams."""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "0011_timed_study_mock_exams"
down_revision = "0010_verified_study_assessments"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "study_mock_exams",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("topic", sa.String(length=255), nullable=False),
        sa.Column("duration_seconds", sa.Integer(), nullable=False),
        sa.Column(
            "started_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("submitted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "report_json",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_study_mock_exams_user_id",
        "study_mock_exams",
        ["user_id"],
    )
    op.add_column(
        "study_quiz_questions",
        sa.Column("mock_exam_id", postgresql.UUID(as_uuid=True), nullable=True),
    )
    op.create_foreign_key(
        "fk_study_quiz_questions_mock_exam_id",
        "study_quiz_questions",
        "study_mock_exams",
        ["mock_exam_id"],
        ["id"],
        ondelete="CASCADE",
    )
    op.create_index(
        "ix_study_quiz_questions_mock_exam_id",
        "study_quiz_questions",
        ["mock_exam_id"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_study_quiz_questions_mock_exam_id",
        table_name="study_quiz_questions",
    )
    op.drop_constraint(
        "fk_study_quiz_questions_mock_exam_id",
        "study_quiz_questions",
        type_="foreignkey",
    )
    op.drop_column("study_quiz_questions", "mock_exam_id")
    op.drop_index("ix_study_mock_exams_user_id", table_name="study_mock_exams")
    op.drop_table("study_mock_exams")
