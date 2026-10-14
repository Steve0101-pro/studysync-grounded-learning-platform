"""Enforce case-insensitive user email uniqueness."""

from alembic import op
import sqlalchemy as sa


revision = "0007_case_insensitive_email"
down_revision = "0006_study_practice"
branch_labels = None
depends_on = None


def upgrade() -> None:
    duplicate = op.get_bind().execute(
        sa.text(
            "SELECT 1 FROM users "
            "GROUP BY lower(email) HAVING count(*) > 1 LIMIT 1"
        )
    ).first()
    if duplicate:
        raise RuntimeError(
            "Cannot enforce case-insensitive email uniqueness while duplicate "
            "accounts exist; resolve duplicate email accounts before upgrading."
        )
    op.create_index(
        "uq_users_email_lower",
        "users",
        [sa.text("lower(email)")],
        unique=True,
    )


def downgrade() -> None:
    op.drop_index("uq_users_email_lower", table_name="users")
