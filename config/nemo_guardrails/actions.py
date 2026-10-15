from nemoguardrails.actions import action
from nemoguardrails.actions.rail_outcome import RailOutcome, TransformTarget

from app.services.safety import (
    SafetyViolation,
    assert_safe_text,
    redact_sensitive_text,
)


@action(name="studysync_check_input")
async def studysync_check_input(text: str) -> RailOutcome:
    try:
        assert_safe_text(text, "student input")
    except SafetyViolation:
        return RailOutcome.block(
            reason="The input matched the instruction-override policy."
        )

    redacted = redact_sensitive_text(text)
    if redacted != text:
        return RailOutcome.transform(
            [(TransformTarget.USER_MESSAGE, redacted)],
            reason="Sensitive values were redacted.",
        )
    return RailOutcome.allow()


@action(name="studysync_check_output")
async def studysync_check_output(text: str) -> RailOutcome:
    redacted = redact_sensitive_text(text)
    if redacted != text:
        return RailOutcome.transform(
            [(TransformTarget.BOT_MESSAGE, redacted)],
            reason="Sensitive values were redacted.",
        )
    return RailOutcome.allow()
