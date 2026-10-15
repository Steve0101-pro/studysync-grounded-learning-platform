"""NeMo Guardrails adapter for deterministic StudySync input and output checks."""

from functools import lru_cache
from pathlib import Path

from nemoguardrails import LLMRails, RailsConfig
from nemoguardrails.rails.llm.options import RailType

from app.services.safety import SafetyViolation

_CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "config" / "nemo_guardrails"
)


@lru_cache(maxsize=1)
def _rails() -> LLMRails:
    return LLMRails(RailsConfig.from_path(str(_CONFIG_PATH)))


def _checked_content(result: object, original: str) -> str:
    status = getattr(getattr(result, "status", None), "name", "")
    if status == "BLOCKED":
        raise SafetyViolation("Content was blocked by the StudySync safety policy")
    if status not in {"PASSED", "MODIFIED"}:
        raise RuntimeError(f"NeMo Guardrails returned unexpected status: {status or 'missing'}")
    content = getattr(result, "content", None)
    if not isinstance(content, str) or not content:
        raise RuntimeError("NeMo Guardrails returned empty checked content")
    return content if status == "MODIFIED" else original


async def check_studysync_input(text: str) -> str:
    """Block unsafe questions and return PII-redacted user text."""
    result = await _rails().check_async(
        [{"role": "user", "content": text}],
        rail_types=[RailType.INPUT],
    )
    return _checked_content(result, text)


async def check_studysync_output(text: str) -> str:
    """Return assistant text after output PII checks and redaction."""
    result = await _rails().check_async(
        [{"role": "assistant", "content": text}],
        rail_types=[RailType.OUTPUT],
    )
    return _checked_content(result, text)
