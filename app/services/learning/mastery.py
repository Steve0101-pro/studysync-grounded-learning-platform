"""Practice-history mastery estimates and simple review scheduling."""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import datetime, timezone

_INITIAL_MASTERY = 0.2
_LEARNING_PROBABILITY = 0.12
_GUESS_PROBABILITY = 0.2
_SLIP_PROBABILITY = 0.1
_FORGETTING_HALF_LIFE_DAYS = 21


def estimate_mastery(
    events: Sequence[tuple[bool, datetime]],
    *,
    as_of: datetime | None = None,
) -> float:
    """Estimate a topic's latent mastery from answered practice questions."""
    if not events:
        return _INITIAL_MASTERY

    observations = sorted(events, key=lambda event: event[1])
    mastery = _INITIAL_MASTERY
    previous_at: datetime | None = None

    for correct, answered_at in observations:
        answered_at = _as_utc(answered_at)
        if previous_at is not None:
            mastery = _decay_mastery(mastery, (answered_at - previous_at).total_seconds())

        if correct:
            numerator = mastery * (1 - _SLIP_PROBABILITY)
            denominator = numerator + (1 - mastery) * _GUESS_PROBABILITY
        else:
            numerator = mastery * _SLIP_PROBABILITY
            denominator = numerator + (1 - mastery) * (1 - _GUESS_PROBABILITY)
        posterior = numerator / denominator if denominator else mastery
        mastery = posterior + (1 - posterior) * _LEARNING_PROBABILITY
        previous_at = answered_at

    reference_time = _as_utc(as_of or datetime.now(timezone.utc))
    if previous_at is not None:
        mastery = _decay_mastery(mastery, (reference_time - previous_at).total_seconds())
    return min(1.0, max(0.0, mastery))


def recommended_review_days(mastery: float) -> int:
    """Return a conservative review interval from the estimated mastery."""
    if mastery < 0.4:
        return 1
    if mastery < 0.65:
        return 3
    if mastery < 0.85:
        return 7
    return 14


def _decay_mastery(mastery: float, elapsed_seconds: float) -> float:
    days = max(0.0, elapsed_seconds / 86_400)
    retention = math.exp(-math.log(2) * days / _FORGETTING_HALF_LIFE_DAYS)
    return _INITIAL_MASTERY + (mastery - _INITIAL_MASTERY) * retention


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)
