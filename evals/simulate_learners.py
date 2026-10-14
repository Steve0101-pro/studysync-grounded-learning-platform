"""Compare random, spaced-review, and adaptive study policies reproducibly."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.services.learning.mastery import (  # noqa: E402
    estimate_mastery,
    recommended_review_days,
)

CONDITIONS = ("random", "spaced_repetition", "adaptive")
TOPICS = ("foundations", "application", "analysis", "synthesis")
ITEMS_PER_TOPIC = 16


def simulate(
    profiles: dict[str, list[float]],
    *,
    sessions: int = 5,
    questions_per_session: int = 3,
    seed: int = 1729,
) -> dict[str, Any]:
    if sessions < 5:
        raise ValueError("Track D learner simulation requires at least five sessions")
    if questions_per_session < 1:
        raise ValueError("questions_per_session must be positive")
    if any(len(mastery) != len(TOPICS) for mastery in profiles.values()):
        raise ValueError(f"Each profile must define mastery for exactly {len(TOPICS)} topics")
    if any(not 0 <= value <= 1 for mastery in profiles.values() for value in mastery):
        raise ValueError("Initial topic mastery values must be between zero and one")

    result_rows = []
    for profile_name, initial_mastery in sorted(profiles.items()):
        for condition in CONDITIONS:
            rng = random.Random(f"{seed}:{profile_name}:{condition}")
            actual = dict(zip(TOPICS, initial_mastery, strict=True))
            history: dict[str, list[tuple[bool, datetime]]] = defaultdict(list)
            last_seen: dict[str, int] = {}
            question_counts: dict[str, int] = defaultdict(int)
            selected_questions: list[str] = []
            weak_topic_picks = 0
            total_picks = 0

            for session_index in range(sessions):
                for topic in TOPICS:
                    actual[topic] *= 0.99
                estimated = {
                    topic: estimate_mastery(
                        history[topic],
                        as_of=datetime(2026, 1, 1, tzinfo=timezone.utc)
                        + timedelta(days=session_index),
                    )
                    for topic in TOPICS
                }
                weak_topics = set(
                    sorted(estimated, key=lambda topic: (estimated[topic], topic))[
                        : max(1, len(TOPICS) // 2)
                    ]
                )
                if condition == "random":
                    selected_topics = [
                        rng.choice(TOPICS) for _ in range(questions_per_session)
                    ]
                elif condition == "spaced_repetition":
                    selected_topics = sorted(
                        TOPICS,
                        key=lambda topic: (
                            -(
                                (session_index - last_seen.get(topic, -sessions))
                                / max(1, recommended_review_days(estimated[topic]))
                            ),
                            topic,
                        ),
                    )[:questions_per_session]
                else:
                    selected_topics = sorted(
                        TOPICS,
                        key=lambda topic: (estimated[topic], actual[topic], topic),
                    )[:questions_per_session]

                for topic in selected_topics:
                    pool = [f"{topic}-{index}" for index in range(ITEMS_PER_TOPIC)]
                    unseen = [item for item in pool if question_counts[item] == 0]
                    if condition == "random":
                        item = rng.choice(pool)
                    elif unseen:
                        item = rng.choice(unseen)
                    else:
                        item = min(pool, key=lambda candidate: question_counts[candidate])
                    question_counts[item] += 1
                    selected_questions.append(item)

                    if topic in weak_topics:
                        weak_topic_picks += 1
                    total_picks += 1

                    correct = rng.random() < actual[topic]
                    actual[topic] = min(
                        0.98,
                        actual[topic] + (0.08 if correct else 0.04),
                    )
                    history[topic].append(
                        (
                            correct,
                            datetime(2026, 1, 1, tzinfo=timezone.utc)
                            + timedelta(days=session_index),
                        )
                    )
                    last_seen[topic] = session_index

            final_estimates = {
                topic: estimate_mastery(
                    history[topic],
                    as_of=datetime(2026, 1, 1, tzinfo=timezone.utc)
                    + timedelta(days=sessions - 1),
                )
                for topic in TOPICS
            }
            initial_mean = sum(initial_mastery) / len(TOPICS)
            final_mean = sum(actual.values()) / len(TOPICS)
            normalized_gain = (
                (final_mean - initial_mean) / (1 - initial_mean)
                if initial_mean < 1
                else 0.0
            )
            repeats = len(selected_questions) - len(set(selected_questions))
            result_rows.append(
                {
                    "profile": profile_name,
                    "condition": condition,
                    "sessions": sessions,
                    "initial_mean_mastery": initial_mean,
                    "final_mean_mastery": final_mean,
                    "normalized_gain": normalized_gain,
                    "question_repetition_rate": repeats / len(selected_questions),
                    "weak_topic_targeting_rate": weak_topic_picks / total_picks,
                    "mastery_estimation_mae": sum(
                        abs(final_estimates[topic] - actual[topic]) for topic in TOPICS
                    )
                    / len(TOPICS),
                }
            )

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in result_rows:
        grouped[row["condition"]].append(row)
    summary = {
        condition: {
            metric: sum(row[metric] for row in rows) / len(rows)
            for metric in (
                "normalized_gain",
                "question_repetition_rate",
                "weak_topic_targeting_rate",
                "mastery_estimation_mae",
            )
        }
        for condition, rows in grouped.items()
    }
    return {
        "simulation_type": "synthetic_policy_comparison_not_learner_outcome_evidence",
        "seed": seed,
        "sessions_per_profile_condition": sessions,
        "questions_per_session": questions_per_session,
        "profile_count": len(profiles),
        "conditions": list(CONDITIONS),
        "summary": summary,
        "results": result_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profiles",
        type=Path,
        default=Path(__file__).with_name("simulated_profiles.json"),
    )
    parser.add_argument("--sessions", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()
    profiles = json.loads(args.profiles.read_text(encoding="utf-8"))
    print(json.dumps(simulate(profiles, sessions=args.sessions, seed=args.seed), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
