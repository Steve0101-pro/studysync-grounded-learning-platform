"""Statistical helpers for reporting sparse practice outcomes honestly."""

from __future__ import annotations

import math
from typing import Any

import altair as alt


def wilson_interval(
    correct: int,
    answered: int,
    *,
    z_score: float = 1.96,
) -> tuple[float, float] | None:
    """Return a Wilson score interval for a binomial accuracy proportion."""
    if answered < 0 or correct < 0 or correct > answered:
        raise ValueError("correct and answered must satisfy 0 <= correct <= answered")
    if not math.isfinite(z_score) or z_score <= 0:
        raise ValueError("z_score must be a finite positive value")
    if answered == 0:
        return None

    observed = correct / answered
    z_squared = z_score**2
    denominator = 1 + z_squared / answered
    center = (observed + z_squared / (2 * answered)) / denominator
    margin = (
        z_score
        * math.sqrt(
            observed * (1 - observed) / answered
            + z_squared / (4 * answered**2)
        )
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def practice_accuracy_chart(topics: list[dict[str, Any]]) -> alt.LayerChart | None:
    """Build a topic accuracy forest plot with sample-aware confidence intervals."""
    chart_rows = []
    for topic in topics:
        answered = int(topic["answered"])
        if answered == 0:
            continue
        interval = wilson_interval(int(topic["correct"]), answered)
        if interval is None:
            continue
        chart_rows.append(
            {
                "topic": str(topic["topic"]),
                "accuracy": float(topic["accuracy"]),
                "lower": interval[0],
                "upper": interval[1],
                "correct": int(topic["correct"]),
                "answered": answered,
            }
        )
    if not chart_rows:
        return None

    base = alt.Chart(alt.Data(values=chart_rows)).encode(
        y=alt.Y(
            "topic:N",
            sort=alt.SortField(field="accuracy", order="descending"),
            title=None,
        )
    )
    percent_axis = alt.Axis(format=".0%", title="Observed accuracy · 95% interval")
    interval_chart = base.mark_rule(color="#5580b5", strokeWidth=5).encode(
        x=alt.X("lower:Q", scale=alt.Scale(domain=[0, 1]), axis=percent_axis),
        x2="upper:Q",
        tooltip=[
            alt.Tooltip("topic:N", title="Topic"),
            alt.Tooltip("correct:Q", title="Correct answers"),
            alt.Tooltip("answered:Q", title="Answered"),
            alt.Tooltip("lower:Q", title="95% lower bound", format=".0%"),
            alt.Tooltip("upper:Q", title="95% upper bound", format=".0%"),
        ],
    )
    accuracy_points = base.mark_circle(
        color="#35a7ff",
        size=130,
        stroke="white",
        strokeWidth=1,
    ).encode(
        x=alt.X("accuracy:Q", scale=alt.Scale(domain=[0, 1]), axis=percent_axis),
        tooltip=[
            alt.Tooltip("topic:N", title="Topic"),
            alt.Tooltip("accuracy:Q", title="Observed accuracy", format=".0%"),
            alt.Tooltip("correct:Q", title="Correct answers"),
            alt.Tooltip("answered:Q", title="Answered"),
        ],
    )
    return (
        (interval_chart + accuracy_points)
        .properties(height=max(140, 44 * len(chart_rows)))
        .configure_view(stroke=None)
        .configure_axis(labelColor="#d6e5fa", titleColor="#9aacc8", gridColor="#1b3154")
    )
