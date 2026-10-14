"""Run the deterministic baseline comparison on a reviewed gold set."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from evals.benchmark import run_benchmark  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("responses", type=Path, help="Baseline response records JSON")
    parser.add_argument(
        "--cases",
        type=Path,
        default=Path(__file__).with_name("golden_rag.json"),
    )
    parser.add_argument("--minimum-cases", type=int, default=100)
    args = parser.parse_args()

    cases: list[dict[str, Any]] = json.loads(args.cases.read_text(encoding="utf-8"))
    responses: dict[str, Any] = json.loads(args.responses.read_text(encoding="utf-8"))
    if len(cases) < args.minimum_cases:
        parser.error(
            f"Gold set has {len(cases)} cases; Track D requires at least "
            f"{args.minimum_cases} reviewed cases."
        )
    report = run_benchmark(cases, responses)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
