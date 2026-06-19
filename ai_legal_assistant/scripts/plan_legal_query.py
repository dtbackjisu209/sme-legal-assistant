from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from enum import Enum
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ai_legal_assistant.infrastructure.bootstrap.query_planning import (
    QueryPlannerConfig,
    build_query_planner,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize, analyze, and conditionally expand one Vietnamese legal query."
    )
    parser.add_argument("query")
    parser.add_argument(
        "--planner-model",
        default=os.getenv("QUERY_PLANNER_MODEL", "Qwen/Qwen3-0.6B"),
    )
    parser.add_argument("--planner-device", default=os.getenv("QUERY_PLANNER_DEVICE"))
    parser.add_argument("--planner-max-input-tokens", type=int, default=4096)
    parser.add_argument("--planner-max-new-tokens", type=int, default=500)
    parser.add_argument("--planner-trust-remote-code", action="store_true")
    return parser.parse_args()


def json_default(value: Any) -> str:
    if isinstance(value, Enum):
        return value.value
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def main() -> int:
    args = parse_args()
    plan = build_query_planner(
        QueryPlannerConfig(
            model_name_or_path=args.planner_model,
            device=args.planner_device,
            max_input_tokens=args.planner_max_input_tokens,
            max_new_tokens=args.planner_max_new_tokens,
            trust_remote_code=args.planner_trust_remote_code,
        )
    ).execute(args.query)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps(asdict(plan), ensure_ascii=False, indent=2, default=json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
