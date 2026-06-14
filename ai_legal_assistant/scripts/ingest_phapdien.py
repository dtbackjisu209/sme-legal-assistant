from __future__ import annotations

import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ai_legal_assistant.infrastructure.bootstrap.container import build_container


def main() -> int:
    container = build_container()
    use_case = container.ingest_phapdien_use_case
    result = use_case.execute()
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
