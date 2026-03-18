"""Thin Harbor grading entrypoint that imports the shared Portex runtime."""

from __future__ import annotations

import sys
from pathlib import Path

RUNTIME_DIR = Path(__file__).resolve().parent / "runtime"
sys.path.insert(0, str(RUNTIME_DIR))

from portex_eval.benchmark.harbor.verifier import main  # noqa: E402


if __name__ == "__main__":
    main()
