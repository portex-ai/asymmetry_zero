"""Reporting helpers for Portex eval outputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from portex_eval.reporting import tables


def load(path: str) -> pd.DataFrame:
    """Load a CSV report into a pandas DataFrame."""
    resolved = Path(path).expanduser().resolve()
    return pd.read_csv(resolved)


__all__ = ["load", "tables"]
