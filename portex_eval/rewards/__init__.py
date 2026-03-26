"""Rewards extraction for RL training pipelines.

This module provides utilities to extract and write reward signals from
evaluation results for use in reinforcement learning training pipelines.
"""

from __future__ import annotations

from portex_eval.rewards.writer import (
    build_rewards,
    build_training_data,
    extract_rewards,
    write_rewards,
    write_training_data,
)

__all__ = [
    "build_rewards",
    "build_training_data",
    "extract_rewards",
    "write_rewards",
    "write_training_data",
]
