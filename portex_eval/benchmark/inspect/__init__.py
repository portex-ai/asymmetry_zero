"""Inspect AI integration for portex_eval.

This module provides the Inspect AI task definitions, solvers, and scorers
for running Portex evaluations.

Key components:
- dataset: Dataset loading from bundle format
- eval: Task definitions (portex_qa_eval, create_eval_task)
- scorer: Multi-judge scoring with provider support
- solver: Candidate generation with provider support
"""

from portex_eval.benchmark.inspect.dataset import dataset_generator
from portex_eval.benchmark.inspect.eval import (
    create_eval_task,
    portex_qa_eval,
    portex_qa_eval_with_providers,
)
from portex_eval.benchmark.inspect.scorer import portex_scorer, provider_scorer
from portex_eval.benchmark.inspect.solver import candidate_generate, provider_generate

__all__ = [
    "dataset_generator",
    "create_eval_task",
    "portex_qa_eval",
    "portex_qa_eval_with_providers",
    "portex_scorer",
    "provider_scorer",
    "candidate_generate",
    "provider_generate",
]
