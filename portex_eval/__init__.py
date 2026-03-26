"""Portex Eval - Lightweight evaluation framework for LLM judges and candidates."""

from __future__ import annotations

from typing import TYPE_CHECKING

from dotenv import load_dotenv

__version__ = "0.1.0"

if TYPE_CHECKING:
    from portex_eval import reporting as reports
    from portex_eval.api import agent_eval, create_agent_eval, create_benchmark, eval
    from portex_eval.config import Config
    from portex_eval.errors import PortexEvalError
    from portex_eval.run_spec import RunSpec, load_run_spec
    from portex_eval.types import (
        AgentEvalBundle,
        AgentEvalResults,
        Benchmark,
        EvalResults,
        ReportPaths,
    )

__all__ = [
    "__version__",
    # API
    "agent_eval",
    "create_agent_eval",
    "create_benchmark",
    "eval",
    "format_bundle",
    "induce_criteria",
    # Types
    "AgentEvalBundle",
    "AgentEvalResults",
    "Benchmark",
    "EvalResults",
    "ReportPaths",
    "reports",
    "PortexEvalError",
    # Config
    "Config",
    "RunSpec",
    "load_run_spec",
]

load_dotenv()


# Lazy imports for optional subpackages to avoid hard dependencies
def __getattr__(name: str) -> object:
    """Lazy load optional subpackages."""
    if name == "reports":
        from portex_eval import reporting as reports

        return reports
    if name in {"agent_eval", "create_agent_eval", "create_benchmark", "eval"}:
        from portex_eval import api

        return getattr(api, name)
    if name == "Config":
        from portex_eval.config import Config

        return Config
    if name == "PortexEvalError":
        from portex_eval.errors import PortexEvalError

        return PortexEvalError
    if name in {"RunSpec", "load_run_spec"}:
        from portex_eval import run_spec

        return getattr(run_spec, name)
    if name in {
        "AgentEvalBundle",
        "AgentEvalResults",
        "Benchmark",
        "EvalResults",
        "ReportPaths",
    }:
        from portex_eval import types

        return getattr(types, name)
    if name == "get_provider":
        from portex_eval.providers import get_provider

        return get_provider
    if name == "Provider":
        from portex_eval.providers import Provider

        return Provider
    if name == "Response":
        from portex_eval.providers import Response

        return Response
    if name == "format_bundle":
        from portex_eval.bundle import format_bundle

        return format_bundle
    if name == "induce_criteria":
        from portex_eval.bundle import induce_criteria

        return induce_criteria
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
