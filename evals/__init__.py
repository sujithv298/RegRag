"""Eval harness — load gold sets, run pipeline, compute metrics.

The harness is intentionally separate from the `regrag` package: it
*depends on* `regrag` but isn't part of the importable library. Forkers
who don't need eval can ignore this directory entirely.
"""

from __future__ import annotations

from evals.comparison import (
    CaseComparison,
    ComparisonReport,
    compare_reports,
    format_comparison,
)
from evals.runner import compute_metrics, load_gold_set, run_eval
from evals.types import (
    Category,
    EvalCaseResult,
    EvalMetrics,
    EvalReport,
    GoldEntry,
)

__all__ = [
    "CaseComparison",
    "Category",
    "ComparisonReport",
    "EvalCaseResult",
    "EvalMetrics",
    "EvalReport",
    "GoldEntry",
    "compare_reports",
    "compute_metrics",
    "format_comparison",
    "load_gold_set",
    "run_eval",
]
