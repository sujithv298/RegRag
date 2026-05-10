"""Compare deterministic vs agentic eval reports.

Two pure functions plus a pydantic model:

  - `CaseComparison`        — per-case side-by-side: did each pipeline pass,
                              regress, or progress?
  - `ComparisonReport`      — aggregate over all cases, with the four
                              headline counts (both-passed / both-failed /
                              progressed / regressed) plus the underlying
                              EvalReports.
  - `compare_reports(...)`  — pair up cases by entry_id and build a
                              ComparisonReport. Pure function.
  - `format_comparison(...)`— render a human-readable summary suitable for
                              `regrag eval-compare` CLI output.

This is the highest-leverage artifact in v2: it answers empirically when
the agent helps and when it hurts, on real gold-set questions. The output
is the headline of the "agentic vs deterministic in regulated finance"
LinkedIn post.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from evals.types import EvalCaseResult, EvalReport

CaseDelta = Literal["both_passed", "both_failed", "progressed", "regressed"]


class CaseComparison(BaseModel):
    """One gold entry's outcome under both pipelines."""

    model_config = ConfigDict(frozen=True)

    entry_id: str
    category: str
    deterministic: EvalCaseResult
    agent: EvalCaseResult

    @property
    def delta(self) -> CaseDelta:
        det_passed = self.deterministic.passed
        agent_passed = self.agent.passed
        if det_passed and agent_passed:
            return "both_passed"
        if not det_passed and not agent_passed:
            return "both_failed"
        if not det_passed and agent_passed:
            return "progressed"  # agentic helped
        return "regressed"  # agentic hurt


class ComparisonReport(BaseModel):
    """Side-by-side report comparing two pipelines on the same gold set."""

    model_config = ConfigDict(frozen=True)

    timestamp: datetime
    gold_set_path: str
    deterministic_report: EvalReport
    agent_report: EvalReport
    case_comparisons: list[CaseComparison] = Field(default_factory=list)

    @property
    def n_both_passed(self) -> int:
        return sum(1 for c in self.case_comparisons if c.delta == "both_passed")

    @property
    def n_both_failed(self) -> int:
        return sum(1 for c in self.case_comparisons if c.delta == "both_failed")

    @property
    def n_progressed(self) -> int:
        """Cases where agentic passed but deterministic failed."""
        return sum(1 for c in self.case_comparisons if c.delta == "progressed")

    @property
    def n_regressed(self) -> int:
        """Cases where deterministic passed but agentic failed."""
        return sum(1 for c in self.case_comparisons if c.delta == "regressed")


def compare_reports(
    *,
    deterministic: EvalReport,
    agent: EvalReport,
    gold_set_path: str = "",
) -> ComparisonReport:
    """Pair cases by entry_id, build the comparison report.

    Cases that appear in only one report (shouldn't happen for well-formed
    inputs, but defensive) are dropped from the per-case comparison.
    """
    det_by_id = {c.entry_id: c for c in deterministic.cases}
    agent_by_id = {c.entry_id: c for c in agent.cases}
    common_ids = [c.entry_id for c in deterministic.cases if c.entry_id in agent_by_id]

    comparisons = [
        CaseComparison(
            entry_id=entry_id,
            category=det_by_id[entry_id].category,
            deterministic=det_by_id[entry_id],
            agent=agent_by_id[entry_id],
        )
        for entry_id in common_ids
    ]

    return ComparisonReport(
        timestamp=datetime.now(UTC),
        gold_set_path=gold_set_path,
        deterministic_report=deterministic,
        agent_report=agent,
        case_comparisons=comparisons,
    )


def format_comparison(report: ComparisonReport) -> str:
    """Human-readable summary for the CLI."""
    det = report.deterministic_report.overall
    agent = report.agent_report.overall

    lines: list[str] = []
    lines.append(f"Gold set: {report.gold_set_path}")
    lines.append(f"Deterministic model: {report.deterministic_report.model_name}")
    lines.append(f"Agent model:         {report.agent_report.model_name}")
    lines.append("")
    lines.append("                              deterministic   agent     delta")
    lines.append(_row("citation_accuracy", det.citation_accuracy, agent.citation_accuracy))
    lines.append(_row("answer_correctness", det.answer_correctness, agent.answer_correctness))
    lines.append(_row("refusal_rate_correct", det.refusal_rate_correct, agent.refusal_rate_correct))
    lines.append(
        _row(
            "refusal_rate_false_pos",
            det.refusal_rate_false_positive,
            agent.refusal_rate_false_positive,
            lower_is_better=True,
        )
    )
    lines.append(_row("pass_rate", det.pass_rate, agent.pass_rate))
    lines.append("")
    lines.append(
        f"Per-case deltas:"
        f" both_passed={report.n_both_passed}"
        f"  both_failed={report.n_both_failed}"
        f"  progressed={report.n_progressed}  (agentic helped)"
        f"  regressed={report.n_regressed}  (agentic hurt)"
    )

    if report.n_regressed:
        lines.append("")
        lines.append("Regressions (agentic broke a deterministic-pass case):")
        for c in report.case_comparisons:
            if c.delta == "regressed":
                lines.append(
                    f"  - {c.entry_id} ({c.category}): "
                    f"deterministic={c.deterministic.actual_outcome}, "
                    f"agent={c.agent.actual_outcome}"
                )
    if report.n_progressed:
        lines.append("")
        lines.append("Improvements (agentic fixed a deterministic-fail case):")
        for c in report.case_comparisons:
            if c.delta == "progressed":
                lines.append(
                    f"  - {c.entry_id} ({c.category}): "
                    f"deterministic={c.deterministic.actual_outcome}, "
                    f"agent={c.agent.actual_outcome}"
                )
    return "\n".join(lines)


def _row(label: str, det: float, agent: float, *, lower_is_better: bool = False) -> str:
    delta = agent - det
    arrow = ""
    if abs(delta) >= 0.005:
        improving = (delta < 0) if lower_is_better else (delta > 0)
        arrow = " ↑" if improving else " ↓"
    return f"  {label:<26}    {det:>5.2f}        {agent:>5.2f}    {delta:+.2f}{arrow}"
