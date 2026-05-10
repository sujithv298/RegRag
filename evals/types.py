"""Pydantic models for the eval harness.

`GoldEntry`           — one hand-written test case from the gold-set JSONL.
`EvalCaseResult`      — what the runner records per case: actual outcome,
                        actual citations, and a pass/fail per dimension.
`EvalMetrics`         — the headline numbers: citation_accuracy,
                        answer_correctness, refusal_rate_correct,
                        refusal_rate_false_positive. Reported overall and
                        per-category.
`EvalReport`          — case results + metrics + run metadata.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Category = Literal[
    "exact_citation",
    "multi_section",
    "intentional_refusal",
    "rule_vs_interpretation",
]


class GoldEntry(BaseModel):
    """One hand-written test case."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(description="Stable id, e.g. 'regE-001'.")
    category: Category
    question: str
    expected_citations: list[str] = Field(
        default_factory=list,
        description="Canonical citation paths the system should emit. Empty for refusal cases.",
    )
    expected_answer_keywords: list[str] = Field(
        default_factory=list,
        description="Substrings that must appear in the answer text. Case-insensitive.",
    )
    should_refuse: bool = Field(
        default=False,
        description="True if the gold answer is 'refuse' (out-of-scope, not-in-corpus, etc.).",
    )
    notes: str = Field(default="", description="Author notes; ignored by the runner.")


class EvalCaseResult(BaseModel):
    """Per-case outcome from the runner."""

    model_config = ConfigDict(frozen=True)

    entry_id: str
    category: Category
    question: str

    actual_outcome: Literal["answered", "refused"]
    actual_refusal_reason: str | None = None
    actual_citations: list[str] = Field(default_factory=list)
    actual_answer: str

    # Per-dimension pass/fail. The aggregate `passed` rolls these up.
    citations_correct: bool
    keywords_present: bool
    outcome_matches_expected: bool

    @property
    def passed(self) -> bool:
        """A case passes only when every relevant dimension passed.

        For should_refuse cases, only outcome alignment matters.
        For should-answer cases, outcome + citations + keywords all matter.
        """
        if self.actual_outcome == "refused" and not self.outcome_matches_expected:
            return False  # false-positive refusal
        if self.actual_outcome == "refused" and self.outcome_matches_expected:
            return True  # correctly refused
        # outcome == "answered"
        return self.outcome_matches_expected and self.citations_correct and self.keywords_present


class EvalMetrics(BaseModel):
    """Aggregate metrics over a slice of cases (overall or one category)."""

    model_config = ConfigDict(frozen=True)

    n_cases: int
    n_should_answer: int
    n_should_refuse: int

    citation_accuracy: float = Field(
        description="Among should-answer cases, fraction whose actual citations "
        "include all expected citations."
    )
    answer_correctness: float = Field(
        description="Among should-answer cases, fraction whose actual answer "
        "contains all expected keywords."
    )
    refusal_rate_correct: float = Field(
        description="Among should-refuse cases, fraction the system correctly refused."
    )
    refusal_rate_false_positive: float = Field(
        description="Among should-answer cases, fraction the system wrongly refused. "
        "This is the over-refusal noise floor."
    )
    pass_rate: float = Field(description="Fraction of cases where `EvalCaseResult.passed` is True.")


class EvalReport(BaseModel):
    """Output of one run of the eval harness."""

    model_config = ConfigDict(frozen=True)

    timestamp: datetime
    gold_set_path: str
    model_name: str
    prompt_template_version: str

    overall: EvalMetrics
    by_category: dict[str, EvalMetrics]
    cases: list[EvalCaseResult]
