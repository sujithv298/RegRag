"""PII scrubbers — strip sensitive data before logging or LLM calls.

Two implementations:

  - `RegexPIIScrubber`     — conservative regex patterns for SSN, credit
                              card, email, phone. Pure Python, zero deps.
                              Good enough for v1 dev/test. Will miss
                              names, addresses, and account numbers.

  - `PresidioPIIScrubber`  — Microsoft Presidio analyzer + anonymizer.
                              Covers ~30 PII categories including names,
                              addresses, IBANs, dates of birth. Heavyweight
                              (downloads spaCy English model on first use).
                              The production default for any bank deployment.

Both expose the same `scrub(text) -> ScrubResult` API.

The audit log records `pii_redaction_count`, never the raw values.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

# ---- Patterns (order matters: most specific first to avoid mis-matching) ----

# SSN: 123-45-6789
_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")

# Credit card: 4 groups of 4 digits, optionally separated by space or dash.
# Conservative — won't catch unspaced 16-digit runs to reduce false positives.
_CARD_RE = re.compile(r"\b(?:\d{4}[-\s]){3}\d{4}\b")

# Email: standard local@domain.tld
_EMAIL_RE = re.compile(
    r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"
)

# US phone: optional +1 / 1, optional parens, separators of -, ., space.
_PHONE_RE = re.compile(
    r"(?<!\d)(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b"
)


@dataclass(frozen=True)
class ScrubResult:
    """Output of a scrub call. Holds redacted text and a count, never the values."""

    text: str
    redaction_count: int


@runtime_checkable
class PIIScrubber(Protocol):
    """The scrubber contract."""

    @property
    def name(self) -> str:
        """Stable identifier; recorded in the audit log."""

    def scrub(self, text: str) -> ScrubResult:
        ...


# ---- RegexPIIScrubber ----


class RegexPIIScrubber:
    """Conservative regex-based scrubber. Default for offline / sandboxed envs."""

    @property
    def name(self) -> str:
        return "regex"

    def scrub(self, text: str) -> ScrubResult:
        count = 0
        # Order: most specific first.
        for pattern, label in (
            (_SSN_RE, "[REDACTED:SSN]"),
            (_CARD_RE, "[REDACTED:CARD]"),
            (_EMAIL_RE, "[REDACTED:EMAIL]"),
            (_PHONE_RE, "[REDACTED:PHONE]"),
        ):
            text, n = pattern.subn(label, text)
            count += n
        return ScrubResult(text=text, redaction_count=count)


# ---- PresidioPIIScrubber ----


class PresidioPIIScrubber:
    """Real PII scrubbing via Microsoft Presidio. Lazy-loaded.

    Loads `presidio-analyzer` + `presidio-anonymizer` on first `scrub`.
    Will pull the spaCy English model (~50MB) the first time it runs,
    which requires network access. If you're running in a sandboxed
    environment, use `RegexPIIScrubber` instead.
    """

    @property
    def name(self) -> str:
        return "presidio"

    def __init__(self) -> None:
        self._analyzer: object | None = None
        self._anonymizer: object | None = None

    def scrub(self, text: str) -> ScrubResult:
        self._ensure_loaded()
        assert self._analyzer is not None  # noqa: S101
        assert self._anonymizer is not None  # noqa: S101
        results = self._analyzer.analyze(text=text, language="en")  # type: ignore[attr-defined]
        if not results:
            return ScrubResult(text=text, redaction_count=0)
        anonymized = self._anonymizer.anonymize(text=text, analyzer_results=results)  # type: ignore[attr-defined]
        return ScrubResult(text=anonymized.text, redaction_count=len(results))

    def _ensure_loaded(self) -> None:
        if self._analyzer is not None:
            return
        try:
            from presidio_analyzer import AnalyzerEngine  # noqa: PLC0415
            from presidio_anonymizer import AnonymizerEngine  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "PresidioPIIScrubber requires presidio-analyzer and "
                "presidio-anonymizer. Install via `uv sync`, or use "
                "RegexPIIScrubber for offline-only environments."
            ) from exc
        self._analyzer = AnalyzerEngine()
        self._anonymizer = AnonymizerEngine()
