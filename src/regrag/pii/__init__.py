"""PII scrubbing.

Applied to user input *before* the LLM call and *before* anything is
written to the audit log. The scrubbed text is what gets persisted; the
raw input never leaves the process.
"""

from __future__ import annotations

from regrag.pii.scrubber import (
    PIIScrubber,
    PresidioPIIScrubber,
    RegexPIIScrubber,
    ScrubResult,
)

__all__ = ["PIIScrubber", "PresidioPIIScrubber", "RegexPIIScrubber", "ScrubResult"]
