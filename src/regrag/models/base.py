"""LLMAdapter protocol — the contract every model implementation honors.

Every adapter:
  - has a stable `name` and `version` recorded in the audit log
  - exposes `generate(system, user, max_tokens, temperature)` returning a
    `GenerationResult`

The adapter abstraction is the load-bearing abstraction for "model-agnostic":
a bank with data-residency restrictions swaps `AnthropicAdapter` for
`LlamaLocalAdapter` and nothing else in the pipeline changes.

Future adapters (Bedrock, Vertex, internal bank-hosted models) implement
this protocol — no inheritance, no edits elsewhere.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from regrag.models.types import GenerationResult


@runtime_checkable
class LLMAdapter(Protocol):
    """The model contract."""

    @property
    def name(self) -> str:
        """Stable identifier; recorded in the audit log."""

    @property
    def version(self) -> str:
        """Provider-reported version, or a stable string if unavailable."""

    def generate(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> GenerationResult:
        """Run one completion. Default temperature 0.0 — determinism matters
        for audit-log replay; callers can raise it for exploratory work."""
