"""RegRAG — Reg-Compliant RAG Template for US banking regulations.

Public surface lives at module top-level for forker convenience:

    from regrag import answer_query, ingest_part

Most users go through `regrag.cli` or the CLI entrypoint `regrag`.
See ARCHITECTURE.md for the full module map.
"""

from __future__ import annotations

__version__ = "0.1.0"

# Public re-exports land here once the underlying modules are implemented.
# Kept empty for v0.1 so import-time side effects stay zero.
__all__: list[str] = []
