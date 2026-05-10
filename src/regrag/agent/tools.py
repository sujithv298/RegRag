"""Tools the agent LLM can call.

Each tool is a class implementing `Tool`. The orchestrator looks up tools
by `name`, validates the arguments roughly, and invokes `run`. Results are
serialized to a string before being handed back to the LLM.

We define three tools for v2.0:

  - `SearchRegulations`     — the bread-and-butter retrieval call. Same
                              hybrid retriever the deterministic pipeline
                              uses. The agent typically calls this first.
  - `GetChunkByCitation`    — direct lookup of a specific citation_path,
                              for cases where the LLM already knows what
                              section it wants to read in full.
  - `ExpandChunkContext`    — given a chunk_id, return the surrounding
                              chunks in the same Section. Useful for
                              follow-up "what does the rest of this
                              section say?" questions.
"""

from __future__ import annotations

import json
from typing import Any, Protocol, runtime_checkable

from regrag.chunking import Chunk
from regrag.retrieval.hybrid import HybridRetriever


@runtime_checkable
class Tool(Protocol):
    """Tool contract."""

    @property
    def name(self) -> str:
        """Name the LLM uses to invoke this tool."""

    @property
    def description(self) -> str:
        """Plain-English description shown to the LLM in the prompt."""

    @property
    def schema(self) -> dict[str, Any]:
        """JSON Schema for arguments (mirrors Anthropic's tool-use shape)."""

    def run(self, arguments: dict[str, Any]) -> str:
        """Execute and return a string for the LLM. Errors should be
        returned as descriptive strings, not raised."""


# ---- SearchRegulations ----


class SearchRegulations:
    """Hybrid-retrieve top-k chunks for a query."""

    def __init__(self, retriever: HybridRetriever) -> None:
        self._retriever = retriever

    @property
    def name(self) -> str:
        return "search_regulations"

    @property
    def description(self) -> str:
        return (
            "Search the indexed regulation corpus for chunks matching a query. "
            "Use this when you need to find regulation text by topic, paraphrase, "
            "or partial citation. Returns up to k chunks with their citations and text."
        )

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural-language query or partial citation.",
                },
                "k": {
                    "type": "integer",
                    "description": "Number of results to return (default 5).",
                    "default": 5,
                },
            },
            "required": ["query"],
        }

    def run(self, arguments: dict[str, Any]) -> str:
        query = arguments.get("query", "")
        k = int(arguments.get("k", 5))
        if not query:
            return "Error: 'query' argument is required and must be non-empty."
        results = self._retriever.search(query, k=k)
        if not results:
            return "No matching regulation chunks found for that query."
        return _format_chunks_for_llm(
            [r.chunk for r in results],
            heading=f"Top {len(results)} chunks for query: {query!r}",
        )


# ---- GetChunkByCitation ----


class GetChunkByCitation:
    """Direct lookup of a chunk by its canonical citation path."""

    def __init__(self, corpus: list[Chunk]) -> None:
        self._by_citation: dict[str, Chunk] = {}
        for c in corpus:
            # Multiple chunks can share a citation_path (split paragraphs).
            # Keep the first; the orchestrator can call ExpandChunkContext for more.
            self._by_citation.setdefault(c.citation_path, c)

    @property
    def name(self) -> str:
        return "get_chunk_by_citation"

    @property
    def description(self) -> str:
        return (
            "Look up a regulation chunk by its canonical citation, e.g. "
            "'12 CFR 1005.6(b)(1)' or '12 CFR 1005, Comment 6(b)-1'. "
            "Use when you already know the citation you want to read."
        )

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "citation_path": {
                    "type": "string",
                    "description": "Canonical citation, e.g. '12 CFR 1005.6(b)(1)'.",
                },
            },
            "required": ["citation_path"],
        }

    def run(self, arguments: dict[str, Any]) -> str:
        citation = arguments.get("citation_path", "").strip()
        if not citation:
            return "Error: 'citation_path' argument is required."
        chunk = self._by_citation.get(citation)
        if chunk is None:
            return (
                f"No chunk found for citation '{citation}'. "
                "The citation may not exist in the corpus or may use a non-canonical form."
            )
        return _format_chunks_for_llm([chunk], heading=f"Chunk for {citation}")


# ---- ExpandChunkContext ----


class ExpandChunkContext:
    """Return the chunks adjacent to a given chunk_id within the same Section."""

    def __init__(self, corpus: list[Chunk]) -> None:
        self._by_id: dict[str, Chunk] = {c.chunk_id: c for c in corpus}
        # Pre-group by section for fast neighbor lookup.
        self._by_section: dict[str, list[Chunk]] = {}
        for c in corpus:
            key = c.section or f"part-{c.part}"
            self._by_section.setdefault(key, []).append(c)

    @property
    def name(self) -> str:
        return "expand_chunk_context"

    @property
    def description(self) -> str:
        return (
            "Given a chunk_id, return that chunk plus its neighbors in the same "
            "Section (in document order). Use when one chunk's context is "
            "insufficient and you want surrounding paragraphs."
        )

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "chunk_id": {"type": "string"},
                "neighbors": {
                    "type": "integer",
                    "description": "How many neighbors on each side (default 1).",
                    "default": 1,
                },
            },
            "required": ["chunk_id"],
        }

    def run(self, arguments: dict[str, Any]) -> str:
        chunk_id = arguments.get("chunk_id", "")
        neighbors = int(arguments.get("neighbors", 1))
        if not chunk_id:
            return "Error: 'chunk_id' argument is required."
        target = self._by_id.get(chunk_id)
        if target is None:
            return f"No chunk found with chunk_id '{chunk_id}'."
        section_key = target.section or f"part-{target.part}"
        siblings = self._by_section.get(section_key, [])
        # Find the target's position in siblings.
        try:
            idx = siblings.index(target)
        except ValueError:
            return _format_chunks_for_llm([target], heading="Requested chunk")
        start = max(0, idx - neighbors)
        end = min(len(siblings), idx + neighbors + 1)
        return _format_chunks_for_llm(
            siblings[start:end], heading=f"Chunk and {end - start - 1} neighbor(s)"
        )


# ---- Helpers ----


def _format_chunks_for_llm(chunks: list[Chunk], *, heading: str) -> str:
    """Render a list of chunks as a string for the LLM.

    We emit JSON-friendly text rather than a real JSON object because the
    LLM has to reason over it; structured but human-readable wins."""
    parts: list[str] = [heading]
    for i, c in enumerate(chunks, start=1):
        parts.append(
            f"\n[{i}] chunk_id={c.chunk_id}\n"
            f"    citation: {c.citation_path}\n"
            f"    heading:  {c.section_heading or '(none)'}\n"
            f"    text:     {c.source_text}"
        )
    return "\n".join(parts)


def serialize_tool_result_for_audit(result: Any) -> str:
    """Stable string form of a tool result for the audit log."""
    if isinstance(result, str):
        return result
    try:
        return json.dumps(result, default=str)
    except (TypeError, ValueError):
        return str(result)
