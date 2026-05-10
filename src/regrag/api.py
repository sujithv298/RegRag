"""FastAPI service exposing the RegRAG pipeline over HTTP.

Single endpoint plus a static-file mount for the playground UI:

  - POST /query              one-shot agentic query, returns AnswerResult JSON
  - GET  /healthz            liveness check
  - GET  /                   serves examples/web/index.html
  - GET  /static/*           served from examples/web/

Run:

    uv sync --extra playground
    make setup
    uv run uvicorn regrag.api:app --reload
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

import contextlib

from regrag.audit import AuditLogger
from regrag.chunking import Chunk
from regrag.models import FakeAdapter
from regrag.pipeline_agent import answer_query_agentic
from regrag.retrieval import (
    BM25Index,
    DenseRetriever,
    HashingEmbedder,
    HybridRetriever,
    LexicalOverlapReranker,
)
from regrag.store import InMemoryVectorStore

# ---- Paths ----

_ROOT = Path(__file__).resolve().parents[2]
_WEB_DIR = _ROOT / "examples" / "web"
_CHUNKS_PATH = _ROOT / "data" / "reg_e_excerpt.chunks.jsonl"
_AUDIT_PATH = _ROOT / "audit-playground.jsonl"
_CACHE_PATH = _ROOT / "data" / "query_cache.jsonl"

# Cache invalidation pinned to the corpus + prompt template version. Bump
# either of these and old cache entries become unreachable (their keys
# won't match new lookups).
_CORPUS_SNAPSHOT_DATE = "2025-01-01"
_AGENT_PROMPT_VERSION = "agent-0.1.0"


# ---- Lazy-built pipeline (built on first request) ----


class _Pipeline:
    def __init__(self) -> None:
        self.chunks: list[Chunk] | None = None
        self.retriever: HybridRetriever | None = None
        self.logger: AuditLogger | None = None

    def build(self) -> None:
        if self.chunks is not None:
            return
        if not _CHUNKS_PATH.exists():
            raise RuntimeError(
                f"Chunks file not found at {_CHUNKS_PATH}. "
                "Run `make setup` from the project root first."
            )
        chunks = [
            Chunk.model_validate_json(line)
            for line in _CHUNKS_PATH.read_text().splitlines()
            if line.strip()
        ]
        retriever = HybridRetriever(
            bm25=BM25Index(),
            dense=DenseRetriever(embedder=HashingEmbedder(), store=InMemoryVectorStore()),
            reranker=LexicalOverlapReranker(),
        )
        retriever.add(chunks)
        self.chunks = chunks
        self.retriever = retriever
        self.logger = AuditLogger(log_path=_AUDIT_PATH)


_pipeline = _Pipeline()


# ---- Response cache ----
#
# Keyed on (normalized question, model, corpus_snapshot_date, prompt_template_version).
# Same question + same model + same corpus → return cached answer, skip the
# LLM call entirely. Persisted as JSONL so the cache survives server restarts.
#
# Trade-offs:
#   - We cache the *full* response, including the audit record. So a cached
#     reply has a stale `query_id` and `timestamp` from the original run.
#     This is fine for cost-savings + latency demos but not for compliance
#     audit replay — set `?fresh=1` on /query to bypass cache when you need
#     a fresh audit record.
#   - Cache is invalidated by changing the corpus snapshot date or bumping
#     the prompt template version (both are baked into the key).


class _ResponseCache:
    def __init__(self) -> None:
        self._mem: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._loaded = False

    @staticmethod
    def _key(question: str, model: str) -> str:
        raw = json.dumps(
            {
                "q": question.strip().lower(),
                "model": model,
                "corpus": _CORPUS_SNAPSHOT_DATE,
                "prompt": _AGENT_PROMPT_VERSION,
            },
            sort_keys=True,
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    def _load(self) -> None:
        if self._loaded:
            return
        if _CACHE_PATH.exists():
            for line in _CACHE_PATH.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                    self._mem[entry["key"]] = entry["response"]
                except (json.JSONDecodeError, KeyError):
                    continue
        self._loaded = True

    def get(self, question: str, model: str) -> dict | None:
        with self._lock:
            self._load()
            return self._mem.get(self._key(question, model))

    def set(self, question: str, model: str, response: dict) -> None:
        with self._lock:
            self._load()
            key = self._key(question, model)
            self._mem[key] = response
            _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            with _CACHE_PATH.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"key": key, "response": response}) + "\n")

    def clear(self) -> int:
        with self._lock:
            n = len(self._mem)
            self._mem.clear()
            if _CACHE_PATH.exists():
                try:
                    _CACHE_PATH.unlink()
                except OSError:
                    # Best-effort: if we can't delete (e.g. read-only fs),
                    # at least empty it.
                    with contextlib.suppress(OSError):
                        _CACHE_PATH.write_text("")
            self._loaded = True
            return n

    def size(self) -> int:
        with self._lock:
            self._load()
            return len(self._mem)


_cache = _ResponseCache()


# ---- Request / response models ----


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    model: Literal["fake-good", "fake-bad", "anthropic", "openai"] = "fake-good"


class CitationOut(BaseModel):
    citation_path: str
    chunk_id: str | None
    exists: bool
    supports: bool
    support_score: float
    passed: bool
    failure_reason: str | None


class QueryResponse(BaseModel):
    outcome: Literal["answered", "refused"]
    refusal_reason: str | None
    answer: str
    citations: list[CitationOut]
    retrieved_chunk_ids: list[str]
    model_name: str
    audit_query_id: str
    has_anthropic_key: bool
    cached: bool = False


# ---- Adapter builder ----


def _agent_adapter(model_id: str):
    if model_id == "fake-good":
        from regrag.cli import _fake_good_agent_responder

        return FakeAdapter(
            agent_responder=_fake_good_agent_responder,
            name="fake-good",
            version="v0",
        )
    if model_id == "fake-bad":
        from regrag.cli import _fake_bad_agent_responder

        return FakeAdapter(
            agent_responder=_fake_bad_agent_responder,
            name="fake-bad",
            version="v0",
        )
    if model_id == "anthropic":
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise HTTPException(
                status_code=400,
                detail="ANTHROPIC_API_KEY not set. Add it to .env to use real Claude.",
            )
        from regrag.models import AnthropicAgentAdapter

        return AnthropicAgentAdapter()
    if model_id == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise HTTPException(
                status_code=400,
                detail="OPENAI_API_KEY not set. Add it to .env to use real GPT-4.",
            )
        from regrag.models import OpenAIAgentAdapter

        return OpenAIAgentAdapter()
    raise HTTPException(status_code=400, detail=f"unknown model {model_id}")


# ---- FastAPI app ----

app = FastAPI(
    title="RegRAG",
    description="Verifiable RAG over US banking regulations.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/info")
def info() -> dict[str, object]:
    """Static info the UI can use to enable/disable real-model options."""
    return {
        "version": "0.1.0",
        "has_anthropic_key": bool(os.getenv("ANTHROPIC_API_KEY")),
        "has_openai_key": bool(os.getenv("OPENAI_API_KEY")),
        "corpus_size": len(_pipeline.chunks) if _pipeline.chunks else None,
        "cache_size": _cache.size(),
    }


@app.post("/cache/clear")
def clear_cache() -> dict[str, int]:
    """Clear the response cache (in-memory + on-disk JSONL)."""
    n = _cache.clear()
    return {"cleared": n}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest, fresh: bool = False) -> QueryResponse:
    """Run an agentic query, optionally bypassing the cache.

    Pass `?fresh=1` to bypass cache (e.g., when you need a fresh audit
    record or are debugging). Default is to use cache.
    """
    # ---- Cache lookup ----
    if not fresh:
        cached = _cache.get(req.question, req.model)
        if cached is not None:
            payload = {**cached, "cached": True}
            return QueryResponse.model_validate(payload)

    # ---- Cache miss → run the pipeline ----
    _pipeline.build()
    assert _pipeline.retriever is not None
    assert _pipeline.chunks is not None
    assert _pipeline.logger is not None

    adapter = _agent_adapter(req.model)
    result = answer_query_agentic(
        req.question,
        retriever=_pipeline.retriever,
        model=adapter,
        audit_logger=_pipeline.logger,
        corpus=_pipeline.chunks,
        corpus_snapshot_date=_CORPUS_SNAPSHOT_DATE,
    )

    response_payload = {
        "outcome": result.outcome,
        "refusal_reason": result.refusal_reason,
        "answer": result.answer,
        "citations": [
            {
                "citation_path": c.citation_path,
                "chunk_id": c.chunk_id,
                "exists": c.exists,
                "supports": c.supports,
                "support_score": c.support_score,
                "passed": c.passed,
                "failure_reason": c.failure_reason,
            }
            for c in result.citations
        ],
        "retrieved_chunk_ids": list(result.audit_record.retrieved_chunk_ids),
        "model_name": result.audit_record.model_name,
        "audit_query_id": result.audit_query_id,
        "has_anthropic_key": bool(os.getenv("ANTHROPIC_API_KEY")),
        "cached": False,
    }
    # Persist to cache for next time
    _cache.set(req.question, req.model, response_payload)

    # `model_validate` instead of `**dict` spread so mypy can resolve the
    # field types from the pydantic schema rather than the inferred union.
    return QueryResponse.model_validate(response_payload)


# ---- Static files (the UI itself) ----


@app.get("/")
def index() -> FileResponse:
    index_path = _WEB_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(
            status_code=500,
            detail=f"UI index.html not found at {index_path}",
        )
    return FileResponse(index_path)


# Serve examples/web/* as /static/*
if _WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=_WEB_DIR), name="static")
