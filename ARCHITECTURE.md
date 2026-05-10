# Architecture

This document explains the shape of the system and the rationale for each piece. Decisions that were close calls live in `docs/adr/`.

## Design principles

1. **Determinism over cleverness.** The pipeline is one-shot retrieve-then-generate, not agentic. Every additional LLM decision is another thing an examiner has to reason about. v1 keeps the LLM's role narrow: ground in retrieved context, emit answer + citations, refuse if unsupported.
2. **Citations are part of the contract, not a post-hoc nicety.** The citation extractor and verifier are first-class modules, not utilities. An answer that doesn't pass citation verification doesn't get returned — it gets converted to a refusal.
3. **Structure-preserving from end to end.** Regulatory text has hierarchy that carries semantic weight (a paragraph in §1005.6(b)(1) means something specific). The ingestion, chunking, retrieval, and citation modules all carry that structure forward; nothing is flattened to "just text."
4. **Auditability as a build-time invariant.** The audit log schema is checked-in, versioned, and validated. Every code path that produces an answer also produces an audit record; this is enforced in `pipeline.py`, not left to discipline.
5. **Swappability at every external boundary.** Vector store, LLM, embedding model, and reranker are all behind interfaces. A bank with a data-residency mandate should be able to swap the LLM adapter without touching the retrieval logic.

## Data flow

```
┌───────────────────────────────────────────────────────────────────────┐
│                         INGESTION (offline, one-shot)                 │
│                                                                       │
│   eCFR XML API  ──►  ecfr_client  ──►  parser  ──►  hierarchy nodes   │
│                                                            │          │
│                                                            ▼          │
│                                                       chunker         │
│                                                            │          │
│                                                            ▼          │
│                                  store (Chroma)  +  BM25 index        │
└───────────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────────┐
│                        QUERY (online, per request)                    │
│                                                                       │
│   user query                                                          │
│       │                                                               │
│       ▼                                                               │
│   pii.scrubber  ──►  scrubbed query  ──┐                              │
│       │                                │                              │
│       ▼                                ▼                              │
│   audit.logger (start)        retrieval.hybrid                        │
│                                       │                               │
│                                       ▼                               │
│                              retrieval.rerank (cross-encoder)         │
│                                       │                               │
│                                       ▼                               │
│                              prompt template (versioned)              │
│                                       │                               │
│                                       ▼                               │
│                              models.{anthropic,openai,llama_local}    │
│                                       │                               │
│                                       ▼                               │
│                              citations.extractor                      │
│                                       │                               │
│                                       ▼                               │
│                              citations.verifier  ──► refuse?          │
│                                       │                               │
│                                       ▼                               │
│                              audit.logger (commit)                    │
│                                       │                               │
│                                       ▼                               │
│                              verified answer  |  refusal              │
└───────────────────────────────────────────────────────────────────────┘
```

## Module map

Each module owns one concern. The five differentiators map onto modules so that "where does X live?" has a single answer.

| Module | Owns | Maps to differentiator |
|--------|------|-----------------------|
| `regrag.ingest` | Fetching eCFR XML, parsing it into typed hierarchy nodes | (2) Hierarchy-preserving chunking — upstream half |
| `regrag.chunking` | Converting hierarchy nodes into retrievable chunks while keeping structural metadata attached | (2) Hierarchy-preserving chunking — downstream half |
| `regrag.retrieval` | BM25, dense embeddings, hybrid orchestration, cross-encoder rerank | Foundation for (1) Citation guarantee — better recall = more verifiable answers |
| `regrag.store` | Vector + metadata persistence (Chroma in v1, pgvector path documented) | Infrastructure |
| `regrag.models` | LLM adapter protocol + Claude / GPT-4 / local-Llama implementations | (5) Model-agnostic |
| `regrag.citations` | Extracting citations from generated text, verifying them against the ingested corpus | (1) Citation guarantee |
| `regrag.pii` | Pre-LLM PII scrubbing via Microsoft Presidio | Compliance hygiene + audit-log safety |
| `regrag.audit` | Audit-log JSON schema + writer; one record per query, examiner-grade | (3) Reproducible audit log |
| `regrag.pipeline` | The single entrypoint that wires the above modules together for one query | Enforces the invariant: every answer produces an audit record |
| `regrag.cli` | `regrag ingest`, `regrag query`, `regrag eval` | Developer surface |
| `evals/` | Eval runner + gold set + metrics reports | (4) Eval harness |

## Key external dependencies and why

| Dep | Why this one |
|-----|--------------|
| `httpx` | eCFR API client. Async-capable, modern, good test ergonomics with `respx`. |
| `lxml` | XML parsing. eCFR XML is structurally rich; lxml's XPath support is the cleanest way to walk it. |
| `pydantic` v2 | Typed hierarchy nodes, audit-log records, eval gold-set entries. JSON serialization for free. |
| `rank-bm25` | Lightweight BM25 implementation. Good enough for v1 corpus size; trivial to swap if needed. |
| `sentence-transformers` | BGE embeddings + cross-encoder rerank in one package. Local inference path means no API call required for retrieval. |
| `chromadb` | Vector store. Embedded mode = zero infra for forkers to start. pgvector adapter documented for production. |
| `presidio-analyzer` / `presidio-anonymizer` | PII scrubbing. Microsoft-maintained, deny-list-extensible, used by compliance-conscious teams already. |
| `anthropic` / `openai` / `llama-cpp-python` | LLM adapters. Optional `[local]` extra keeps the on-prem path opt-in to avoid forcing a heavy dep on cloud-only forkers. |
| `structlog` | Structured JSON logging. The audit log uses this; aligns with what bank observability stacks expect. |
| `click` + `rich` | CLI ergonomics. |

## What is explicitly *not* in v1

- **Agentic retrieval.** No multi-hop, no LLM-driven query refinement. v1 is one-shot.
- **Streaming responses.** Returns are all-or-nothing because verification has to run on the complete output.
- **Multi-tenancy / auth.** This is a template, not a service.
- **Web UI.** A Streamlit playground is on the v1.1 roadmap; the slot is reserved at `examples/playground/`.
- **Fine-tuning.** Off the table for v1 — undermines reproducibility and forkability.

## Where the agentic future plugs in (if and when)

If/when v2 adds an agentic mode, it lives as a new entrypoint (e.g., `regrag.pipeline_agent`) that uses the same retrieval, citation, and audit modules underneath. The classic-RAG `pipeline.py` stays as the default and the contract surface for examiners stays unchanged.
