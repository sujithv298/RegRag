# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Phase 9 — Streamlit playground (`examples/playground/app.py`).** Three-tab demo surface (Ask, Try-to-break-it, How-it-works) wrapping the existing pipelines. Animated agent reasoning trace, green/red citation badges (✓/✗), expandable per-query audit log JSON, preset attack prompts. Sidebar lets you swap between deterministic and agent mode and between offline `fake-good` / `fake-bad` adapters and real Anthropic (auto-detected via `ANTHROPIC_API_KEY` in `.env`). Launches with `make playground` or `streamlit run examples/playground/app.py`. **The wow surface — what you show on a screen, not in a terminal.**

- Project skeleton: repo structure, packaging via `uv` + `hatchling`, lint via `ruff`, type-checking via `mypy`, tests via `pytest`.
- Module stubs for ingest, chunking, retrieval, store, models, citations, pii, audit.
- Architecture documentation and ADR scaffolding.
- Eval harness directory layout and gold-set placeholder for Reg E v1.
- **Phase 2 — eCFR ingestion.**
  - `regrag.chunking.hierarchy.HierarchyNode` pydantic model with computed `citation_path` and `chunk_id`.
  - `regrag.ingest.ecfr_client.fetch_part_xml` async httpx client with snapshot-date pinning for audit-log reproducibility.
  - `regrag.ingest.parser.parse_part_xml` lxml-based parser handling DIV5/DIV6/DIV8/DIV9 and nested paragraph designators `(a)` / `(b)(1)` / `(b)(1)(i)`.
  - Handcrafted Reg E excerpt fixture at `tests/fixtures/reg_e_part_1005_excerpt.xml`.
  - 19 unit tests covering parser invariants and HTTP-client behavior (mocked with `respx`).
  - CLI `regrag ingest` and `regrag inspect` commands wired end-to-end.
- **Phase 3 — hierarchy-preserving chunking.**
  - `regrag.chunking.chunk.Chunk` pydantic model with separate `text` (for retrieval, with citation + heading prefix) and `source_text` (for verification, no prefix).
  - `regrag.chunking.chunker.chunk_nodes` algorithm. One chunk per leaf paragraph by default; in-paragraph splitting only when a leaf exceeds the embedder's context budget.
  - 10 chunker tests covering all five invariants (every leaf produces a chunk, no straddling, citation prefix present, unique chunk_ids, splits partition the original text).
  - `regrag ingest` CLI extended to chunk after parsing; new `regrag inspect-chunks` debugging command.
- **Phase 4a — BM25 keyword search.**
  - `regrag.retrieval.bm25.BM25Index` backed by `rank-bm25`'s BM25Plus (avoids negative-IDF on small corpora).
  - Legal-text-aware tokenizer: keeps `1005.6` intact, splits `(b)(1)` cleanly, no stemming.
  - Conservative English stopword list applied at query time only (modal verbs, negation, and quantifiers preserved for legal correctness).
  - `regrag.retrieval.types.ScoredChunk` model with `retriever` label for audit-log provenance.
  - 12 BM25 + tokenizer tests; documented honest limit (BM25 alone can't always pick the exact paragraph for citation-only queries — Phase 4b/4c address that).
  - `regrag search` CLI command runs end-to-end against a chunks JSONL.
- **Phase 4b — dense embeddings + vector store.**
  - `regrag.retrieval.embeddings.Embedder` protocol with two implementations: `BGEEmbedder` (real semantic, lazy-loaded `BAAI/bge-small-en-v1.5`) and `HashingEmbedder` (deterministic offline-only stub for tests/sandboxed environments).
  - `regrag.store.base.VectorStore` protocol with two implementations: `InMemoryVectorStore` (pure-Python brute-force cosine, default for v1) and `ChromaStore` (chromadb-backed, real-vector-database swap-in).
  - `regrag.retrieval.dense.DenseRetriever` thin composer that ties Embedder + VectorStore.
  - 22 new unit tests; 63 total tests pass.
  - `regrag search` CLI extended with `--retriever bm25|dense|both` to show BM25 and dense retrieval side-by-side on the same query.
- **Phase 4c — hybrid retrieval + cross-encoder rerank.**
  - `regrag.retrieval.fusion.rrf_fuse` — Reciprocal Rank Fusion. Parameter-free, scale-independent merge of multiple ranked lists.
  - `regrag.retrieval.rerank` — `Reranker` protocol with three implementations: `NoOpReranker` (identity), `LexicalOverlapReranker` (offline-only stub re-scoring by query-token overlap with `source_text`), `CrossEncoderReranker` (real, lazy-loaded `BAAI/bge-reranker-base`).
  - `regrag.retrieval.hybrid.HybridRetriever` — composes BM25 + dense + reranker into one search API. Default candidate_k=20, rrf_k=60.
  - 27 new unit tests; 90 total pass.
  - `regrag search --retriever hybrid` (and `all`) wired into the CLI; offline demo shows that hybrid+rerank lands the canonical answer chunk at #1 even when BM25 and dense individually rank it #2.
- **Phase 5 — citations + audit + PII (the regulatory differentiators).**
  - `regrag.citations.extractor.CitationExtractor` parses `[CFR:...]` tagged citations and free-form `12 CFR 1005.6(b)(1)` / `12 CFR 1005, Comment 6(b)-1` citations from model output, pairing each with the surrounding sentence (the claim).
  - `regrag.citations.verifier.CitationVerifier` is the **fail-closed gate**: every citation must (a) resolve to a real chunk in the corpus and (b) have its claim supported by the chunk's source_text above a token-overlap threshold (default 0.3). Any failure → response converted to a refusal. A response with zero citations also fails (`no_citations_emitted`).
  - `regrag.audit.schema.AuditRecord` pydantic model — versioned (`SCHEMA_VERSION = "1"`) examiner-grade JSON schema with prompt/response SHA-256 hashes for tamper detection.
  - `regrag.audit.logger.AuditLogger` append-only JSONL writer with `make_record(...)` helper.
  - `regrag.pii.scrubber.RegexPIIScrubber` (offline default, redacts SSN / credit card / email / phone) and `PresidioPIIScrubber` (production, lazy-loaded Microsoft Presidio).
  - 34 new unit tests; 124 total pass.
  - End-to-end offline demo shows: PII scrubbed before logging, valid citation answered, hallucinated citation refused with `refusal_reason="unverified_citation"`.
- **Phase 6 — model adapters + top-level pipeline.**
  - `regrag.models.LLMAdapter` protocol with four implementations: `FakeAdapter` (deterministic stand-in for tests/offline demos), `AnthropicAdapter` (Claude), `OpenAIAdapter` (GPT-4), `LlamaLocalAdapter` (on-prem llama-cpp-python). All cloud + local adapters lazy-load.
  - `regrag.prompt` — versioned prompt template (`PROMPT_TEMPLATE_VERSION = "0.1.0"`) with strict instructions: use only retrieved excerpts, emit `[CFR:...]` inline citations, exact-phrase refusal when unanswerable. Bumped versions are recorded in the audit log.
  - `regrag.pipeline.answer_query` — the top-level integration point. Linear, explicit: PII scrub → hybrid retrieve → render prompt → call LLM → extract citations → verify (fail-closed) → write audit record → return AnswerResult. **Invariant: one query = one audit record, enforced by structure.**
  - `_surrounding_sentence` heuristic in citation extractor improved to handle the common pattern of citations appended after a complete sentence (model emits citation as a trailing annotation).
  - 19 new unit tests including end-to-end pipeline tests through `FakeAdapter` covering answered, refused-by-hallucination, refused-by-no-citations, and PII-redaction-before-logging paths. **143 total tests pass.**
  - `regrag query` CLI command runs the full pipeline with `--model fake-good | fake-bad | anthropic | openai`. Demo shows answered + refused outcomes plus the per-record JSONL audit log on disk.
- **API-key safety guards.**
  - `python-dotenv` auto-load in CLI; `.env` is the canonical location for keys (gitignored).
  - `detect-secrets` pre-commit hook config to catch accidental key commits.
  - `SECURITY.md` documenting the key-handling protocol, including rotation guidance for already-leaked keys.
- **Phase 7 — eval harness + gold set.**
  - `evals.types` — `GoldEntry`, `EvalCaseResult`, `EvalMetrics`, `EvalReport` pydantic models.
  - `evals.runner` — `run_eval()` and `compute_metrics()` pure functions. Each gold entry runs through the full pipeline, producing one audit record + one EvalCaseResult.
  - `evals/gold/reg_e_v1.jsonl` — 25 hand-written gold-set entries across `exact_citation` (12), `multi_section` (5), `intentional_refusal` (5), `rule_vs_interpretation` (3). Expansion to ~50 entries flagged as v1.1 work.
  - Headline metrics: `citation_accuracy`, `answer_correctness`, `refusal_rate_correct`, `refusal_rate_false_positive`, `pass_rate` — overall and per-category.
  - 13 new unit tests; **152 total tests pass.**
  - `regrag eval` CLI runs the harness end-to-end, prints metrics, writes a JSON report to `evals/reports/`. Smoke run on the test gold-set fixture achieves **pass_rate 1.00**.
