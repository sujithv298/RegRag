# RegRAG — Reg-Compliant RAG Template for US Banking Regulations

A production-grade, MIT-licensed reference implementation of retrieval-augmented generation over US banking regulations, designed for engineering teams at banks, credit unions, and fintechs who need to ship LLM-powered compliance tooling without reinventing the plumbing.

**v1 scope:** Regulation E (Electronic Fund Transfers, 12 CFR Part 1005), including Official Interpretations.

**Status:** Alpha. Build-in-public — see [CHANGELOG.md](CHANGELOG.md) for what's landed.

---

## Why this exists

Off-the-shelf LLMs are not usable for regulatory work in a regulated environment. They hallucinate citations to nonexistent CFR sections, have stale knowledge of amended regulations, produce no audit trail for examiners, can't meet bank data-residency requirements, and have no built-in PII handling.

So banks today either avoid LLMs for compliance work, spend 12–18 months building internal tools, or pay enterprise prices for general-purpose vendors that aren't tuned for banking. The hard problems — hierarchy-preserving chunking of CFR text, verifiable citations, audit logging schemas, PII scrubbing, eval methodology for legal Q&A — are patterns, not proprietary IP. Every team rebuilds them, badly. No strong open-source reference exists.

This repo is that reference. Fork it, replace what's bank-specific, ship in weeks instead of quarters.

## What makes this different from a generic RAG tutorial

Five properties, treated as load-bearing requirements rather than nice-to-haves:

1. **Citation guarantee.** Every factual claim resolves to a verified citation in the corpus, or the system refuses to answer. Generated citations are validated against the ingested text, not just emitted by the model.
2. **Hierarchy-preserving chunking.** Chunks carry full structural metadata (Part → Subpart → Section → Paragraph → Interpretation) so retrieval respects the document structure of legal text.
3. **Reproducible audit log.** Every query persists PII-scrubbed input, retrieved chunk IDs, prompt template version, model name and version, response, timestamp, and content hashes — sufficient to reconstruct any answer months later.
4. **Eval harness with gold set.** ~50 hand-written questions spanning exact-citation lookups, multi-section synthesis, intentional-refusal cases, and rule-vs-interpretation distinctions. Reported metrics: citation accuracy, answer correctness, refusal rate.
5. **Model-agnostic.** Adapters for Claude, GPT-4, and a local Llama path, so banks with data-residency restrictions can run the template entirely on-premises.

## Quickstart

```bash
# Requires Python 3.11+ and uv (https://github.com/astral-sh/uv)
uv sync --extra dev
cp .env.example .env  # add your API keys

# Pull Reg E from the eCFR XML API
uv run regrag ingest --part 1005

# Ask a question
uv run regrag query "What are the consumer's liability limits for unauthorized EFTs?"

# Run the eval harness against the gold set
uv run regrag eval --gold evals/gold/reg_e_v1.jsonl
```

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for the data flow and module-by-module breakdown. The short version:

```
eCFR XML  →  Hierarchy-preserving chunker  →  Hybrid index (BM25 + BGE)
                                                       │
   User query  →  PII scrub  →  Retrieve  →  Cross-encoder rerank
                                                       │
                            Prompt with retrieved context  →  LLM adapter
                                                       │
                          Citation extraction  →  Citation verifier
                                                       │
                            Audit log writer  →  Verified answer | Refusal
```

## Roadmap

- **v1.0** — Reg E (12 CFR 1005) end-to-end with all five properties above.
- **v1.1** — Streamlit playground (`examples/playground/`) for non-engineer demos.
- **v1.2** — pgvector store adapter as drop-in for Chroma.
- **v2.0** — Add Reg Z (12 CFR 1026), then BSA/AML, then FFIEC handbooks.
- **v2.x** — Agentic retrieval mode (model-driven query refinement, multi-hop).

## Non-goals

This is **not** legal advice. **Not** a replacement for a compliance officer. **Not** production-ready for any specific bank without adaptation. **Not** a closed-source SaaS product.

If you fork this and ship it, you own the regulatory exposure. The template gives you the plumbing and the patterns; the legal and risk review of how you use it is on you.

## Security

API keys for cloud LLMs (Anthropic, OpenAI) are read from environment variables — never from source. Put them in `.env` (gitignored, auto-loaded by `python-dotenv`) or inject through your CI's secrets store. A `detect-secrets` pre-commit hook catches accidental commits.

See [SECURITY.md](SECURITY.md) for the full key-handling protocol, including what to do if a key ever lands in a commit (rotate, don't scrub).

## License

MIT. See [LICENSE](LICENSE).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). PRs that add a new regulation under `src/regrag/ingest/`, expand the gold set, or improve eval methodology are especially welcome.
