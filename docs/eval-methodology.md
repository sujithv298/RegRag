# Eval methodology

How RegRAG measures whether it's working.

## The three headline metrics

1. **Citation accuracy** — of the citations the system emits, what fraction resolve to the correct gold-set citation? Penalizes hallucinated and off-by-one citations equally.
2. **Answer correctness** — of the answers the system produces (excluding refusals), what fraction match the gold-set answer keywords? Substring or rubric-based depending on the question category.
3. **Refusal rate** — what fraction of queries the system refused. This is reported in two pieces: refusals on `intentional_refusal` questions (good) and refusals on answerable questions (bad).

## Why not just "correctness"

A system that answers everything with confident hallucination scores 100% on coverage and 0% on usefulness in a regulated context. A system that refuses everything scores 0% on coverage and produces no audit-log noise. Neither is what we want. The three metrics together force the trade-off into the open: a tuning change that increases answer correctness at the cost of refusing more answerable questions is visible in the report, not hidden inside an aggregate score.

## Question categories (set by the gold set)

See `evals/gold/README.md` for the four categories: `exact_citation`, `multi_section`, `intentional_refusal`, `rule_vs_interpretation`.

The per-category breakdown is the load-bearing output of the report. A regression in `multi_section` while `exact_citation` improves usually means the reranker shifted in a way that helps single-source queries and hurts synthesis ones — that's an actionable signal, not just a number that went down.

## Retrieval-only metrics (reported alongside)

- **Chunk recall@k** for k ∈ {5, 10, 20}: did the gold-set citation chunk make it into the top-k retrieved set?
- **Rerank lift**: did the cross-encoder rerank improve the ordering relative to RRF-fused?

These let us tune retrieval independently from the LLM.

## What's *not* measured (yet)

- Latency. Real, but not the v1 priority. Phase 7+ task.
- Cost per query. Same.
- Adversarial robustness (prompt injection, jailbreak resistance). v2 — needs its own gold set.
