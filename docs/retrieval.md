# Retrieval

Hybrid: BM25 + BGE dense embeddings, fused with reciprocal rank fusion, then reranked with a cross-encoder.

## Why hybrid

Pure dense retrieval underweights rare exact-match tokens (regulation numbers, defined terms in quotes) that legal questions hinge on. Pure BM25 misses paraphrase. Hybrid gets both.

## Defaults (single-Part corpus, ~1k chunks)

- BM25 top-k: 20
- Dense top-k: 20
- RRF constant: 60 (the standard)
- Cross-encoder rerank top-N: 10
- Final top-M passed to the LLM: 5

## Why these k values

- Small corpus (~1k chunks). Top-20 from each retriever covers the relevant slice without dragging in noise.
- 5 chunks to the LLM keeps prompt token cost reasonable and gives the citation verifier a manageable set to check against.

## Tuning

Phase 7's eval harness reports retrieval-only metrics (chunk-recall@k for the gold-set citations) so these defaults can be tuned with data, not guesswork.

## Reranker model

`BAAI/bge-reranker-base` by default. Local inference. Forkers with stricter latency budgets can swap in a smaller cross-encoder via the LLMAdapter-style interface.
