# Chunking

How structured CFR text becomes retrievable chunks without losing structure.

## The naive approach (don't do this)

A recursive character splitter at 1024 tokens with 200-token overlap will work for blog posts. For legal text it will:

- Split mid-paragraph, breaking the citation address. A chunk that starts halfway through § 1005.6(b)(1) and ends halfway through § 1005.6(b)(2) cannot be cited correctly.
- Lose section headers. The model retrieves the body of (b)(1) but doesn't know it's about consumer liability limits.
- Smear Official Interpretations into rule text. The Interpretation comments to § 1005.6 are formally separate from § 1005.6 itself; conflating them is a real correctness bug.

## What we do instead

1. **Walk the hierarchy.** Each leaf node (paragraph, comment) is the chunking unit. Containers (Part, Subpart, Section) contribute metadata, not text.
2. **Split within leaf nodes only when needed.** A paragraph that exceeds the embedding context gets split — but only inside that paragraph, never across paragraphs.
3. **Duplicate ancestor headers into each chunk's text.** A chunk for § 1005.6(b)(1) starts with "§ 1005.6 Liability of consumer for unauthorized transfers / (b) Limitations on amount of liability" before the body. This costs tokens and gains retrieval recall.
4. **Tag every chunk with full metadata.** `title`, `part`, `subpart`, `section`, `paragraph`, `is_interpretation`, `citation_path`. The verifier downstream uses `citation_path` as the canonical address.

## Invariants enforced by tests

- Every chunk has exactly one `citation_path`.
- Chunk text is a contiguous substring of the source XML's text content (no paraphrasing during chunking).
- No chunk straddles a Section boundary.
- For every leaf in the source XML, at least one chunk descends from it.

These get tested in `tests/unit/test_chunker.py` once Phase 3 lands.
