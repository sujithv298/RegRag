"""BM25 keyword retrieval.

Backed by `rank-bm25`'s BM25Okapi. The algorithm itself is a well-understood
30-year-old keyword scorer that weights rare words higher than common ones
and saturates the contribution of repeated terms. We don't reimplement it.

What we *do* care about here is **tokenization** — how we split chunk text
and queries into terms. Generic tokenizers stem aggressively (turning
"liability" into "liabl") and strip punctuation, which destroys matching on
the things that matter most for legal text:

  - Regulation numbers like ``1005.6`` (a stemmer would split on the dot)
  - Paragraph designators like ``(b)(1)`` (stripped by most tokenizers)
  - Defined terms in title case (lowercased, fine, but not stemmed)

Our tokenizer is intentionally minimal: lowercase, then extract runs of
alphanumeric characters with optional `.<digits>` extensions for numeric
tokens like ``1005.6``. No stemming, no stopword removal. Legal text doesn't
benefit from either.
"""

from __future__ import annotations

import re

from rank_bm25 import BM25Plus

from regrag.chunking import Chunk
from regrag.retrieval.stopwords import ENGLISH_STOPWORDS
from regrag.retrieval.types import ScoredChunk

# Token = a run of letters/digits, optionally followed by `.digits`.
# This keeps "1005.6" as one token and "(b)(1)" as ["b", "1"].
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")


def tokenize(text: str) -> list[str]:
    """Lowercase + extract alphanumeric tokens, preserving numeric `.` chains.

    Exposed at module level so tests can verify tokenization independently.
    Stopwords are NOT removed here — chunks index everything so that
    citation strings and quoted phrases remain searchable. Use
    `tokenize_query` for the query-side stripping.
    """
    return _TOKEN_RE.findall(text.lower())


def tokenize_query(query: str) -> list[str]:
    """Tokenize a query AND strip English stopwords.

    Stripping at query time only (not at index time) keeps the index
    complete while preventing the failure mode where an accidental match
    on a high-IDF stopword like "does" outranks a deliberate match on
    a rare topical token like "1005.6".
    """
    return [t for t in tokenize(query) if t not in ENGLISH_STOPWORDS]


class BM25Index:
    """In-memory BM25 index over a fixed set of Chunks.

    Usage:
        index = BM25Index()
        index.add(chunks)
        results = index.search("consumer liability unauthorized transfer", k=5)

    Adding chunks rebuilds the underlying BM25 model (rank-bm25 doesn't
    support incremental updates). For v1 corpus sizes (~1k chunks) that's
    fine; for larger corpora we'd switch to a persistent inverted index.
    """

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        # BM25Plus instead of BM25Okapi: avoids negative IDFs on terms that
        # appear in >50% of the corpus, which is common in small/specialized
        # corpora like a single CFR Part where the part number appears in
        # most chunks. With BM25Okapi, matching such a term would *lower*
        # the score instead of raising it.
        self._bm25: BM25Plus | None = None

    def add(self, chunks: list[Chunk]) -> None:
        """Add chunks to the index. Rebuilds the BM25 model from scratch."""
        self._chunks.extend(chunks)
        if not self._chunks:
            return
        tokenized = [tokenize(c.text) for c in self._chunks]
        self._bm25 = BM25Plus(tokenized)

    def search(self, query: str, *, k: int = 10) -> list[ScoredChunk]:
        """Return the top-k chunks for `query`, ranked by BM25 score.

        Chunks with a non-positive score (no query-term overlap) are filtered
        out — returning them would just be noise.
        """
        if self._bm25 is None or not self._chunks:
            return []
        query_tokens = tokenize_query(query)
        if not query_tokens:
            return []
        scores = self._bm25.get_scores(query_tokens)
        ranked = sorted(
            enumerate(scores), key=lambda pair: pair[1], reverse=True
        )
        results: list[ScoredChunk] = []
        for idx, score in ranked[:k]:
            if score <= 0:
                continue
            results.append(
                ScoredChunk(
                    chunk=self._chunks[idx],
                    score=float(score),
                    retriever="bm25",
                )
            )
        return results

    def __len__(self) -> int:
        return len(self._chunks)
