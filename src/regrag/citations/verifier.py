"""Verify extracted citations against the ingested corpus. Fail closed.

For every `ExtractedCitation` from `CitationExtractor.extract`:

    1. Existence — does `citation_path` resolve to a chunk in the corpus?
       If no, the citation was hallucinated. Fail.

    2. Support — does the chunk's `source_text` actually contain language
       supporting the surrounding claim? v1 uses a token-overlap heuristic
       (fraction of meaningful claim tokens that appear in the chunk).
       Phase 7's eval harness tunes the threshold against the gold set;
       for now the default is conservative (0.3).

If any citation in the response fails either check, the response is
refused — even if other citations passed. **Bank-grade systems return one
unverified-claim refusal rather than three verified plus one hallucinated.**

Future work (post-v1):
- Replace token-overlap support check with a small NLI/entailment model.
- Span-level support: which exact substring of the chunk supports each
  claim, for a "highlight" UI in the playground.
"""

from __future__ import annotations

from regrag.chunking import Chunk
from regrag.citations.types import (
    ExtractedCitation,
    VerificationResult,
    VerifiedCitation,
)
from regrag.retrieval.bm25 import tokenize, tokenize_query

DEFAULT_SUPPORT_THRESHOLD = 0.3


class CitationVerifier:
    """Verify extracted citations against a fixed corpus.

    The corpus is indexed once at construction; verification is O(N) per
    response in the number of extracted citations.
    """

    def __init__(
        self,
        *,
        corpus: list[Chunk],
        support_threshold: float = DEFAULT_SUPPORT_THRESHOLD,
    ) -> None:
        # Index by citation_path. If a citation maps to multiple chunks
        # (split paragraphs), join the source_texts so support checks see
        # the full original paragraph.
        self._chunk_id_by_citation: dict[str, str] = {}
        self._source_text_by_citation: dict[str, str] = {}
        for c in corpus:
            cp = c.citation_path
            if cp in self._source_text_by_citation:
                self._source_text_by_citation[cp] = (
                    self._source_text_by_citation[cp] + " " + c.source_text
                )
            else:
                self._source_text_by_citation[cp] = c.source_text
                self._chunk_id_by_citation[cp] = c.chunk_id
        self._support_threshold = support_threshold

    def verify(
        self, extracted: list[ExtractedCitation]
    ) -> VerificationResult:
        verified: list[VerifiedCitation] = []
        for ec in extracted:
            source_text = self._source_text_by_citation.get(ec.citation_path)
            if source_text is None:
                verified.append(
                    VerifiedCitation(
                        citation_path=ec.citation_path,
                        claim=ec.claim,
                        chunk_id=None,
                        exists=False,
                        supports=False,
                        support_score=0.0,
                    )
                )
                continue
            score = self._support_score(ec.claim, source_text)
            verified.append(
                VerifiedCitation(
                    citation_path=ec.citation_path,
                    claim=ec.claim,
                    chunk_id=self._chunk_id_by_citation.get(ec.citation_path),
                    exists=True,
                    supports=score >= self._support_threshold,
                    support_score=score,
                )
            )
        return VerificationResult(citations=verified)

    # ---- Internal ----

    def _support_score(self, claim: str, source_text: str) -> float:
        """Fraction of meaningful claim tokens that appear in the source.

        We strip stopwords from the claim (so "what does the regulation
        say about..." doesn't dominate) but not from the source — chunks
        are indexed in full and we want to match against the full body.
        """
        claim_tokens = set(tokenize_query(claim))
        if not claim_tokens:
            return 0.0
        source_tokens = set(tokenize(source_text))
        return len(claim_tokens & source_tokens) / len(claim_tokens)
