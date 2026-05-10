"""Citation extraction and verification — the citation guarantee.

`CitationExtractor` parses the model's response for citations.
`CitationVerifier` checks each one against the ingested corpus.

The verifier is the **fail-closed safety gate**. An answer whose citations
don't all verify is converted to a refusal upstream in `regrag.pipeline`,
not returned to the user.
"""

from __future__ import annotations

from regrag.citations.extractor import CitationExtractor
from regrag.citations.types import (
    ExtractedCitation,
    VerificationResult,
    VerifiedCitation,
)
from regrag.citations.verifier import CitationVerifier

__all__ = [
    "CitationExtractor",
    "CitationVerifier",
    "ExtractedCitation",
    "VerificationResult",
    "VerifiedCitation",
]
