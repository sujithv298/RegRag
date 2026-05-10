"""The `Chunk` model — what chunking produces and what retrieval indexes.

A `Chunk` is one searchable unit. In v1, one chunk = one CFR paragraph
(or one Official Interpretation comment). Long paragraphs get split into
multiple chunks; short ones become a single chunk. Either way, every chunk
carries enough metadata to (a) be cited correctly and (b) be reconstructed
back to its source for the verifier.

Two text fields, one for retrieval and one for verification:

    text         — what the embedder sees and what BM25 indexes. Includes a
                   citation prefix and the section heading prepended to the
                   body, so a search for either the citation string or the
                   heading topic matches.

    source_text  — the original paragraph body, with no prefix, no heading.
                   This is what the citation verifier later compares model
                   output against. Keeping it separate from `text` means the
                   prefix we added for retrieval doesn't pollute verification.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class Chunk(BaseModel):
    """One searchable unit. Most v1 leaves produce exactly one Chunk."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str = Field(
        description="Stable unique identifier. For un-split chunks, equals "
        "the source HierarchyNode's chunk_id. For split chunks, suffixed "
        "with `::N` where N is the split index."
    )
    text: str = Field(description="What gets embedded and indexed (prefix + heading + body).")
    source_text: str = Field(
        description="The original paragraph body, no prefix. Verifier uses this."
    )
    citation_path: str = Field(description="Canonical CFR citation, copied from source node.")

    # Source-node metadata, denormalized for retrieval-time access without
    # round-tripping back to ingest output.
    title: int
    part: int
    subpart: str | None = None
    section: str | None = None
    section_heading: str | None = None
    paragraph_path: tuple[str, ...] = ()
    is_interpretation: bool = False
    comment_id: str | None = None

    # Split-tracking. When a paragraph is too long for the embedding context
    # we split it; these fields make the split traceable.
    is_split: bool = False
    split_index: int = 0
    split_total: int = 1
