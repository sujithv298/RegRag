"""Typed hierarchy node model — the contract between ingest and chunking.

A `HierarchyNode` represents one *leaf* in the CFR hierarchy: a paragraph in
the rule text, or a comment in the Official Interpretations. Container
elements (Title, Part, Subpart, Section) are not modeled as nodes — their
identity is carried as fields on the leaves they contain.

That design choice (flat list of leaves, not a tree) is deliberate:

  - The chunker consumes a flat sequence; modeling the tree would force it
    to flatten anyway.
  - The retriever treats each chunk as a unit; the tree is a parsing artifact.
  - The audit log references nodes by `chunk_id`, which is a string, not a
    tree path. A flat list keeps that lookup O(1).

If a future feature needs the tree (e.g., expanding a chunk to "show me the
whole Section it came from"), reconstructing it from the leaves is cheap
because every leaf carries its full address.

Two computed properties are the load-bearing API:

  - `citation_path` — the canonical citation address; what the verifier
    checks model output against.
  - `chunk_id` — a stable string ID; what the audit log records.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class HierarchyNode(BaseModel):
    """One leaf in the CFR hierarchy: a rule paragraph or an interpretation comment.

    Field naming mirrors how a human cites these things — `12 CFR 1005.6(b)(1)`
    breaks down into title=12, part=1005, section="1005.6", paragraph_path=("b", "1").
    """

    model_config = ConfigDict(frozen=True)

    # Address fields (every node carries the full breadcrumb)
    title: int = Field(description="CFR Title number, e.g. 12 for banking.")
    part: int = Field(description="CFR Part number, e.g. 1005 for Regulation E.")
    subpart: str | None = Field(
        default=None, description="Subpart letter, e.g. 'A'. None for Part-level matter."
    )
    section: str | None = Field(
        default=None, description="Section identifier, e.g. '1005.6'. None for non-section matter."
    )
    section_heading: str | None = Field(
        default=None,
        description="Heading text of the containing section, duplicated onto every leaf so "
        "retrieval can match against it.",
    )
    paragraph_path: tuple[str, ...] = Field(
        default=(),
        description="Paragraph designator path, e.g. ('b', '1') for §1005.6(b)(1). "
        "Empty for section-level introductory matter.",
    )

    # Interpretation fields (mutually exclusive with paragraph_path being meaningful;
    # is_interpretation=True means this node lives in Supplement I, not the rule itself)
    is_interpretation: bool = Field(
        default=False, description="True if this leaf comes from Supplement I (Official Interp)."
    )
    comment_id: str | None = Field(
        default=None,
        description="Official Interpretation comment identifier, e.g. '6(b)(1)-1'. "
        "Set iff is_interpretation is True.",
    )

    # Content
    text: str = Field(description="The actual paragraph or comment text content.")

    # ---- Computed identity ----

    @property
    def citation_path(self) -> str:
        """Canonical citation address — what the verifier checks against.

        Examples:
            12 CFR 1005.6(b)(1)
            12 CFR 1005, Comment 6(b)(1)-1
        """
        if self.is_interpretation:
            return f"{self.title} CFR {self.part}, Comment {self.comment_id}"
        if self.section is None:
            return f"{self.title} CFR {self.part}"
        para = "".join(f"({p})" for p in self.paragraph_path)
        return f"{self.title} CFR {self.section}{para}"

    @property
    def chunk_id(self) -> str:
        """Stable identifier for this node, used as chunk_id by the store and audit log.

        Format is grep-friendly so a human reading an audit log can find the
        referenced chunk by eye.
        """
        if self.is_interpretation:
            return f"comment:{self.part}:{self.comment_id}"
        section = self.section or f"part-{self.part}"
        para = ".".join(self.paragraph_path) or "intro"
        return f"section:{section}:{para}"
