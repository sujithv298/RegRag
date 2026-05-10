"""Parse citations out of model output.

The prompt template (Phase 6) instructs the model to emit citations inline
in one of two formats:

    Tagged (preferred):  "...consumer's liability is $50 [CFR:1005.6(b)(1)]."
    Free-form fallback:  "...per 12 CFR 1005.6(b)(1), the consumer..."

Tagged is preferred because it's unambiguous and easy to parse. Free-form
exists because models occasionally drop the tag despite instructions, and
we'd rather catch a real citation late than miss it entirely. Both produce
the same canonical `12 CFR <body>` form for the verifier.

Each extracted citation is paired with the *claim* — the sentence the
citation sits inside. The verifier checks whether the cited chunk
substantively supports that claim, not just whether the chunk exists.
"""

from __future__ import annotations

import re

from regrag.citations.types import ExtractedCitation

# Tagged: [CFR:1005.6(b)(1)] or [CFR:Comment 6(b)-1]
_TAGGED_RE = re.compile(r"\[CFR:([^\]]+)\]")

# Free-form rule citation: "12 CFR 1005.6(b)(1)" / "12 CFR 1005.6"
_FREE_RULE_RE = re.compile(r"\b(\d+)\s+CFR\s+(\d+\.\d+(?:\([A-Za-z0-9]+\))*)")

# Free-form Official Interpretation citation:
# "12 CFR 1005, Comment 6(b)(1)-1" or "12 CFR Part 1005, Comment 6(b)-1"
_FREE_COMMENT_RE = re.compile(
    r"\b(\d+)\s+CFR\s+(?:Part\s+)?(\d+)\s*,\s*Comment\s+([0-9A-Za-z()\-.]+)"
)


class CitationExtractor:
    """Pull citations out of free-form text in canonical form."""

    def __init__(self, *, default_title: int = 12, default_part: int = 1005) -> None:
        self._default_title = default_title
        self._default_part = default_part

    def extract(self, text: str) -> list[ExtractedCitation]:
        seen: set[tuple[str, int, int]] = set()
        results: list[ExtractedCitation] = []

        for m in _TAGGED_RE.finditer(text):
            citation_path = self._canonicalize_tagged(m.group(1).strip())
            self._add(results, seen, citation_path, m.start(), m.end(), text)

        for m in _FREE_COMMENT_RE.finditer(text):
            title, part, comment = m.groups()
            citation_path = f"{title} CFR {part}, Comment {comment}"
            self._add(results, seen, citation_path, m.start(), m.end(), text)

        for m in _FREE_RULE_RE.finditer(text):
            title, body = m.groups()
            citation_path = f"{title} CFR {body}"
            self._add(results, seen, citation_path, m.start(), m.end(), text)

        return results

    # ---- Internal ----

    def _canonicalize_tagged(self, raw: str) -> str:
        """Normalize a `[CFR:...]` payload to a `12 CFR ...` canonical form."""
        raw = raw.strip()
        # Already canonical
        if re.match(r"^\d+\s+CFR\s+", raw):
            return raw
        # "Comment 6(b)-1" inside tag
        if raw.lower().startswith("comment "):
            return f"{self._default_title} CFR {self._default_part}, {raw}"
        # Bare "1005.6(b)(1)" or "Comment 6(b)-1" without prefix
        return f"{self._default_title} CFR {raw}"

    def _add(
        self,
        results: list[ExtractedCitation],
        seen: set[tuple[str, int, int]],
        citation_path: str,
        start: int,
        end: int,
        text: str,
    ) -> None:
        key = (citation_path, start, end)
        if key in seen:
            return
        seen.add(key)
        results.append(
            ExtractedCitation(
                citation_path=citation_path,
                claim=_surrounding_sentence(text, start, end),
                span=(start, end),
            )
        )


# ---- Sentence-window helper ----

# Boundary chars: the claim is the sentence the citation lives in.
_SENTENCE_END = re.compile(r"[.!?]\s")


def _surrounding_sentence(text: str, start: int, end: int) -> str:
    """Return the sentence containing the [start:end] span, plus — if needed —
    the preceding sentence.

    Two cases the heuristic distinguishes:

    1. Citation is mid-sentence ("Liability is $50 [CFR:...].").
       The claim is the sentence the citation lives in.

    2. Citation immediately follows a sentence-end with nothing between
       ("Liability is $50. [CFR:...]" — common when an LLM emits the
       citation as a trailing footnote-style annotation).
       In this case the claim is the *previous* sentence — the one the
       citation is actually annotating.

    We detect case 2 by checking whether there's any non-whitespace text
    between the most recent sentence-end and the citation start. If not,
    walk one more sentence back.
    """
    sentence_ends = list(_SENTENCE_END.finditer(text, 0, start))
    if not sentence_ends:
        s_start = 0
    else:
        last_end = sentence_ends[-1].end()
        between = text[last_end:start].strip()
        if between:
            # Citation is inside the sentence that started at last_end.
            s_start = last_end
        elif len(sentence_ends) >= 2:
            # Citation is at the start of a new "sentence" — include the
            # previous sentence (the actual claim).
            s_start = sentence_ends[-2].end()
        else:
            s_start = 0

    next_end = _SENTENCE_END.search(text, end)
    s_end = next_end.start() + 1 if next_end else len(text)
    return text[s_start:s_end].strip()
