"""Parse eCFR Part XML into typed `HierarchyNode` leaves.

The eCFR XML schema in one paragraph: a Part is a `<DIV5 TYPE="PART">`,
containing `<DIV6 TYPE="SUBPART">` subparts, containing `<DIV8 TYPE="SECTION">`
sections, containing `<P>` paragraphs. The Official Interpretations live in a
sibling `<DIV9 TYPE="APPENDIX">` titled "Supplement I to Part {N}".

Paragraph hierarchy is encoded by leading designators in the `<P>` text
(e.g. `(a)`, `(b)`, `(b)(1)`, `(b)(1)(i)`), not by XML nesting. The parser
strips those designators from the text and lifts them into `paragraph_path`.

Important caveat (Phase 2 v0.1): this parser was developed against a
handcrafted XML fixture modeled on the documented eCFR schema, because the
sandbox could not reach www.ecfr.gov. When run against live eCFR data, edge
cases may surface — most likely candidates: paragraphs whose designators use
context (e.g. a `<P>` inside Section 1005.6(b) that just says `(1) ...` and
inherits the `(b)` from the previous `<P>`), embedded tables (`<GPOTABLE>`),
and `<EXTRACT>` elements for indented quoted text. The extractor below
handles full-path designators robustly; contextual-designator handling is
flagged TODO and falls back to a best-effort previous-path heuristic.
"""

from __future__ import annotations

import re

from lxml import etree

from regrag.chunking.hierarchy import HierarchyNode

# Regex matching one or more parenthesized designators at the start of a
# paragraph: "(a)", "(b)(1)", "(b)(1)(i)", "(b)(1)(i)(A)", etc.
# Each captured group is one level. Permits letters, digits, lowercase roman.
_DESIGNATOR_RE = re.compile(r"^((?:\([A-Za-z0-9ivxlcIVXLC]+\))+)\s*")

# Regex for Official Interpretation comment numbers like "1.", "2.", "10."
_COMMENT_NUM_RE = re.compile(r"^(\d+)\.\s+")

# Regex for the comment-context line like "6(b) Limitations on amount..."
# which tells us which section/paragraph the following numbered comments
# attach to.
_COMMENT_CONTEXT_RE = re.compile(r"^(\d+(?:\([^)]+\))*)\s")


def parse_part_xml(xml_bytes: bytes, *, title: int = 12) -> list[HierarchyNode]:
    """Parse eCFR Part XML into a flat list of leaf nodes in document order.

    Args:
        xml_bytes: Raw XML returned by `regrag.ingest.ecfr_client.fetch_part_xml`.
        title: CFR Title number. Defaults to 12 (banking).

    Returns:
        Leaf `HierarchyNode`s in document order. Both rule paragraphs and
        Official Interpretation comments are returned in one list; consumers
        distinguish via `node.is_interpretation`.

    Raises:
        ValueError: if no DIV5 (PART) element is found.
    """
    root = etree.fromstring(xml_bytes)
    part_div = _find_part_div(root)
    if part_div is None:
        raise ValueError("No <DIV5 TYPE='PART'> element found in XML.")

    part_n = part_div.get("N")
    if part_n is None:
        raise ValueError("DIV5 element missing required 'N' attribute (Part number).")
    part = int(part_n)

    nodes: list[HierarchyNode] = []
    nodes.extend(_parse_rule_text(part_div, title=title, part=part))
    nodes.extend(_parse_interpretations(part_div, title=title, part=part))
    return nodes


# ---- Internal helpers ----


def _find_part_div(root: etree._Element) -> etree._Element | None:
    """Locate the `<DIV5 TYPE='PART'>` element. The eCFR endpoint may return
    it as the document root or wrapped in higher-level DIVs; handle both."""
    if root.tag == "DIV5" and root.get("TYPE") == "PART":
        return root
    return root.find(".//DIV5[@TYPE='PART']")


def _parse_rule_text(part_div: etree._Element, *, title: int, part: int) -> list[HierarchyNode]:
    """Walk DIV6 (Subpart) → DIV8 (Section) → P (Paragraph)."""
    nodes: list[HierarchyNode] = []
    for subpart_div in part_div.findall("DIV6"):
        subpart = subpart_div.get("N")
        for section_div in subpart_div.findall("DIV8"):
            section = _normalize_section_id(section_div.get("N"))
            section_heading = _text_content(section_div.find("HEAD"))
            last_path: tuple[str, ...] = ()
            for p_elem in section_div.findall("P"):
                raw_text = _text_content(p_elem)
                designators, body = _extract_designators(raw_text, last_path)
                last_path = designators
                nodes.append(
                    HierarchyNode(
                        title=title,
                        part=part,
                        subpart=subpart,
                        section=section,
                        section_heading=section_heading,
                        paragraph_path=designators,
                        text=body,
                    )
                )
    return nodes


def _parse_interpretations(
    part_div: etree._Element, *, title: int, part: int
) -> list[HierarchyNode]:
    """Walk Supplement I (DIV9 APPENDIX) → per-section DIV9 → P comments."""
    nodes: list[HierarchyNode] = []
    supp = part_div.find("DIV9[@TYPE='APPENDIX']")
    if supp is None:
        return nodes

    for section_block in supp.findall("DIV9"):
        # The section heading is "Section 1005.6 Liability of ..."; we only
        # need the section number for comment_id construction.
        head_text = _text_content(section_block.find("HEAD"))
        section_num = _section_num_from_interp_head(head_text) or str(part)

        # Walk paragraphs. Lines of the form "6(b) ..." establish context;
        # lines of the form "1. ..." are numbered comments that attach to
        # the most recent context.
        current_context: str | None = None
        for p_elem in section_block.findall("P"):
            text = _text_content(p_elem).strip()
            ctx_match = _COMMENT_CONTEXT_RE.match(text)
            num_match = _COMMENT_NUM_RE.match(text)
            if num_match:
                num = num_match.group(1)
                body = text[num_match.end() :].strip()
                comment_id = (
                    f"{current_context}-{num}" if current_context else f"{section_num}-{num}"
                )
                nodes.append(
                    HierarchyNode(
                        title=title,
                        part=part,
                        is_interpretation=True,
                        comment_id=comment_id,
                        text=body,
                    )
                )
            elif ctx_match:
                current_context = ctx_match.group(1)
    return nodes


def _extract_designators(raw_text: str, last_path: tuple[str, ...]) -> tuple[tuple[str, ...], str]:
    """Pull leading paragraph designators off raw text.

    Returns (designator_path, body). If no designator is found, falls back to
    `last_path` (best-effort contextual handling for paragraphs that omit the
    parent designator).
    """
    text = raw_text.lstrip()
    match = _DESIGNATOR_RE.match(text)
    if not match:
        return last_path, text
    designator_block = match.group(1)
    body = text[match.end() :].strip()
    # Split "(b)(1)(i)" into ("b", "1", "i")
    designators = tuple(re.findall(r"\(([^)]+)\)", designator_block))
    return designators, body


def _normalize_section_id(raw: str | None) -> str | None:
    """`N="§ 1005.6"` → `"1005.6"`. Strip the section symbol and whitespace."""
    if raw is None:
        return None
    # Section symbol U+00A7 plus surrounding whitespace
    return raw.replace("§", "").strip() or None


def _section_num_from_interp_head(head: str) -> str | None:
    """`"Section 1005.6 Liability ..."` → `"6"` (just the part-relative section number).

    The interpretation comment_id format we want is e.g. `6(b)(1)-1`, where
    `6` is the section number relative to the Part. So we extract just the
    trailing `.6` from `1005.6`.
    """
    m = re.search(r"\b\d+\.(\d+)\b", head)
    return m.group(1) if m else None


def _text_content(elem: etree._Element | None) -> str:
    """Flatten an element's text content, including text from inline children
    like `<I>` italic spans. Strips only the structural noise; preserves the
    designators because the caller relies on them."""
    if elem is None:
        return ""
    # itertext() yields all text in document order, recursively.
    return "".join(elem.itertext()).strip()
