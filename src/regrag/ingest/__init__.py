"""Ingestion: fetch CFR XML and parse it into typed hierarchy nodes.

The output of this module is the input to `regrag.chunking`. The contract
between them is the `HierarchyNode` pydantic model defined in
`regrag.chunking.hierarchy` (so chunking owns the schema; ingestion produces it).

Why XML over HTML scraping or PDFs:
- eCFR XML carries the structural elements as XML tags (DIV5 = Part,
  DIV6 = Subpart, DIV8 = Section, DIV9 = Supplement I). Parsing HTML loses
  that without a fragile mapping.
- PDFs are out — they lose paragraph-level addressing, which is the whole
  basis of citation in legal text.

See `docs/ingestion.md` for endpoint specifics and known quirks.
"""

from __future__ import annotations

from regrag.ingest.ecfr_client import ECFRError, fetch_part_xml
from regrag.ingest.parser import parse_part_xml

__all__ = ["ECFRError", "fetch_part_xml", "parse_part_xml"]
