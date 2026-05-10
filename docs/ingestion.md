# Ingestion

How regulation text gets into the system.

## Source

eCFR XML API. Endpoint pattern:

```
GET https://www.ecfr.gov/api/versioner/v1/full/{date}/title-{title}.xml?part={part}
```

Example for Reg E as of 2026-01-01:

```
https://www.ecfr.gov/api/versioner/v1/full/2026-01-01/title-12.xml?part=1005
```

## Why XML over HTML or PDF

XML carries the structural elements as tags — `DIV3` (Part), `DIV4` (Subpart), `DIV5` (Section), `DIV9` (Official Interpretations), `<P>` paragraphs with `N` attributes giving the paragraph identifier. That structure is what the chunker preserves and what the citation verifier later checks against.

HTML scraping loses the tag semantics under inconsistent CSS. PDFs lose paragraph addressing entirely.

## Snapshot dates

Every ingestion is pinned to a specific eCFR snapshot date. The audit log records this as `corpus_snapshot_date`. Re-ingesting against a newer date is a deliberate operation that creates a new corpus version — old audit-log records still resolve against the old snapshot if you keep it on disk.

## Quirks (will be filled in during Phase 2)

- *(placeholder — will document any DIV-tag oddities or missing-attribute cases we hit during real ingestion)*
