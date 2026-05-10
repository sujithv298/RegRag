"""Async HTTP client for the eCFR XML API.

Endpoint:
    GET https://www.ecfr.gov/api/versioner/v1/full/{date}/title-{title}.xml?part={part}

Two design points worth flagging:

1. Snapshot-date pinning. Every fetch takes a `date` (ISO-8601). The eCFR
   API serves the corpus *as of* that date. The audit log later records
   this date as `corpus_snapshot_date`, which is what makes audit-log
   replay reproducible. Drifting dates between ingest and replay would
   break the citation verifier silently — same `12 CFR 1005.6(b)(1)`
   citation but different text behind it.

2. Async even though we typically make one call. Async keeps the door
   open for the obvious next step (batch ingestion of N parts in parallel
   when v2.0 expands beyond Reg E) without forcing a rewrite. Cost today
   is one extra `asyncio.run` in the CLI; benefit later is meaningful.
"""

from __future__ import annotations

import httpx

ECFR_BASE_URL = "https://www.ecfr.gov/api/versioner/v1/full"


class ECFRError(RuntimeError):
    """Raised on a non-2xx response from the eCFR API."""


async def fetch_part_xml(
    *,
    title: int,
    part: int,
    date: str,
    client: httpx.AsyncClient | None = None,
    timeout: float = 30.0,
) -> bytes:
    """Fetch the raw XML for a CFR Part at a given eCFR snapshot date.

    Args:
        title: CFR title number, e.g. 12.
        part: CFR part number, e.g. 1005 for Regulation E.
        date: ISO date string, e.g. "2025-01-01". Must be a date the eCFR
            API recognizes (typically any historical date with a daily
            snapshot, or "current" sentinel — see eCFR docs).
        client: Optional injected httpx client. Tests use this with respx;
            production code passes None and we open a one-shot client.
        timeout: Per-request timeout in seconds. Default 30s; CFR Parts can
            be hundreds of KB.

    Returns:
        Raw XML bytes. Caller passes to `regrag.ingest.parser.parse_part_xml`.

    Raises:
        ECFRError: on a non-2xx response.
    """
    url = f"{ECFR_BASE_URL}/{date}/title-{title}.xml"
    params = {"part": str(part)}

    if client is None:
        async with httpx.AsyncClient(timeout=timeout) as new_client:
            return await _fetch(new_client, url, params)
    return await _fetch(client, url, params)


async def _fetch(
    client: httpx.AsyncClient, url: str, params: dict[str, str]
) -> bytes:
    response = await client.get(url, params=params)
    if response.status_code != 200:
        raise ECFRError(
            f"eCFR returned {response.status_code} for {response.request.url}: "
            f"{response.text[:200]}"
        )
    return response.content
