"""Tests for `regrag.ingest.ecfr_client`. Mocks HTTP with respx so no real
network hit is required."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
import respx

from regrag.ingest import ECFRError, fetch_part_xml

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "reg_e_part_1005_excerpt.xml"


@pytest.mark.asyncio
@respx.mock
async def test_fetch_part_xml_returns_response_bytes() -> None:
    fixture_bytes = FIXTURE.read_bytes()
    route = respx.get("https://www.ecfr.gov/api/versioner/v1/full/2025-01-01/title-12.xml").mock(
        return_value=httpx.Response(200, content=fixture_bytes)
    )

    result = await fetch_part_xml(title=12, part=1005, date="2025-01-01")

    assert result == fixture_bytes
    assert route.called
    # Verify the part query param made it through.
    assert route.calls.last.request.url.params["part"] == "1005"


@pytest.mark.asyncio
@respx.mock
async def test_fetch_part_xml_raises_on_non_200() -> None:
    respx.get("https://www.ecfr.gov/api/versioner/v1/full/2025-01-01/title-12.xml").mock(
        return_value=httpx.Response(404, text="Not Found")
    )

    with pytest.raises(ECFRError) as exc_info:
        await fetch_part_xml(title=12, part=9999, date="2025-01-01")
    assert "404" in str(exc_info.value)


@pytest.mark.asyncio
@respx.mock
async def test_fetch_part_xml_pins_to_provided_date() -> None:
    """Reproducibility hinge: the URL must include the date the caller asked for."""
    route = respx.get("https://www.ecfr.gov/api/versioner/v1/full/2024-06-15/title-12.xml").mock(
        return_value=httpx.Response(200, content=b"<DIV5 N='1005' TYPE='PART'/>")
    )

    await fetch_part_xml(title=12, part=1005, date="2024-06-15")

    assert route.called
    # If the client ever silently substituted a different date, this assertion
    # would fail because the route above wouldn't match.
