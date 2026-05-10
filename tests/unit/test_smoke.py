"""Smoke test — proves the package imports and the version is wired up.

Real tests land per phase: test_ecfr_client.py and test_parser.py in Phase 2,
test_chunker.py in Phase 3, etc.
"""

from __future__ import annotations

import regrag


def test_version_is_set() -> None:
    assert regrag.__version__ == "0.1.0"
