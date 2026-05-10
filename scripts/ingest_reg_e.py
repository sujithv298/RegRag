"""Convenience wrapper: pull Reg E and build the indices.

Equivalent to `uv run regrag ingest --part 1005`. Useful in CI smoke runs
and for the README quickstart's copy-paste path.
"""

from __future__ import annotations

from regrag.cli import main

if __name__ == "__main__":
    main(["ingest", "--part", "1005"])
