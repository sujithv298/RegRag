"""Convenience wrapper: run the Reg E v1 gold set.

Equivalent to `uv run regrag eval --gold evals/gold/reg_e_v1.jsonl`.
"""

from __future__ import annotations

from regrag.cli import main

if __name__ == "__main__":
    main(["eval", "--gold", "evals/gold/reg_e_v1.jsonl"])
