# Contributing

Thanks for considering a contribution. RegRAG is small on purpose — its value is in being a clean reference implementation, not in being a kitchen-sink framework. PRs that simplify the patterns are as welcome as PRs that add capability.

## Especially welcome

- A new regulation under `src/regrag/ingest/` (e.g., Reg Z, BSA/AML, FFIEC). Each new corpus is its own ADR-worthy decision: which sections, which interpretations, what to skip.
- Expanded gold-set entries for the eval harness, with rationale for the question category.
- Improvements to the eval methodology — better metrics, additional refusal-case patterns, harder multi-hop synthesis questions.
- A pgvector store adapter to sit alongside the Chroma one.

## Workflow

1. Fork and branch from `main`.
2. `uv sync --extra dev` to set up the environment.
3. `pre-commit install` so lint and type-check run on commit.
4. Make the change. Add or update tests. If you change the audit-log schema or the prompt templates, bump the version in the relevant module and update the ADR.
5. `uv run pytest` and `uv run regrag eval --gold evals/gold/reg_e_v1.jsonl` should both pass.
6. PR description should explain what the change is and what it does to eval metrics (citation accuracy / answer correctness / refusal rate). "No effect" is a fine answer if measured.

## Things to discuss in an issue first

- New external dependencies. The current dep list is intentional; each addition has a maintenance cost.
- Changes to the audit-log schema. It's a contract; breaking it is fine, but it should be deliberate and versioned.
- Changes that move the project toward agentic retrieval as the default. The deterministic one-shot pipeline is the v1 contract; agentic mode is a parallel entrypoint, not a replacement.

## Code of conduct

See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
