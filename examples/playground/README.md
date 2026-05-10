# Playground

Streamlit demo surface for RegRAG. Three tabs:

- **Ask** — free-form question against the ingested Reg E corpus. Pipeline runs with a live-animated agent reasoning view; final answer renders with green/red citation badges; the per-query audit log is one click below.
- **Try to break it** — preset attack prompts designed to challenge the citation verifier. Pick one, watch the fail-closed gate fire in real time.
- **How it works** — architecture explainer.

## Run

```bash
# From the project root
uv sync --extra playground
make setup    # produces data/reg_e_excerpt.chunks.jsonl if you haven't already
streamlit run examples/playground/app.py
```

Open the URL Streamlit prints (usually `http://localhost:8501`).

## Models

The sidebar lets you pick which adapter the pipeline uses:

| Adapter | Notes |
|---------|-------|
| `fake-good (offline)` | Default. Deterministic stub that produces a well-behaved answer. No API key required. |
| `fake-bad (offline)`  | Deterministic stub that hallucinates a citation. Use this with the "Hallucinated section number" preset to see the verifier catch it. |
| `Anthropic (real Claude)` | Only appears if `ANTHROPIC_API_KEY` is set in `.env`. Calls real Claude. Requires network access to `api.anthropic.com`. |

## What you'll see

The Streamlit app emphasizes the parts of RegRAG that don't read well in terminal output:

- The agent's reasoning streams as a series of step cards (`→ search_regulations(...)`, `← tool result: 5 chunks retrieved`, `→ final answer`).
- Each citation in the final answer renders as a pill — **green ✓** if verification passed, **red ✗** if it failed.
- The audit-log record for the query is one expander click away. JSON pretty-printed.

## What's a good demo flow for a colleague?

1. Open the **Ask** tab. Run the default question. They see an answer with a green citation badge.
2. Switch to the **Try to break it** tab. Run *"Hallucinated section number"* with `fake-bad`. They see a red citation badge and a refusal.
3. Switch to **How it works** for the architecture explainer.

That's the 90-second guided tour.
