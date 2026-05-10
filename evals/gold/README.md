# Gold sets

Each gold-set file is a JSONL of questions for the eval harness. v1 ships
`reg_e_v1.jsonl` covering 12 CFR Part 1005.

## Status — v1.0

`reg_e_v1.jsonl` ships with **25 hand-written entries** across four
categories. Expansion to ~50 entries is on the v1.1 roadmap. The 25
entries are distributed:

| Category                  | Count |
|---------------------------|-------|
| `exact_citation`          | 12    |
| `multi_section`           | 5     |
| `intentional_refusal`     | 5     |
| `rule_vs_interpretation`  | 3     |

**Important:** the `expected_citations` and `expected_answer_keywords`
were drafted from training-knowledge of Reg E and need to be validated
against the live eCFR text the first time you ingest the real corpus.
A bad gold-set entry produces misleading metrics.

## Categories

Every entry is tagged with one of:

- `exact_citation` — has a single right answer with a single right citation. Tests retrieval + extraction.
- `multi_section` — requires synthesizing two or more sections. Tests reranking + multi-citation handling.
- `intentional_refusal` — about something Reg E doesn't cover (e.g., a Reg Z question). The system *should* refuse. Tests the verifier's fail-closed behavior.
- `rule_vs_interpretation` — the rule and its Official Interpretation differ in nuance. Tests that the chunker preserved the distinction and the model uses the right one.

## Schema (per line)

```json
{
  "id": "regE-001",
  "category": "exact_citation",
  "question": "What is the consumer's liability limit when an unauthorized EFT is reported within two business days?",
  "expected_citations": ["12 CFR 1005.6(b)(1)"],
  "expected_answer_keywords": ["$50"],
  "should_refuse": false,
  "notes": "Standard tier-1 liability cap."
}
```

## Adding entries

1. Pick a category. Distribute new entries roughly proportionally to the existing balance.
2. Verify the `expected_citations` against the live eCFR XML. Don't ship a citation you haven't seen.
3. `expected_answer_keywords` should be specific enough to fail on a wrong answer (e.g., `"$50"` not `"liability"`).
4. For refusal cases, leave both citation and keyword arrays empty.
5. Run `regrag eval --gold evals/gold/reg_e_v1.jsonl` and confirm the new entry behaves as designed.
