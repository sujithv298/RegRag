# Audit log schema (v1)

Every query produces exactly one record, appended to the JSONL audit log.

## Fields

| Field | Type | Notes |
|-------|------|-------|
| `schema_version` | string | Currently `"1"`. Bumped on breaking changes. |
| `query_id` | UUID v4 | Unique per query. |
| `timestamp` | ISO-8601 UTC | Time the query started. |
| `scrubbed_input` | string | PII-redacted user query. The raw query is never persisted. |
| `pii_redactions` | int | Count of PII entities Presidio redacted (no values). |
| `corpus_snapshot_date` | ISO date | The eCFR date the index was built against. |
| `retrieved_chunk_ids` | array[string] | Chunk IDs in rank order, post-rerank. |
| `prompt_template_version` | string | Semver of the prompt template used. |
| `model_name` | string | E.g. `"claude-sonnet-4-6"`, `"gpt-4o"`, `"llama-3.1-70b-instruct.Q4_K_M"`. |
| `model_version` | string | API-reported version where available. |
| `response_text` | string | The full model output (raw, before refusal conversion). |
| `citations` | array[object] | One per extracted citation: `{path, supports_claim, verification_status}`. |
| `outcome` | enum | `"answered"` \| `"refused"`. |
| `refusal_reason` | string \| null | Set when `outcome == "refused"`. Examples: `"unverified_citation"`, `"no_supporting_chunk"`, `"low_support_score"`. |
| `prompt_hash` | sha256 hex | Hash of the rendered prompt, for tamper detection. |
| `response_hash` | sha256 hex | Hash of `response_text`. |

## Versioning policy

- Adding a field is non-breaking; the schema version stays.
- Renaming or removing a field bumps the major version of `schema_version`.
- Forkers piping to a SIEM should pin to a major version.

## Storage seam

The default writer appends to a JSONL file at `REGRAG_AUDIT_LOG`. Replace `regrag.audit.logger.AuditLogger` with an S3 / Splunk / Kafka writer to integrate with bank observability stacks; nothing else in the pipeline changes.
