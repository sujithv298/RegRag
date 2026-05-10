// Mirrors the shape of regrag.api.QueryResponse and friends.

export type ModelId = "openai" | "anthropic" | "fake-good" | "fake-bad";

export interface InfoResponse {
  version: string;
  has_anthropic_key: boolean;
  has_openai_key: boolean;
  corpus_size: number | null;
  cache_size: number;
}

export interface CitationOut {
  citation_path: string;
  chunk_id: string | null;
  exists: boolean;
  supports: boolean;
  support_score: number;
  passed: boolean;
  failure_reason: string | null;
}

export type Outcome = "answered" | "refused";

export interface QueryResponse {
  outcome: Outcome;
  refusal_reason: string | null;
  answer: string;
  citations: CitationOut[];
  retrieved_chunk_ids: string[];
  model_name: string;
  audit_query_id: string;
  has_anthropic_key: boolean;
  cached: boolean;
}

export interface QueryRequest {
  question: string;
  model: ModelId;
}
