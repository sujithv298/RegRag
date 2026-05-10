import type { InfoResponse, QueryRequest, QueryResponse } from "./types";

async function jsonFetch<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`${res.status}: ${text || res.statusText}`);
  }
  return (await res.json()) as T;
}

export const api = {
  info: () => jsonFetch<InfoResponse>("/api/info"),

  query: (req: QueryRequest, opts?: { fresh?: boolean }) =>
    jsonFetch<QueryResponse>(
      `/api/query${opts?.fresh ? "?fresh=1" : ""}`,
      { method: "POST", body: JSON.stringify(req) },
    ),

  clearCache: () =>
    jsonFetch<{ cleared: number }>("/api/cache/clear", { method: "POST" }),
};
