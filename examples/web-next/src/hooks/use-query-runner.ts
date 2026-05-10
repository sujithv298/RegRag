"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "@/lib/api";
import type {
  InfoResponse,
  ModelId,
  QueryResponse,
} from "@/lib/types";
import type { Step } from "@/components/pipeline-trace";
import { sleep } from "@/lib/utils";

const STEPS_TEMPLATE = (model: ModelId): Step[] => [
  { label: "PII scrub", detail: "redact SSN / email / phone", state: "active", time: null },
  {
    label: "Hybrid retrieval",
    detail: "BM25 + dense embeddings · RRF · cross-encoder rerank",
    state: "pending",
    time: null,
  },
  { label: "LLM", detail: `agent loop · ${model}`, state: "pending", time: null },
  {
    label: "Citation verifier",
    detail: "existence + support checks · fail-closed",
    state: "pending",
    time: null,
  },
  {
    label: "Audit log",
    detail: "JSONL append · SHA-256 prompt + response hashes",
    state: "pending",
    time: null,
  },
];

const FAKE_BEATS = [55, 240, 740, 90, 12];

export function useQueryRunner() {
  const [info, setInfo] = useState<InfoResponse | null>(null);
  const [steps, setSteps] = useState<Step[]>([]);
  const [traceMeta, setTraceMeta] = useState<string>("");
  const [result, setResult] = useState<QueryResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [recents, setRecents] = useState<{ question: string; outcome: "answered" | "refused" }[]>([]);
  const cancelRef = useRef(false);

  const refreshInfo = useCallback(() => {
    api.info().then(setInfo).catch(() => {});
  }, []);

  useEffect(() => {
    refreshInfo();
  }, [refreshInfo]);

  const run = useCallback(
    async (question: string, model: ModelId) => {
      if (!question.trim() || isRunning) return;
      setIsRunning(true);
      setError(null);
      setResult(null);
      setTraceMeta("running");
      cancelRef.current = false;

      const localSteps = STEPS_TEMPLATE(model);
      setSteps([...localSteps]);

      const t0 = performance.now();

      // Kick off the API request in parallel with the animated step pacing.
      const apiPromise = api.query({ question, model }).catch((e: Error) => {
        setError(e.message);
        return null;
      });

      // Animate step progression
      for (let i = 0; i < localSteps.length; i++) {
        if (cancelRef.current) return;
        await sleep(FAKE_BEATS[i] / 3);
        localSteps[i].state = "done";
        localSteps[i].time = FAKE_BEATS[i].toFixed(1);
        if (i + 1 < localSteps.length) localSteps[i + 1].state = "active";
        setSteps([...localSteps]);
      }

      const data = await apiPromise;
      const elapsed = (performance.now() - t0).toFixed(0);
      setTraceMeta(`complete · ${elapsed} ms`);

      if (data) {
        setResult(data);
        setRecents((prev) => [
          ...prev.slice(-7),
          { question, outcome: data.outcome },
        ]);
        // Refresh info so the cache_size in the badge updates
        refreshInfo();
      }
      setIsRunning(false);
    },
    [isRunning, refreshInfo],
  );

  return { info, steps, traceMeta, result, error, isRunning, recents, run };
}
