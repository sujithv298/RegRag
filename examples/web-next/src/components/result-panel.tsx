"use client";

import { useEffect, useState } from "react";
import { AnimatePresence, motion } from "motion/react";
import { Check, X, ChevronRight } from "lucide-react";
import { CitationList } from "./citation-list";
import { Badge } from "./ui/badge";
import { cn, sleep } from "@/lib/utils";
import type { QueryResponse } from "@/lib/types";

interface Props {
  result: QueryResponse | null;
  error: string | null;
}

/** Word-by-word reveal of text. Mimics LLM streaming. */
function useTypewriter(text: string, msPerWord = 18, enabled = true) {
  const [out, setOut] = useState("");
  const [done, setDone] = useState(false);

  useEffect(() => {
    if (!text) {
      setOut("");
      setDone(true);
      return;
    }
    if (!enabled) {
      setOut(text);
      setDone(true);
      return;
    }
    let cancelled = false;
    setOut("");
    setDone(false);
    (async () => {
      const tokens = text.split(/(\s+)/);
      let buf = "";
      for (const token of tokens) {
        if (cancelled) return;
        buf += token;
        setOut(buf);
        if (token.trim().length > 0) {
          await sleep(msPerWord + Math.random() * 12);
        }
      }
      if (!cancelled) setDone(true);
    })();
    return () => {
      cancelled = true;
    };
  }, [text, msPerWord, enabled]);

  return { text: out, done };
}

export function ResultPanel({ result, error }: Props) {
  // Skip the typewriter when the response is cached — instant feel makes more sense.
  const skipAnimation = !!result?.cached;
  const { text: streamedAnswer, done: animationDone } = useTypewriter(
    result?.answer ?? "",
    18,
    !skipAnimation,
  );
  const [auditOpen, setAuditOpen] = useState(false);

  if (error) {
    return (
      <motion.section
        initial={{ opacity: 0, y: 6 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.25 }}
        className="rounded-[var(--radius-md)] border border-[var(--color-danger-border)] bg-[var(--color-danger-bg)] p-4 text-sm text-[var(--color-danger)]"
      >
        <div className="font-medium">Request failed</div>
        <div className="mt-1 font-mono text-xs">{error}</div>
      </motion.section>
    );
  }

  if (!result) return null;

  const isAnswered = result.outcome === "answered";

  return (
    <AnimatePresence mode="wait">
      <motion.div
        key={result.audit_query_id}
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0, y: -4 }}
        transition={{ duration: 0.4, ease: [0.22, 1, 0.36, 1] }}
        className="space-y-6"
      >
        {/* Verdict + result meta */}
        <section>
          <div className="mb-2 flex items-baseline justify-between border-b border-[var(--color-border)] pb-2">
            <span className="text-[11px] font-semibold uppercase tracking-wider text-[var(--color-text-3)]">
              Result
            </span>
            <span className="font-mono text-[11px] text-[var(--color-text-3)]">
              {result.audit_query_id}
            </span>
          </div>

          <div
            className={cn(
              "inline-flex items-center gap-2 rounded-[var(--radius-md)] border px-4 py-2 text-sm font-semibold",
              isAnswered
                ? "border-[var(--color-success-border)] text-[var(--color-success)]"
                : "border-[var(--color-danger-border)] text-[var(--color-danger)]",
            )}
            style={{
              background: isAnswered
                ? "linear-gradient(135deg, var(--color-success-bg) 0%, #ffffff 100%)"
                : "linear-gradient(135deg, var(--color-danger-bg) 0%, #ffffff 100%)",
              boxShadow: isAnswered
                ? "0 1px 0 rgba(255,255,255,0.6) inset, 0 2px 6px rgba(5,150,105,0.15)"
                : "0 1px 0 rgba(255,255,255,0.6) inset, 0 2px 6px rgba(225,29,72,0.1)",
            }}
          >
            <span
              className={cn(
                "grid h-4 w-4 place-items-center rounded-[4px] text-[11px] font-bold text-white",
                isAnswered ? "bg-[var(--color-success)]" : "bg-[var(--color-danger)]",
              )}
            >
              {isAnswered ? <Check className="h-3 w-3" strokeWidth={3} /> : <X className="h-3 w-3" strokeWidth={3} />}
            </span>
            <span>{isAnswered ? "Answered" : "Refused"}</span>
            <span className="ml-2 text-xs font-medium opacity-75">
              {isAnswered
                ? `via agent · ${result.model_name}`
                : `reason: ${result.refusal_reason ?? "unknown"}`}
              {result.cached && " · from cache"}
            </span>
          </div>

          {/* Answer block with gradient bar */}
          <div className="relative mt-3 overflow-hidden rounded-[var(--radius-md)] border border-[var(--color-border)] bg-[var(--color-surface)] p-4 shadow-[0_1px_2px_rgba(15,23,42,0.04)]">
            <div
              className="absolute inset-y-0 left-0 w-[3px]"
              style={{
                background:
                  "linear-gradient(180deg, var(--color-accent), var(--color-success))",
                opacity: 0.7,
              }}
            />
            <div
              className={cn(
                "ml-2 text-[14.5px] leading-[1.65] text-[var(--color-text)]",
                !animationDone && "caret-blink",
              )}
            >
              {streamedAnswer}
            </div>
          </div>
        </section>

        {/* Citations */}
        <section>
          <div className="mb-2 flex items-baseline justify-between border-b border-[var(--color-border)] pb-2">
            <span className="text-[11px] font-semibold uppercase tracking-wider text-[var(--color-text-3)]">
              Citations
            </span>
            <span className="font-mono text-[11px] text-[var(--color-text-3)]">
              {result.citations.length === 0
                ? "none emitted"
                : `${result.citations.length} ${result.citations.length === 1 ? "citation" : "citations"}`}
            </span>
          </div>
          <CitationList citations={result.citations} />
        </section>

        {/* Audit JSON */}
        <section>
          <button
            onClick={() => setAuditOpen((v) => !v)}
            className="flex items-center gap-1.5 text-xs font-medium text-[var(--color-text-2)] hover:text-[var(--color-text)]"
          >
            <ChevronRight
              className={cn(
                "h-3.5 w-3.5 transition-transform",
                auditOpen && "rotate-90",
              )}
            />
            Audit-log record (JSON)
          </button>
          <AnimatePresence initial={false}>
            {auditOpen && (
              <motion.div
                initial={{ height: 0, opacity: 0 }}
                animate={{ height: "auto", opacity: 1 }}
                exit={{ height: 0, opacity: 0 }}
                transition={{ duration: 0.2 }}
                style={{ overflow: "hidden" }}
              >
                <pre className="code-block mt-2">{JSON.stringify(result, null, 2)}</pre>
              </motion.div>
            )}
          </AnimatePresence>
        </section>

        {result.cached && (
          <div className="text-[11px] text-[var(--color-text-3)]">
            <Badge tone="amber">cached</Badge>{" "}
            <span className="ml-1">Served from cache — no LLM call made.</span>
          </div>
        )}
      </motion.div>
    </AnimatePresence>
  );
}
