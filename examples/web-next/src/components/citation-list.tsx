"use client";

import { motion } from "motion/react";
import { cn } from "@/lib/utils";
import type { CitationOut } from "@/lib/types";

export function CitationList({ citations }: { citations: CitationOut[] }) {
  if (citations.length === 0) {
    return (
      <div className="rounded-[var(--radius-md)] border border-[var(--color-border)] bg-[var(--color-surface)] p-3 text-xs text-[var(--color-text-3)]">
        No citations were emitted by the model.
      </div>
    );
  }

  return (
    <div className="overflow-hidden rounded-[var(--radius-md)] border border-[var(--color-border)] bg-[var(--color-surface)] shadow-[0_1px_2px_rgba(15,23,42,0.04)]">
      {citations.map((c, i) => (
        <motion.div
          key={c.citation_path + i}
          initial={{ opacity: 0, y: 4 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.2, delay: i * 0.04 }}
          className={cn(
            "grid grid-cols-[1fr_auto_auto] items-center gap-3 px-3 py-2.5 text-[12.5px]",
            i < citations.length - 1 && "border-b border-[var(--color-border)]",
          )}
        >
          <span className="font-mono font-medium text-[var(--color-text)]" title={c.chunk_id ?? ""}>
            {c.citation_path}
          </span>
          <span className="font-mono text-[11px] text-[var(--color-text-3)]">
            support {c.support_score.toFixed(2)}
          </span>
          <span
            className={cn(
              "rounded-[4px] border px-2 py-0.5 text-[11px] font-medium uppercase tracking-wide",
              c.passed
                ? "border-[var(--color-success-border)] bg-[var(--color-success-bg)] text-[var(--color-success)]"
                : "border-[var(--color-danger-border)] bg-[var(--color-danger-bg)] text-[var(--color-danger)]",
            )}
          >
            {c.passed ? "verified" : "failed"}
          </span>
        </motion.div>
      ))}
    </div>
  );
}
