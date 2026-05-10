"use client";

import { motion } from "motion/react";
import { cn } from "@/lib/utils";

export interface RecentQuery {
  question: string;
  outcome: "answered" | "refused";
}

interface Props {
  recents: RecentQuery[];
}

export function Sidebar({ recents }: Props) {
  return (
    <aside
      className="flex h-full flex-col border-r border-[var(--color-border)] p-3"
      style={{
        background:
          "linear-gradient(180deg, var(--color-surface-2) 0%, var(--color-bg) 100%)",
      }}
    >
      <div className="px-2 py-3 text-[11px] font-semibold uppercase tracking-wider text-[var(--color-text-3)]">
        Recent queries
      </div>

      <div className="flex-1 space-y-0.5 overflow-y-auto">
        {recents.length === 0 ? (
          <div className="rounded-[6px] border border-dashed border-[var(--color-border)] bg-[var(--color-surface)]/40 px-3 py-3 text-[11px] leading-relaxed text-[var(--color-text-3)]">
            <div className="mb-1 font-medium text-[var(--color-text-2)]">No queries yet</div>
            Each query you run will show up here, with a green or red dot indicating whether it was answered or refused.
          </div>
        ) : (
          recents
            .slice()
            .reverse()
            .map((r, i) => (
              <motion.div
                key={`${r.question}-${i}`}
                initial={{ opacity: 0, x: -4 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.18, delay: i * 0.02 }}
                className="cursor-default truncate rounded-[5px] px-2 py-1.5 text-xs text-[var(--color-text-2)] hover:bg-black/5"
                title={r.question}
              >
                <span
                  className={cn(
                    "mr-2 inline-block h-1.5 w-1.5 rounded-full align-middle",
                    r.outcome === "answered"
                      ? "bg-[var(--color-success)]"
                      : "bg-[var(--color-danger)]",
                  )}
                />
                {r.question.length > 38 ? `${r.question.slice(0, 38)}…` : r.question}
              </motion.div>
            ))
        )}
      </div>

      <div className="mt-auto border-t border-[var(--color-border)] px-2 py-3 text-[11px] text-[var(--color-text-3)]">
        Corpus: 12 CFR 1005
        <br />
        Snapshot: 2025-01-01
      </div>
    </aside>
  );
}
