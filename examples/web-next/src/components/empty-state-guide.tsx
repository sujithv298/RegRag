"use client";

import { motion } from "motion/react";
import { Search, Cpu, ShieldCheck } from "lucide-react";

const STEPS = [
  {
    icon: Search,
    title: "Retrieve",
    body: "Hybrid search (BM25 + dense embeddings) finds the most relevant paragraphs from the regulation.",
  },
  {
    icon: Cpu,
    title: "Generate",
    body: "An agent loop drafts an answer, citing each claim with [CFR:…] tags inline.",
  },
  {
    icon: ShieldCheck,
    title: "Verify",
    body: "Every citation is independently checked. If it doesn't resolve to real text, the system refuses.",
  },
];

export function EmptyStateGuide() {
  return (
    <motion.section
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, delay: 0.1 }}
      className="mt-8"
    >
      <div className="mb-3 text-[11px] font-semibold uppercase tracking-wider text-[var(--color-text-3)]">
        How it works
      </div>
      <div className="grid grid-cols-1 gap-3 md:grid-cols-3">
        {STEPS.map((step, i) => (
          <motion.div
            key={step.title}
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3, delay: 0.15 + i * 0.06 }}
            className="rounded-[var(--radius-md)] border border-[var(--color-border)] bg-[var(--color-surface)] p-4 shadow-[0_1px_2px_rgba(15,23,42,0.04)]"
          >
            <div className="mb-2.5 flex items-center gap-2">
              <span
                className="grid h-7 w-7 place-items-center rounded-[6px]"
                style={{
                  background: "var(--color-accent-bg)",
                  color: "var(--color-accent-strong)",
                  border: "1px solid var(--color-accent-border)",
                }}
              >
                <step.icon className="h-3.5 w-3.5" strokeWidth={2.5} />
              </span>
              <span className="text-[13px] font-semibold text-[var(--color-text)]">
                <span className="font-mono text-[10px] text-[var(--color-text-3)]">
                  0{i + 1}
                </span>{" "}
                {step.title}
              </span>
            </div>
            <p className="text-[12px] leading-[1.55] text-[var(--color-text-2)]">
              {step.body}
            </p>
          </motion.div>
        ))}
      </div>

      <div className="mt-5 flex items-center gap-2 text-[12px] text-[var(--color-text-3)]">
        <span className="font-mono text-[10px]">↑</span>
        Type a question above, or click a preset to see the pipeline run.
      </div>
    </motion.section>
  );
}
