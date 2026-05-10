"use client";

import { motion } from "motion/react";
import { cn } from "@/lib/utils";

export interface Step {
  label: string;
  detail: string;
  state: "pending" | "active" | "done";
  time: string | null;
}

interface Props {
  steps: Step[];
  meta: React.ReactNode;
}

export function PipelineTrace({ steps, meta }: Props) {
  return (
    <section>
      <div className="mb-2 flex items-baseline justify-between border-b border-[var(--color-border)] pb-2">
        <span className="text-[11px] font-semibold uppercase tracking-wider text-[var(--color-text-3)]">
          Pipeline
        </span>
        <span className="font-mono text-[11px] text-[var(--color-text-3)]">{meta}</span>
      </div>
      <div className="overflow-hidden rounded-[var(--radius-md)] border border-[var(--color-border)] bg-[var(--color-surface)] shadow-[0_1px_2px_rgba(15,23,42,0.04)]">
        {steps.map((s, i) => (
          <motion.div
            key={s.label}
            initial={false}
            animate={{
              opacity: s.state === "pending" ? 0.4 : 1,
              backgroundColor:
                s.state === "active" ? "rgba(109,74,255,0.05)" : "rgba(0,0,0,0)",
            }}
            transition={{ duration: 0.25 }}
            className={cn(
              "grid grid-cols-[28px_1fr_auto] items-center gap-3 px-3 py-2.5 text-[12.5px]",
              i < steps.length - 1 && "border-b border-[var(--color-border)]",
            )}
          >
            <span
              className={cn(
                "text-right font-mono text-[11px] font-medium",
                s.state === "active" && "text-[var(--color-accent)]",
                s.state === "done" && "text-[var(--color-success)]",
                s.state === "pending" && "text-[var(--color-text-3)]",
              )}
            >
              {String(i + 1).padStart(2, "0")}
            </span>
            <div className="min-w-0">
              <span className="text-[var(--color-text)]">{s.label}</span>
              {s.detail && (
                <span className="ml-2 font-mono text-[11px] text-[var(--color-text-3)]">
                  {s.detail}
                </span>
              )}
            </div>
            <span className="min-w-[60px] text-right font-mono text-[11px] text-[var(--color-text-3)]">
              {s.time != null ? `${s.time} ms` : "—"}
            </span>
          </motion.div>
        ))}
      </div>
    </section>
  );
}
