"use client";

import { Topbar } from "@/components/topbar";
import { Sidebar } from "@/components/sidebar";
import { QueryForm } from "@/components/query-form";
import { PipelineTrace } from "@/components/pipeline-trace";
import { ResultPanel } from "@/components/result-panel";
import { EmptyStateGuide } from "@/components/empty-state-guide";
import { useQueryRunner } from "@/hooks/use-query-runner";

export default function Home() {
  const { info, steps, traceMeta, result, error, isRunning, recents, run } = useQueryRunner();

  return (
    <div className="grid min-h-screen grid-rows-[52px_1fr] grid-cols-[240px_1fr] [grid-template-areas:'topbar_topbar''sidebar_main'] max-md:grid-cols-1 max-md:[grid-template-areas:'topbar''main']">
      <div className="[grid-area:topbar]">
        <Topbar />
      </div>

      <div className="[grid-area:sidebar] max-md:hidden">
        <Sidebar recents={recents} />
      </div>

      <main className="[grid-area:main] overflow-y-auto">
        <div className="mx-auto max-w-[760px] px-6 py-9 pb-24 md:px-8">
          {/* Page heading */}
          <div className="mb-6">
            <span
              className="mb-3 inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-[11px] font-medium"
              style={{
                color: "var(--color-accent-strong)",
                background: "var(--color-accent-bg)",
                borderColor: "var(--color-accent-border)",
              }}
            >
              <span
                className="h-1.5 w-1.5 rounded-full"
                style={{
                  background: "var(--color-accent)",
                  boxShadow: "0 0 0 2px var(--color-accent-bg)",
                }}
              />
              Verifiable RAG · Regulation E
            </span>
            <h1 className="text-[28px] font-bold tracking-[-0.025em] text-[var(--color-text)]">
              Ask the regulation.
            </h1>
            <p className="mt-2 max-w-[620px] text-sm leading-relaxed text-[var(--color-text-2)]">
              Every citation the model emits is independently verified against the actual regulation
              text. If verification fails, the system refuses — no hallucinated citations leave the gate.
            </p>
          </div>

          {/* Compose */}
          <QueryForm info={info} isRunning={isRunning} onSubmit={run} />

          {/* Trace + Result, OR an empty-state guide when nothing has run yet */}
          {steps.length > 0 || result || error ? (
            <div className="mt-6 space-y-6">
              {steps.length > 0 && <PipelineTrace steps={steps} meta={traceMeta} />}
              {(result || error) && <ResultPanel result={result} error={error} />}
            </div>
          ) : (
            <EmptyStateGuide />
          )}
        </div>
      </main>
    </div>
  );
}
