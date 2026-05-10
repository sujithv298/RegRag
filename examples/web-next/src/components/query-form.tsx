"use client";

import { useState, useEffect, useCallback } from "react";
import { ArrowRight } from "lucide-react";
import { Card } from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { ModelSelect } from "./model-select";
import type { InfoResponse, ModelId } from "@/lib/types";

interface Props {
  info: InfoResponse | null;
  isRunning: boolean;
  onSubmit: (question: string, model: ModelId) => void;
}

const PRESETS: { label: string; question: string; model: ModelId; tone: "muted" | "danger" }[] = [
  {
    label: "Standard liability",
    question:
      "What is the consumer's maximum liability if they report an unauthorized transfer within two business days?",
    model: "openai",
    tone: "muted",
  },
  {
    label: "⚠ Hallucinated section",
    question: "What does 12 CFR 9999.99(z) say about cryptocurrency reporting?",
    model: "fake-bad",
    tone: "danger",
  },
  {
    label: "Out of scope",
    question: "What is the FDIC deposit insurance coverage limit per depositor?",
    model: "openai",
    tone: "muted",
  },
  {
    label: "Error resolution",
    question:
      "Within how many days must an institution complete an error investigation under Regulation E?",
    model: "openai",
    tone: "muted",
  },
];

export function QueryForm({ info, isRunning, onSubmit }: Props) {
  const [question, setQuestion] = useState("");
  const [model, setModel] = useState<ModelId>("fake-good");

  // When info loads, default the model to OpenAI if the key is available.
  useEffect(() => {
    if (info && model === "fake-good") {
      if (info.has_openai_key) setModel("openai");
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [info?.has_openai_key]);

  const handleRun = useCallback(() => {
    const q = question.trim();
    if (!q || isRunning) return;
    onSubmit(q, model);
  }, [question, model, isRunning, onSubmit]);

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
      e.preventDefault();
      handleRun();
    }
  };

  const runPreset = (p: (typeof PRESETS)[number]) => {
    if (p.model === "openai" && !info?.has_openai_key) {
      // Fall back to fake-good if the user hasn't set their OpenAI key.
      setQuestion(p.question);
      setModel("fake-good");
      onSubmit(p.question, "fake-good");
      return;
    }
    setQuestion(p.question);
    setModel(p.model);
    onSubmit(p.question, p.model);
  };

  return (
    <Card className="p-5 transition-shadow focus-within:shadow-[0_8px_24px_rgba(15,23,42,0.06),0_0_0_4px_rgba(109,74,255,0.18)]">
      <label className="mb-2 block text-[11px] font-semibold uppercase tracking-wider text-[var(--color-text-3)]">
        Question
      </label>
      <Textarea
        value={question}
        onChange={(e) => setQuestion(e.target.value)}
        onKeyDown={handleKeyDown}
        rows={3}
        placeholder="e.g. What is the consumer's maximum liability if they report an unauthorized transfer within two business days?"
      />

      <div className="mt-4 flex flex-wrap items-center gap-3">
        <ModelSelect value={model} onChange={setModel} info={info} />
        <div className="flex-1" />
        <Button onClick={handleRun} disabled={isRunning || !question.trim()}>
          {isRunning ? "Running…" : "Run query"}
          <ArrowRight className="h-4 w-4" />
        </Button>
      </div>

      <div className="mt-5 border-t border-dashed border-[var(--color-border)] pt-4">
        <div className="mb-2 text-[11px] font-semibold uppercase tracking-wider text-[var(--color-text-3)]">
          Or try a preset
        </div>
        <div className="flex flex-wrap gap-2">
          {PRESETS.map((p) => (
            <button
              key={p.label}
              type="button"
              onClick={() => runPreset(p)}
              disabled={isRunning}
              className="rounded-full border border-[var(--color-border)] bg-[var(--color-surface)] px-3 py-1 text-xs font-medium text-[var(--color-text-2)] transition-all hover:-translate-y-px hover:border-[var(--color-border-strong)] hover:bg-[var(--color-surface-2)] hover:text-[var(--color-text)] disabled:opacity-50"
            >
              {p.label}
            </button>
          ))}
        </div>
      </div>

      {info && info.cache_size > 0 && (
        <div className="mt-3 flex items-center gap-2 text-[11px] text-[var(--color-text-3)]">
          <Badge tone="amber">cache: {info.cache_size}</Badge>
          <span>cached responses skip the LLM call</span>
        </div>
      )}
    </Card>
  );
}
