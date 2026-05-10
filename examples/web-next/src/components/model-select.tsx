"use client";

import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import type { InfoResponse, ModelId } from "@/lib/types";

interface Props {
  value: ModelId;
  onChange: (m: ModelId) => void;
  info: InfoResponse | null;
}

export function ModelSelect({ value, onChange, info }: Props) {
  const openaiEnabled = info?.has_openai_key ?? false;
  const anthropicEnabled = info?.has_anthropic_key ?? false;

  return (
    <Select value={value} onValueChange={(v) => onChange(v as ModelId)}>
      <SelectTrigger className="w-[260px]">
        <SelectValue />
      </SelectTrigger>
      <SelectContent>
        <SelectItem value="openai" disabled={!openaiEnabled}>
          openai · gpt-4o {openaiEnabled ? "" : "(set OPENAI_API_KEY)"}
        </SelectItem>
        <SelectItem value="anthropic" disabled={!anthropicEnabled}>
          anthropic · claude {anthropicEnabled ? "" : "(set ANTHROPIC_API_KEY)"}
        </SelectItem>
        <SelectItem value="fake-good">fake-good · offline demo</SelectItem>
        <SelectItem value="fake-bad">fake-bad · hallucinating demo</SelectItem>
      </SelectContent>
    </Select>
  );
}
