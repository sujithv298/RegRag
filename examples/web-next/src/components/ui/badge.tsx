import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

const badgeVariants = cva(
  "inline-flex items-center gap-1.5 rounded-full border text-xs font-medium font-mono px-2.5 py-0.5",
  {
    variants: {
      tone: {
        accent:
          "bg-[var(--color-accent-bg)] text-[var(--color-accent-strong)] border-[var(--color-accent-border)]",
        success:
          "bg-[var(--color-success-bg)] text-[var(--color-success)] border-[var(--color-success-border)]",
        danger:
          "bg-[var(--color-danger-bg)] text-[var(--color-danger)] border-[var(--color-danger-border)]",
        amber:
          "bg-[var(--color-amber-bg)] text-[var(--color-amber)] border-[var(--color-amber-border)]",
        muted:
          "bg-[var(--color-surface-2)] text-[var(--color-text-2)] border-[var(--color-border)]",
      },
    },
    defaultVariants: { tone: "muted" },
  },
);

export interface BadgeProps
  extends React.HTMLAttributes<HTMLSpanElement>,
    VariantProps<typeof badgeVariants> {}

export function Badge({ className, tone, ...props }: BadgeProps) {
  return <span className={cn(badgeVariants({ tone, className }))} {...props} />;
}
