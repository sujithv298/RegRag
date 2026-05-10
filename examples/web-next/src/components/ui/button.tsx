"use client";

import * as React from "react";
import { Slot } from "@radix-ui/react-slot";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

const buttonVariants = cva(
  "inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-[var(--radius-sm)] text-sm font-medium transition-all disabled:pointer-events-none disabled:opacity-50 focus-ring",
  {
    variants: {
      variant: {
        primary:
          "bg-gradient-to-b from-[#1e293b] to-[#0f172a] text-white border border-black shadow-[0_1px_0_rgba(255,255,255,0.08)_inset,0_2px_6px_rgba(15,23,42,0.18)] hover:from-[#334155] hover:to-[#1e293b] hover:shadow-[0_1px_0_rgba(255,255,255,0.08)_inset,0_4px_12px_rgba(15,23,42,0.22)] active:translate-y-[0.5px]",
        secondary:
          "bg-[var(--color-surface)] text-[var(--color-text)] border border-[var(--color-border)] hover:bg-[var(--color-surface-2)] hover:border-[var(--color-border-strong)]",
        ghost:
          "bg-transparent text-[var(--color-text-2)] hover:bg-[var(--color-surface-2)] hover:text-[var(--color-text)]",
      },
      size: {
        sm: "h-8 px-3 text-xs",
        md: "h-9 px-4",
        lg: "h-10 px-5",
      },
    },
    defaultVariants: { variant: "primary", size: "md" },
  },
);

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {
  asChild?: boolean;
}

export const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, size, asChild = false, ...props }, ref) => {
    const Comp = asChild ? Slot : "button";
    return (
      <Comp
        ref={ref}
        className={cn(buttonVariants({ variant, size, className }))}
        {...props}
      />
    );
  },
);
Button.displayName = "Button";
