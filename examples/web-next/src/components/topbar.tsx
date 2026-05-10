import Link from "next/link";

export function Topbar() {
  return (
    <header
      className="sticky top-0 z-20 flex h-[52px] items-center justify-between border-b border-[var(--color-border)] px-5"
      style={{
        background: "rgba(255, 255, 255, 0.85)",
        backdropFilter: "saturate(140%) blur(8px)",
        WebkitBackdropFilter: "saturate(140%) blur(8px)",
      }}
    >
      <div className="flex items-center gap-2.5">
        <span
          className="grid h-[26px] w-[26px] place-items-center rounded-[7px] text-[13px] font-bold text-white"
          style={{
            background:
              "linear-gradient(135deg, var(--color-accent) 0%, var(--color-accent-strong) 100%)",
            boxShadow:
              "0 2px 8px rgba(109,74,255,0.18), inset 0 1px 0 rgba(255,255,255,0.2)",
            letterSpacing: "-0.04em",
          }}
        >
          R
        </span>
        <span className="text-[15px] font-semibold tracking-tight">RegRAG</span>
        <span className="ml-1 rounded-[4px] border border-[var(--color-accent-border)] bg-[var(--color-accent-bg)] px-2 py-0.5 text-[11px] font-medium text-[var(--color-accent-strong)]">
          v0.1
        </span>
      </div>

      <div className="flex items-center gap-4 text-sm">
        <Link
          href="https://github.com/sujithv298/regrag"
          target="_blank"
          className="text-[var(--color-text-2)] transition-colors hover:text-[var(--color-text)]"
        >
          GitHub
        </Link>
      </div>
    </header>
  );
}
