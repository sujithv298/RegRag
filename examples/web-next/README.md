# RegRAG Web (Next.js)

Modern web UI for RegRAG. Built with Next.js 15 (App Router), React 19, TypeScript 5, Tailwind 4, Radix UI primitives, Framer Motion, and Lucide icons.

## Run locally

You need two terminals: one for the FastAPI backend, one for the Next dev server.

**Terminal 1 — start the backend (from the project root):**

```bash
cd ~/Documents/Claude/Projects/Agent
PYTHONPATH=src uv run --extra playground uvicorn regrag.api:app \
  --host 127.0.0.1 --port 8000 --reload --reload-dir src
```

**Terminal 2 — start the frontend (this directory):**

```bash
cd examples/web-next
npm install            # one time
npm run dev
```

Open `http://localhost:3000`. The Next dev server proxies `/api/*` to `http://127.0.0.1:8000` (configured in `next.config.ts`), so the SPA can use same-origin fetches.

Or use the consolidated make target from the project root:

```bash
make web-next
```

## Architecture

- **`src/app/`** — Next.js App Router (layout, root page, global CSS).
- **`src/components/`** — feature components (topbar, sidebar, query-form, pipeline-trace, result-panel, citation-list, model-select).
- **`src/components/ui/`** — primitives (button, badge, card, textarea, select). Hand-rolled in shadcn/ui style — built on Radix UI + class-variance-authority + tailwind-merge.
- **`src/hooks/use-query-runner.ts`** — central state for the query lifecycle (info fetch, animated pipeline steps, result, recents).
- **`src/lib/api.ts`** — typed API client.
- **`src/lib/types.ts`** — TS types mirroring `regrag.api.QueryResponse`.

## Design tokens

In `src/app/globals.css` under `@theme inline` (Tailwind 4 inline theme):

- **Surface system**: `--color-bg`, `--color-surface`, `--color-surface-2`, `--color-surface-3` for layered depth.
- **Accent**: `--color-accent` (`#6d4aff`) — distinctive violet, intentionally not the indigo/coral every AI demo uses.
- **Verification**: emerald `--color-success` and rose `--color-danger`.
- **Highlight**: amber `--color-amber` for the cache pill.

## Production build

```bash
npm run build
npm start
```

Or deploy to Vercel — the `next.config.ts` rewrites point at `process.env.FASTAPI_URL ?? "http://127.0.0.1:8000"`. Set `FASTAPI_URL` to your deployed FastAPI service URL in the Vercel env config.
