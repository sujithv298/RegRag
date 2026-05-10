.PHONY: setup test demo demo-good demo-bad demo-agent demo-agent-bad eval eval-compare playground web web-next clean clear-cache help

help:
	@echo "Common commands:"
	@echo "  make setup           install deps + build chunks (one time)"
	@echo "  make test            run the unit-test suite (182 tests)"
	@echo "  make demo            fail-closed gate catching a hallucination (deterministic)"
	@echo "  make demo-good       verified-citation answered query (deterministic)"
	@echo "  make demo-agent      agent loop on the answered path (tool calling)"
	@echo "  make demo-agent-bad  agent loop with fail-closed catching a hallucination"
	@echo "  make eval            run the smoke gold-set through the pipeline"
	@echo "  make eval-compare    run the gold-set through both pipelines, print comparison"
	@echo "  make playground      launch the Streamlit demo at http://localhost:8501"
	@echo "  make web             launch the polished Tailwind UI at http://localhost:8000"
	@echo "  make web-next        launch the Next.js modern UI at http://localhost:3000 (+ FastAPI on :8000)"
	@echo "  make clean           remove generated data and audit logs"

setup:
	uv sync --extra dev
	@uv run python -c "\
from pathlib import Path; \
from regrag.ingest import parse_part_xml; \
from regrag.chunking import chunk_nodes; \
nodes = parse_part_xml(Path('tests/fixtures/reg_e_part_1005_excerpt.xml').read_bytes()); \
chunks = chunk_nodes(nodes); \
Path('data').mkdir(exist_ok=True); \
f = open('data/reg_e_excerpt.chunks.jsonl', 'w'); \
[f.write(c.model_dump_json() + '\n') for c in chunks]; \
f.close(); \
print(f'wrote {len(chunks)} chunks to data/reg_e_excerpt.chunks.jsonl')"

test:
	uv run pytest tests/unit/

demo: demo-bad

demo-bad:
	@echo "=== fail-closed: hallucinated citation gets caught ==="
	@uv run regrag query "What does Reg E say about cryptocurrency reporting?" \
		--chunks data/reg_e_excerpt.chunks.jsonl --model fake-bad

demo-good:
	@echo "=== verified citation (deterministic): answer returned ==="
	@uv run regrag query "What is the consumer's maximum liability for an unauthorized transfer reported within two business days?" \
		--chunks data/reg_e_excerpt.chunks.jsonl --model fake-good

demo-agent:
	@echo "=== agent loop (search -> finalize -> verify): answer returned ==="
	@uv run regrag query "What is the consumer's maximum liability for an unauthorized transfer reported within two business days?" \
		--chunks data/reg_e_excerpt.chunks.jsonl --model fake-good --mode agent

demo-agent-bad:
	@echo "=== agent loop with hallucination: fail-closed gate refuses ==="
	@uv run regrag query "What does Reg E say about cryptocurrency reporting?" \
		--chunks data/reg_e_excerpt.chunks.jsonl --model fake-bad --mode agent

eval:
	uv run regrag eval --gold tests/fixtures/eval_smoke.jsonl \
		--chunks data/reg_e_excerpt.chunks.jsonl --model fake-good

eval-compare:
	uv run regrag eval-compare --gold tests/fixtures/eval_smoke.jsonl \
		--chunks data/reg_e_excerpt.chunks.jsonl --model fake-good

playground:
	@echo "Launching Streamlit playground at http://localhost:8501 (Ctrl-C to stop)"
	PYTHONPATH=src uv run --extra playground streamlit run examples/playground/app.py

web:
	@echo "Launching polished web UI at http://localhost:8000 (Ctrl-C to stop)"
	PYTHONPATH=src uv run --extra playground uvicorn regrag.api:app \
		--host 127.0.0.1 --port 8000 \
		--reload --reload-dir src --reload-dir examples/web

web-next:
	@echo "Launching FastAPI on :8000 + Next.js dev server on :3000 (Ctrl-C to stop)"
	@if [ ! -d examples/web-next/node_modules ]; then \
		echo "Installing Next.js deps (one time)..."; \
		cd examples/web-next && npm install; \
	fi
	@(PYTHONPATH=src uv run --extra playground uvicorn regrag.api:app \
		--host 127.0.0.1 --port 8000 --reload --reload-dir src &) ; \
	cd examples/web-next && npm run dev

clean:
	rm -rf data/ audit.jsonl audit-eval.jsonl evals/reports/*.json

clear-cache:
	@echo "Clearing query cache..."
	rm -f data/query_cache.jsonl
	@echo "  done."
