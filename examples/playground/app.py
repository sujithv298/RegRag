"""RegRAG Playground — minimal, focused.

One page. Question in, answer out, citation verified or refused.
Technical detail is one click away if you want it. No tabs, no sidebar.

Run:

    uv sync --extra playground
    make setup
    streamlit run examples/playground/app.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import streamlit as st

# Allow running from project root or this directory.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from regrag.audit import AuditLogger  # noqa: E402
from regrag.chunking import Chunk  # noqa: E402
from regrag.models import FakeAdapter  # noqa: E402
from regrag.pipeline_agent import answer_query_agentic  # noqa: E402
from regrag.retrieval import (  # noqa: E402
    BM25Index,
    DenseRetriever,
    HashingEmbedder,
    HybridRetriever,
    LexicalOverlapReranker,
)
from regrag.store import InMemoryVectorStore  # noqa: E402

# ---- Page config: centered, sidebar collapsed ----

st.set_page_config(
    page_title="RegRAG",
    page_icon="📜",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
      .cit-badge {
        display: inline-block;
        font-family: ui-monospace, "SF Mono", Menlo, monospace;
        font-size: 0.85em;
        font-weight: 500;
        padding: 3px 12px;
        border-radius: 999px;
        margin: 4px 6px 4px 0;
        border: 1px solid;
        white-space: nowrap;
      }
      .cit-pass { background: #ecfdf5; color: #047857; border-color: #6ee7b7; }
      .cit-fail { background: #fef2f2; color: #b91c1c; border-color: #fca5a5; }
      .stButton button { font-weight: 500; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---- Cached pipeline ----


@st.cache_resource
def _load() -> tuple[list[Chunk], HybridRetriever]:
    chunks_path = _ROOT / "data" / "reg_e_excerpt.chunks.jsonl"
    if not chunks_path.exists():
        st.error("Run `make setup` from the project root first.")
        st.stop()
    chunks = [
        Chunk.model_validate_json(line)
        for line in chunks_path.read_text().splitlines()
        if line.strip()
    ]
    retriever = HybridRetriever(
        bm25=BM25Index(),
        dense=DenseRetriever(embedder=HashingEmbedder(), store=InMemoryVectorStore()),
        reranker=LexicalOverlapReranker(),
    )
    retriever.add(chunks)
    return chunks, retriever


def _agent_adapter(model_id: str):
    if model_id == "fake-good":
        from regrag.cli import _fake_good_agent_responder  # noqa: PLC0415

        return FakeAdapter(agent_responder=_fake_good_agent_responder, name="fake-good", version="v0")
    if model_id == "fake-bad":
        from regrag.cli import _fake_bad_agent_responder  # noqa: PLC0415

        return FakeAdapter(agent_responder=_fake_bad_agent_responder, name="fake-bad", version="v0")
    if model_id == "anthropic":
        from regrag.models import AnthropicAgentAdapter  # noqa: PLC0415

        return AnthropicAgentAdapter()
    raise ValueError(model_id)


# ---- Run a query and render the result ----


def _run(question: str, *, model_id: str = "fake-good") -> None:
    if not question.strip():
        st.warning("Type a question first.")
        return

    chunks, retriever = _load()
    logger = AuditLogger(log_path=_ROOT / "audit-playground.jsonl")
    adapter = _agent_adapter(model_id)

    with st.status("Thinking…", expanded=False) as status:
        st.write(f"→ searching the regulation for context")
        time.sleep(0.3)
        result = answer_query_agentic(
            question,
            retriever=retriever,
            model=adapter,
            audit_logger=logger,
            corpus=chunks,
            corpus_snapshot_date="2025-01-01",
        )
        st.write(f"← retrieved {len(result.audit_record.retrieved_chunk_ids)} chunks")
        time.sleep(0.2)
        st.write("→ generating answer")
        time.sleep(0.2)
        st.write("→ verifying citations against the regulation text")
        time.sleep(0.2)
        status.update(label="Done", state="complete", expanded=False)

    # Verdict
    if result.outcome == "answered":
        st.success(f"✓  Answered")
    else:
        st.error(f"✗  Refused — {result.refusal_reason}")

    # Answer
    st.markdown(f"> {result.answer}")

    # Citations
    if result.citations:
        badges = ""
        for c in result.citations:
            cls = "cit-pass" if c.passed else "cit-fail"
            icon = "✓" if c.passed else "✗"
            badges += f'<span class="cit-badge {cls}">{icon} &nbsp;{c.citation_path}</span>'
        st.markdown(badges, unsafe_allow_html=True)

    # Single expander for everything technical
    with st.expander("Technical detail"):
        st.markdown("**Audit-log record** (what an examiner would replay)")
        st.json(json.loads(result.audit_record.model_dump_json()))


# ---- Page layout ----

st.title("RegRAG")
st.caption(
    "Ask a question about Regulation E (12 CFR Part 1005). "
    "Every citation is verified against the actual regulation text — "
    "if the model invents a fake citation, the system refuses instead of answering."
)

st.write("")

question = st.text_area(
    "Your question",
    value="",
    placeholder="e.g. What is the consumer's maximum liability if they report an unauthorized transfer within two business days?",
    height=80,
    label_visibility="collapsed",
)

if st.button("Run query", type="primary", use_container_width=True):
    _run(question)

st.write("")
st.caption("Or try one of these:")

c1, c2, c3 = st.columns(3)
with c1:
    if st.button("Standard liability question", use_container_width=True):
        _run(
            "What is the consumer's maximum liability if they report an unauthorized "
            "transfer within two business days?"
        )
with c2:
    if st.button("⚠ Try to trick it", use_container_width=True):
        _run(
            "What does 12 CFR 9999.99(z) say about cryptocurrency reporting?",
            model_id="fake-bad",
        )
with c3:
    if st.button("Out of scope", use_container_width=True):
        _run("What is the FDIC deposit insurance coverage limit per depositor?")

st.write("")
st.divider()
st.caption(
    "RegRAG · MIT licensed · "
    "[GitHub](https://github.com/your-handle/regrag)"
)
