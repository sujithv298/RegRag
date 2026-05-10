"""Prompt template — versioned.

`PROMPT_TEMPLATE_VERSION` bumps with any change to system prompt, user
message structure, or the citation tag format. The audit log records the
version that produced each response, so a future replay knows which
prompt the model saw.

Format choices that matter:
  - `[CFR:<citation>]` inline tags. Easy to parse, hard to confuse with
    paragraph designators that already use parens.
  - Explicit refusal phrase. Gives the model a way to opt out without
    fabricating citations. The verifier converts any unverifiable response
    to a refusal anyway, but a model-side refusal is cleaner.
  - "Use ONLY the regulatory text" + "Do not draw on prior training
    knowledge." LLMs trained on the internet have memorized parts of
    Reg E; we want them to ignore that and ground in the retrieved chunks.
"""

from __future__ import annotations

from regrag.chunking import Chunk

PROMPT_TEMPLATE_VERSION = "0.1.0"

REFUSAL_PHRASE = "I cannot answer this question based on the provided regulatory text."

SYSTEM_PROMPT = """You are a regulatory compliance assistant for US banking regulations. \
You answer questions using only the regulatory text excerpts provided in each conversation.

Rules:

1. Use ONLY the regulatory text provided in the conversation. Do not draw on prior \
training knowledge of these regulations, even if you remember it.

2. Every factual claim must be followed by an inline citation in the format \
[CFR:<citation>]. Examples: [CFR:1005.6(b)(1)] for a rule paragraph, or \
[CFR:Comment 6(b)-1] for an Official Interpretation comment. Place the citation in \
the same sentence as the claim it supports.

3. If the provided regulatory text does not contain enough information to answer the \
question, respond with exactly this phrase and nothing else: \
"{refusal_phrase}"

4. Paraphrase in plain English. Quote regulation language only when the exact wording \
matters. Do not speculate, extrapolate, or give legal advice.

5. Do not invent citations. If language you want to cite is not in the provided text, \
do not write it.""".format(refusal_phrase=REFUSAL_PHRASE)


def render_user_message(question: str, chunks: list[Chunk]) -> str:
    """Render the user-side message: retrieved chunks + the question."""
    if not chunks:
        excerpts_block = "(no regulatory text retrieved for this query)"
    else:
        excerpts = []
        for i, chunk in enumerate(chunks, start=1):
            heading = chunk.section_heading or ""
            if chunk.is_interpretation:
                heading = (
                    f"Official Interpretation, 12 CFR Part {chunk.part}, "
                    f"Comment {chunk.comment_id}"
                )
            excerpts.append(
                f"[Excerpt {i}] Citation: {chunk.citation_path}\n"
                f"{heading}\n\n"
                f"{chunk.source_text}"
            )
        excerpts_block = "\n\n---\n\n".join(excerpts)

    return (
        f"Regulatory text excerpts (in order of relevance):\n\n"
        f"{excerpts_block}\n\n"
        f"---\n\n"
        f"Question: {question}\n\n"
        f"Answer the question using only the excerpts above. Cite each factual "
        f"claim with [CFR:<citation>] inline."
    )


def render_full_prompt(question: str, chunks: list[Chunk]) -> str:
    """The combined prompt text used for hashing in the audit log."""
    return f"<<system>>\n{SYSTEM_PROMPT}\n\n<<user>>\n{render_user_message(question, chunks)}"
