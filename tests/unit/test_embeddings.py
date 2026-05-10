"""Tests for the Embedder protocol and HashingEmbedder.

We don't test BGEEmbedder here — it requires a model download from
huggingface.co which CI runs and the sandbox can't do hermetically. A
manual integration smoke test against the real BGE model belongs in
tests/integration/.
"""

from __future__ import annotations

import math

from regrag.retrieval.embeddings import Embedder, HashingEmbedder

# ---- Protocol conformance ----


def test_hashing_embedder_implements_protocol() -> None:
    embedder = HashingEmbedder()
    assert isinstance(embedder, Embedder)
    assert embedder.dim == 256
    assert embedder.name == "hashing-256"


# ---- Determinism ----


def test_hashing_embedder_is_deterministic() -> None:
    e = HashingEmbedder()
    a = e.embed_query("consumer liability")
    b = e.embed_query("consumer liability")
    assert a == b


def test_hashing_embedder_same_input_same_output() -> None:
    e = HashingEmbedder()
    a = e.embed_texts(["one text", "another text"])
    b = e.embed_texts(["one text", "another text"])
    assert a == b


# ---- Shape ----


def test_embedding_dimensionality() -> None:
    e = HashingEmbedder(dim=128)
    v = e.embed_query("anything")
    assert len(v) == 128


def test_embed_texts_returns_one_vector_per_input() -> None:
    e = HashingEmbedder()
    out = e.embed_texts(["alpha", "bravo", "charlie"])
    assert len(out) == 3
    assert all(len(v) == e.dim for v in out)


# ---- Normalization ----


def test_embeddings_are_l2_normalized() -> None:
    """Both default embedders L2-normalize so cosine similarity reduces to
    a dot product. Tests downstream depend on this."""
    e = HashingEmbedder()
    v = e.embed_query("consumer liability for unauthorized transfers")
    norm = math.sqrt(sum(x * x for x in v))
    assert abs(norm - 1.0) < 1e-6


def test_empty_query_returns_zero_vector() -> None:
    e = HashingEmbedder()
    v = e.embed_query("")
    assert v == [0.0] * e.dim


# ---- Token-overlap behavior (sanity, not semantic) ----


def test_overlapping_texts_have_higher_similarity_than_disjoint() -> None:
    """The hashing embedder is lexical — it should give higher similarity
    between texts sharing tokens than between texts with none in common.
    This is the architectural guarantee even though the embedder isn't
    semantic. (BGE in production gives semantic similarity instead.)"""
    e = HashingEmbedder()
    # All L2-normalized, so dot product == cosine similarity.
    v1 = e.embed_query("consumer liability unauthorized transfer")
    v2 = e.embed_query("liability unauthorized transfer consumer")  # same tokens
    v3 = e.embed_query("xylophone narwhal mango")  # disjoint
    sim_overlap = sum(a * b for a, b in zip(v1, v2, strict=True))
    sim_disjoint = sum(a * b for a, b in zip(v1, v3, strict=True))
    assert sim_overlap > sim_disjoint
    assert sim_overlap > 0.99  # essentially identical bag-of-words
    assert sim_disjoint < 0.1
