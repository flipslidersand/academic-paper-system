"""Tests for nugget extraction module."""

import pytest

from academic_paper.nugget import (
    bm25_scores,
    cosine_similarity,
    extract_nuggets,
    split_sentences,
)


def test_split_sentences_basic():
    text = "This is sentence one. This is sentence two. And three!"
    parts = split_sentences(text)
    assert len(parts) == 3


def test_split_sentences_empty():
    assert split_sentences("") == []


def test_split_sentences_single():
    assert split_sentences("Only one sentence") == ["Only one sentence"]


def test_bm25_scores_length():
    sentences = ["KV cache reuse", "Diffusion models", "RAG retrieval"]
    scores = bm25_scores("KV cache", sentences)
    assert len(scores) == 3


def test_bm25_relevant_scores_higher():
    sentences = ["KV cache reuse reduces inference cost", "Diffusion models generate images"]
    scores = bm25_scores("KV cache", sentences)
    assert scores[0] > scores[1]


def test_bm25_empty_sentences():
    assert bm25_scores("query", []) == []


def test_extract_nuggets_returns_top_k():
    text = (
        "KV cache reuse reduces inference cost in transformers. "
        "Diffusion models iteratively denoise images. "
        "RAG combines retrieval with generation. "
        "KV caching is essential for long-context LLMs."
    )
    nugget = extract_nuggets("KV cache transformer", text, top_k=2)
    assert "KV" in nugget or "kv" in nugget.lower()
    # Should not be the full text (nugget is shorter)
    assert len(nugget.split()) < len(text.split())


def test_extract_nuggets_fallback_empty():
    assert extract_nuggets("query", "") == ""


def test_extract_nuggets_single_sentence():
    text = "KV cache reuse is the core mechanism."
    result = extract_nuggets("KV cache", text, top_k=3)
    assert result == text


def test_extract_nuggets_top_k_respected():
    text = (
        "Sentence one about KV cache. "
        "Sentence two about RAG. "
        "Sentence three about diffusion. "
        "Sentence four about transformers."
    )
    nugget = extract_nuggets("KV cache RAG", text, top_k=2)
    # top_k=2 → at most 2 sentences returned
    sentences = [s.strip() for s in nugget.split(".") if s.strip()]
    assert len(sentences) <= 2


# --- hybrid (BM25 + embedding) scoring (#84) ---


def test_cosine_similarity_basic():
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_similarity_zero_vector():
    assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0


def test_extract_nuggets_embedding_shifts_ranking():
    # BM25 favors the lexical-overlap sentence; embedding vectors are crafted so
    # the *second* sentence is closest to the query vector. With embed_weight=1.0
    # the embedding-favored sentence must win.
    text = "Alpha beta gamma. Delta epsilon zeta."
    query = "alpha"  # lexically overlaps only sentence one
    query_vec = [0.0, 1.0]
    sentence_vecs = [[1.0, 0.0], [0.0, 1.0]]  # sentence two aligns with query
    nugget = extract_nuggets(
        query,
        text,
        top_k=1,
        embed_weight=1.0,
        query_vec=query_vec,
        sentence_vecs=sentence_vecs,
    )
    assert nugget == "Delta epsilon zeta."


def test_extract_nuggets_weight_zero_is_bm25():
    # embed_weight=0.0 must ignore vectors entirely → identical to BM25-only.
    text = "Alpha beta gamma. Delta epsilon zeta."
    query = "alpha"
    query_vec = [0.0, 1.0]
    sentence_vecs = [[1.0, 0.0], [0.0, 1.0]]
    hybrid = extract_nuggets(query, text, top_k=1, embed_weight=0.0, query_vec=query_vec, sentence_vecs=sentence_vecs)
    bm25_only = extract_nuggets(query, text, top_k=1)
    assert hybrid == bm25_only == "Alpha beta gamma."


def test_extract_nuggets_mismatched_vecs_falls_back_to_bm25():
    # Wrong number of sentence vectors → safe BM25 fallback, no crash.
    text = "Alpha beta gamma. Delta epsilon zeta."
    result = extract_nuggets(
        "alpha",
        text,
        top_k=1,
        embed_weight=0.7,
        query_vec=[0.0, 1.0],
        sentence_vecs=[[1.0, 0.0]],  # only 1 vec for 2 sentences
    )
    assert result == "Alpha beta gamma."
