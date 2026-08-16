"""Nugget extraction: select query-relevant sentences from a chunk.

Based on the CoinRAG idea: instead of returning full chunks to the LLM,
return only the top-N sentences most relevant to the query (nuggets).
This reduces context length while maintaining or improving Recall@k.

No external dependencies — uses BM25 and sentence splitting only.
"""

from __future__ import annotations

import math
import re
from collections import Counter


def split_sentences(text: str) -> list[str]:
    """Split text into sentences on '. ', '? ', '! ' boundaries."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


def bm25_scores(query: str, sentences: list[str], k1: float = 1.5, b: float = 0.75) -> list[float]:
    """Return BM25 score for each sentence relative to query."""
    if not sentences:
        return []
    tokenized = [_tokenize(s) for s in sentences]
    avgdl = sum(len(t) for t in tokenized) / len(tokenized)
    q_terms = _tokenize(query)
    scores = []
    for doc in tokenized:
        tf = Counter(doc)
        dl = len(doc)
        score = 0.0
        for term in q_terms:
            f = tf.get(term, 0)
            idf = math.log(1 + (len(sentences) - f + 0.5) / (f + 0.5))
            numerator = f * (k1 + 1)
            denominator = f + k1 * (1 - b + b * dl / max(avgdl, 1))
            score += idf * (numerator / denominator)
        scores.append(score)
    return scores


def _normalize(scores: list[float]) -> list[float]:
    """Min-max normalize to [0, 1]. Returns zeros if all scores are equal."""
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    if hi == lo:
        return [0.0] * len(scores)
    return [(s - lo) / (hi - lo) for s in scores]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors. Returns 0.0 for a zero vector."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def extract_nuggets(
    query: str,
    text: str,
    top_k: int = 3,
    embed_weight: float = 0.0,
    query_vec: list[float] | None = None,
    sentence_vecs: list[list[float]] | None = None,
) -> str:
    """Return the top-k most query-relevant sentences from text joined as a string.

    Scoring is BM25 by default. When ``query_vec`` and ``sentence_vecs`` are
    supplied (one vector per split sentence), the score is a min-max blend of
    normalized BM25 and cosine similarity, controlled by ``embed_weight``
    (0 = BM25-only, 1 = embedding-only). nugget-rag-eval #11 found w=0.7 optimal
    (Recall 1.0, ~75% token reduction).

    Falls back to the full text if it cannot be split into sentences, and to
    BM25-only if embeddings are absent or mismatched in length.
    """
    sentences = split_sentences(text)
    if not sentences:
        return text

    bm25 = _normalize(bm25_scores(query, sentences))
    if (
        embed_weight > 0.0
        and query_vec is not None
        and sentence_vecs is not None
        and len(sentence_vecs) == len(sentences)
    ):
        emb = _normalize([cosine_similarity(query_vec, sv) for sv in sentence_vecs])
        combined = [(1 - embed_weight) * b + embed_weight * e for b, e in zip(bm25, emb)]
    else:
        combined = bm25

    ranked = sorted(zip(combined, sentences), reverse=True)
    return " ".join(s for _, s in ranked[:top_k])
