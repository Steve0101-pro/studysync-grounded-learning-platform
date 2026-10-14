"""Sparse BM25 ranking and hybrid fusion for study passages."""

from __future__ import annotations

import math
from collections import Counter

from app.services.study_materials import tokenize


def bm25_scores(query: str, documents: list[str]) -> list[float]:
    """Return non-negative BM25 scores aligned with the supplied documents."""
    query_terms = tokenize(query)
    tokenized_documents = [tokenize(document) for document in documents]
    if not query_terms or not tokenized_documents:
        return [0.0] * len(documents)

    document_lengths = [len(tokens) for tokens in tokenized_documents]
    average_length = sum(document_lengths) / len(document_lengths) or 1.0
    term_frequencies = [Counter(tokens) for tokens in tokenized_documents]
    document_frequency: Counter[str] = Counter()
    for frequencies in term_frequencies:
        document_frequency.update(frequencies.keys())

    count = len(tokenized_documents)
    k1 = 1.5
    b = 0.75
    scores = []
    for frequencies, document_length in zip(term_frequencies, document_lengths, strict=True):
        score = 0.0
        for term in query_terms:
            frequency = frequencies.get(term, 0)
            if not frequency:
                continue
            inverse_frequency = math.log(
                1 + (count - document_frequency[term] + 0.5)
                / (document_frequency[term] + 0.5)
            )
            length_normalizer = k1 * (
                1 - b + b * document_length / average_length
            )
            score += inverse_frequency * (
                frequency * (k1 + 1) / (frequency + length_normalizer)
            )
        scores.append(score)
    return scores


def fuse_bm25_and_cosine(
    lexical_scores: list[float],
    cosine_scores: list[float],
    *,
    lexical_weight: float = 0.6,
) -> list[float]:
    """Fuse BM25 scores with cosine similarities on a normalized 0..1 scale."""
    if len(lexical_scores) != len(cosine_scores):
        raise ValueError("BM25 and cosine score lists must have the same length")
    if not 0 <= lexical_weight <= 1:
        raise ValueError("lexical_weight must be between 0 and 1")

    max_lexical = max(lexical_scores, default=0.0)
    fused = []
    for lexical, cosine in zip(lexical_scores, cosine_scores, strict=True):
        normalized_lexical = lexical / max_lexical if lexical and max_lexical else 0.0
        normalized_cosine = min(1.0, max(0.0, cosine))
        fused.append(
            lexical_weight * normalized_lexical
            + (1 - lexical_weight) * normalized_cosine
        )
    return fused
