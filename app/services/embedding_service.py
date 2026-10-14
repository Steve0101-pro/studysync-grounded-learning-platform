"""Shared embedding and cosine-similarity helpers for learning retrieval."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from app.config import settings

_embedder = None
_late_embedder = None


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    global _embedder
    if _embedder is None:
        from fastembed import TextEmbedding

        _embedder = TextEmbedding(model_name=settings.STUDY_EMBEDDING_MODEL)
    vectors = [vector.tolist() for vector in _embedder.embed(texts)]
    if len(vectors) != len(texts):
        raise RuntimeError("Embedding model returned an unexpected number of vectors")
    dimensions = {len(vector) for vector in vectors}
    if len(dimensions) != 1 or not next(iter(dimensions), 0):
        raise RuntimeError("Embedding model returned inconsistent vector dimensions")
    return vectors


def _get_late_embedder():
    global _late_embedder
    if _late_embedder is None:
        from fastembed.late_interaction.token_embeddings import TokenEmbeddingsModel

        _late_embedder = TokenEmbeddingsModel(
            model_name=settings.STUDY_LATE_EMBEDDING_MODEL,
            threads=2,
        )
        _late_embedder.load_onnx_model()
    return _late_embedder


def _offsets_for_token_vectors(encoding: Any, token_count: int) -> list[tuple[int, int]]:
    offsets = encoding.offsets
    attention_mask = getattr(encoding, "attention_mask", None)
    if attention_mask is not None:
        if len(attention_mask) != len(offsets):
            raise RuntimeError("Late embedding attention mask does not match tokenizer offsets")
        offsets = [
            offset
            for offset, is_attended in zip(offsets, attention_mask, strict=True)
            if is_attended
        ]
    if len(offsets) != token_count:
        raise RuntimeError("Late embedding token offsets do not match model output")
    return offsets


def embed_late_chunks(chunks: list[dict[str, Any]]) -> list[list[float]]:
    """Pool contextual token vectors into vectors for their child text spans."""
    if not chunks:
        return []

    grouped: dict[str, dict[str, Any]] = {}
    for index, chunk in enumerate(chunks):
        context = str(chunk.get("parent_content") or chunk["content"])
        context_id = str(chunk.get("parent_unit_id") or f"unparented-{index}")
        start = int(chunk.get("char_start") or 0)
        end = int(chunk.get("char_end") or len(context))
        if start < 0 or end <= start or end > len(context):
            raise ValueError("Late chunk span must be a valid range inside its parent")
        group = grouped.setdefault(context_id, {"context": context, "items": []})
        if group["context"] != context:
            raise ValueError("A parent ID cannot refer to multiple parent texts")
        group["items"].append((index, start, end))

    contexts = [group["context"] for group in grouped.values()]
    embedder = _get_late_embedder()
    encodings = embedder.tokenize(contexts)
    token_vectors = list(embedder.embed(contexts, batch_size=8))
    if len(token_vectors) != len(contexts):
        raise RuntimeError("Late embedding model returned an unexpected number of contexts")

    vectors: list[list[float] | None] = [None] * len(chunks)
    for group, encoding, tokens in zip(grouped.values(), encodings, token_vectors, strict=True):
        offsets = _offsets_for_token_vectors(encoding, len(tokens))
        token_array = np.asarray(tokens, dtype=np.float32)
        for index, start, end in group["items"]:
            selected = [
                token_array[token_index]
                for token_index, (token_start, token_end) in enumerate(offsets)
                if token_start < token_end and token_start < end and token_end > start
            ]
            if not selected:
                raise RuntimeError("Late chunk span did not align with any embedding tokens")
            pooled = np.mean(selected, axis=0)
            magnitude = float(np.linalg.norm(pooled))
            if not magnitude:
                raise RuntimeError("Late embedding model returned a zero child vector")
            vectors[index] = (pooled / magnitude).astype(np.float32).tolist()

    if any(vector is None for vector in vectors):
        raise RuntimeError("Late embedding model did not return every child vector")
    return [vector for vector in vectors if vector is not None]


def embed_late_query(query: str) -> list[float]:
    if not query.strip():
        raise ValueError("A non-empty query is required")
    embedder = _get_late_embedder()
    encoding = embedder.tokenize([query])[0]
    tokens = np.asarray(next(iter(embedder.embed([query]))), dtype=np.float32)
    offsets = _offsets_for_token_vectors(encoding, len(tokens))
    lexical_tokens = [
        tokens[index]
        for index, (start, end) in enumerate(offsets)
        if start < end
    ]
    if not lexical_tokens:
        raise RuntimeError("Late embedding model returned no query tokens")
    pooled = np.mean(lexical_tokens, axis=0)
    magnitude = float(np.linalg.norm(pooled))
    if not magnitude:
        raise RuntimeError("Late embedding model returned a zero query vector")
    return (pooled / magnitude).astype(np.float32).tolist()


def cosine_similarity(left: list[float], right: list[float]) -> float:
    left_vector = np.asarray(left, dtype=np.float64)
    right_vector = np.asarray(right, dtype=np.float64)
    if left_vector.ndim != 1 or right_vector.ndim != 1:
        raise ValueError("Cosine similarity expects one-dimensional vectors")
    if left_vector.shape != right_vector.shape:
        raise ValueError("Cosine similarity vectors must have equal dimensions")
    denominator = float(np.linalg.norm(left_vector) * np.linalg.norm(right_vector))
    if not denominator:
        return 0.0
    return float(np.dot(left_vector, right_vector) / denominator)
