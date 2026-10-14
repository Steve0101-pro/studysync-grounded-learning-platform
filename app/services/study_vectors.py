"""Qdrant-backed vector retrieval for source-grounded study passages."""

from __future__ import annotations

import logging
from pathlib import Path
from threading import RLock
from typing import Any
from uuid import UUID

from app.config import BASE_DIR, settings
from app.services.embedding_service import embed_texts

logger = logging.getLogger(__name__)

COLLECTION_NAME = "study_content_units"
LATE_COLLECTION_NAME = "study_content_units_late_v1"
MIN_RELEVANCE_SCORE = 0.35
_lock = RLock()
_client = None
_embedder = None
_vector_size: int | None = None


def _get_components():
    global _client, _embedder, _vector_size
    with _lock:
        if _client is None:
            from qdrant_client import QdrantClient

            if settings.QDRANT_URL:
                _client = QdrantClient(
                    url=settings.QDRANT_URL,
                    api_key=settings.QDRANT_API_KEY.get_secret_value() or None,
                    timeout=settings.DATABASE_CONNECT_TIMEOUT_SECONDS,
                )
            else:
                local_path = Path(settings.QDRANT_LOCAL_PATH)
                if not local_path.is_absolute():
                    local_path = BASE_DIR / local_path
                _client = QdrantClient(path=str(local_path))
        return _client, _embedder


def _embed(texts: list[str]) -> list[list[float]]:
    global _vector_size
    _get_components()
    vectors = (
        [
            vector.tolist() if hasattr(vector, "tolist") else list(vector)
            for vector in _embedder.embed(texts)
        ]
        if _embedder is not None
        else embed_texts(texts)
    )
    if not vectors:
        return []
    dimensions = len(vectors[0])
    if any(len(vector) != dimensions for vector in vectors):
        raise RuntimeError("Embedding model returned inconsistent vector dimensions")
    if _vector_size is not None and _vector_size != dimensions:
        raise RuntimeError("Embedding dimensions changed after Qdrant collection initialization")
    _vector_size = dimensions
    return vectors


def _ensure_collection(collection_name: str, vector_size: int) -> None:
    from qdrant_client.models import Distance, VectorParams

    client, _ = _get_components()
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        return

    info = client.get_collection(collection_name)
    existing = info.config.params.vectors
    if isinstance(existing, dict):
        sizes = {vector.size for vector in existing.values()}
        compatible = vector_size in sizes and len(sizes) == 1
    else:
        compatible = existing.size == vector_size
    if not compatible:
        raise RuntimeError(
            "Qdrant collection vector size does not match the configured embedding model"
        )


def index_study_passages(
    user_id: UUID,
    source_id: UUID,
    filename: str,
    passages: list[dict[str, Any]],
) -> int:
    """Dual-index searchable child chunks with scoped provenance and parent context."""
    if not passages:
        return 0
    from qdrant_client.models import PointStruct

    with _lock:
        client, _ = _get_components()
        payloads = [
            {
                "user_id": str(user_id),
                "source_id": str(source_id),
                "filename": filename,
                "location_type": passage["location_type"],
                "location_number": passage["location_number"],
                "chunk_index": passage["chunk_index"],
                "content": passage["content"],
            }
            for passage in passages
        ]
        legacy_vectors = _embed([passage["content"] for passage in passages])
        _ensure_collection(COLLECTION_NAME, len(legacy_vectors[0]))
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=[
                PointStruct(
                    id=str(passage["id"]),
                    vector=vector,
                    payload=payload,
                )
                for passage, vector, payload in zip(
                    passages,
                    legacy_vectors,
                    payloads,
                    strict=True,
                )
            ],
        )

        if _embedder is not None:
            late_vectors = _embed([passage["content"] for passage in passages])
        else:
            from app.services.embedding_service import embed_late_chunks

            late_vectors = embed_late_chunks(passages)
        _ensure_collection(LATE_COLLECTION_NAME, len(late_vectors[0]))
        client.upsert(
            collection_name=LATE_COLLECTION_NAME,
            points=[
                PointStruct(
                    id=str(passage["id"]),
                    vector=vector,
                    payload={
                        **payload,
                        "parent_unit_id": passage.get("parent_unit_id"),
                        "parent_content": passage.get("parent_content", passage["content"]),
                        "char_start": passage.get("char_start", 0),
                        "char_end": passage.get("char_end", len(passage["content"])),
                    },
                )
                for passage, vector, payload in zip(
                    passages,
                    late_vectors,
                    payloads,
                    strict=True,
                )
            ],
        )
    return len(passages)


def search_study_passages(
    user_id: UUID,
    query: str,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Search the Qdrant index while enforcing per-user payload filtering."""
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    def query_collection(collection_name: str, vector: list[float]):
        if not client.collection_exists(collection_name):
            return []
        return client.query_points(
            collection_name=collection_name,
            query=vector,
            query_filter=Filter(
                must=[
                    FieldCondition(
                        key="user_id",
                        match=MatchValue(value=str(user_id)),
                    )
                ]
            ),
            limit=limit,
            with_payload=True,
        ).points

    with _lock:
        client, _ = _get_components()
        results_by_id: dict[str, dict[str, Any]] = {}
        failure: Exception | None = None
        if client.collection_exists(LATE_COLLECTION_NAME):
            try:
                from app.services.embedding_service import embed_late_query

                late_vector = (
                    _embed([query])[0]
                    if _embedder is not None
                    else embed_late_query(query)
                )
                for match in query_collection(LATE_COLLECTION_NAME, late_vector):
                    payload = match.payload or {}
                    if payload.get("user_id") != str(user_id):
                        logger.error("Qdrant returned a study passage outside the requested user scope")
                        continue
                    score = float(match.score)
                    if score >= MIN_RELEVANCE_SCORE:
                        results_by_id[str(match.id)] = {
                            "source_id": payload["source_id"],
                            "filename": payload["filename"],
                            "location_type": payload["location_type"],
                            "location_number": payload["location_number"],
                            "chunk_index": payload["chunk_index"],
                            "parent_content": payload.get("parent_content", payload["content"]),
                            "content": payload["content"],
                            "score": score,
                        }
            except Exception as exc:
                failure = exc
                logger.exception("Late-chunk Qdrant search failed")

        if client.collection_exists(COLLECTION_NAME):
            try:
                legacy_vector = _embed([query])[0]
                for match in query_collection(COLLECTION_NAME, legacy_vector):
                    payload = match.payload or {}
                    if payload.get("user_id") != str(user_id):
                        logger.error("Qdrant returned a study passage outside the requested user scope")
                        continue
                    score = float(match.score)
                    if score >= MIN_RELEVANCE_SCORE:
                        key = str(match.id)
                        existing = results_by_id.get(key)
                        if existing is None:
                            results_by_id[key] = {
                                "source_id": payload["source_id"],
                                "filename": payload["filename"],
                                "location_type": payload["location_type"],
                                "location_number": payload["location_number"],
                                "chunk_index": payload["chunk_index"],
                                "parent_content": payload["content"],
                                "content": payload["content"],
                                "score": score,
                            }
                        else:
                            existing["score"] = max(existing["score"], score)
            except Exception as exc:
                if not results_by_id:
                    if failure is not None:
                        raise RuntimeError("Both late and standard Qdrant searches failed") from exc
                    raise
                logger.exception("Standard Qdrant search failed; returning late-chunk matches")
        if failure is not None and not results_by_id:
            raise RuntimeError("Late-chunk Qdrant search failed") from failure

    return sorted(results_by_id.values(), key=lambda result: result["score"], reverse=True)[:limit]
