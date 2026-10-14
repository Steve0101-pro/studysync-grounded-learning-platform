"""Public service interface for the Qdrant-backed study vector index."""

from app.services.study_vectors import (
    COLLECTION_NAME,
    MIN_RELEVANCE_SCORE,
    index_study_passages,
    search_study_passages,
)

__all__ = [
    "COLLECTION_NAME",
    "MIN_RELEVANCE_SCORE",
    "index_study_passages",
    "search_study_passages",
]
