import uuid

import numpy as np
import pytest
from qdrant_client import QdrantClient

from app.services import study_vectors


class _FakeEmbedding:
    def embed(self, documents):
        for document in documents:
            words = document.casefold()
            yield np.array(
                [
                    1.0 if "mitochondria" in words else 0.0,
                    1.0 if "membrane" in words else 0.0,
                    1.0,
                ],
                dtype=np.float32,
            )


@pytest.fixture
def memory_qdrant(monkeypatch):
    monkeypatch.setattr(study_vectors, "_client", QdrantClient(location=":memory:"))
    monkeypatch.setattr(study_vectors, "_embedder", _FakeEmbedding())
    monkeypatch.setattr(study_vectors, "_vector_size", None)
    yield
    study_vectors._client.close()
    monkeypatch.setattr(study_vectors, "_client", None)
    monkeypatch.setattr(study_vectors, "_embedder", None)
    monkeypatch.setattr(study_vectors, "_vector_size", None)


def test_qdrant_indexes_passages_and_filters_by_user(memory_qdrant):
    owner_id = uuid.uuid4()
    other_user_id = uuid.uuid4()
    source_id = uuid.uuid4()
    passages = [
        {
            "id": uuid.uuid4(),
            "location_type": "page",
            "location_number": 2,
            "chunk_index": 0,
            "content": "Mitochondria produce ATP.",
        },
        {
            "id": uuid.uuid4(),
            "location_type": "page",
            "location_number": 4,
            "chunk_index": 0,
            "content": "The cell membrane controls transport.",
        },
    ]

    assert study_vectors.index_study_passages(
        owner_id, source_id, "biology.pdf", passages
    ) == 2

    owner_results = study_vectors.search_study_passages(owner_id, "mitochondria ATP")
    other_user_results = study_vectors.search_study_passages(other_user_id, "mitochondria ATP")

    assert len(owner_results) == 2
    assert owner_results[0]["location_number"] == 2
    assert owner_results[0]["source_id"] == str(source_id)
    assert other_user_results == []


def test_qdrant_discards_low_relevance_passages(memory_qdrant, monkeypatch):
    owner_id = uuid.uuid4()
    source_id = uuid.uuid4()
    passages = [
        {
            "id": uuid.uuid4(),
            "location_type": "page",
            "location_number": 2,
            "chunk_index": 0,
            "content": "Mitochondria produce ATP.",
        },
        {
            "id": uuid.uuid4(),
            "location_type": "page",
            "location_number": 4,
            "chunk_index": 0,
            "content": "The cell membrane controls transport.",
        },
    ]
    study_vectors.index_study_passages(owner_id, source_id, "biology.pdf", passages)
    monkeypatch.setattr(study_vectors, "MIN_RELEVANCE_SCORE", 0.75)

    results = study_vectors.search_study_passages(owner_id, "mitochondria ATP")

    assert len(results) == 1
    assert results[0]["location_number"] == 2
