import numpy as np

from app.services import embedding_service


class _CharacterTokenEncoder:
    def tokenize(self, contexts):
        encodings = []
        for context in contexts:
            offsets = [(0, 0)] + [
                (index, index + 1) for index in range(len(context))
            ]
            attention_mask = [1] * len(offsets)
            offsets.extend([(0, 0)] * 3)
            attention_mask.extend([0] * 3)
            encodings.append(
                type(
                    "Encoding",
                    (),
                    {"offsets": offsets, "attention_mask": attention_mask},
                )()
            )
        return encodings

    def embed(self, contexts, batch_size=256):
        del batch_size
        for context in contexts:
            yield np.array(
                [[0.0, 1.0]]
                + [[float(ord(character)), 1.0] for character in context],
                dtype=np.float32,
            )


def test_late_chunking_pools_child_spans_from_shared_parent_context(monkeypatch):
    monkeypatch.setattr(embedding_service, "_late_embedder", _CharacterTokenEncoder())
    passages = [
        {
            "parent_unit_id": "parent-1",
            "parent_content": "alpha beta",
            "char_start": 0,
            "char_end": 5,
            "content": "alpha",
        },
        {
            "parent_unit_id": "parent-1",
            "parent_content": "alpha beta",
            "char_start": 6,
            "char_end": 10,
            "content": "beta",
        },
    ]

    vectors = embedding_service.embed_late_chunks(passages)

    assert len(vectors) == 2
    assert all(len(vector) == 2 for vector in vectors)
    assert vectors[0] != vectors[1]


def test_late_query_embedding_ignores_padded_and_special_token_offsets(monkeypatch):
    monkeypatch.setattr(embedding_service, "_late_embedder", _CharacterTokenEncoder())

    vector = embedding_service.embed_late_query("abc")
    expected = np.mean(
        [[float(ord(character)), 1.0] for character in "abc"],
        axis=0,
    )
    expected /= np.linalg.norm(expected)

    assert np.allclose(vector, expected)
