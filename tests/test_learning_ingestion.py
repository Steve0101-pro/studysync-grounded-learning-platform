import pytest

from app.services.learning import ingestion
from app.services import study_materials
from app.services.study_materials import StudyMaterialError


@pytest.mark.asyncio
async def test_video_ingestion_preserves_timestamps_and_hierarchical_chunks(monkeypatch):
    async def transcribe(*_args):
        return [
            {"start": 12.2, "text": "The mitochondria produce ATP."},
            {"start": 12.8, "text": "The cell membrane regulates transport."},
            {"start": 68.1, "text": "Ribosomes assemble proteins."},
        ]

    monkeypatch.setattr(ingestion, "transcribe_audio_segments", transcribe)
    monkeypatch.setattr(
        study_materials,
        "embed_texts",
        lambda texts: [[1.0, 0.0] for _ in texts],
    )
    content_type, unit_count, units = await ingestion.ingest_lecture_video(
        b"video bytes", "lecture.mp4", "video/mp4"
    )

    assert content_type == "video/mp4"
    children = [unit for unit in units if unit.chunk_level == "child"]
    assert unit_count == len(children) == 3
    assert len(units) == 6
    assert [(unit.location_type, unit.location_number, unit.chunk_index) for unit in children] == [
        ("timestamp", 12, 0),
        ("timestamp", 12, 1),
        ("timestamp", 68, 0),
    ]


@pytest.mark.asyncio
async def test_video_ingestion_rejects_empty_and_oversize_uploads():
    with pytest.raises(StudyMaterialError, match="empty"):
        await ingestion.ingest_lecture_video(b"", "lecture.mp4", "video/mp4")

    oversized = b"x" * (ingestion.MAX_SOURCE_BYTES + 1)
    with pytest.raises(StudyMaterialError, match="25 MB"):
        await ingestion.ingest_lecture_video(oversized, "lecture.mp4", "video/mp4")
