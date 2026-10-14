"""Source ingestion for PDF, PowerPoint, and lecture-video course materials."""

from app.services.study_materials import (
    MAX_SOURCE_BYTES,
    PDF_CONTENT_TYPE,
    PPTX_CONTENT_TYPE,
    StudyMaterialError,
    ContentUnit,
    MAX_CONTENT_UNITS,
    MAX_EXTRACTED_CHARACTERS,
    extract_study_material,
    semantic_parent_child_chunks,
)
from app.services.stt import transcribe_audio_segments


def ingest_course_file(file_bytes: bytes, filename: str):
    """Extract bounded, page/slide-addressable units for course indexing."""
    return extract_study_material(file_bytes, filename)


async def ingest_lecture_video(
    file_bytes: bytes,
    filename: str,
    content_type: str,
) -> tuple[str, int, list[ContentUnit]]:
    """Transcribe supported lecture video audio into timestamped source units."""
    if len(file_bytes) > MAX_SOURCE_BYTES:
        raise StudyMaterialError("The uploaded video exceeds the 25 MB limit")
    if not file_bytes:
        raise StudyMaterialError("The uploaded video is empty")

    segments = await transcribe_audio_segments(file_bytes, filename, content_type)
    units: list[ContentUnit] = []
    extracted_characters = 0
    next_child_index: dict[int, int] = {}
    next_parent_index: dict[int, int] = {}
    for segment in segments:
        timestamp = max(0, int(segment["start"]))
        text = str(segment["text"])
        extracted_characters += len(text)
        if extracted_characters > MAX_EXTRACTED_CHARACTERS:
            raise StudyMaterialError("Video transcript exceeds the 2 MB processing limit")
        segment_units = semantic_parent_child_chunks(
            text,
            "timestamp",
            timestamp,
            child_index_start=next_child_index.get(timestamp, 0),
            parent_index_start=next_parent_index.get(timestamp, 0),
        )
        units.extend(segment_units)
        next_child_index[timestamp] = (
            next_child_index.get(timestamp, 0)
            + sum(unit.chunk_level == "child" for unit in segment_units)
        )
        next_parent_index[timestamp] = (
            next_parent_index.get(timestamp, 0)
            + sum(unit.chunk_level == "parent" for unit in segment_units)
        )
        if len(units) > MAX_CONTENT_UNITS:
            raise StudyMaterialError("The video creates too many parent and child chunks")
    if not any(unit.chunk_level == "child" for unit in units):
        raise StudyMaterialError("No speech was found in the uploaded lecture video")
    return (
        content_type,
        sum(unit.chunk_level == "child" for unit in units),
        units,
    )


__all__ = [
    "MAX_SOURCE_BYTES",
    "PDF_CONTENT_TYPE",
    "PPTX_CONTENT_TYPE",
    "StudyMaterialError",
    "ingest_lecture_video",
    "ingest_course_file",
]
