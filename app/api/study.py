"""Authenticated endpoints for course-material ingestion and cited retrieval."""

from __future__ import annotations

import hashlib
import json
import logging
import secrets
import uuid
import base64
from collections import defaultdict
from functools import partial
from datetime import datetime, timezone
from typing import Literal
from urllib.parse import urljoin

import anyio
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile, status
from fastapi.responses import StreamingResponse
from sqlalchemy import case, func, or_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import aliased

from app.config import settings
from app.database import SessionLocal, get_db
from app.dependencies.auth import get_current_user
from app.models import Conversation, Message, StudyContentUnit, StudyQuizQuestion, StudySource, User
from app.rate_limit import limiter, RATE_LIMIT_STUDY
from app.schema import (
    StudyProgressResponse,
    StudyQuizAnswer,
    StudyQuizCreate,
    StudyQuizResponse,
    StudySearchResponse,
    StudySourceResponse,
    StudyTutorResponse,
)
from app.services.storage import create_signed_url, delete_file, download_file, upload_file
from app.services.study_orchestration import run_study_tutor, stream_study_tutor
from app.services.qdrant_service import index_study_passages, search_study_passages
from app.services.neo4j_service import index_source_passages
from app.services.media_validation import (
    ALLOWED_IMAGE_TYPES,
    ALLOWED_TRANSCRIPTION_TYPES,
    MAX_MULTIMODAL_FILE_SIZE,
    has_valid_media_signature,
)
from app.services.learning.ingestion import (
    MAX_SOURCE_BYTES,
    PDF_CONTENT_TYPE,
    PPTX_CONTENT_TYPE,
    StudyMaterialError,
    ingest_course_file,
    ingest_lecture_video,
)
from app.services.learning.mastery import estimate_mastery, recommended_review_days
from app.services.study_materials import (
    StudyEmbeddingUnavailable,
    _prepare_figure_image,
    _pptx_slides,
    rank_content,
    tokenize,
)
from app.services.learning.retrieval import bm25_scores, fuse_bm25_and_cosine
from app.services.safety import SafetyViolation, assert_safe_text, redact_sensitive_text
from app.services.stt import transcribe_audio

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/study", tags=["Study materials"])

VIDEO_CONTENT_TYPES = {
    "mp4": "video/mp4",
    "webm": "video/webm",
}
ALLOWED_CONTENT_TYPES = {
    PDF_CONTENT_TYPE,
    PPTX_CONTENT_TYPE,
    *VIDEO_CONTENT_TYPES.values(),
    "application/octet-stream",
}


async def _cleanup_storage_path(storage_path: str | None) -> None:
    if not storage_path:
        return
    try:
        await anyio.to_thread.run_sync(delete_file, storage_path)
    except Exception:
        logger.warning("Failed to clean up study source storage object", exc_info=True)


async def _retrieve_study_units(
    db: AsyncSession,
    user_id,
    query: str,
    limit: int = 10,
):
    terms = list(dict.fromkeys(tokenize(query)))
    if not terms:
        raise HTTPException(status_code=422, detail="Query must contain searchable text")
    if len(terms) > 25:
        raise HTTPException(status_code=422, detail="Query must contain no more than 25 distinct terms")

    predicates = [StudyContentUnit.content.ilike(f"%{term}%") for term in terms]
    parent_unit = aliased(StudyContentUnit)
    result = await db.execute(
        select(StudyContentUnit, StudySource, parent_unit.content)
        .join(StudySource, StudyContentUnit.source_id == StudySource.id)
        .outerjoin(parent_unit, StudyContentUnit.parent_unit_id == parent_unit.id)
        .where(StudySource.user_id == user_id)
        .where(StudyContentUnit.chunk_level == "child")
        .where(or_(*predicates))
        .limit(500)
    )
    rows = [
        (row[0], row[1], row[2] if len(row) > 2 and row[2] else row[0].content)
        for row in result.all()
    ]
    lexical_scores = bm25_scores(query, [unit.content for unit, _source, _parent in rows])
    candidates = [
        (score, unit, source, parent_content)
        for score, (unit, source, parent_content) in zip(lexical_scores, rows, strict=True)
    ]
    candidates = [candidate for candidate in candidates if candidate[0] > 0]
    candidates.sort(
        key=lambda item: (
            -item[0],
            item[2].filename,
            item[1].location_number,
            item[1].chunk_index,
        )
    )
    return candidates[:limit]


def _passage_key(passage: dict) -> tuple[str, str, int, int]:
    return (
        str(passage["source_id"]),
        passage["location_type"],
        int(passage["location_number"]),
        int(passage["chunk_index"]),
    )


async def _hybrid_retrieve_study_passages(
    db: AsyncSession,
    user_id,
    query: str,
    limit: int,
) -> tuple[list[dict], str]:
    lexical_candidates = await _retrieve_study_units(db, user_id, query, limit=500)
    passages: dict[tuple[str, str, int, int], dict] = {}
    for score, unit, source, parent_content in lexical_candidates:
        passage = {
            "source_id": source.id,
            "filename": source.filename,
            "location_type": unit.location_type,
            "location_number": unit.location_number,
            "chunk_index": unit.chunk_index,
            "content": unit.content,
            "parent_content": parent_content,
            "bm25_score": score,
            "cosine_score": 0.0,
        }
        passages[_passage_key(passage)] = passage

    try:
        vector_results = await anyio.to_thread.run_sync(
            search_study_passages,
            user_id,
            query,
            max(limit * 2, 10),
        )
    except Exception:
        logger.exception("Qdrant study retrieval failed; using BM25 retrieval")
        vector_results = []

    for result in vector_results:
        key = _passage_key(result)
        existing = passages.get(key)
        if existing is None:
            passages[key] = {
                **result,
                "bm25_score": 0.0,
                "cosine_score": float(result["score"]),
            }
        else:
            existing["cosine_score"] = max(
                existing["cosine_score"],
                float(result["score"]),
            )

    if not passages:
        return [], "bm25+cosine" if vector_results else "bm25"

    ranked = list(passages.values())
    lexical_scores = [float(passage["bm25_score"]) for passage in ranked]
    cosine_scores = [float(passage["cosine_score"]) for passage in ranked]
    hybrid_scores = fuse_bm25_and_cosine(lexical_scores, cosine_scores)
    for passage, score in zip(ranked, hybrid_scores, strict=True):
        passage["score"] = score
    ranked.sort(
        key=lambda passage: (
            -passage["score"],
            passage["filename"],
            passage["location_number"],
            passage["chunk_index"],
        )
    )
    return ranked[:limit], "bm25+cosine" if vector_results else "bm25"


def _source_response(source: StudySource) -> dict:
    return {
        "id": source.id,
        "filename": source.filename,
        "content_type": source.content_type,
        "page_count": source.page_count,
        "unit_count": source.unit_count,
        "created_at": source.created_at,
        "source_type": "video" if source.content_type in VIDEO_CONTENT_TYPES.values() else "document",
    }


@router.post("/sources", response_model=StudySourceResponse, status_code=201)
@limiter.limit(RATE_LIMIT_STUDY)
async def upload_study_source(
    request: Request,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    user_id = current_user.id
    filename = (file.filename or "").replace("\\", "/").rsplit("/", 1)[-1]
    filename = "".join(character for character in filename if character >= " " and character != "\x7f").strip()
    if not filename or filename in {".", ".."}:
        raise HTTPException(status_code=400, detail="A valid filename is required")
    if len(filename) > 255:
        raise HTTPException(status_code=400, detail="Filename exceeds the 255-character limit")

    extension = filename.rsplit(".", 1)[-1].casefold() if "." in filename else ""
    expected_content_type = {
        "pdf": PDF_CONTENT_TYPE,
        "pptx": PPTX_CONTENT_TYPE,
        **VIDEO_CONTENT_TYPES,
    }.get(extension)
    if expected_content_type is None:
        raise HTTPException(
            status_code=415,
            detail="Supported course files are PDF, PowerPoint (.pptx), MP4, and WebM.",
        )
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=415, detail="Unsupported content type")
    if file.content_type not in {expected_content_type, "application/octet-stream"}:
        raise HTTPException(status_code=415, detail="File extension and content type do not match")

    storage_path = None
    committed = False
    try:
        file_bytes = await file.read(MAX_SOURCE_BYTES + 1)
        if len(file_bytes) > MAX_SOURCE_BYTES:
            raise HTTPException(status_code=413, detail="The uploaded file exceeds the 25 MB limit")
        checksum = hashlib.sha256(file_bytes).hexdigest()
        existing = await db.execute(
            select(StudySource.id).where(
                StudySource.user_id == user_id,
                StudySource.content_sha256 == checksum,
            )
        )
        if existing.scalar_one_or_none() is not None:
            raise HTTPException(status_code=409, detail="This file is already in your study library")

        try:
            if extension in VIDEO_CONTENT_TYPES:
                content_type, page_count, content_units = await ingest_lecture_video(
                    file_bytes, filename, expected_content_type
                )
            else:
                content_type, page_count, content_units = await anyio.to_thread.run_sync(
                    ingest_course_file, file_bytes, filename
                )
        except StudyMaterialError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except StudyEmbeddingUnavailable as exc:
            logger.exception("Semantic chunking unavailable for study source %s", filename)
            raise HTTPException(
                status_code=503,
                detail="Semantic chunking is temporarily unavailable. Please try again.",
            ) from exc
        except RuntimeError as exc:
            logger.exception("Study source preprocessing failed for %s", filename)
            raise HTTPException(
                status_code=503,
                detail=(
                    "Study embeddings are temporarily unavailable. Please try again."
                    if extension not in VIDEO_CONTENT_TYPES
                    else "Lecture transcription or study embeddings are temporarily unavailable. Please try again."
                ),
            ) from exc

        storage_path = await anyio.to_thread.run_sync(
            upload_file, file_bytes, filename, content_type
        )
        child_count = sum(unit.chunk_level == "child" for unit in content_units)
        source = StudySource(
            user_id=user_id,
            filename=filename[:255],
            content_type=content_type,
            storage_path=storage_path,
            content_sha256=checksum,
            page_count=page_count,
            unit_count=child_count,
        )
        db.add(source)
        await db.flush()
        parent_rows = [
            StudyContentUnit(
                source_id=source.id,
                location_type=unit.location_type,
                location_number=unit.location_number,
                chunk_level="parent",
                chunk_index=unit.chunk_index,
                content=unit.content,
            )
            for unit in content_units
            if unit.chunk_level == "parent"
        ]
        db.add_all(parent_rows)
        await db.flush()
        parents_by_location_index = {
            (parent.location_type, parent.location_number, parent.chunk_index): parent
            for parent in parent_rows
        }
        parents_by_id = {parent.id: parent for parent in parent_rows}
        child_rows = [
            StudyContentUnit(
                source_id=source.id,
                location_type=unit.location_type,
                location_number=unit.location_number,
                chunk_level="child",
                chunk_index=unit.chunk_index,
                parent_unit_id=parents_by_location_index[
                    (
                        unit.location_type,
                        unit.location_number,
                        unit.parent_chunk_index,
                    )
                ].id,
                char_start=unit.char_start,
                char_end=unit.char_end,
                content=unit.content,
            )
            for unit in content_units
            if unit.chunk_level == "child"
        ]
        db.add_all(child_rows)
        await db.flush()
        await db.refresh(source)
        response_payload = _source_response(source)
        response_payload["chunking_strategy"] = "semantic-parent-child+late-chunking"
        response_payload["figure_descriptions"] = sum(
            unit.chunk_level == "parent" and "[Figure " in unit.content
            for unit in content_units
        )
        passages = [
            {
                "id": unit.id,
                "location_type": unit.location_type,
                "location_number": unit.location_number,
                "chunk_index": unit.chunk_index,
                "parent_unit_id": str(unit.parent_unit_id),
                "parent_content": parents_by_id[unit.parent_unit_id].content,
                "char_start": unit.char_start,
                "char_end": unit.char_end,
                "content": unit.content,
            }
            for unit in child_rows
        ]
        await db.commit()
        committed = True
        try:
            indexed_count = await anyio.to_thread.run_sync(
                index_study_passages,
                user_id,
                source.id,
                source.filename,
                passages,
            )
            response_payload["vector_indexed"] = indexed_count == len(passages)
        except Exception:
            logger.exception("Qdrant indexing failed for study source %s", source.id)
            response_payload["vector_indexed"] = False
        try:
            response_payload["graph_indexed"] = await anyio.to_thread.run_sync(
                partial(
                    index_source_passages,
                    user_id=str(user_id),
                    source_id=str(source.id),
                    filename=source.filename,
                    passages=passages,
                )
            )
        except Exception:
            logger.exception("Neo4j graph indexing failed for study source %s", source.id)
            response_payload["graph_indexed"] = False
        return response_payload
    except HTTPException:
        await db.rollback()
        if not committed:
            await _cleanup_storage_path(storage_path)
        raise
    except IntegrityError as exc:
        await db.rollback()
        if not committed:
            await _cleanup_storage_path(storage_path)
        original_error = getattr(exc, "orig", None)
        constraint_name = getattr(original_error, "constraint_name", None)
        if constraint_name is None:
            constraint_name = getattr(getattr(original_error, "diag", None), "constraint_name", None)
        if constraint_name == "uq_study_sources_user_checksum":
            raise HTTPException(status_code=409, detail="This file is already in your study library") from exc
        logger.error("Failed to save study source for user %s", user_id, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to save the study source") from exc
    except Exception as exc:
        await db.rollback()
        if not committed:
            await _cleanup_storage_path(storage_path)
        logger.error("Failed to upload study source for user %s", user_id, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to process the study source") from exc
    finally:
        await file.close()


@router.get("/sources", response_model=list[StudySourceResponse])
async def list_study_sources(
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0, le=100_000),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    result = await db.execute(
        select(StudySource)
        .where(StudySource.user_id == current_user.id)
        .order_by(StudySource.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    return [_source_response(source) for source in result.scalars().all()]


@router.get("/sources/{source_id}/open")
@limiter.limit(RATE_LIMIT_STUDY)
async def open_study_source(
    request: Request,
    source_id: uuid.UUID,
    location_type: str | None = Query(default=None, pattern="^(page|slide|timestamp)$"),
    location_number: int | None = Query(default=None, ge=0),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    source = await db.scalar(
        select(StudySource).where(
            StudySource.id == source_id,
            StudySource.user_id == current_user.id,
        )
    )
    if source is None:
        raise HTTPException(status_code=404, detail="Study source not found")

    if (location_type is None) != (location_number is None):
        raise HTTPException(
            status_code=422,
            detail="Location type and location number must be supplied together",
        )
    if location_type in {"page", "slide"}:
        expected_type = PDF_CONTENT_TYPE if location_type == "page" else PPTX_CONTENT_TYPE
        if source.content_type != expected_type or location_number == 0:
            raise HTTPException(status_code=422, detail="Location does not match this source type")
        if location_number > source.page_count:
            raise HTTPException(status_code=422, detail="Location is beyond the end of this source")
    elif location_type == "timestamp" and source.content_type not in VIDEO_CONTENT_TYPES.values():
        raise HTTPException(status_code=422, detail="Timestamp locations require a video source")

    try:
        signed_url = await anyio.to_thread.run_sync(
            partial(create_signed_url, source.storage_path, 300)
        )
    except Exception as exc:
        logger.exception("Could not create a temporary source URL for %s", source.id)
        raise HTTPException(
            status_code=503,
            detail="The source viewer is temporarily unavailable",
        ) from exc

    url = urljoin(settings.SUPABASE_URL.rstrip("/") + "/", signed_url)
    viewer_supported = False
    if location_number is not None:
        if source.content_type == PDF_CONTENT_TYPE and location_type == "page":
            url = f"{url}#page={location_number}"
            viewer_supported = True
        elif source.content_type in VIDEO_CONTENT_TYPES.values() and location_type == "timestamp":
            url = f"{url}#t={location_number}"
            viewer_supported = True

    return {
        "url": url,
        "expires_in_seconds": 300,
        "viewer_supported": viewer_supported,
        "location_type": location_type,
        "location_number": location_number,
    }


@router.get("/sources/{source_id}/slides/{slide_number}")
@limiter.limit(RATE_LIMIT_STUDY)
async def get_study_source_slide(
    request: Request,
    source_id: uuid.UUID,
    slide_number: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    source = await db.scalar(
        select(StudySource).where(
            StudySource.id == source_id,
            StudySource.user_id == current_user.id,
        )
    )
    if source is None:
        raise HTTPException(status_code=404, detail="Study source not found")
    if source.content_type != PPTX_CONTENT_TYPE:
        raise HTTPException(status_code=415, detail="Slide viewing is only available for PowerPoint sources")
    if slide_number < 1 or slide_number > source.page_count:
        raise HTTPException(status_code=404, detail="Slide not found")

    try:
        file_bytes = await anyio.to_thread.run_sync(download_file, source.storage_path)
        slides = await anyio.to_thread.run_sync(_pptx_slides, file_bytes)
    except (RuntimeError, ValueError) as exc:
        logger.exception("Could not load PowerPoint slide %s from source %s", slide_number, source.id)
        raise HTTPException(
            status_code=503,
            detail="The PowerPoint slide viewer is temporarily unavailable",
        ) from exc

    selected_slide = next(
        (slide for slide in slides if slide[0] == slide_number),
        None,
    )
    if selected_slide is None:
        raise HTTPException(status_code=404, detail="Slide not found in the stored presentation")

    images = []
    total_image_bytes = 0
    for image_bytes in selected_slide[2]:
        preview = await anyio.to_thread.run_sync(_prepare_figure_image, image_bytes)
        total_image_bytes += len(preview)
        if len(preview) > 2 * 1024 * 1024 or total_image_bytes > 6 * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail="Slide image previews exceed the viewer size limit",
            )
        images.append(
            {
                "media_type": "image/jpeg",
                "data": base64.b64encode(preview).decode("ascii"),
            }
        )

    return {
        "source_id": str(source.id),
        "filename": source.filename,
        "slide_number": slide_number,
        "slide_count": source.page_count,
        "text": selected_slide[1],
        "images": images,
    }


@router.get("/search", response_model=StudySearchResponse)
@limiter.limit(RATE_LIMIT_STUDY)
async def search_study_sources(
    request: Request,
    q: str = Query(min_length=2, max_length=500),
    limit: int = Query(default=10, ge=1, le=20),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    candidates, retrieval = await _hybrid_retrieve_study_passages(
        db, current_user.id, q, limit
    )

    return {
        "query": q,
        "retrieval": retrieval,
        "results": [
            {
                "source_id": passage["source_id"],
                "filename": passage["filename"],
                "location_type": passage["location_type"],
                "location_number": passage["location_number"],
                "chunk_index": passage["chunk_index"],
                "score": passage["score"],
                "content": passage["content"],
            }
            for passage in candidates
        ],
    }


@router.get("/tutor", response_model=StudyTutorResponse)
@limiter.limit(RATE_LIMIT_STUDY)
async def tutor_on_study_sources(
    request: Request,
    q: str = Query(min_length=2, max_length=500),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    async def retrieve(query: str) -> tuple[list[dict], str]:
        return await _hybrid_retrieve_study_passages(
            db, current_user.id, query, limit=5
        )

    result = await run_study_tutor(q, retrieve)
    passages = result.get("passages", [])
    return {
        "query": q,
        "answer": result.get("answer", ""),
        "grounded": bool(passages),
        "citations": [
            {
                "source_id": passage["source_id"],
                "filename": passage["filename"],
                "location_type": passage["location_type"],
                "location_number": passage["location_number"],
                "content": passage.get("parent_content") or passage["content"],
            }
            for passage in passages
        ],
        "retrieval_backend": result.get("retrieval_backend", "lexical"),
        "generation_provider": result.get("generation_provider", "source_excerpt"),
    }


@router.get("/conversations")
async def list_study_conversations(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    study_conversation_ids = (
        select(Message.conversation_id)
        .where(Message.metadata_json["workflow"].as_string() == "study_tutor")
        .distinct()
    )
    result = await db.execute(
        select(Conversation)
        .where(Conversation.user_id == current_user.id)
        .where(Conversation.id.in_(study_conversation_ids))
        .order_by(Conversation.created_at.desc(), Conversation.id.desc())
        .limit(50)
    )
    return [
        {
            "id": str(conversation.id),
            "title": conversation.title or "Study conversation",
            "created_at": conversation.created_at,
        }
        for conversation in result.scalars().all()
    ]


@router.get("/conversations/{conversation_id}/messages")
async def get_study_conversation_messages(
    conversation_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    try:
        conversation_uuid = uuid.UUID(conversation_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid conversation_id format") from exc

    result = await db.execute(
        select(Conversation).where(
            Conversation.id == conversation_uuid,
            Conversation.user_id == current_user.id,
        )
    )
    conversation = result.scalar_one_or_none()
    if conversation is None:
        raise HTTPException(status_code=404, detail="Study conversation not found")

    result = await db.execute(
        select(Message)
        .where(Message.conversation_id == conversation.id)
        .order_by(Message.created_at.asc(), Message.id.asc())
        .limit(200)
    )
    return [
        {
            "id": str(message.id),
            "role": message.role,
            "content": message.content,
            "metadata": message.metadata_json or {},
            "created_at": message.created_at,
        }
        for message in result.scalars().all()
    ]


def _sse_event(event_type: str, payload: dict) -> str:
    return f"event: {event_type}\ndata: {json.dumps(payload, default=str)}\n\n"


def _describe_uploaded_image(image_bytes: bytes) -> str:
    prepared_image = _prepare_figure_image(image_bytes)
    if settings.GEMINI_API_KEY.get_secret_value():
        from app.services.vlm_service import describe_study_figure

        try:
            description = describe_study_figure(prepared_image, "image/jpeg").strip()
            if description:
                return description
            logger.warning("Study tutor vision model returned an empty image description")
        except Exception:
            logger.exception("Study tutor vision model failed; falling back to OCR")

    from app.services.ocr_services import extract_text_from_image

    extracted_text = extract_text_from_image(prepared_image)
    if not extracted_text.strip():
        raise RuntimeError(
            "Image understanding is unavailable: configure the vision model or install OCR."
        )
    return f"Readable text in the uploaded image:\n{extracted_text.strip()}"


@router.post("/tutor/transcribe")
@limiter.limit(RATE_LIMIT_STUDY)
async def transcribe_study_tutor_question(
    request: Request,
    audio: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
):
    if audio.content_type not in ALLOWED_TRANSCRIPTION_TYPES:
        raise HTTPException(status_code=400, detail="Unsupported audio or video format")

    audio_bytes = await audio.read(MAX_MULTIMODAL_FILE_SIZE + 1)
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Audio file payload is empty")
    if len(audio_bytes) > MAX_MULTIMODAL_FILE_SIZE:
        raise HTTPException(status_code=413, detail="Audio file exceeds the 25 MB limit")
    if not has_valid_media_signature(audio_bytes, audio.content_type):
        raise HTTPException(
            status_code=400,
            detail="Audio content does not match its declared type",
        )

    try:
        transcript = await transcribe_audio(
            audio_bytes,
            audio.filename or "study-question",
            audio.content_type,
        )
    except RuntimeError as exc:
        logger.exception("Study tutor speech transcription failed")
        raise HTTPException(
            status_code=503,
            detail=(
                "Speech transcription is temporarily unavailable. "
                "Check the configured speech-provider credentials and retry."
            ),
        ) from exc

    transcript = transcript.strip()
    if not transcript:
        raise HTTPException(status_code=422, detail="No speech was detected in the recording.")
    if len(transcript) > 500:
        raise HTTPException(
            status_code=422,
            detail="Transcription exceeds the 500 character question limit.",
        )
    return {"transcript": transcript}


@router.post("/tutor/stream")
@limiter.limit(RATE_LIMIT_STUDY)
async def stream_study_tutor_turn(
    request: Request,
    query: str | None = Form(None, max_length=500),
    conversation_id: str | None = Form(None),
    tutoring_mode: Literal["adaptive", "hints_first", "direct"] = Form("adaptive"),
    audio: UploadFile | None = File(None),
    image: UploadFile | None = File(None),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if not (query and query.strip()) and audio is None:
        raise HTTPException(
            status_code=422,
            detail="Enter a question or provide an audio question.",
        )

    clean_query = query.strip() if query and query.strip() else ""
    transcript = ""
    if audio is not None:
        if audio.content_type not in ALLOWED_TRANSCRIPTION_TYPES:
            raise HTTPException(status_code=400, detail="Unsupported audio or video format")
        audio_bytes = await audio.read(MAX_MULTIMODAL_FILE_SIZE + 1)
        if not audio_bytes:
            raise HTTPException(status_code=400, detail="Audio file payload is empty")
        if len(audio_bytes) > MAX_MULTIMODAL_FILE_SIZE:
            raise HTTPException(status_code=413, detail="Audio file exceeds the 25 MB limit")
        if not has_valid_media_signature(audio_bytes, audio.content_type):
            raise HTTPException(
                status_code=400,
                detail="Audio content does not match its declared type",
            )
        try:
            transcript = await transcribe_audio(
                audio_bytes,
                audio.filename or "study-question",
                audio.content_type,
            )
        except RuntimeError as exc:
            logger.exception("Study tutor speech transcription failed")
            raise HTTPException(
                status_code=503,
                detail="Speech transcription is temporarily unavailable.",
            ) from exc

    visual_context = ""
    if image is not None:
        if image.content_type not in ALLOWED_IMAGE_TYPES:
            raise HTTPException(status_code=400, detail="Unsupported image format")
        image_bytes = await image.read(MAX_MULTIMODAL_FILE_SIZE + 1)
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Image file payload is empty")
        if len(image_bytes) > MAX_MULTIMODAL_FILE_SIZE:
            raise HTTPException(status_code=413, detail="Image file exceeds the 25 MB limit")
        if not has_valid_media_signature(image_bytes, image.content_type):
            raise HTTPException(
                status_code=400,
                detail="Image content does not match its declared type",
            )
        try:
            visual_context = await anyio.to_thread.run_sync(
                _describe_uploaded_image,
                image_bytes,
            )
        except StudyMaterialError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("Study tutor image understanding failed")
            raise HTTPException(
                status_code=503,
                detail="Image understanding is temporarily unavailable.",
            ) from exc

    if not clean_query and not transcript:
        raise HTTPException(status_code=422, detail="The audio did not contain a question.")
    try:
        assert_safe_text(clean_query, "student question")
        assert_safe_text(transcript, "student transcript")
    except SafetyViolation as exc:
        raise HTTPException(status_code=400, detail="Request blocked by safety policy") from exc

    student_question_parts = [part for part in (clean_query, transcript) if part]
    student_question = redact_sensitive_text("\n".join(student_question_parts))
    visual_context = redact_sensitive_text(visual_context)[:5_000]
    if len(student_question) > 5_000:
        raise HTTPException(status_code=422, detail="Question transcript exceeds the 5,000 character limit")
    if not student_question.strip():
        raise HTTPException(status_code=422, detail="Enter a question or provide an audio question.")

    conversation = None
    if conversation_id:
        try:
            conversation_uuid = uuid.UUID(conversation_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Invalid conversation_id format") from exc
        result = await db.execute(
            select(Conversation).where(
                Conversation.id == conversation_uuid,
                Conversation.user_id == current_user.id,
            )
        )
        conversation = result.scalar_one_or_none()
        if conversation is None:
            raise HTTPException(status_code=404, detail="Study conversation not found")
    else:
        conversation = Conversation(
            user_id=current_user.id,
            title=student_question[:100],
        )
        db.add(conversation)
        await db.flush()

    result = await db.execute(
        select(Message)
        .where(Message.conversation_id == conversation.id)
        .order_by(Message.created_at.desc(), Message.id.desc())
        .limit(12)
    )
    previous_messages = list(reversed(result.scalars().all()))
    history: list[dict[str, str]] = []
    for message in previous_messages:
        content = message.content
        metadata = message.metadata_json or {}
        prior_visual = metadata.get("visual_context")
        if message.role == "user" and prior_visual:
            content += f"\n[Image description]\n{prior_visual}"
        history.append({"role": message.role, "content": content})

    practice_result = await db.execute(
        select(
            StudyQuizQuestion.topic,
            StudyQuizQuestion.answered_index == StudyQuizQuestion.answer_index,
            StudyQuizQuestion.answered_at,
        ).where(
            StudyQuizQuestion.user_id == current_user.id,
            StudyQuizQuestion.answered_at.is_not(None),
        )
        .order_by(StudyQuizQuestion.answered_at.desc())
        .limit(200)
    )
    practice_rows = [
        row for row in practice_result.all()
        if row[2] is not None
    ]
    topic_scores = [
        (rank_content(student_question, str(row[0])), str(row[0]))
        for row in practice_rows
    ]
    best_topic_score, relevant_topic = max(topic_scores, default=(0.0, ""))
    if best_topic_score <= 0:
        relevant_topic = ""
    relevant_events = [
        (bool(correct), answered_at)
        for topic, correct, answered_at in practice_rows
        if topic == relevant_topic
    ] if relevant_topic else []
    mastery_estimate = estimate_mastery(relevant_events) if relevant_events else None

    modality = []
    if transcript:
        modality.append("audio")
    if visual_context:
        modality.append("image")
    if not modality:
        modality.append("text")
    db.add(
        Message(
            conversation_id=conversation.id,
            role="user",
            content=student_question,
            metadata_json={
                "workflow": "study_tutor",
                "modalities": modality,
                "transcript": transcript or None,
                "visual_context": visual_context or None,
                "tutoring_mode": tutoring_mode,
                "practice_evidence_count": len(relevant_events),
            },
        )
    )
    await db.commit()

    async def retrieve(retrieval_query: str):
        async with SessionLocal() as retrieval_db:
            return await _hybrid_retrieve_study_passages(
                retrieval_db,
                current_user.id,
                retrieval_query,
                limit=5,
            )

    conversation_id_value = str(conversation.id)

    async def event_stream():
        yield _sse_event(
            "conversation",
            {
                "conversation_id": conversation_id_value,
                "transcript": transcript or None,
                "visual_context": visual_context or None,
            },
        )
        answer_parts: list[str] = []
        final_state: dict = {}
        stream_context: dict = {}
        try:
            async for event in stream_study_tutor(
                student_question,
                retrieve,
                conversation_history=history,
                visual_context=visual_context or None,
                tutoring_mode=tutoring_mode,
                mastery_estimate=mastery_estimate,
            ):
                if event["type"] == "token":
                    answer_parts.append(event["text"])
                elif event["type"] == "replace":
                    answer_parts = [event["text"]] if event["text"] else []
                elif event["type"] == "context":
                    stream_context = event
                elif event["type"] == "complete":
                    final_state = event
                    continue
                yield _sse_event(event["type"], event)

            final_answer = "".join(answer_parts).strip()
            if not final_state or not final_answer:
                raise RuntimeError("Tutor stream ended without a complete answer")
            async with SessionLocal() as persist_db:
                persist_db.add(
                    Message(
                        conversation_id=conversation.id,
                        role="assistant",
                        content=final_answer,
                        metadata_json={
                            "workflow": "study_tutor",
                            "grounded": bool(final_state.get("grounded")),
                            "generation_provider": final_state.get(
                                "generation_provider",
                                "source_excerpt",
                            ),
                            "citations": stream_context.get("passages", []),
                            "retrieval_backend": stream_context.get(
                                "retrieval_backend",
                                "bm25",
                            ),
                            "tutoring_mode": final_state.get(
                                "tutoring_mode",
                                tutoring_mode,
                            ),
                            "mastery_band": final_state.get("mastery_band"),
                        },
                    )
                )
                await persist_db.commit()
        except Exception:
            logger.exception(
                "Study tutor stream failed for conversation %s",
                conversation_id_value,
            )
            yield _sse_event(
                "error",
                {"detail": "The tutor stream failed. Your question has been saved; please retry."},
            )
            return

        yield _sse_event("complete", final_state)
        yield _sse_event(
            "done",
            {
                "conversation_id": conversation_id_value,
                "generation_provider": final_state.get("generation_provider", "source_excerpt"),
                "grounded": bool(final_state.get("grounded")),
                "tutoring_mode": final_state.get("tutoring_mode", tutoring_mode),
                "mastery_band": final_state.get("mastery_band", "cold_start"),
            },
        )

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/quizzes", response_model=StudyQuizResponse, status_code=status.HTTP_201_CREATED)
@limiter.limit(RATE_LIMIT_STUDY)
async def create_study_quiz(
    request: Request,
    payload: StudyQuizCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    candidates = await _retrieve_study_units(db, current_user.id, payload.topic, limit=500)
    distinct_units = []
    seen_ids = set()
    for score, unit, source, _parent_content in candidates:
        if unit.id not in seen_ids:
            distinct_units.append((score, unit, source))
            seen_ids.add(unit.id)
    if not distinct_units:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No matching course passage was found for this topic.",
        )

    used_result = await db.execute(
        select(
            StudyQuizQuestion.source_id,
            StudyQuizQuestion.location_type,
            StudyQuizQuestion.location_number,
        ).where(StudyQuizQuestion.user_id == current_user.id)
    )
    used_passages = {
        (source_id, location_type, location_number)
        for source_id, location_type, location_number in used_result.all()
    }
    distinct_units = [
        candidate
        for candidate in distinct_units
        if (
            candidate[1].source_id,
            candidate[1].location_type,
            candidate[1].location_number,
        )
        not in used_passages
    ][:4]
    if not distinct_units:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="You have already practised all matching passages for this topic.",
        )

    correct_candidate = distinct_units[0]
    all_units_result = await db.execute(
        select(StudyContentUnit, StudySource)
        .join(StudySource, StudyContentUnit.source_id == StudySource.id)
        .where(StudySource.user_id == current_user.id)
        .where(StudyContentUnit.chunk_level == "child")
        .limit(500)
    )
    other_units = [
        (rank_content(payload.topic, unit.content), unit, source)
        for unit, source in all_units_result.all()
        if unit.id != correct_candidate[1].id
    ]
    unrelated_units = [candidate for candidate in other_units if candidate[0] == 0]
    distractors = unrelated_units or sorted(other_units, key=lambda item: item[0])
    option_candidates = [correct_candidate, *distractors[:3]]
    if len(option_candidates) < 2:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Add at least two distinct course passages to create a practice question.",
        )

    secrets.SystemRandom().shuffle(option_candidates)
    answer_index = option_candidates.index(correct_candidate)
    options = [
        {
            "content": unit.content,
            "filename": source.filename,
            "source_id": source.id,
            "location_type": unit.location_type,
            "location_number": unit.location_number,
        }
        for _, unit, source in option_candidates
    ]
    question = StudyQuizQuestion(
        user_id=current_user.id,
        topic=payload.topic.strip(),
        prompt=f"Which passage best supports the topic “{payload.topic.strip()}”?",
        options=options,
        answer_index=answer_index,
        source_id=correct_candidate[2].id,
        location_type=correct_candidate[1].location_type,
        location_number=correct_candidate[1].location_number,
        deduplication_key=hashlib.sha256(
            (
                f"{correct_candidate[2].id}:"
                f"{correct_candidate[1].location_type}:"
                f"{correct_candidate[1].location_number}"
            ).encode("utf-8")
        ).hexdigest(),
    )
    db.add(question)
    try:
        await db.commit()
    except IntegrityError as exc:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="This source passage was just used in another practice question.",
        ) from exc
    await db.refresh(question)
    return {
        "id": question.id,
        "topic": question.topic,
        "prompt": question.prompt,
        "options": question.options,
    }


@router.post("/quizzes/{question_id}/answer", response_model=StudyQuizResponse)
async def answer_study_quiz(
    question_id: uuid.UUID,
    payload: StudyQuizAnswer,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    question = await db.scalar(
        select(StudyQuizQuestion).where(
            StudyQuizQuestion.id == question_id,
            StudyQuizQuestion.user_id == current_user.id,
        )
    )
    if question is None:
        raise HTTPException(status_code=404, detail="Practice question not found")
    if question.answered_index is not None:
        raise HTTPException(status_code=409, detail="This practice question has already been answered")
    if payload.selected_index >= len(question.options):
        raise HTTPException(status_code=422, detail="Selected option does not exist")

    question.answered_index = payload.selected_index
    question.answered_at = datetime.now(timezone.utc)
    await db.commit()
    return {
        "id": question.id,
        "topic": question.topic,
        "prompt": question.prompt,
        "options": question.options,
        "answered_index": question.answered_index,
        "correct": question.answered_index == question.answer_index,
        "explanation": question.options[question.answer_index],
    }


@router.get("/progress", response_model=StudyProgressResponse)
async def get_study_progress(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    source_count = await db.scalar(
        select(func.count(StudySource.id)).where(StudySource.user_id == current_user.id)
    )
    totals = await db.execute(
        select(
            StudyQuizQuestion.topic,
            func.count(StudyQuizQuestion.id),
            func.sum(case((StudyQuizQuestion.answered_index.is_not(None), 1), else_=0)),
            func.sum(
                case(
                    (StudyQuizQuestion.answered_index == StudyQuizQuestion.answer_index, 1),
                    else_=0,
                )
            ),
        )
        .where(StudyQuizQuestion.user_id == current_user.id)
        .group_by(StudyQuizQuestion.topic)
        .order_by(StudyQuizQuestion.topic)
    )
    history_rows = await db.execute(
        select(
            StudyQuizQuestion.topic,
            StudyQuizQuestion.answered_index == StudyQuizQuestion.answer_index,
            StudyQuizQuestion.answered_at,
        )
        .where(
            StudyQuizQuestion.user_id == current_user.id,
            StudyQuizQuestion.answered_index.is_not(None),
            StudyQuizQuestion.answered_at.is_not(None),
        )
        .order_by(StudyQuizQuestion.topic, StudyQuizQuestion.answered_at)
    )
    mastery_history: dict[str, list[tuple[bool, datetime]]] = defaultdict(list)
    for topic, correct, answered_at in history_rows.all():
        mastery_history[topic].append((correct, answered_at))

    topics = []
    total_questions = answered_questions = correct_answers = 0
    for topic, total, answered, correct in totals.all():
        answered = answered or 0
        correct = correct or 0
        mastery = estimate_mastery(mastery_history[topic])
        topics.append(
            {
                "topic": topic,
                "answered": answered,
                "correct": correct,
                "accuracy": correct / answered if answered else 0.0,
                "mastery_estimate": mastery,
                "recommended_review_in_days": (
                    recommended_review_days(mastery) if answered else None
                ),
            }
        )
        total_questions += total
        answered_questions += answered
        correct_answers += correct

    return {
        "total_sources": source_count or 0,
        "total_questions": total_questions,
        "answered_questions": answered_questions,
        "correct_answers": correct_answers,
        "practice_accuracy": (
            correct_answers / answered_questions if answered_questions else 0.0
        ),
        "topics": topics,
    }
