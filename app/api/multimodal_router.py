import uuid
import asyncio
import base64
import hashlib
import logging
import anyio
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

# Core application imports (Adjust paths to match your layout exactly)
from app.database import get_db
from app.dependencies.auth import get_current_user
from app.models import Conversation, Message, ResponseFeedback, User
from app.schema import MultimodalResponse
from app.services.multimodal import process_multimodal_request
from app.api.redis_service import create_cache_key, get_cache, set_cache, invalidate_cache_pattern
from app.services.storage import upload_file
from app.services.safety import SafetyViolation
from app.services.experiments import assign_variant
from app.services.safety import redact_sensitive_text
from app.services.online_eval import evaluate_response
from app.services.audit import record_audit
from app.config import settings
from app.rate_limit import limiter, RATE_LIMIT_MULTIMODAL
from app.services.media_validation import (
    ALLOWED_AUDIO_TYPES,
    ALLOWED_IMAGE_TYPES,
    MAX_MULTIMODAL_FILE_SIZE,
    has_valid_media_signature as _has_valid_signature,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/multimodal", tags=["Multimodal"])


def _normalize_cache_text(value: str | None) -> str:
    """Normalize user-entered prompts so harmless whitespace changes still hit cache."""
    return " ".join((value or "").split()).casefold()


@router.get("/conversations")
async def list_conversations(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """List conversation IDs available to the authenticated user."""
    result = await db.execute(
        select(Conversation)
        .where(Conversation.user_id == current_user.id)
        .order_by(Conversation.created_at.desc())
    )
    conversations = result.scalars().all()
    return [
        {
            "id": str(conversation.id),
            "title": conversation.title,
            "created_at": conversation.created_at,
        }
        for conversation in conversations
    ]

@router.post("/voice-image", response_model=MultimodalResponse)
@limiter.limit(RATE_LIMIT_MULTIMODAL)
async def voice_image_pipeline(
    request: Request,
    audio: UploadFile = File(...),
    image: UploadFile | None = File(None),
    conversation_id: str | None = Form(None),
    user_prompt: str | None = Form(None),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    # ----------------------------------------------------------------
    # 1. Validate Audio Input
    # ----------------------------------------------------------------
    if audio.content_type not in ALLOWED_AUDIO_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio format: {audio.content_type}",
        )

    audio_bytes = await audio.read(MAX_MULTIMODAL_FILE_SIZE + 1)
    if not audio_bytes:
        raise HTTPException(
            status_code=400,
            detail="Audio file payload is empty",
        )
    if len(audio_bytes) > MAX_MULTIMODAL_FILE_SIZE:
        raise HTTPException(status_code=413, detail="Audio file exceeds the 25MB limit")
    if not _has_valid_signature(audio_bytes, audio.content_type):
        raise HTTPException(status_code=400, detail="Audio content does not match its declared type")

    # ----------------------------------------------------------------
    # 2. Validate Image Input (Optional)
    # ----------------------------------------------------------------
    image_bytes = None
    image_content_type = None

    if image:
        if image.content_type not in ALLOWED_IMAGE_TYPES:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported image format: {image.content_type}",
            )

        image_bytes = await image.read(MAX_MULTIMODAL_FILE_SIZE + 1)
        if not image_bytes:
            raise HTTPException(
                status_code=400,
                detail="Image file payload is empty",
            )
        if len(image_bytes) > MAX_MULTIMODAL_FILE_SIZE:
            raise HTTPException(status_code=413, detail="Image file exceeds the 25MB limit")
        if not _has_valid_signature(image_bytes, image.content_type):
            raise HTTPException(status_code=400, detail="Image content does not match its declared type")
        image_content_type = image.content_type

    # ----------------------------------------------------------------
    # 3. Fetch or Initialize Conversation Session
    # ----------------------------------------------------------------
    if conversation_id:
        try:
            conversation_uuid = uuid.UUID(conversation_id)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail="Invalid conversation_id format",
            )

        result = await db.execute(
            select(Conversation).where(
                Conversation.id == conversation_uuid,
                Conversation.user_id == current_user.id,
            )
        )
        conversation = result.scalar_one_or_none()
        
        if not conversation:
            raise HTTPException(
                status_code=404,
                detail="Requested conversation session not found",
            )
    else:
        # Create a placeholder title; will update with transcript later if available
        conversation = Conversation(
            user_id=current_user.id,
            title=(user_prompt or "Voice conversation")[:255],
        )
        db.add(conversation)
        await db.flush()  # Populates conversation.id immediately

    # ----------------------------------------------------------------
    # 4. Load Slidewindow Conversation Context
    # ----------------------------------------------------------------
    result = await db.execute(
        select(Message)
        .where(Message.conversation_id == conversation.id)
        .order_by(Message.created_at.desc())
        .limit(20)
    )
    messages = list(reversed(result.scalars().all()))
    
    conversation_context = [
        {"role": message.role, "content": message.content}
        for message in messages
    ]

    # ----------------------------------------------------------------
    # 5. Process Fingerprint Cache Key (Redis Verification)
    # ----------------------------------------------------------------
    exact_cache_key = create_cache_key(
        str(current_user.id),
        str(conversation.id),
        audio.filename or "",
        hashlib.sha256(audio_bytes).hexdigest(),
        user_prompt or "",
        image.filename if image else "",
        hashlib.sha256(image_bytes).hexdigest() if image_bytes else "",
    )

    cache_keys = [exact_cache_key]
    if user_prompt and user_prompt.strip():
        cache_keys.insert(
            0,
            create_cache_key(
                str(current_user.id),
                "prompt",
                _normalize_cache_text(user_prompt),
                hashlib.sha256(image_bytes).hexdigest() if image_bytes else "",
            ),
        )

    cached_response = None
    for cache_key in cache_keys:
        cached_response = await get_cache(cache_key)
        if cached_response:
            break
    if cached_response:
        return MultimodalResponse(**cached_response)

    # ----------------------------------------------------------------
    # 6. Execute Async Multimodal Core Pipeline Tasks
    # ----------------------------------------------------------------
    try:
        pipeline_result = await process_multimodal_request(
            audio_bytes=audio_bytes,
            audio_filename=audio.filename,
            audio_content_type=audio.content_type,
            image_bytes=image_bytes,
            image_content_type=image_content_type,
            conversation_context=conversation_context,
            user_prompt=user_prompt,
        )
    except SafetyViolation as e:
        logger.warning("Blocked unsafe multimodal request: %s", e)
        raise HTTPException(status_code=400, detail="Request blocked by safety policy")
    except Exception as e:
        logger.exception("Multimodal backend execution failed")
        raise HTTPException(
            status_code=500,
            detail="Internal processing error within AI core services",
        )

    transcript = pipeline_result["transcript"]
    response_text = pipeline_result["response_text"]
    generated_audio = pipeline_result["audio"]
    safe_transcript = redact_sensitive_text(transcript)
    safe_response_text = redact_sensitive_text(response_text)
    experiment_variant = assign_variant(str(current_user.id))
    evaluation = await evaluate_response(
        request_text=redact_sensitive_text(
            f"{transcript}\n{user_prompt or ''}"
        ),
        response_text=safe_response_text,
        context=redact_sensitive_text(
            "\n".join(message["content"] for message in conversation_context)
        ),
        sample_key=f"{current_user.id}:{conversation.id}:{experiment_variant}",
        variant=experiment_variant,
    )

    await record_audit(
        db,
        action="online_evaluation.completed",
        resource_type="multimodal_response",
        actor_id=current_user.id,
        resource_id=str(conversation.id),
        metadata={
            "status": evaluation["status"],
            "scores": evaluation.get("scores"),
            "variant": experiment_variant,
            "model_version": settings.GEMINI_MODEL,
            "prompt_version": settings.PROMPT_VERSION,
        },
    )

    scores = evaluation.get("scores") or {}
    if scores.get("safety", 5) <= 2 or scores.get("correctness", 5) <= 2:
        db.add(
            ResponseFeedback(
                user_id=current_user.id,
                conversation_id=conversation.id,
                rating=None,
                comment="Automatically escalated by online evaluation.",
                model_version=settings.GEMINI_MODEL,
                prompt_version=settings.PROMPT_VERSION,
                review_status="escalated",
            )
        )

    # Dynamic fallback: Set descriptive title using freshly generated transcript
    if not conversation_id and not user_prompt:
        conversation.title = transcript[:255]

    # ----------------------------------------------------------------
    # 7. Persist Dialogue History Records
    # ----------------------------------------------------------------
    user_content = f"{transcript}\n{user_prompt}" if user_prompt else transcript
    user_message = Message(
        conversation_id=conversation.id,
        role="user",
        content=redact_sensitive_text(user_content),
    )
    db.add(user_message)

    assistant_message = Message(
        conversation_id=conversation.id,
        role="assistant",
        content=safe_response_text,
    )
    db.add(assistant_message)

    # ----------------------------------------------------------------
    # 8. Dispatch Concurrent Storage Uploads via AnyIO Threads
    # ----------------------------------------------------------------
    # Converts synchronous I/O functions into non-blocking AnyIO task coroutines
    audio_upload_task = anyio.to_thread.run_sync(
        upload_file, audio_bytes, audio.filename, audio.content_type
    )
    
    generated_audio_filename = f"response_{uuid.uuid4()}.wav"
    generated_audio_upload_task = anyio.to_thread.run_sync(
        upload_file, generated_audio, generated_audio_filename, "audio/wav"
    )

    if image_bytes and image:
        image_upload_task = anyio.to_thread.run_sync(
            upload_file, image_bytes, image.filename, image.content_type
        )
        # Execute all three file writing streams concurrently without blocking the loop
        uploaded_paths = await asyncio.gather(
            audio_upload_task, 
            generated_audio_upload_task, 
            image_upload_task
        )
    else:
        # Execute only the two primary audio pipelines
        uploaded_paths = await asyncio.gather(
            audio_upload_task, 
            generated_audio_upload_task
        )

    # Extract the storage reference path index returned from the upload tasks
    generated_audio_storage_path = uploaded_paths[1]

    # ----------------------------------------------------------------
    # 9. Commit Working DB Transactions
    # ----------------------------------------------------------------
    await db.commit()

    # ----------------------------------------------------------------
    # 10. Construct Final Response Payload & Populate Redis Cache
    # ----------------------------------------------------------------
    response_payload = {
        "conversation_id": conversation.id,
        "transcript": safe_transcript,
        "response_text": safe_response_text,
        "audio_storage_path": generated_audio_storage_path,
        "audio_content_type": "audio/wav",
        "audio_base64": base64.b64encode(generated_audio).decode("ascii"),
        "experiment_variant": experiment_variant,
    }

    # Format serialization compatible data variant for Redis text insertion
    cache_payload = {
        **response_payload, 
        "conversation_id": str(conversation.id)
    }
    for cache_key in cache_keys:
        await set_cache(cache_key, cache_payload, ttl=600)

    return MultimodalResponse(**response_payload)
