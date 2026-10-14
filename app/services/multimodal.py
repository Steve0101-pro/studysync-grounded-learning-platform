import logging
import asyncio
import time
from langsmith import traceable

# Ensure these point to the new async implementations we created
from app.config import settings
from app.services.gemini_vlm_service import generate_response
from app.services.stt import transcribe_audio
from app.services.tts_services import text_to_speech
from app.services.portkeys_service import portkeys_client
from app.services.safety import assert_safe_text
from app.services.metrics import LATENCY, REQUESTS
from app.services.safety import redact_sensitive_text

logger = logging.getLogger(__name__)

@traceable(
    name="multimodal_voice_vision_pipeline",
    run_type="chain",
)
async def process_multimodal_request(
    audio_bytes: bytes,
    audio_filename: str,
    audio_content_type: str,
    image_bytes: bytes | None,
    image_content_type: str | None,
    conversation_context: list[dict],
    user_prompt: str | None = None,
) -> dict:
    """Orchestrates audio transcription, multimodal vision analysis, and text-to-speech asynchronously."""
    
    workflow_timer = LATENCY.labels("multimodal").time()
    workflow_timer.__enter__()
    try:
        # 1. Gemini transcription
        stt_start = time.time()
        try:
            transcript = await transcribe_audio(
                audio_bytes,
                audio_filename,
                audio_content_type
            )
        except Exception:
            logger.exception("Multimodal STT stage failed")
            raise
        stt_latency_ms = (time.time() - stt_start) * 1000

        assert_safe_text(transcript, "transcript")
        assert_safe_text(user_prompt, "user prompt")
        user_prompt = redact_sensitive_text(user_prompt) if user_prompt else user_prompt
        
        # Track STT call to Portkeys (non-blocking)
        asyncio.create_task(
            portkeys_client.track_llm_call(
                provider="google",
                model=settings.GEMINI_MODEL,
                request_type="speech-to-text",
                input_tokens=0,
                output_tokens=len(transcript.split()),
                latency_ms=stt_latency_ms,
                status="success",
                metadata={"audio_file": audio_filename, "duration_sec": len(audio_bytes) / 16000},
            )
        )

        # 2. Multimodal response generation (Gemini)
        response_start = time.time()
        try:
            response_text = await generate_response(
                transcript=transcript,
                image_bytes=image_bytes,
                image_content_type=image_content_type,
                conversation_context=conversation_context,
                user_prompt=user_prompt,
            )
        except Exception:
            logger.exception("Multimodal Gemini response stage failed")
            raise
        response_latency_ms = (time.time() - response_start) * 1000
        
        # Track Gemini call to Portkeys (non-blocking)
        asyncio.create_task(
            portkeys_client.track_llm_call(
                provider="google",
                model=settings.GEMINI_MODEL,
                request_type="multimodal",
                input_tokens=len(transcript.split()),
                output_tokens=len(response_text.split()),
                latency_ms=response_latency_ms,
                status="success",
                metadata={"has_image": bool(image_bytes), "context_messages": len(conversation_context)},
            )
        )

        # 3. Text-to-speech synthesis (Gemini TTS)
        tts_start = time.time()
        try:
            audio_response = await text_to_speech(
                response_text
            )
        except Exception:
            logger.exception("Multimodal TTS stage failed")
            raise
        tts_latency_ms = (time.time() - tts_start) * 1000
        
        # Track TTS call to Portkeys (non-blocking)
        asyncio.create_task(
            portkeys_client.track_llm_call(
                provider="google",
                model=settings.GEMINI_TTS_MODEL,
                request_type="text-to-speech",
                input_tokens=len(response_text.split()),
                output_tokens=0,
                latency_ms=tts_latency_ms,
                status="success",
                metadata={"audio_duration_sec": len(audio_response) / (16000 * 2)},
            )
        )

        REQUESTS.labels("multimodal", "success").inc()
        return {
            "transcript": redact_sensitive_text(transcript),
            "response_text": response_text,
            "audio": audio_response,
        }

    except Exception as e:
        REQUESTS.labels("multimodal", "error").inc()
        logger.error(f"Error inside multimodal voice-vision pipeline chain: {e}")
        raise
    finally:
        workflow_timer.__exit__(None, None, None)
