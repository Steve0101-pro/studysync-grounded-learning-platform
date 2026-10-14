import logging
from fastapi import Depends, HTTPException, APIRouter, UploadFile, File, Form, Path, Request
from uuid import UUID
from app.database import get_db
from app.models import Document
from app.schema import DocumentResponse
from app.services.ocr_services import extract_text_from_image, extract_layout_from_image
from app.services.storage import upload_file, delete_file
from app.services.vlm_service import extract_invoice_via_vlm
from sqlalchemy.ext.asyncio import AsyncSession 
import anyio
from app.services.extracted_service import extract_structured_data
from app.services.safety import assert_safe_text, redact_sensitive_data
from app.dependencies.auth import get_current_user
from app.models import User
from app.api.redis_service import invalidate_cache_pattern
from app.rate_limit import limiter, RATE_LIMIT_DOCUMENTS
from sqlalchemy import select

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["Documents"])

ALLOWED_CONTENT_TYPES = {
    "image/png",
    "image/jpeg",
    "image/jpg",
}

MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB


def _has_document_signature(file_bytes: bytes, content_type: str) -> bool:
    if content_type in {"image/jpeg", "image/jpg"}:
        return file_bytes.startswith(b"\xff\xd8\xff")
    if content_type == "image/png":
        return file_bytes.startswith(b"\x89PNG\r\n\x1a\n")
    return False


async def _cleanup_storage_path(storage_path: str |  None) -> None:
    if not storage_path:
        return
    try:
        await anyio.to_thread.run_sync(delete_file, storage_path)
    except Exception:
        logger.warning("Failed to clean up storage object", exc_info=True)


@router.post("/", response_model=DocumentResponse)
@limiter.limit(RATE_LIMIT_DOCUMENTS)
async def upload_process_document(
    request: Request,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Upload and process a document with OCR.
    
    - **file**: Image file (PNG, JPEG)
    - Returns: Extracted text and layout data
    """
    storage_path = None
    try:
        # Validate file type
        if file.content_type not in ALLOWED_CONTENT_TYPES:
            logger.warning(f"Invalid content type uploaded: {file.content_type} by user {current_user.id}")
            raise HTTPException(
                status_code=400,
                detail="Only PNG and JPEG images are supported",
            )
        
        # Read file
        file_bytes = await file.read(MAX_FILE_SIZE + 1)
        
        # Validate file size
        if len(file_bytes) > MAX_FILE_SIZE:
            logger.warning(f"File too large: {len(file_bytes)} bytes")
            raise HTTPException(
                status_code=413,
                detail=f"File size exceeds maximum allowed size of {MAX_FILE_SIZE / 1024 / 1024:.0f}MB",
            )
        
        if not file_bytes:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty",
            )
        if not _has_document_signature(file_bytes, file.content_type):
            raise HTTPException(
                status_code=400,
                detail="Document content does not match its declared type",
            )

        logger.info(f"Processing document: {file.filename} for user {current_user.id}")

        # Offload blocking operations to thread pool
        extracted_text = await anyio.to_thread.run_sync(extract_text_from_image, file_bytes)
        assert_safe_text(extracted_text, "document text")
        layout_data = await anyio.to_thread.run_sync(extract_layout_from_image, file_bytes)
        structured_data = await anyio.to_thread.run_sync(extract_structured_data, extracted_text)
        storage_path = await anyio.to_thread.run_sync(
            upload_file, file_bytes, file.filename, file.content_type
        )
        
        combined_metadata = {
            "raw_layout": layout_data,
            "invoice_data": structured_data
        }
        
        document = Document(
            filename=file.filename,
            content_type=file.content_type,
            storage_path=storage_path,
            extracted_text=redact_sensitive_data(extracted_text),
            layout_data=redact_sensitive_data(combined_metadata),
            user_id=current_user.id,
        )
        
        db.add(document)
        await db.commit()
        await db.refresh(document)
        
        logger.info(f"Document processed successfully: {document.id}")
        return document
    
    except HTTPException:
        await _cleanup_storage_path(storage_path)
        raise
    except Exception as e:
        await db.rollback()
        await _cleanup_storage_path(storage_path)
        logger.error(f"Error processing document: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Failed to process document",
        )


@router.post("/vlm", response_model=DocumentResponse)
@limiter.limit(RATE_LIMIT_DOCUMENTS)
async def upload_process_document_vlm(
    request: Request,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Upload and process a document with Vision Language Model.
    
    - **file**: Image file (PNG, JPEG)
    - Returns: VLM-extracted structured data
    """
    storage_path = None
    try:
        # Validate file type
        if file.content_type not in ALLOWED_CONTENT_TYPES:
            logger.warning(f"Invalid content type for VLM: {file.content_type}")
            raise HTTPException(
                status_code=400,
                detail="Only PNG and JPEG images are supported"
            )
        
        # Read and validate file
        file_bytes = await file.read()
        
        if len(file_bytes) > MAX_FILE_SIZE:
            logger.warning(f"VLM file too large: {len(file_bytes)} bytes")
            raise HTTPException(
                status_code=413,
                detail=f"File size exceeds maximum allowed size of {MAX_FILE_SIZE / 1024 / 1024:.0f}MB",
            )
        
        if not file_bytes:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty",
            )

        logger.info(f"Processing document with VLM: {file.filename} for user {current_user.id}")

        if not _has_document_signature(file_bytes, file.content_type):
            raise HTTPException(status_code=400, detail="Document content does not match its declared type")

        vlm_payload = await anyio.to_thread.run_sync(
            extract_invoice_via_vlm, file_bytes, file.content_type
        )

        storage_path = await anyio.to_thread.run_sync(
            upload_file, file_bytes, file.filename, file.content_type
        )

        combined_metadata = {
            "raw_layout": [],
            "invoice_data": vlm_payload.get("invoice_data")
        }

        document = Document(
            filename=file.filename,
            content_type=file.content_type,
            storage_path=storage_path,
            extracted_text=redact_sensitive_data(vlm_payload.get("extracted_text")),
            layout_data=redact_sensitive_data(combined_metadata),
            user_id=current_user.id,
        )
        
        db.add(document)
        await db.commit()
        await db.refresh(document)
        
        logger.info(f"VLM document processed successfully: {document.id}")
        return document
    
    except HTTPException:
        await _cleanup_storage_path(storage_path)
        raise
    except Exception as e:
        await db.rollback()
        await _cleanup_storage_path(storage_path)
        logger.error(f"Error processing document with VLM: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Failed to process document with VLM",
        )


@router.delete("/{document_id}", status_code=204)
async def delete_document(
    document_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Delete a document by ID. Only the owner can delete their documents.
    Also invalidates associated cache.
    """
    try:
        # Query and verify ownership
        stmt = select(Document).where(Document.id == document_id)
        result = await db.execute(stmt)
        document = result.scalar_one_or_none()
        
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        if document.user_id != current_user.id:
            logger.warning(f"User {current_user.id} attempted to delete document {document_id} owned by {document.user_id}")
            raise HTTPException(status_code=403, detail="Not authorized to delete this document")
        
        await db.delete(document)
        await db.commit()
        
        # Invalidate cache for this user's conversations
        deleted_keys = await invalidate_cache_pattern(
            f"multimodal:{current_user.id}:*"
        )
        logger.info(f"Deleted document {document_id} and invalidated {deleted_keys} cache keys")
        
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        logger.error(f"Error deleting document {document_id}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Failed to delete document",
        )


@router.post("/cache/invalidate", status_code=200)
async def invalidate_cache(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Manually invalidate all cache entries. Useful for forcing fresh processing.
    This is an admin-like operation that affects the entire cache layer.
    """
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")

    try:
        deleted_keys = await invalidate_cache_pattern("multimodal:*")
        logger.info(f"User {current_user.id} invalidated cache; removed {deleted_keys} keys")
        return {
            "message": "Cache invalidated successfully",
            "keys_removed": deleted_keys,
        }
    except Exception as e:
        logger.error(f"Error invalidating cache: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Failed to invalidate cache",
        )
