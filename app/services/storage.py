import uuid
import logging
import threading
from pathlib import PurePath
from supabase import create_client
from app.config import settings

logger = logging.getLogger(__name__)

supabase_url = settings.SUPABASE_URL
supabase_service_key=settings.SUPABASE_SERVICE_KEY.get_secret_value()
supabase_bucket=settings.SUPABASE_BUCKET_NAME
storage_lock = threading.Lock()


supabase = create_client(
    supabase_url,
    supabase_service_key
)

def upload_file(file_bytes: bytes,file_name:str,content_type:str):
    if not file_bytes:
        raise ValueError(
                "Cannot upload an empty file"
            )
    
    if not file_name:
        file_name = "uploaded_file"
    
    if not content_type:
        content_type = "application/octet-stream"
        
    safe_name = PurePath(file_name).name.replace("\x00", "")[:120]
    if not safe_name or safe_name in {".", ".."}:
        safe_name = "uploaded_file"
    upload_filename = f"{uuid.uuid4()}_{safe_name}"
    
    try:
        # The synchronous Supabase client is shared by async worker threads.
        # Serialize requests because its underlying client is not thread-safe.
        with storage_lock:
            supabase.storage.from_(supabase_bucket).upload(
                path=upload_filename,
                file=file_bytes,
                file_options={"content-type": content_type},
            )
    except Exception as exc:
        logger.error(
            "Supabase upload failed for bucket %s and content type %s: %s",
            supabase_bucket,
            content_type,
            exc,
        )
        raise RuntimeError("Supabase storage upload failed") from exc
    return upload_filename


def delete_file(storage_path: str) -> None:
    """Remove an object from the configured storage bucket."""
    if storage_path:
        with storage_lock:
            supabase.storage.from_(supabase_bucket).remove([storage_path])


def download_file(storage_path: str) -> bytes:
    """Download an object from the configured private storage bucket."""
    if not storage_path:
        raise ValueError("Cannot download an object with an empty storage path")
    with storage_lock:
        response = supabase.storage.from_(supabase_bucket).download(storage_path)
    if not isinstance(response, bytes) or not response:
        raise RuntimeError("Storage provider returned an empty or invalid object")
    return response


def create_signed_url(storage_path: str, expires_in: int) -> str:
    """Create a short-lived URL for an authenticated, owner-checked source."""
    if not storage_path:
        raise ValueError("Cannot create a URL for an empty storage path")
    with storage_lock:
        response = supabase.storage.from_(supabase_bucket).create_signed_url(
            storage_path,
            expires_in,
        )
    signed_url = response.get("signedURL") or response.get("signedUrl")
    if not isinstance(signed_url, str) or not signed_url:
        raise RuntimeError("Storage provider did not return a signed URL")
    return signed_url
