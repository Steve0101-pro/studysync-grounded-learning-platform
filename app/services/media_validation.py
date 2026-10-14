"""Shared upload limits and content-signature checks for multimodal inputs."""

MAX_MULTIMODAL_FILE_SIZE = 25 * 1024 * 1024

ALLOWED_AUDIO_TYPES = {
    "audio/mpeg",
    "audio/mp3",
    "audio/mp4",
    "audio/m4a",
    "audio/wav",
    "audio/x-wav",
    "audio/ogg",
    "audio/webm",
}
ALLOWED_TRANSCRIPTION_TYPES = ALLOWED_AUDIO_TYPES | {"video/mp4", "video/webm"}

ALLOWED_IMAGE_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
}


def has_valid_media_signature(file_bytes: bytes, content_type: str) -> bool:
    """Reject payloads whose leading bytes do not match the declared media type."""
    signatures = {
        "audio/mpeg": file_bytes.startswith((b"ID3", b"\xff\xfb", b"\xff\xf3", b"\xff\xf2")),
        "audio/mp3": file_bytes.startswith((b"ID3", b"\xff\xfb", b"\xff\xf3", b"\xff\xf2")),
        "audio/wav": file_bytes.startswith(b"RIFF") and file_bytes[8:12] == b"WAVE",
        "audio/x-wav": file_bytes.startswith(b"RIFF") and file_bytes[8:12] == b"WAVE",
        "audio/ogg": file_bytes.startswith(b"OggS"),
        "audio/webm": file_bytes.startswith(b"\x1a\x45\xdf\xa3"),
        "audio/mp4": b"ftyp" in file_bytes[:16],
        "audio/m4a": b"ftyp" in file_bytes[:16],
        "video/mp4": b"ftyp" in file_bytes[:16],
        "video/webm": file_bytes.startswith(b"\x1a\x45\xdf\xa3"),
        "image/jpeg": file_bytes.startswith(b"\xff\xd8\xff"),
        "image/jpg": file_bytes.startswith(b"\xff\xd8\xff"),
        "image/png": file_bytes.startswith(b"\x89PNG\r\n\x1a\n"),
        "image/webp": file_bytes.startswith(b"RIFF") and file_bytes[8:12] == b"WEBP",
    }
    return signatures.get(content_type, False)
