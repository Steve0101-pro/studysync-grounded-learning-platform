from app.api.learning._shared import router_for_paths

router = router_for_paths(
    {
        "/study/search",
        "/study/tutor",
        "/study/tutor/transcribe",
        "/study/tutor/stream",
        "/study/conversations",
        "/study/conversations/{conversation_id}/messages",
    }
)
